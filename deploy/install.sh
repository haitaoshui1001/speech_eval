#!/usr/bin/env bash
# 演讲评分系统 —— Ubuntu 22.04 一键安装（systemd + nginx，单进程）
#
# 用法（在 /opt/speech-assist 下执行）：
#   sudo bash deploy/install.sh
#   sudo ADMIN_PASSWORD='你的管理员口令' bash deploy/install.sh
#   sudo bash deploy/install.sh --listen-port 8080
#   sudo bash deploy/install.sh --no-nginx            # 不要反代，让 uvicorn 直接监听公网端口
#
# 幂等：重复执行不会覆盖已有 .env、不会删 data/、不会重建 venv（只补装依赖）。
# 完整流程见 deploy/部署手册.md。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

APP_DIR="/opt/speech-assist"
SERVICE="speech-assist"
RUN_USER="speech"
LISTEN_PORT="80"
APP_PORT="8000"
BIND_HOST="127.0.0.1"
USE_NGINX="1"
WITH_SWAP="1"
SKIP_DEPS="0"

log()  { printf '\033[1;32m[安装]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[注意]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[中止]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
用法：sudo bash deploy/install.sh [选项]
  --listen-port N     nginx 对外端口（默认 80）
  --app-port N        uvicorn 本机端口（默认 8000；--no-nginx 时等于 --listen-port）
  --app-dir PATH      安装目录（默认 /opt/speech-assist）
  --admin-user NAME   管理员账号（默认沿用 .env 的 admin）
  --admin-password PW 管理员口令（省略则随机生成并打印一次）
  --no-nginx          不装反代配置，uvicorn 直接监听 0.0.0.0
  --no-swap           不创建 swap
  --skip-deps         不执行 apt-get install
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --listen-port)    LISTEN_PORT="$2"; shift 2 ;;
        --app-port)       APP_PORT="$2"; shift 2 ;;
        --app-dir)        APP_DIR="$2"; shift 2 ;;
        --admin-user)     ADMIN_USERNAME_ARG="$2"; export ADMIN_USERNAME_ARG; shift 2 ;;
        --admin-password) ADMIN_PASSWORD_ARG="$2"; export ADMIN_PASSWORD_ARG; shift 2 ;;
        --no-nginx)       USE_NGINX="0"; shift ;;
        --no-swap)        WITH_SWAP="0"; shift ;;
        --skip-deps)      SKIP_DEPS="1"; shift ;;
        -h|--help)        usage; exit 0 ;;
        *)                usage; die "未知参数：$1" ;;
    esac
done

if [ "$USE_NGINX" = "0" ]; then
    BIND_HOST="0.0.0.0"
    APP_PORT="$LISTEN_PORT"
fi

[ "$(id -u)" = "0" ] || die "请用 root 或 sudo 运行：sudo bash deploy/install.sh"

if [ -r /etc/os-release ]; then
    . /etc/os-release
    case "${ID:-}" in
        ubuntu|debian) log "发行版：${PRETTY_NAME:-$ID}" ;;
        *) warn "发行版 ${ID:-unknown} 未经验证，脚本按 Ubuntu/Debian 的包名与路径写。" ;;
    esac
fi

command -v python3 >/dev/null 2>&1 || die "找不到 python3。"

# ---------------------------------------------------------------- 同步代码到 APP_DIR
if [ "$(readlink -f "$REPO_DIR")" != "$(readlink -f "$APP_DIR")" ]; then
    log "同步代码：$REPO_DIR -> $APP_DIR"
    mkdir -p "$APP_DIR"
    # .env / api_key.txt / data / .venv 被 exclude，因此既是"不覆盖"也是"不被 --delete 误删"。
    rsync -a --delete \
        --exclude '/data/' --exclude '/.venv/' --exclude '/.env' --exclude '/api_key.txt' \
        --exclude '/__pycache__/' --exclude '/.git/' --exclude '*.pyc' --exclude '*.log' \
        --exclude 'logs_*.txt' --exclude '/提示词.json' \
        "$REPO_DIR/" "$APP_DIR/"
fi
[ -d "$APP_DIR/app" ] || die "$APP_DIR/app 不存在：发布包解压位置不对，或包不完整。"
[ -f "$APP_DIR/评价标准.txt" ] || die "缺少 评价标准.txt —— 评分标准文件名硬编码在应用根目录，不能靠环境变量改。"

# ---------------------------------------------------------------- 系统依赖
if [ "$SKIP_DEPS" = "0" ]; then
    export DEBIAN_FRONTEND=noninteractive
    log "apt 安装运行依赖（python3-venv / ffmpeg / nginx / rsync ...）"
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends \
        python3 python3-venv python3-pip python3-dev \
        ffmpeg nginx rsync curl ca-certificates openssl tar
else
    warn "--skip-deps：跳过 apt 安装，需自行保证 ffmpeg / nginx / python3-venv 可用。"
fi

PIP_HOST="mirrors.aliyun.com/pypi/simple"
if getent hosts mirrors.cloud.aliyuncs.com >/dev/null 2>&1; then
    PIP_HOST="mirrors.cloud.aliyuncs.com/pypi/simple"
fi
log "pip 源：https://$PIP_HOST/（阿里云内网源不占公网带宽）"

# ---------------------------------------------------------------- swap（2GiB 内存的保险）
if [ "$WITH_SWAP" = "1" ]; then
    mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
    if swapon --noheadings 2>/dev/null | grep -q .; then
        log "已有 swap，跳过。"
    elif [ "$mem_kb" -lt 4000000 ]; then
        log "内存 $((mem_kb / 1024)) MB 且无 swap：创建 2G /swapfile"
        rm -f /swapfile
        fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
        chmod 600 /swapfile
        mkswap /swapfile >/dev/null
        swapon /swapfile
        grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
        if [ -d /etc/sysctl.d ]; then
            echo 'vm.swappiness=10' >| /etc/sysctl.d/99-speech-swap.conf
            sysctl --quiet --load /etc/sysctl.d/99-speech-swap.conf || true
        fi
    else
        log "内存充足（$((mem_kb / 1024)) MB），不建 swap。"
    fi
fi

# ---------------------------------------------------------------- 运行账号
getent group "$RUN_USER" >/dev/null || groupadd --system "$RUN_USER"
id "$RUN_USER" >/dev/null 2>&1 || \
    useradd --system --gid "$RUN_USER" --home-dir "$APP_DIR" --shell /usr/sbin/nologin "$RUN_USER"
log "运行账号：$RUN_USER"

# ---------------------------------------------------------------- venv + Python 依赖
VENV="$APP_DIR/.venv"
if [ ! -x "$VENV/bin/python" ]; then
    log "创建虚拟环境 $VENV"
    python3 -m venv "$VENV"
fi
"$VENV/bin/python" - <<'PY' || die "Python 版本过低：需要 >= 3.10。Ubuntu 20.04 自带 3.8，请换 22.04 镜像。"
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY
log "安装 requirements（fastapi / uvicorn / openpyxl / opencv ...）"
"$VENV/bin/pip" install --quiet --disable-pip-version-check --no-input \
    --index-url "https://$PIP_HOST/" -r "$APP_DIR/requirements.txt"
log "Python：$("$VENV/bin/python" -V 2>&1)   uvicorn：$("$VENV/bin/pip" show uvicorn 2>/dev/null | awk '/^Version/{print $2}')"

# ---------------------------------------------------------------- .env
ENV_TARGET="$APP_DIR/.env"
CREATED_ENV="0"
if [ ! -f "$ENV_TARGET" ]; then
    install -m 600 "$SCRIPT_DIR/env.production" "$ENV_TARGET"
    CREATED_ENV="1"
    log "生成 $ENV_TARGET（模板 deploy/env.production，已按 2C2G 收紧并发）"
else
    log "已存在 $ENV_TARGET：保留你改过的值，只补占位符"
fi

GEN_INFO="$("$VENV/bin/python" - "$ENV_TARGET" <<'PY'
"""把占位符换成真实值；已经是真值的行一律不动（注释与顺序保持不变）。"""
import os
import re
import secrets
import string
import sys
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_bytes().decode("utf-8")
newline = "\r\n" if "\r\n" in raw else "\n"
lines = raw.replace("\r\n", "\n").split("\n")

rnd = secrets.SystemRandom()
gen_pw = "".join(rnd.choices(string.ascii_lowercase + string.digits, k=12))
want_pw = os.environ.get("ADMIN_PASSWORD_ARG", "").strip()
want_user = os.environ.get("ADMIN_USERNAME_ARG", "").strip()
notes = []
pw_applied = False
user_applied = False

for i, line in enumerate(lines):
    m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line)
    if not m:
        continue
    key, val = m.group(1), m.group(2)
    if key == "SECRET_KEY" and (not val.strip() or "__GENERATE__" in val or "change-me" in val):
        lines[i] = "SECRET_KEY=" + secrets.token_hex(32)
        notes.append("SECRET_KEY 已生成随机值（以后改它会让所有人重新登录）")
    elif key == "ADMIN_USERNAME" and want_user:
        # 只填模板默认值。已有真实用户名时不覆盖：.env 里的名字改了，库里的账号不会跟着改，
        # 重启后 init_db 找不到同名用户，于是**多建一个管理员**而不是改名。
        if not val.strip() or "__SET_ME__" in val or val.strip() in ("admin", want_user):
            lines[i] = "ADMIN_USERNAME=" + want_user
            notes.append("ADMIN_USERNAME 已设为 " + want_user)
            user_applied = True
        else:
            notes.append("__WARN__=--admin-user 未生效：%s 里已是 ADMIN_USERNAME=%s；"
                         "改名请登录后到 /settings 改（直接改 .env 会多出一个管理员，不是改名）" % (path.name, val))
    elif key == "ADMIN_PASSWORD" and (not val.strip() or "__SET_ME__" in val or val.strip() == "admin123"):
        # 上面这些条件决定"要不要写"，下面这块决定"怎么写才不会被 dotenv 读回去时变形"（均为实测结论）：
        #   'x#y' 原样保留 / 'x #y' 与 'x<TAB>#y' 被当行内注释截成 'x' /
        #   双引号会被剥掉（"x #y" -> x #y）但内部转义会生效（\n 变换行）/
        #   值里有 " 且不闭合（"x #y）→ 整行解析失败，键直接消失，口令退回默认值。
        chosen = want_pw or gen_pw
        inline_hash = " #" in chosen or "\t#" in chosen
        serializable = not re.search(r'["\\\r\n]', chosen)
        if inline_hash and serializable:
            lines[i] = 'ADMIN_PASSWORD="%s"' % chosen
            notes.append("__WARN__=ADMIN_PASSWORD 含 ' #'，已加引号写入（dotenv 会剥掉引号，口令原样生效）")
        elif inline_hash or not serializable:
            lines[i] = "ADMIN_PASSWORD=" + chosen
            notes.append("__WARN__=ADMIN_PASSWORD 含 dotenv 无法原样表达的字符（' #' / 引号 / 反斜杠 / 换行），"
                         "读回来时会被静默改写甚至整行丢掉。建议先用纯字母数字口令，装完登录后到 /settings 改成真正想要的")
        else:
            lines[i] = "ADMIN_PASSWORD=" + chosen
        notes.append("__ADMIN__=" + ("你指定的" if want_pw else "随机生成") + ":" + chosen)
        pw_applied = True

# 重跑 install.sh 时 .env 已存在，ADMIN_PASSWORD 通常已是真实值 —— 这时 --admin-password 会被跳过，
# 必须说出来，否则用户以为口令改了、实际还是旧的。
if want_pw and not pw_applied:
    notes.append("__WARN__=--admin-password 未生效：%s 里已有真实 ADMIN_PASSWORD。"
                 "改口令要登录后到 /settings（编辑 .env 不会回写库里的哈希）" % path.name)

path.write_bytes(newline.join(lines).encode("utf-8"))
path.chmod(0o600)
print("\n".join(notes))
PY
)"

ADMIN_LINE=""
while IFS= read -r line; do
    [ -n "$line" ] || continue
    case "$line" in
        __ADMIN__=*) ADMIN_LINE="${line#__ADMIN__=}" ;;
        __WARN__=*)  warn "${line#__WARN__=}" ;;
        *) log "$line" ;;
    esac
done <<< "$GEN_INFO"
chmod 600 "$ENV_TARGET"
# 收尾要打印的是 .env 里的实际用户名（--admin-user 可能被跳过，不能直接回显参数）
ADMIN_USER_EFFECTIVE="$(awk -F= '/^ADMIN_USERNAME=/{sub(/\r$/,"");print $2; exit}' "$ENV_TARGET")"

MAX_VIDEO_MB="$("$VENV/bin/python" - "$ENV_TARGET" <<'PY'
import re
import sys
from pathlib import Path
raw = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
m = re.search(r"^MAX_VIDEO_MB\s*=\s*(\d+)", raw, re.M)
print(m.group(1) if m else "200")
PY
)"

# api_key.txt 在 rsync 的 exclude 列表里（这样服务器上的密钥既不会被覆盖、也不会被 --delete 删掉），
# 所以首次部署时要显式从发布包目录带过来，否则永远是 mock 模式。
if [ ! -f "$APP_DIR/api_key.txt" ] \
   && [ -f "$REPO_DIR/api_key.txt" ] \
   && [ "$(readlink -f "$REPO_DIR")" != "$(readlink -f "$APP_DIR")" ]; then
    install -m 600 -o "$RUN_USER" -g "$RUN_USER" "$REPO_DIR/api_key.txt" "$APP_DIR/api_key.txt"
    log "已从发布目录复制 api_key.txt 到 $APP_DIR"
fi

if [ -f "$APP_DIR/api_key.txt" ]; then
    chmod 600 "$APP_DIR/api_key.txt"
    log "api_key.txt 已就位（600）"
else
    warn "缺少 $APP_DIR/api_key.txt —— 系统会进 mock 演示模式，报告不会真的调用千问。"
    warn "补齐：scp api_key.txt root@服务器:$APP_DIR/ 然后 chmod 600 $APP_DIR/api_key.txt"
fi

# ---------------------------------------------------------------- 目录与权限
mkdir -p "$APP_DIR/data/videos" "$APP_DIR/data/artifacts"
chown -R "$RUN_USER:$RUN_USER" "$APP_DIR"
# .venv 里是 pip 建好的可执行文件与软链，绝不能被批量 chmod 打回 644。
find "$APP_DIR" -path "$APP_DIR/.venv" -prune -o -type d -exec chmod 0755 {} +
find "$APP_DIR" -path "$APP_DIR/.venv" -prune -o -type f -exec chmod 0644 {} +
chmod 600 "$ENV_TARGET"
if [ -f "$APP_DIR/api_key.txt" ]; then chmod 600 "$APP_DIR/api_key.txt"; fi
chmod 755 "$SCRIPT_DIR"/*.sh 2>/dev/null || true
log "权限已设置（目录 755 / 代码 644 / 密钥 600）；数据目录 $APP_DIR/data"

# ---------------------------------------------------------------- systemd
sed -e "s#__APPDIR__#$APP_DIR#g" \
    -e "s#__PORT__#$APP_PORT#g" \
    -e "s#__HOST__#$BIND_HOST#g" \
    "$SCRIPT_DIR/$SERVICE.service" > "/etc/systemd/system/$SERVICE.service"
# 只检查非注释行：模板顶部的注释里就写着"--workers"这个词，整文件 grep 会误伤自己并中止安装。
if grep -vE '^[[:space:]]*#' "/etc/systemd/system/$SERVICE.service" | grep -q -- '--workers'; then
    die "unit 的 ExecStart 里出现 --workers：单进程是硬要求，请改回 deploy/$SERVICE.service"
fi
systemctl daemon-reload
systemctl enable "$SERVICE" >/dev/null 2>&1
log "systemd 单元已安装（单进程 + Restart=always）"

# ---------------------------------------------------------------- nginx
if [ "$USE_NGINX" = "1" ]; then
    BODY_MB=$((MAX_VIDEO_MB + 20))
    sed -e "s#__APPDIR__#$APP_DIR#g" \
        -e "s#__PORT__#$APP_PORT#g" \
        -e "s#__LISTEN__#$LISTEN_PORT#g" \
        -e "s#__BODY_MB__#$BODY_MB#g" \
        "$SCRIPT_DIR/nginx-$SERVICE.conf" > "/etc/nginx/conf.d/$SERVICE.conf"
    if [ -e /etc/nginx/sites-enabled/default ]; then
        rm -f /etc/nginx/sites-enabled/default
        log "已移除 Ubuntu 自带 default 站点（它会成为 80 端口的默认服务器）。"
    fi
    nginx -t
    systemctl enable nginx >/dev/null 2>&1 || true
    systemctl restart nginx
    log "nginx：0.0.0.0:$LISTEN_PORT -> 127.0.0.1:$APP_PORT，client_max_body_size ${BODY_MB}m"
else
    warn "--no-nginx：uvicorn 直接监听 0.0.0.0:$APP_PORT，上传体积只受 MAX_VIDEO_MB 限制。"
fi

# ---------------------------------------------------------------- 本机防火墙（若启用）
if command -v ufw >/dev/null 2>&1 && ufw status 2>/dev/null | grep -qi 'active'; then
    ufw allow "$LISTEN_PORT/tcp" >/dev/null || true
    if [ "$USE_NGINX" = "0" ] && [ "$APP_PORT" != "$LISTEN_PORT" ]; then
        ufw allow "$APP_PORT/tcp" >/dev/null || true
    fi
    log "ufw 已放行 $LISTEN_PORT/tcp"
fi

# ---------------------------------------------------------------- 启动 + 验收
systemctl restart "$SERVICE"
PROBE_PORT="$LISTEN_PORT"
[ "$USE_NGINX" = "1" ] || PROBE_PORT="$APP_PORT"
HEALTH=""
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    sleep 2
    if HEALTH="$(curl -fsS --max-time 5 "http://127.0.0.1:$PROBE_PORT/health" 2>/dev/null)"; then
        break
    fi
done
[ -n "$HEALTH" ] || { systemctl --no-pager -l status "$SERVICE" || true; die "健康检查没通过，见上面服务日志。"; }

HEALTH_JSON="$HEALTH" "$VENV/bin/python" - <<'PY'
import json
import os

d = json.loads(os.environ["HEALTH_JSON"])
c = d.get("concurrency", {})
print("       模式     :", d.get("mode"), "（key 来源：%s）" % d.get("key_source"))
print("       ffmpeg   :", d.get("ffmpeg"), "   人脸层:", d.get("face_metrics"),
      ("（%s）" % d.get("face_metrics_note") if d.get("face_metrics_note") else ""))
print("       ASR      :", d.get("asr_engine_label"))
print("       评分标准 :", d.get("rubric"))
print("       并发     : analyzers=%s pool=%s running=%s llm=%s web_threads=%s"
      % (c.get("analyzers"), c.get("pool_size"), c.get("running"),
         c.get("llm_requests"), c.get("web_threads")))
if not d.get("ffmpeg"):
    print("       !! ffmpeg 不在 PATH，分析会失败：apt-get install -y ffmpeg")
if d.get("mode") != "real":
    print("       !! 当前不是 real 模式，产出的是演示数据")
if str(c.get("web_threads")) != "32":
    print("       （web_threads 取自进程启动时的 WEB_THREADS，改这项要重启服务）")
PY

if [ "$USE_NGINX" = "1" ]; then
    curl -fsS --max-time 5 "http://127.0.0.1:$LISTEN_PORT/login" >/dev/null \
        || warn "nginx 侧 /login 未返回 200，看 /var/log/nginx/$SERVICE.error.log"
fi

# ---------------------------------------------------------------- 收尾提示
PUB="$(curl -fsS --max-time 3 'http://100.100.100.200/latest/meta-data/eipv2' 2>/dev/null || true)"
[ -n "$PUB" ] || PUB="$(curl -fsS --max-time 3 'http://100.100.100.200/latest/meta-data/public-ipv4' 2>/dev/null || true)"
PRIV="$(hostname -I 2>/dev/null | awk '{print $1}')"

echo
log "部署完成"
echo "  公网地址   : http://${PUB:-<弹性公网IP>}:${LISTEN_PORT}/"
echo "  内网地址   : http://${PRIV:-<内网IP>}:${LISTEN_PORT}/ （同 VPC / 服务器本机可用）"
if [ -n "$ADMIN_LINE" ]; then
    echo "  管理员账号 : ${ADMIN_USER_EFFECTIVE:-admin}"
    case "$ADMIN_LINE" in
        随机生成:*) warn "随机管理员口令：${ADMIN_LINE#随机生成:}  ← 立刻记下来，登录后到 /settings 改掉" ;;
        你指定的:*) echo "  管理员口令 : 你指定的那个（仅用于**首次建号**，哈希入库存着）" ;;
    esac
else
    echo "  管理员账号 : 见 $ENV_TARGET 的 ADMIN_USERNAME / ADMIN_PASSWORD（首次启动时建号）"
fi
warn "改管理员口令只能登录后在 /settings 改（保存时会回写数据库哈希）。"
warn "  直接编辑 .env 的 ADMIN_PASSWORD 再重启**不会**改口令 —— init_db 只在账号不存在时建号。"
warn "  忘了口令的救法见部署手册第 11 节末尾。"
if [ "$CREATED_ENV" = "1" ]; then
    echo "  配置文件   : $ENV_TARGET（大部分项可登录后在 /settings 改并热生效）"
fi
echo "  常用命令   : systemctl status $SERVICE | journalctl -u $SERVICE -n 100 --no-pager"
echo "             : curl http://127.0.0.1/health | bash deploy/backup.sh"
echo
warn "阿里云控制台还要做两件事："
warn "  1) 安全组入方向放行 TCP ${LISTEN_PORT}（0.0.0.0/0）。**不要**放行 ${APP_PORT}——它只应监听 127.0.0.1。"
warn "  2) 磁盘没有自动清理：系统盘 40GB，容量请盯 $APP_DIR/data，见部署手册第 7 节。"
warn "已知限制：这台机器出网 3Mbps（约 375KB/s），两人同时回看视频会明显卡顿。"
