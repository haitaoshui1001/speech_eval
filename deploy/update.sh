#!/usr/bin/env bash
# 升级：用新的发布包覆盖代码，保留 .env / api_key.txt / data/ / 提示词.json / .venv
#
# 用法：
#   sudo bash deploy/update.sh /root/speech-assist.tar.gz
#   sudo bash deploy/update.sh /opt/speech-assist-new --no-backup
#
# 参数可以是 tar.gz（会自动解压），也可以是已经解压好的目录。
set -euo pipefail

APP_DIR="/opt/speech-assist"
SERVICE="speech-assist"
RUN_USER="speech"
DO_BACKUP="1"
NEW_SRC=""

log()  { printf '\033[1;32m[升级]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[注意]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[中止]\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --app-dir)  APP_DIR="$2"; shift 2 ;;
        --no-backup) DO_BACKUP="0"; shift ;;
        -h|--help)  echo "用法：sudo bash deploy/update.sh <新发布包.tar.gz 或 已解压目录> [--app-dir PATH] [--no-backup]"; exit 0 ;;
        *)          NEW_SRC="$1"; shift ;;
    esac
done

[ "$(id -u)" = "0" ] || die "请用 root 或 sudo 运行。"
[ -n "$NEW_SRC" ] || die "缺少参数：新发布包路径。"
[ -d "$APP_DIR/app" ] || die "未安装：$APP_DIR/app 不存在，请先跑 install.sh"

TMP=""
cleanup() { [ -n "$TMP" ] && rm -rf "$TMP" || true; }
trap cleanup EXIT

if [ -f "$NEW_SRC" ]; then
    TMP="$(mktemp -d /tmp/speech-assist-new.XXXXXX)"
    log "解压 $NEW_SRC"
    tar xzf "$NEW_SRC" -C "$TMP"
    NEW_DIR="$(find "$TMP" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    [ -d "$NEW_DIR/app" ] || NEW_DIR="$TMP"
elif [ -d "$NEW_SRC" ]; then
    NEW_DIR="$(readlink -f "$NEW_SRC")"
else
    die "$NEW_SRC 既不是文件也不是目录"
fi
[ -d "$NEW_DIR/app" ] || die "新包里找不到 app/，包不完整？"
if [ "$(readlink -f "$NEW_DIR")" = "$(readlink -f "$APP_DIR")" ]; then
    die "新包目录就是当前安装目录，会自我覆盖。请解压到别处（如 /opt/speech-assist-new）。"
fi

if [ "$DO_BACKUP" = "1" ] && [ -x "$APP_DIR/deploy/backup.sh" ]; then
    log "升级前先备份数据库"
    bash "$APP_DIR/deploy/backup.sh" || warn "备份失败，继续升级（可用 --no-backup 明确跳过）"
fi

log "同步代码 -> $APP_DIR"
rsync -a --delete \
    --exclude '/data/' --exclude '/.venv/' --exclude '/.env' --exclude '/api_key.txt' \
    --exclude '/提示词.json' --exclude '/__pycache__/' --exclude '/.git/' \
    --exclude '*.pyc' --exclude '*.log' --exclude 'logs_*.txt' \
    --exclude '/backups/' \
    "$NEW_DIR/" "$APP_DIR/"

log "补装/更新 Python 依赖"
PIP_HOST="mirrors.aliyun.com/pypi/simple"
if getent hosts mirrors.cloud.aliyuncs.com >/dev/null 2>&1; then
    PIP_HOST="mirrors.cloud.aliyuncs.com/pypi/simple"
fi
"$APP_DIR/.venv/bin/pip" install --quiet --disable-pip-version-check --no-input \
    --index-url "https://$PIP_HOST/" -r "$APP_DIR/requirements.txt"

chown -R "$RUN_USER:$RUN_USER" "$APP_DIR"
find "$APP_DIR" -path "$APP_DIR/.venv" -prune -o -type d -exec chmod 0755 {} +
find "$APP_DIR" -path "$APP_DIR/.venv" -prune -o -type f -exec chmod 0644 {} +
chmod 600 "$APP_DIR/.env"
if [ -f "$APP_DIR/api_key.txt" ]; then chmod 600 "$APP_DIR/api_key.txt"; fi
chmod 755 "$APP_DIR"/deploy/*.sh 2>/dev/null || true

# 单元与 nginx 配置可能被本次改动更新，重新渲染一次（端口沿用已安装的配置）。
if [ -f /etc/systemd/system/$SERVICE.service ]; then
    PORT="$(grep -o -- '--port [0-9]*' "/etc/systemd/system/$SERVICE.service" | awk '{print $2}' || true)"
    HOST="$(grep -o -- '--host [^ ]*' "/etc/systemd/system/$SERVICE.service" | awk '{print $2}' || true)"
    SCRIPT_DIR="$APP_DIR/deploy"
    sed -e "s#__APPDIR__#$APP_DIR#g" -e "s#__PORT__#${PORT:-8000}#g" \
        -e "s#__HOST__#${HOST:-127.0.0.1}#g" \
        "$SCRIPT_DIR/$SERVICE.service" > "/etc/systemd/system/$SERVICE.service.new"
    if ! cmp -s "/etc/systemd/system/$SERVICE.service.new" "/etc/systemd/system/$SERVICE.service"; then
        log "systemd 单元有变化，已更新并 daemon-reload"
        mv "/etc/systemd/system/$SERVICE.service.new" "/etc/systemd/system/$SERVICE.service"
        systemctl daemon-reload
    else
        rm -f "/etc/systemd/system/$SERVICE.service.new"
    fi
    if [ "$HOST" != "127.0.0.1" ]; then
        warn "当前 uvicorn 监听 $HOST:$PORT（无 nginx 模式），端口沿用。"
    elif [ -f /etc/nginx/conf.d/$SERVICE.conf ]; then
        cp -a /etc/nginx/conf.d/$SERVICE.conf /etc/nginx/conf.d/$SERVICE.conf.bak
        LP="$(grep -o 'listen [0-9]*' /etc/nginx/conf.d/$SERVICE.conf | awk '{print $2}' | head -n1 || true)"
        MV="$(awk -F= '/^MAX_VIDEO_MB/{gsub(/[ \t\r]/,"",$2); print $2; exit}' "$APP_DIR/.env")"
        sed -e "s#__APPDIR__#$APP_DIR#g" -e "s#__PORT__#${PORT:-8000}#g" \
            -e "s#__LISTEN__#${LP:-80}#g" -e "s#__BODY_MB__#$(( ${MV:-200} + 20 ))#g" \
            "$SCRIPT_DIR/nginx-$SERVICE.conf" > /etc/nginx/conf.d/$SERVICE.conf
        if nginx -t >/dev/null 2>&1; then
            rm -f /etc/nginx/conf.d/$SERVICE.conf.bak
            systemctl reload nginx
            log "nginx 配置已按 ${MV:-200}MB 上传上限重渲染"
        else
            mv /etc/nginx/conf.d/$SERVICE.conf.bak /etc/nginx/conf.d/$SERVICE.conf
            warn "新 nginx 配置未通过 nginx -t，已回滚为原配置（升级的其余部分仍然有效）。"
        fi
    fi
fi

log "重启服务"
systemctl restart "$SERVICE"
sleep 3
for _ in 1 2 3 4 5 6 7 8 9 10; do
    if curl -fsS --max-time 5 "http://127.0.0.1:${PORT:-8000}/health" >/dev/null 2>&1; then
        log "升级完成，/health 正常。"
        exit 0
    fi
    sleep 2
done
warn "服务已重启但 /health 未通过，回滚办法：journalctl -u $SERVICE -n 100 --no-pager"
exit 1
