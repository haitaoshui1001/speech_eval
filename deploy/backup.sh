#!/usr/bin/env bash
# 备份：SQLite 在线一致快照 + 配置文件，可选带上视频与抽帧产物。
#
# 用法：
#   sudo -u speech bash deploy/backup.sh                # 只备份库和配置（默认）
#   sudo -u speech bash deploy/backup.sh --with-media   # 连 videos/ 与 artifacts/ 一起打包
#   KEEP_DAYS=14 bash deploy/backup.sh                  # 保留 14 天（默认 7）
#
# 为什么用 sqlite3 的 backup API 而不是 cp：库跑在 WAL 模式下，直接 cp 可能拿到
# "主库与 -wal 不一致"的半份快照。backup API 会在读锁内做页级拷贝，服务不用停。
set -euo pipefail

APP_DIR="/opt/speech-assist"
DATA_DIR="${DATA_DIR:-$APP_DIR/data}"
BACKUP_DIR="${BACKUP_DIR:-$APP_DIR/backups}"
KEEP_DAYS="${KEEP_DAYS:-7}"
PY="${APP_DIR}/.venv/bin/python"
WITH_MEDIA="0"

while [ $# -gt 0 ]; do
    case "$1" in
        --with-media) WITH_MEDIA="1"; shift ;;
        --app-dir) APP_DIR="$2"; DATA_DIR="$APP_DIR/data"; shift 2 ;;
        *) shift ;;
    esac
done
[ -x "$PY" ] || PY="$(command -v python3)"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$BACKUP_DIR/$STAMP"
mkdir -p "$OUT"
chmod 700 "$BACKUP_DIR"

DB="$DATA_DIR/speech.db"
[ -f "$DB" ] || { echo "[中止] 找不到数据库 $DB" >&2; exit 1; }

echo "[备份] 数据库 -> $OUT/speech.db"
SRC="$DB" DST="$OUT/speech.db" "$PY" - <<'PY'
import os
import sqlite3

src, dst = os.environ["SRC"], os.environ["DST"]
s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
d = sqlite3.connect(dst)
with d:
    s.backup(d)
d.close()
s.close()
print("       完整性检查:", end=" ")
c = sqlite3.connect(dst)
print(c.execute("PRAGMA integrity_check").fetchone()[0])
n = c.execute("select count(*) from videos").fetchone()[0]
u = c.execute("select count(*) from users").fetchone()[0]
print(f"       videos={n} users={u}")
c.close()
PY

for f in .env api_key.txt 提示词.json 评价标准.txt; do
    if [ -f "$APP_DIR/$f" ]; then
        cp -a "$APP_DIR/$f" "$OUT/$f"
        echo "[备份] 配置 $f"
    fi
done
chmod 600 "$OUT"/* 2>/dev/null || true

if [ "$WITH_MEDIA" = "1" ]; then
    echo "[备份] 打包 videos/ 与 artifacts/（可能很慢、很大）"
    tar czf "$OUT/media.tar.gz" -C "$DATA_DIR" videos artifacts 2>/dev/null || \
        tar czf "$OUT/media.tar.gz" -C "$DATA_DIR" videos artifacts
fi

du -sh "$OUT" | awk '{print "[完成] '"$OUT"'  大小 "$1}'

# 滚动清理
find "$BACKUP_DIR" -mindepth 1 -maxdepth 1 -type d -mtime +"$KEEP_DAYS" -exec rm -rf {} + 2>/dev/null || true
echo "[清理] 保留最近 $KEEP_DAYS 天，当前共 $(find "$BACKUP_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l) 份"
df -h "$APP_DIR" | tail -n 1 | awk '{print "[磁盘] 已用 "$5"（"$3"/"$2"），数据目录在 '"$DATA_DIR"'"}'
