"""SQLite 存储层（标准库 sqlite3，WAL + 外键）。"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import settings
from .rubric import rubric
from .security import hash_password

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL,
    upload_used INTEGER NOT NULL DEFAULT 0,
    upload_limit INTEGER
);

CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    topic TEXT NOT NULL DEFAULT '',
    requirements TEXT NOT NULL DEFAULT '',
    filename TEXT NOT NULL DEFAULT '',
    path TEXT NOT NULL DEFAULT '',
    size INTEGER NOT NULL DEFAULT 0,
    duration REAL NOT NULL DEFAULT 0,
    width INTEGER NOT NULL DEFAULT 0,
    height INTEGER NOT NULL DEFAULT 0,
    thumb TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'uploaded',
    stage TEXT NOT NULL DEFAULT '',
    progress INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    transcript TEXT NOT NULL DEFAULT '',
    segments TEXT NOT NULL DEFAULT '[]',
    engine TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    channels TEXT NOT NULL DEFAULT '{}',
    qc TEXT NOT NULL DEFAULT '[]',
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    api_calls INTEGER NOT NULL DEFAULT 0,
    token_detail TEXT NOT NULL DEFAULT '{}',
    frontal_ratio REAL,
    orig_size INTEGER,
    compress_note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    analyzed_at TEXT
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL,
    rubric_version TEXT NOT NULL DEFAULT '',
    total REAL NOT NULL DEFAULT 0,
    max_total REAL NOT NULL DEFAULT 100,
    band TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0,
    advantages TEXT NOT NULL DEFAULT '[]',
    disadvantages TEXT NOT NULL DEFAULT '[]',
    suggestions TEXT NOT NULL DEFAULT '[]',
    summary TEXT NOT NULL DEFAULT '',
    next_focus TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL DEFAULT '{}',
    model TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dimension_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id INTEGER NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    video_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    dim_key TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    idx INTEGER NOT NULL DEFAULT 0,
    score REAL NOT NULL DEFAULT 0,
    max_score REAL NOT NULL DEFAULT 0,
    ratio REAL NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0,
    evidence TEXT NOT NULL DEFAULT '[]',
    detail TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    video_id INTEGER NOT NULL,
    evaluation_id INTEGER NOT NULL,
    dim_key TEXT NOT NULL DEFAULT '',
    signature TEXT NOT NULL DEFAULT '',
    label TEXT NOT NULL DEFAULT '',
    quote TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS qa_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL DEFAULT '',
    ai_comment TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL DEFAULT '',
    answered_at TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_videos_user ON videos(user_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_eval_video ON evaluations(video_id);
CREATE INDEX IF NOT EXISTS idx_dim_eval ON dimension_scores(evaluation_id);
CREATE INDEX IF NOT EXISTS idx_dim_user ON dimension_scores(user_id, dim_key);
CREATE INDEX IF NOT EXISTS idx_issue_user ON issues(user_id, signature);
CREATE INDEX IF NOT EXISTS idx_qa_video ON qa_turns(video_id, idx);
"""


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# 老库升级用的补列定义：CREATE TABLE IF NOT EXISTS 不会给已存在的表加字段。
_EXTRA_VIDEO_COLUMNS = {
    "completion_tokens": "INTEGER NOT NULL DEFAULT 0",
    "total_tokens": "INTEGER NOT NULL DEFAULT 0",
    "api_calls": "INTEGER NOT NULL DEFAULT 0",
    "token_detail": "TEXT NOT NULL DEFAULT '{}'",
    # 正脸率：可空。NULL 表示「没测或测量未通过门控」，和测出 0% 是两回事，
    # 所以不能给 DEFAULT 0，否则趋势图会把历史空白点画成真实的零。
    "frontal_ratio": "REAL",
    # 自动压缩痕迹：orig_size 可空（NULL = 没压过），compress_note 存给人看的说明。
    # 原件在压缩成功后即被替换删除，这两个字段是唯一能回溯「原来多大」的地方。
    "orig_size": "INTEGER",
    "compress_note": "TEXT NOT NULL DEFAULT ''",
}

# 上传配额同样要补列：upload_limit 可空，NULL 表示「跟随全局 USER_UPLOAD_LIMIT」，
# 和显式填 0（完全禁止上传）是两回事。
_EXTRA_USER_COLUMNS = {
    "upload_used": "INTEGER NOT NULL DEFAULT 0",
    "upload_limit": "INTEGER",
}


def _migrate(conn) -> None:
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(videos)")}
    for name, decl in _EXTRA_VIDEO_COLUMNS.items():
        if name not in cols:
            conn.execute(f"ALTER TABLE videos ADD COLUMN {name} {decl}")
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
    for name, decl in _EXTRA_USER_COLUMNS.items():
        if name not in cols:
            conn.execute(f"ALTER TABLE users ADD COLUMN {name} {decl}")


@contextmanager
def get_conn():
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path, timeout=30, detect_types=0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)
        cur = conn.execute("SELECT id FROM users WHERE username = ?", (settings.admin_username,))
        if cur.fetchone() is None:
            conn.execute(
                "INSERT INTO users(username, password_hash, display_name, role, created_at) "
                "VALUES (?, ?, ?, 'admin', ?)",
                (settings.admin_username, hash_password(settings.admin_password),
                 "管理员", now()))


def sync_admin_login(username: str, password: str = "") -> str:
    """把 .env 里的管理员账号 / 口令同步到数据库，返回给用户看的说明。

    设置页保存时调用：init_db 只在账号不存在时建号，改口令必须显式回写。
    """
    username = (username or "").strip()
    password = password or ""
    if not username and not password:
        return ""
    with get_conn() as conn:
        row = conn.execute("SELECT id, username FROM users WHERE role = 'admin' ORDER BY id LIMIT 1").fetchone()
        if row is None:
            name = username or settings.admin_username
            conn.execute("INSERT INTO users(username, password_hash, display_name, role, created_at) "
                         "VALUES (?, ?, ?, 'admin', ?)",
                         (name, hash_password(password or settings.admin_password), "管理员", now()))
            return f"已创建管理员账号 {name}"
        notes: list[str] = []
        if username and username != row["username"]:
            taken = conn.execute("SELECT id FROM users WHERE username = ? AND id <> ?",
                                 (username, row["id"])).fetchone()
            if taken:
                raise ValueError(f"用户名 {username} 已被其他账号占用")
            conn.execute("UPDATE users SET username = ? WHERE id = ?", (username, row["id"]))
            notes.append(f"管理员账号已改名为 {username}")
        if password:
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                         (hash_password(password), row["id"]))
            notes.append("管理员口令已更新")
        return "；".join(notes)


# ---------------- users ----------------
def create_user(username: str, password: str, display_name: str = "", role: str = "user") -> int:
    with get_conn() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users(username, password_hash, display_name, role, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (username, hash_password(password), display_name or username, role, now()))
            return int(cur.lastrowid)
        except sqlite3.IntegrityError:
            raise ValueError("该用户名已被注册") from None


def get_user(user_id: int) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def set_password(user_id: int, password: str) -> None:
    """管理员重置口令：只换哈希，用户名与角色不动。"""
    with get_conn() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                     (hash_password(password), user_id))


def upload_quota(user_id: int) -> dict:
    """账号的上传额度：used / limit / remaining（remaining 为 None 表示不限）。

    额度取值优先级：账号上的 upload_limit（管理员单独放行过）> 全局 USER_UPLOAD_LIMIT。
    """
    with get_conn() as conn:
        row = conn.execute("SELECT role, upload_used, upload_limit FROM users WHERE id = ?",
                           (user_id,)).fetchone()
    if row is None:
        return {"used": 0, "limit": 0, "remaining": 0}
    used = int(row["upload_used"] or 0)
    if row["role"] == "admin":
        return {"used": used, "limit": None, "remaining": None}
    limit = settings.user_upload_limit if row["upload_limit"] is None else int(row["upload_limit"])
    return {"used": used, "limit": limit, "remaining": max(0, limit - used)}


def set_upload_quota(user_id: int, used: int | None = None,
                     limit: int | None | str = "") -> None:
    """管理员调整额度：used=重置已用次数，limit=单独设上限（None 表示跟随全局）。

    limit 传空字符串 = 不改这一列；传 None = 清空覆盖、回到跟随全局默认。
    """
    sets, args = [], []
    if used is not None:
        sets.append("upload_used = ?")
        args.append(max(0, int(used)))
    if limit != "":
        sets.append("upload_limit = ?")
        args.append(None if limit is None else max(0, int(limit)))
    if not sets:
        return
    args.append(user_id)
    with get_conn() as conn:
        conn.execute(f"UPDATE users SET {', '.join(sets)} WHERE id = ?", args)


def get_user_by_name(username: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()


def list_users() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return list(conn.execute(
            "SELECT u.*, COALESCE(u.upload_limit, ?) AS limit_now, "
            " (SELECT COUNT(*) FROM videos v WHERE v.user_id = u.id) AS video_count, "
            " (SELECT COUNT(*) FROM videos v WHERE v.user_id = u.id AND v.status = 'done') AS done_count, "
            " (SELECT COALESCE(SUM(v.total_tokens), 0) FROM videos v WHERE v.user_id = u.id) AS tokens, "
            " (SELECT COALESCE(SUM(v.api_calls), 0) FROM videos v WHERE v.user_id = u.id) AS api_calls, "
            " (SELECT ROUND(AVG(e.total), 1) FROM evaluations e WHERE e.user_id = u.id) AS avg_score "
            "FROM users u ORDER BY u.role DESC, u.id", (settings.user_upload_limit,)))


def delete_user(user_id: int) -> None:
    import shutil
    with get_conn() as conn:
        for row in conn.execute("SELECT id, path, thumb FROM videos WHERE user_id = ?", (user_id,)):
            for col in ("path", "thumb"):
                p = Path(row[col] or "")
                if p.is_absolute():
                    p.unlink(missing_ok=True)
            art = settings.artifact_dir / str(row["id"])
            if art.exists():
                shutil.rmtree(art, ignore_errors=True)
        conn.execute("DELETE FROM issues WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM dimension_scores WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


# ---------------- videos ----------------
def create_video(user_id: int, title: str, filename: str, path: str, size: int,
                 topic: str = "", requirements: str = "") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO videos(user_id, title, topic, requirements, filename, path, size, "
            "status, stage, progress, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'uploaded', '待分析', 0, ?, ?)",
            (user_id, title, topic, requirements, filename, path, size, now(), now()))
        # 一次上传即一次额度消耗，和插入同事务，避免"建了视频没记账"的漂移。
        conn.execute("UPDATE users SET upload_used = upload_used + 1 WHERE id = ?", (user_id,))
        return int(cur.lastrowid)


def update_video(video_id: int, **fields) -> None:
    fields = dict(fields)
    fields["updated_at"] = now()
    for k, v in list(fields.items()):
        if isinstance(v, (dict, list)):
            fields[k] = json.dumps(v, ensure_ascii=False)
    keys = ", ".join(f"{k} = ?" for k in fields)
    with get_conn() as conn:
        conn.execute(f"UPDATE videos SET {keys} WHERE id = ?", (*fields.values(), video_id))


def set_progress(video_id: int, status: str, stage: str, progress: int, error: str = "") -> None:
    update_video(video_id, status=status, stage=stage, progress=progress, error=error)


def get_video(video_id: int) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute(
            "SELECT v.*, u.username, u.display_name FROM videos v "
            "JOIN users u ON u.id = v.user_id WHERE v.id = ?", (video_id,)).fetchone()


def list_user_videos(user_id: int, limit: int = 100) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return list(conn.execute(
            "SELECT v.*, e.total, e.band, e.confidence FROM videos v "
            "LEFT JOIN evaluations e ON e.video_id = v.id "
            "WHERE v.user_id = ? ORDER BY v.id DESC LIMIT ?", (user_id, limit)))


def list_all_videos(limit: int = 500) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return list(conn.execute(
            "SELECT v.*, u.username, u.display_name, e.total, e.band FROM videos v "
            "JOIN users u ON u.id = v.user_id "
            "LEFT JOIN evaluations e ON e.video_id = v.id "
            "ORDER BY v.id DESC LIMIT ?", (limit,)))


def delete_video(video_id: int) -> None:
    """删除视频及其评价、抽帧产物。"""
    row = get_video(video_id)
    if row is None:
        return
    for p in (Path(row["path"] or ""), Path(row["thumb"] or "")):
        if p.is_absolute():
            p.unlink(missing_ok=True)
    art = settings.artifact_dir / str(video_id)
    if art.exists():
        import shutil
        shutil.rmtree(art, ignore_errors=True)
    with get_conn() as conn:
        conn.execute("DELETE FROM issues WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
        # 删掉即释放名额：否则用户传错文件、删掉重传就会被额度卡死。
        conn.execute("UPDATE users SET upload_used = MAX(0, upload_used - 1) WHERE id = ?",
                     (row["user_id"],))


def latest_done_video(user_id: int) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM videos WHERE user_id = ? AND status = 'done' ORDER BY id DESC LIMIT 1",
            (user_id,)).fetchone()


def admin_stats() -> dict:
    with get_conn() as conn:
        users = conn.execute("SELECT COUNT(*) c FROM users WHERE role = 'user'").fetchone()["c"]
        vids = conn.execute("SELECT COUNT(*) c FROM videos").fetchone()["c"]
        done = conn.execute("SELECT COUNT(*) c FROM videos WHERE status = 'done'").fetchone()["c"]
        running = conn.execute(
            "SELECT COUNT(*) c FROM videos WHERE status IN ('queued','transcribing','analyzing')").fetchone()["c"]
        avg = conn.execute("SELECT ROUND(AVG(total),1) a FROM evaluations").fetchone()["a"]
        by_dim = [dict(r) for r in conn.execute(
            "SELECT name, dim_key, ROUND(AVG(score),2) avg_score, ROUND(AVG(max_score),2) max_score, "
            "COUNT(*) n FROM dimension_scores GROUP BY dim_key ORDER BY idx")]
        week_ago = (datetime.now() - timedelta(days=7)).isoformat(timespec="seconds")
        recent = conn.execute(
            "SELECT COUNT(*) c FROM videos WHERE created_at > ?", (week_ago,)).fetchone()["c"]
        return {"users": users, "videos": vids, "done": done, "running": running,
                "avg": avg, "by_dim": by_dim, "recent": recent}


UNNAMED_TOPIC = "（未填主题）"


def video_topics() -> list[dict]:
    """视频主题清单 + 每个主题下的视频数，供后台筛选下拉框；未填的归到一个占位桶。"""
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT CASE WHEN COALESCE(topic,'') = '' THEN ? ELSE topic END AS topic, "
            "COUNT(*) AS n FROM videos GROUP BY topic ORDER BY COUNT(*) DESC, topic",
            (UNNAMED_TOPIC,))]


def _topic_clause(topic: str) -> tuple[str, tuple]:
    """把主题筛选翻成 SQL 片段；空值 = 全部主题，占位桶匹配空主题。"""
    if not topic:
        return "", ()
    if topic == UNNAMED_TOPIC:
        return "WHERE COALESCE(v.topic,'') = '' ", ()
    return "WHERE v.topic = ? ", (topic,)


def admin_dim_stats(topic: str = "") -> dict:
    """维度均分统计：可先按主题筛选，再按「人 × 维度」摊开。

    只按维度求全站均值会把不同主题、不同人混成一锅——同一个维度在议论文和即兴演讲
    里根本不是同一条尺子。这里同时给出：维度均分（受筛选影响）、每人各维度均分矩阵、
    每人的样本数与平均总分。
    """
    where, args = _topic_clause(topic)
    with get_conn() as conn:
        by_dim = [dict(r) for r in conn.execute(
            "SELECT d.name, d.dim_key, ROUND(AVG(d.score),2) avg_score, "
            "ROUND(AVG(d.max_score),2) max_score, COUNT(*) n FROM dimension_scores d "
            f"JOIN videos v ON v.id = d.video_id {where} "
            "GROUP BY d.dim_key ORDER BY MIN(d.idx)", args)]
        matrix = conn.execute(
            "SELECT v.user_id, u.username, u.display_name, d.dim_key, "
            "ROUND(AVG(d.score),2) avg_score, ROUND(AVG(d.max_score),2) max_score "
            "FROM dimension_scores d "
            "JOIN videos v ON v.id = d.video_id JOIN users u ON u.id = v.user_id "
            f"{where} GROUP BY v.user_id, d.dim_key ORDER BY v.user_id, MIN(d.idx)",
            args).fetchall()
        people: dict[int, dict] = {}
        for r in matrix:
            row = people.setdefault(int(r["user_id"]), {
                "user_id": int(r["user_id"]),
                "who": r["display_name"] or r["username"],
                "username": r["username"], "n": 0, "avg_total": None, "dims": {}})
            row["dims"][r["dim_key"]] = {"avg": r["avg_score"], "max": r["max_score"]}
        for r in conn.execute(
            "SELECT v.user_id, COUNT(*) n, ROUND(AVG(e.total),1) avg_total FROM videos v "
            "JOIN evaluations e ON e.video_id = v.id "
            f"{where} GROUP BY v.user_id", args):
            row = people.get(int(r["user_id"]))
            if row:
                row["n"] = int(r["n"])
                row["avg_total"] = r["avg_total"]
        samples = conn.execute(
            f"SELECT COUNT(*) c FROM videos v {where}", args).fetchone()["c"]
    return {"topic": topic, "topics": video_topics(), "by_dim": by_dim,
            "by_user": sorted(people.values(), key=lambda x: -x["n"]), "samples": samples}


def _merge_usage_axis(prev, new) -> list[dict]:
    """合并两份 by_stage / by_model 明细，按 name 相加后按 token 倒序。"""
    table: dict[str, dict] = {}
    for item in list(prev or []) + list(new or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "—")
        row = table.setdefault(name, {"name": name, "calls": 0, "prompt": 0,
                                      "completion": 0, "total": 0, "estimated": 0})
        for k in ("calls", "prompt", "completion", "total", "estimated"):
            row[k] += int(item.get(k) or 0)
    return sorted(table.values(), key=lambda x: -x["total"])


def add_usage(video_id: int, usage: dict) -> dict:
    """把一次按需调用（导师出题、作答点评）的用量累加进视频行，返回合并结果。

    pipeline._flush_usage 是覆盖写：它汇总的是贯穿整条流水线的那个 client 的全部记录，
    所以只在分析阶段用。分析完成后另起 client 调模型时只能走这里做增量合并，
    否则会把分析阶段已经记下的 token 统计整段抹掉。
    """
    calls = int((usage or {}).get("calls") or 0)
    if not calls:
        return {}
    with get_conn() as conn:
        row = conn.execute(
            "SELECT prompt_tokens, completion_tokens, total_tokens, api_calls, token_detail "
            "FROM videos WHERE id = ?", (video_id,)).fetchone()
    if row is None:
        return {}
    try:
        prev = json.loads(row["token_detail"] or "{}")
    except json.JSONDecodeError:
        prev = {}
    if not isinstance(prev, dict):
        prev = {}
    merged = {
        "calls": int(row["api_calls"] or 0) + calls,
        "prompt": int(row["prompt_tokens"] or 0) + int(usage.get("prompt") or 0),
        "completion": int(row["completion_tokens"] or 0) + int(usage.get("completion") or 0),
        "total": int(row["total_tokens"] or 0) + int(usage.get("total") or 0),
        "estimated_calls": int(prev.get("estimated_calls") or 0) + int(usage.get("estimated_calls") or 0),
        "by_stage": _merge_usage_axis(prev.get("by_stage"), usage.get("by_stage")),
        "by_model": _merge_usage_axis(prev.get("by_model"), usage.get("by_model")),
    }
    update_video(video_id, prompt_tokens=merged["prompt"], completion_tokens=merged["completion"],
                 total_tokens=merged["total"], api_calls=merged["calls"], token_detail=merged)
    return merged


def token_usage(user_id: int | None = None, per_video: int = 0) -> dict:
    """token 用量汇总：user_id 为空时统计全站。

    明细存在 videos.token_detail（每次分析写入的 usage_summary），这里在 Python 侧
    二次聚合，避免依赖 SQLite 的 JSON 扩展。
    """
    sql = ("SELECT v.id, v.title, v.user_id, v.analyzed_at, v.api_calls, v.prompt_tokens, "
           "v.completion_tokens, v.total_tokens, v.token_detail, u.username, u.display_name "
           "FROM videos v JOIN users u ON u.id = v.user_id ")
    args: tuple = ()
    if user_id is not None:
        sql += "WHERE v.user_id = ? "
        args = (user_id,)
    sql += "ORDER BY v.id DESC"
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute(sql, args)]

    out = {"analyzed": 0, "calls": 0, "prompt": 0, "completion": 0, "total": 0,
           "estimated_calls": 0, "by_stage": [], "by_model": [], "by_user": [],
           "videos": []}
    stages: dict[str, dict] = {}
    models: dict[str, dict] = {}
    users: dict[str, dict] = {}
    for r in rows:
        try:
            detail = json.loads(r.get("token_detail") or "{}")
        except json.JSONDecodeError:
            detail = {}
        if not isinstance(detail, dict):
            detail = {}
        calls = int(r.get("api_calls") or detail.get("calls") or 0)
        if not calls and not int(r["total_tokens"] or 0):
            continue
        out["analyzed"] += 1
        out["calls"] += calls
        out["prompt"] += int(r["prompt_tokens"] or 0)
        out["completion"] += int(r["completion_tokens"] or 0)
        out["total"] += int(r["total_tokens"] or 0)
        out["estimated_calls"] += int(detail.get("estimated_calls") or 0)
        who = f"{r['display_name'] or r['username']}（{r['username']}）"
        bucket = users.setdefault(who, {"name": who, "user_id": r["user_id"], "calls": 0,
                                        "total": 0, "videos": 0})
        bucket["calls"] += calls
        bucket["total"] += int(r["total_tokens"] or 0)
        bucket["videos"] += 1
        for axis, table in (("by_stage", stages), ("by_model", models)):
            for item in detail.get(axis) or []:
                if not isinstance(item, dict):
                    continue
                row = table.setdefault(str(item.get("name") or "—"),
                                        {"name": str(item.get("name") or "—"), "calls": 0,
                                         "prompt": 0, "completion": 0, "total": 0,
                                         "estimated": 0})
                row["calls"] += int(item.get("calls") or 0)
                row["prompt"] += int(item.get("prompt") or 0)
                row["completion"] += int(item.get("completion") or 0)
                row["total"] += int(item.get("total") or 0)
                row["estimated"] += int(item.get("estimated") or 0)
        if len(out["videos"]) < per_video:
            out["videos"].append({"id": r["id"], "title": r["title"], "calls": calls,
                                  "total": int(r["total_tokens"] or 0),
                                  "estimated": int(detail.get("estimated_calls") or 0),
                                  "analyzed_at": r["analyzed_at"]})
    out["by_stage"] = sorted(stages.values(), key=lambda x: -x["total"])
    out["by_model"] = sorted(models.values(), key=lambda x: -x["total"])
    out["by_user"] = sorted(users.values(), key=lambda x: -x["total"])
    out["avg_per_video"] = round(out["total"] / out["analyzed"], 1) if out["analyzed"] else 0
    return out


def reset_stale_jobs() -> int:
    """进程重启后把中断的任务标记为失败，避免永远卡在 analyzing。"""
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE videos SET status = 'failed', stage = '已中断', error = '服务重启导致任务中断，请重新点击分析', "
            "updated_at = ? WHERE status IN ('queued','transcribing','analyzing')", (now(),))
        return cur.rowcount


# ---------------- evaluations ----------------
def save_evaluation(video: sqlite3.Row, report: dict, model: str) -> int:
    dims = report["dimensions"]
    payload = report
    with get_conn() as conn:
        conn.execute("DELETE FROM evaluations WHERE video_id = ?", (video["id"],))
        conn.execute("DELETE FROM dimension_scores WHERE video_id = ?", (video["id"],))
        conn.execute("DELETE FROM issues WHERE video_id = ?", (video["id"],))
        conn.execute("DELETE FROM qa_turns WHERE video_id = ?", (video["id"],))
        cur = conn.execute(
            "INSERT INTO evaluations(video_id, user_id, rubric_version, total, max_total, band, "
            "confidence, advantages, disadvantages, suggestions, summary, next_focus, payload, model, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (video["id"], video["user_id"], report.get("rubric_version", rubric.version),
             report["total"], report["max_total"], report.get("band", ""),
             report.get("confidence", 0),
             json.dumps(report.get("advantages", []), ensure_ascii=False),
             json.dumps(report.get("disadvantages", []), ensure_ascii=False),
             json.dumps(report.get("suggestions", []), ensure_ascii=False),
             report.get("summary", ""), report.get("next_focus", ""),
             json.dumps(payload, ensure_ascii=False), model, now()))
        eval_id = int(cur.lastrowid)
        for d in dims:
            conn.execute(
                "INSERT INTO dimension_scores(evaluation_id, video_id, user_id, dim_key, name, idx, "
                "score, max_score, ratio, confidence, evidence, detail) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (eval_id, video["id"], video["user_id"], d["key"], d["name"], d["index"],
                 d["score"], d["max_score"], d["ratio"], d.get("confidence", 0),
                 json.dumps(d.get("evidence", []), ensure_ascii=False),
                 json.dumps(d, ensure_ascii=False)))
            for iss in d.get("issues", []):
                conn.execute(
                    "INSERT INTO issues(user_id, video_id, evaluation_id, dim_key, signature, label, quote, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (video["user_id"], video["id"], eval_id, d["key"],
                     iss.get("signature", ""), iss.get("desc", "")[:160],
                     iss.get("quote", "")[:200], now()))
        return eval_id


def get_evaluation(video_id: int) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM evaluations WHERE video_id = ? ORDER BY id DESC LIMIT 1",
                            (video_id,)).fetchone()


def get_dimension_rows(video_id: int) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return list(conn.execute(
            "SELECT * FROM dimension_scores WHERE evaluation_id = "
            "(SELECT id FROM evaluations WHERE video_id = ? ORDER BY id DESC LIMIT 1) ORDER BY idx",
            (video_id,)))


def history_for_user(user_id: int) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return list(conn.execute(
            "SELECT v.id, v.title, v.duration, v.analyzed_at, v.topic, v.frontal_ratio, "
            "e.total, e.max_total, e.band, "
            "e.confidence, e.id AS evaluation_id FROM videos v JOIN evaluations e ON e.video_id = v.id "
            "WHERE v.user_id = ? ORDER BY v.id", (user_id,)))


def dims_for_video(video_id: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in get_dimension_rows(video_id):
        out[r["dim_key"]] = {"score": r["score"], "max_score": r["max_score"],
                             "name": r["name"], "idx": r["idx"], "ratio": r["ratio"]}
    return out


def recurring_issues(user_id: int, exclude_video: int | None = None) -> list[dict]:
    sql = ("SELECT i.signature, i.dim_key, COUNT(*) n, MAX(i.id) last_id, "
           "(SELECT label FROM issues i2 WHERE i2.signature = i.signature ORDER BY i2.id DESC LIMIT 1) label, "
           "(SELECT video_id FROM issues i3 WHERE i3.signature = i.signature ORDER BY i3.id DESC LIMIT 1) video_id "
           "FROM issues i WHERE i.user_id = ? ")
    args: list = [user_id]
    if exclude_video:
        sql += "AND i.video_id != ? "
        args.append(exclude_video)
    sql += "GROUP BY i.signature HAVING n > 1 ORDER BY n DESC, last_id DESC LIMIT 12"
    with get_conn() as conn:
        rows = list(conn.execute(sql, args))
        total_videos = conn.execute(
            "SELECT COUNT(DISTINCT video_id) c FROM issues WHERE user_id = ?", (user_id,)).fetchone()["c"]
    return [{"label": r["label"], "dim_key": r["dim_key"], "times": r["n"],
             "video_id": r["video_id"], "total_videos": int(total_videos)} for r in rows]


def user_best_worst(user_id: int) -> tuple[dict, dict]:
    with get_conn() as conn:
        rows = list(conn.execute(
            "SELECT dim_key, name, ROUND(AVG(ratio)*100,1) pct, COUNT(*) n FROM dimension_scores "
            "WHERE user_id = ? GROUP BY dim_key ORDER BY pct", (user_id,)))
    data = [{"dim_key": r["dim_key"], "name": r["name"], "pct": r["pct"], "n": r["n"]} for r in rows]
    if not data:
        return {}, {}
    return data[-1], data[0]


# ---------------- 导师提问 ----------------
def save_questions(video_id: int, user_id: int | None, questions: list[str]) -> int:
    """整组重建提问：空文本跳过、只保留前 3 条，重新出题不累积，旧作答随之作废。"""
    cleaned: list[str] = []
    for text in questions:
        q = (text or "").strip()[:500]
        if not q:
            continue
        cleaned.append(q)
        if len(cleaned) >= 3:
            break
    with get_conn() as conn:
        conn.execute("DELETE FROM qa_turns WHERE video_id = ?", (video_id,))
        for i, q in enumerate(cleaned, 1):
            conn.execute(
                "INSERT INTO qa_turns(video_id, user_id, idx, question, created_at) VALUES (?,?,?,?,?)",
                (video_id, user_id, i, q, now()))
        return len(cleaned)


def get_questions(video_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM qa_turns WHERE video_id = ? ORDER BY idx ASC",
                            (video_id,)).fetchall()
    return [{"idx": r["idx"], "question": r["question"], "answer": r["answer"],
             "ai_comment": r["ai_comment"], "status": r["status"]} for r in rows]


def save_qa_answer(video_id: int, idx: int, answer: str, ai_comment: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE qa_turns SET answer = ?, ai_comment = ?, status = 'answered', answered_at = ? "
            "WHERE video_id = ? AND idx = ?",
            (answer, ai_comment, now(), video_id, idx))
        return cur.rowcount > 0
