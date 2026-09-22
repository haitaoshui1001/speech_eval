"""演讲视频评价系统 · Web 入口（FastAPI + Jinja2 服务端渲染）。

页面：登录 → 我的演讲（上传 + 一键分析 + 进度）→ 评价报告 →
历次对比（雷达图 + 维度 Δ + 趋势）→ 管理员后台（开通账号、Excel 批量导入）。
账号一律由管理员开通，站点不开放自助注册。
"""
from __future__ import annotations

import json
import math
import re
import shutil
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

import anyio.to_thread
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import (accounts, config, db, face as face_mod, pipeline,
               prompts, qwen)
from .analyze import (QA_ANSWER_FALLBACK, build_questions, channel_weights,
                      fallback_questions, review_answers)
from .config import settings
from .rubric import rubric as DEFAULT_RUBRIC
from .security import (make_session_token, password_ok, read_session_token,
                       valid_username, verify_password)

BASE = Path(__file__).resolve().parent
COOKIE = "sid"
ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".mpg", ".mpeg", ".wmv", ".3gp"}
ASSET_RE = re.compile(r"^(?:cover\.jpg|audio\.wav|frames/frame_\d{1,3}\.jpg)$")
PAGE_SIZE = 200


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    db.init_db()
    tune_thread_pool()
    stale = db.reset_stale_jobs()
    if stale:
        print(f"[startup] 清理中断任务 {stale} 条")
    print(f"[startup] 模式：{'真实千问' if settings.real_mode else '演示 mock'}；"
          f"评判模型 {settings.chat_model}｜画面 {settings.vlm_model}｜语音 {settings.omni_model}")
    print(f"[startup] 并发：分析 {settings.max_analyzers} 路，千问在途请求 "
          f"{settings.max_llm_requests} 个，网页线程 {anyio_thread_tokens()} 个")
    yield
    pipeline.shutdown()


app = FastAPI(title="演讲视频评价系统", lifespan=lifespan)
templates = Jinja2Templates(directory=str(BASE / "templates"))
(BASE / "static").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")


# ------------------------------------------------------------- 并发与启动
def anyio_thread_tokens() -> int:
    """当前 anyio 默认线程池的执行数；不在事件循环里时返回 0。"""
    try:
        return int(anyio.to_thread.current_default_thread_limiter().total_tokens)
    except Exception:  # noqa: BLE001 - 单测/脚本环境没有运行中的事件循环
        return 0


def tune_thread_pool() -> None:
    """放宽同步路由的工作线程额度，让网页在多人同时操作时不排队。

    本项目路由都写成同步 def，Starlette 把它们交给 anyio 的线程闸门执行，默认只有
    40 个执行位。上传（读盘写文件）、报告页（查库 + 渲染）、进度轮询都占一个位，
    10 个用户同时用很容易占满，表现为页面莫名转圈而不是报错。按 WEB_THREADS 上调即可，
    只增不减，避免把正在处理的请求挤掉。
    """
    want = max(8, settings.web_threads)
    try:
        limiter = anyio.to_thread.current_default_thread_limiter()
        if limiter.total_tokens < want:
            limiter.total_tokens = want
    except Exception:  # noqa: BLE001 - 拿不到闸门就用默认值，不影响启动
        pass


# ------------------------------------------------------------------ 工具
def fmt_ts(seconds: float | None) -> str:
    if not seconds:
        return "—"
    s = int(round(float(seconds)))
    return f"{s // 60:02d}:{s % 60:02d}"


def human_size(n: int | None) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def fmt_date(iso: str | None) -> str:
    return (iso or "")[:16].replace("T", " ")


def as_num(value) -> str:
    """15.0 → 15，0.5 → 0.5：去掉浮点尾巴，报告里更像手写分数。"""
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value if value is not None else "")


def thousands(value) -> str:
    try:
        return f"{int(float(value or 0)):,}"
    except (TypeError, ValueError):
        return "0"


templates.env.filters["ts"] = fmt_ts
templates.env.filters["size"] = human_size
templates.env.filters["dt"] = fmt_date
templates.env.filters["g"] = as_num
templates.env.filters["n"] = thousands


def render(request: Request, name: str, **ctx):
    ctx.setdefault("rubric", DEFAULT_RUBRIC)
    ctx.setdefault("real_mode", settings.real_mode)
    ctx.setdefault("real_mode_model", settings.chat_model)
    ctx.setdefault("nav", "")
    ctx.setdefault("palette", PALETTE)
    ctx.setdefault("asr_engine", settings.asr_engine)
    ctx.setdefault("asr_model", settings.asr_model)
    ctx.setdefault("vlm_model", settings.vlm_model)
    ctx.setdefault("omni_model", settings.omni_model)
    ctx.setdefault("whisper_model_size", settings.whisper_model_size)
    user = session_user(request)
    ctx["me"] = user
    return templates.TemplateResponse(request, name, ctx)


def session_user(request: Request) -> sqlite3.Row | None:
    token = request.cookies.get(COOKIE)
    parsed = read_session_token(token) if token else None
    if not parsed:
        return None
    row = db.get_user(parsed[0])
    return row


def login_redirect(request: Request) -> RedirectResponse:
    nxt = request.url.path
    if request.method != "GET" or nxt in {"/", "/login"}:
        nxt = "/dashboard"
    return RedirectResponse(f"/login?next={nxt}", status_code=303)


def owned_video(request: Request, video_id: int) -> tuple[sqlite3.Row, sqlite3.Row]:
    user = session_user(request)
    if user is None:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    video = db.get_video(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="视频不存在")
    if video["user_id"] != user["id"] and user["role"] != "admin":
        raise HTTPException(status_code=403, detail="无权访问该视频")
    return user, video


# ------------------------------------------------------------------ 认证页面
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    user = session_user(request)
    if user:
        return RedirectResponse("/dashboard", status_code=303)
    return render(request, "index.html")


@app.get("/guide", response_class=HTMLResponse)
def guide(request: Request):
    channels = {"text": "文本通道", "vision": "画面通道", "audio": "语音通道"}
    mapping = []
    for d in DEFAULT_RUBRIC.dimensions:
        weights = channel_weights(d)
        mapping.append({
            "name": d.name,
            "who": " + ".join(channels.get(k, k) for k in weights),
            "detail": "、".join(f"{channels.get(k, k)} {v * 100:.0f}%" for k, v in weights.items()),
        })
    return render(request, "guide.html", nav="guide", dim_channels=mapping)


REGISTER_CLOSED = "账号由管理员统一开通，请把姓名和联系方式交给管理员。"


@app.api_route("/register", methods=["GET", "POST"], response_class=HTMLResponse)
def register_closed():
    """自助注册已下线：老书签和外链一律送回登录页，并说明去哪拿账号。"""
    return RedirectResponse(f"/login?notice={quote(REGISTER_CLOSED)}", status_code=303)


def _set_session(resp: RedirectResponse, user: sqlite3.Row) -> None:
    resp.set_cookie(COOKIE, make_session_token(user["id"], user["role"]),
                    max_age=settings.session_days * 86400, httponly=True, samesite="lax")


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request, next: str = "/dashboard", error: str = "", notice: str = ""):
    return render(request, "login.html", nav="login", nxt=next, error=error, notice=notice)


@app.post("/login")
async def login_submit(request: Request, username: str = Form(""), password: str = Form(""),
                       next: str = Form("/dashboard")):
    row = db.get_user_by_name(username.strip())
    if row is None or not verify_password(password, row["password_hash"]):
        form = await request.form()
        return render(request, "login.html", nav="login", nxt=next, error="用户名或口令不正确",
                      form={"username": username})
    resp = RedirectResponse(next if str(next).startswith("/") and ".." not in str(next)
                            else "/dashboard", status_code=303)
    if row["role"] == "admin" and resp.headers["location"] == "/dashboard":
        # 管理员没有「我的演讲」，默认落点改成后台，省一次跳转
        resp = RedirectResponse("/admin", status_code=303)
    _set_session(resp, row)
    return resp


@app.post("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(COOKIE)
    return resp


# ------------------------------------------------------------------ 我的演讲
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, msg: str = ""):
    user = session_user(request)
    if user is None:
        return login_redirect(request)
    if user["role"] == "admin":
        # 管理员没有「我的演讲」，导航入口已撤掉，直接敲地址也一律送回后台
        return RedirectResponse("/admin", status_code=303)
    videos = db.list_user_videos(user["id"], limit=PAGE_SIZE)
    done = [v for v in videos if v["status"] == "done"]
    return render(request, "dashboard.html", nav="dashboard", videos=videos, done=done,
                  msg=msg, max_mb=settings.max_video_bytes // (1024 * 1024),
                  recurring=db.recurring_issues(user["id"]),
                  quota=db.upload_quota(user["id"]),
                  usage=db.token_usage(user["id"]))


@app.post("/videos")
async def upload(request: Request, title: str = Form(""), topic: str = Form(""),
                 requirements: str = Form(""), auto: str = Form(""),
                 video: UploadFile | None = File(None)):
    user = session_user(request)
    if user is None:
        return login_redirect(request)
    suffix = (Path(video.filename or "").suffix or ".mp4").lower() if video else ".mp4"
    if video is None or video.filename == "":
        return RedirectResponse("/dashboard?msg=请先选择视频文件", status_code=303)
    if suffix not in ALLOWED_SUFFIXES:
        return RedirectResponse(
            "/dashboard?msg=" + f"不支持的文件类型 {suffix}，请上传 mp4/mov/webm/mkv 等视频", status_code=303)
    quota = db.upload_quota(user["id"])
    if quota["remaining"] == 0:
        # 闸门必须在落盘之前：先写几百 MB 再拒绝，等于让额度用完的人照样打满磁盘
        why = ("该账号已被管理员停止上传" if quota["limit"] == 0
               else f"上传次数已用完（{quota['used']}/{quota['limit']}）")
        return RedirectResponse("/dashboard?msg=" + why + "，请联系管理员重置", status_code=303)

    settings.video_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.\-]+", "_", Path(video.filename).name)[:60] or "upload"
    dest = settings.video_dir / f"{user['id']}-{int(time.time())}-{safe}"
    limit = settings.max_video_bytes
    size = 0
    try:
        with dest.open("wb") as fh:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise ValueError(f"文件超过上限 {limit // (1024 * 1024)} MB")
                fh.write(chunk)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        return RedirectResponse("/dashboard?msg=" + str(exc), status_code=303)
    except Exception as exc:  # noqa: BLE001 - 落盘失败不应留半成品
        dest.unlink(missing_ok=True)
        return RedirectResponse("/dashboard?msg=上传失败：" + type(exc).__name__, status_code=303)

    vid = db.create_video(user["id"], title.strip() or Path(video.filename).stem,
                          Path(video.filename).name, str(dest), size,
                          topic=topic.strip(), requirements=requirements.strip())
    if auto == "1":
        pipeline.enqueue(vid)
        return RedirectResponse(f"/videos/{vid}", status_code=303)
    return RedirectResponse("/dashboard?msg=上传成功，点击「开始分析」即可评分", status_code=303)


@app.post("/videos/{video_id}/analyze")
def analyze(video_id: int, request: Request):
    user, video = owned_video(request, video_id)
    if video["status"] in {"queued", "transcribing", "analyzing"}:
        return RedirectResponse(f"/videos/{video_id}", status_code=303)
    db.update_video(video_id, error="")
    pipeline.enqueue(video_id)
    if user["role"] == "admin" and user["id"] != video["user_id"]:
        return RedirectResponse("/admin", status_code=303)
    return RedirectResponse(f"/videos/{video_id}", status_code=303)


@app.post("/videos/{video_id}/delete")
def delete_video(video_id: int, request: Request):
    _, video = owned_video(request, video_id)
    user = session_user(request)
    if user["role"] != "admin" and video["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="无权删除")
    db.delete_video(video_id)
    return RedirectResponse("/admin" if user["role"] == "admin" and user["id"] != video["user_id"]
                            else "/dashboard", status_code=303)


@app.get("/videos/{video_id}/status")
def video_status(video_id: int, request: Request):
    _, video = owned_video(request, video_id)
    return JSONResponse({"id": video_id, "status": video["status"], "stage": video["stage"],
                         "progress": video["progress"], "error": video["error"],
                         "running": pipeline.is_running(video_id)})


def _qa_redirect(video_id: int, msg: str = "") -> RedirectResponse:
    tail = "?msg=" + quote(msg) if msg else ""
    return RedirectResponse(f"/videos/{video_id}{tail}#qa", status_code=303)


@app.post("/videos/{video_id}/qa/generate")
def qa_generate(video_id: int, request: Request):
    _, video = owned_video(request, video_id)
    if video["status"] != "done":
        return _qa_redirect(video_id, "评价完成后才能生成导师提问")
    if db.get_questions(video_id):
        return _qa_redirect(video_id, "导师提问已存在")
    try:
        ev = db.get_evaluation(video_id)
        report = json.loads(ev["payload"] or "{}") if ev else {}
        if settings.real_mode:
            client = qwen.QwenClient().mark("导师提问")
            try:
                questions = build_questions(client, DEFAULT_RUBRIC, report,
                                            video["transcript"], video["topic"],
                                            video["requirements"])
            finally:
                db.add_usage(video_id, client.usage_summary())
        else:
            questions = fallback_questions(report, video["topic"])
        db.save_questions(video_id, video["user_id"], questions)
    except Exception:  # noqa: BLE001 - 出题失败只回执，不抛错页
        return _qa_redirect(video_id, "生成导师提问失败，请稍后重试")
    return _qa_redirect(video_id)


@app.post("/videos/{video_id}/qa/answer")
async def qa_answer(video_id: int, request: Request):
    _, video = owned_video(request, video_id)
    form = await request.form()
    questions = db.get_questions(video_id)
    if not questions:
        return _qa_redirect(video_id, "还没有导师提问")
    answers = {q["idx"]: str(form.get(f"answer_{q['idx']}") or "").strip() for q in questions}
    if any(not answers[q["idx"]] for q in questions):
        return _qa_redirect(video_id, "请把 3 个问题都回答后再提交")
    pairs = [(q["question"], answers[q["idx"]]) for q in questions]
    if settings.real_mode:
        ev = db.get_evaluation(video_id)
        report = json.loads(ev["payload"] or "{}") if ev else {}
        client = qwen.QwenClient().mark("导师点评")
        try:
            comments = review_answers(client, report, pairs, note=[])
        finally:
            db.add_usage(video_id, client.usage_summary())
    else:
        comments = [QA_ANSWER_FALLBACK] * len(pairs)
    for q, comment in zip(questions, comments):
        db.save_qa_answer(video_id, q["idx"], answers[q["idx"]], comment)
    return _qa_redirect(video_id, "已收到你的作答，导师点评已生成")


@app.get("/videos/{video_id}", response_class=HTMLResponse)
def video_page(request: Request, video_id: int, msg: str = ""):
    user, video = owned_video(request, video_id)
    if video["status"] != "done":
        return render(request, "progress.html", nav="dashboard", video=video,
                      can_edit=video["status"] in {"uploaded", "failed"})

    ev = db.get_evaluation(video_id)
    if ev is None:
        raise HTTPException(status_code=404, detail="评价记录缺失，请重新分析")
    report = json.loads(ev["payload"] or "{}")
    dims = [json.loads(r["detail"] or "{}") for r in db.get_dimension_rows(video_id)]
    frames = [f for f in report.get("frames", []) if ASSET_RE.match(f"frames/{f}")]
    face = report.get("face") or {}
    film = build_film(frames, report.get("stamps") or [], face.get("frames") or [])
    history = db.history_for_user(video["user_id"])
    rank = next((i + 1 for i, h in enumerate(history) if h["id"] == video_id), len(history))
    prev = None
    if rank > 1:
        prev = db.get_evaluation(history[rank - 2]["id"])
        prev_dims = db.dims_for_video(history[rank - 2]["id"]) if prev else {}
        prev = {"video": history[rank - 2], "dims": prev_dims}
    segments = json.loads(video["segments"] or "[]")
    qa = db.get_questions(video_id)
    return render(request, "report.html", nav="dashboard", video=video, report=report, dims=dims,
                  frames=frames, film=film, face=face, ev=ev, prev=prev, rank=rank,
                  total_runs=len(history),
                  segments=segments, is_owner=user["id"] == video["user_id"],
                  qa=qa, qa_answered=any(q["status"] == "answered" for q in qa), msg=msg,
                  user_rows=db.list_user_videos(video["user_id"], limit=PAGE_SIZE))


@app.get("/videos/{video_id}/play")
def play(video_id: int, request: Request):
    _, video = owned_video(request, video_id)
    path = Path(video["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="视频文件丢失")
    suffix = path.suffix.lstrip(".").lower() or "mp4"
    return _stream(path, request, f"video/{suffix}")


@app.get("/videos/{video_id}/asset/{name:path}")
def asset(video_id: int, name: str, request: Request):
    _, _video = owned_video(request, video_id)
    if not ASSET_RE.match(name):
        raise HTTPException(status_code=400, detail="非法资源名")
    path = settings.artifact_dir / str(video_id) / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="资源不存在")
    return _stream(path, request, "audio/wav" if name.endswith(".wav") else "image/jpeg")


def _stream(path: Path, request: Request, media_type: str) -> StreamingResponse:
    """支持 Range 的流式响应：浏览器需要 206 才能拖动进度条。"""
    size = path.stat().st_size
    start, end = 0, size - 1
    rng = re.match(r"bytes=(\d*)-(\d*)$", (request.headers.get("range") or "").strip())
    status = 200
    if rng:
        first, last = rng.group(1), rng.group(2)
        if first == "" and last:
            start = max(0, size - int(last))
        elif first:
            start = int(first)
            if last:
                end = min(int(last), size - 1)
        if start > end or start >= size:
            return JSONResponse({"detail": "range 不合法"}, status_code=416,
                                headers={"Content-Range": f"bytes */{size}"})
        status = 206

    def reader():
        with path.open("rb") as fh:
            fh.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = fh.read(min(512 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {"Accept-Ranges": "bytes", "Content-Length": str(end - start + 1),
               "Content-Range": f"bytes {start}-{end}/{size}"}
    return StreamingResponse(reader(), status_code=status, media_type=media_type, headers=headers)


# ------------------------------------------------------------------ 历次对比
@app.get("/compare", response_class=HTMLResponse)
def compare(request: Request, ids: str = "", msg: str = ""):
    user = session_user(request)
    if user is None:
        return login_redirect(request)
    if user["role"] == "admin":
        # 「历次对比」是用户看自己成长曲线的页面，管理员同样不再保留入口
        return RedirectResponse("/admin", status_code=303)
    history = db.history_for_user(user["id"])
    raw = ",".join(request.query_params.getlist("ids")) or ids
    picked = [int(x) for x in re.findall(r"\d+", raw)][:6]
    if not picked:
        picked = [h["id"] for h in history[-3:]]
    chosen = [h for h in history if h["id"] in picked]
    chosen.sort(key=lambda h: str(h["analyzed_at"] or ""))
    data = [dict(video=h, dims=db.dims_for_video(h["id"])) for h in chosen]
    radar = build_radar(data) if len(data) >= 2 else ""
    trend = build_trend(history) if len(history) >= 2 else ""
    trend_gaze = any(h["frontal_ratio"] is not None for h in history)
    deltas = build_deltas(data) if len(data) >= 2 else []
    return render(request, "compare.html", nav="compare", history=history, data=data,
                  radar=radar, trend=trend, trend_gaze=trend_gaze, deltas=deltas,
                  recurring=db.recurring_issues(user["id"]),
                  best_worst=db.user_best_worst(user["id"]), msg=msg)


PALETTE = ["#1f5f8b", "#b3341f", "#3d7a4b", "#8a6d1f", "#6b4a8f", "#2b2b2b"]


def _label_x(x: float, chars: int, anchor: str, size: float, unit: float, pad: float = 6.0) -> float:
    """把雷达标签压回画布内。中文按 1em 估宽（unit 取窄屏放大后的字号），
    长维度名若贴着边界就会被 viewBox 切掉，宁可让它向内压进网格区也不能缺字。"""
    w = chars * unit
    if anchor == "start":
        return min(x, size - w - pad)
    if anchor == "end":
        return max(x, w + pad)
    return max(pad + w / 2, min(x, size - pad - w / 2))


def build_radar(data: list[dict]) -> str:
    dims = DEFAULT_RUBRIC.dimensions
    n = len(dims)
    size = 500.0
    label_unit = 15.0
    cx = cy = size / 2
    r = 165.0
    pts_axes = []
    for i, d in enumerate(dims):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        x, y = cx + r * math.cos(ang), cy + r * math.sin(ang)
        pts_axes.append((d, x, y, ang))

    svg = [f'<svg viewBox="0 0 {size:.0f} {size:.0f}" class="radar" role="img" aria-label="各维度得分率雷达图">']
    for f in (0.25, 0.5, 0.75, 1.0):
        ring = " ".join(f"{cx + r * f * math.cos(-math.pi / 2 + 2 * math.pi * i / n):.1f},"
                        f"{cy + r * f * math.sin(-math.pi / 2 + 2 * math.pi * i / n):.1f}"
                        for i in range(n))
        svg.append(f'<polygon points="{ring}" fill="none" stroke="#ded6c6" stroke-width="1"/>')
    for d, x, y, ang in pts_axes:
        svg.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#ded6c6"/>')
        ly = cy + (r + 30) * math.sin(ang)
        anchor = "middle" if abs(math.cos(ang)) < 0.35 else ("start" if math.cos(ang) > 0 else "end")
        lx = _label_x(cx + (r + 34) * math.cos(ang), len(d.name), anchor, size, label_unit)
        svg.append(f'<text x="{lx:.1f}" y="{ly + 4:.1f}" text-anchor="{anchor}" class="radar-label">'
                   f'{d.name}</text>')
        svg.append(f'<text x="{lx:.1f}" y="{ly + 20:.1f}" text-anchor="{anchor}" class="radar-sub">'
                   f'{d.max_score:g} 分</text>')

    for k, item in enumerate(data):
        color = PALETTE[k % len(PALETTE)]
        coords = []
        for d, _x, _y, ang in pts_axes:
            ratio = float((item["dims"].get(d.key) or {}).get("ratio") or 0.0)
            rr = r * max(0.04, min(1.0, ratio))
            coords.append(f"{cx + rr * math.cos(ang):.1f},{cy + rr * math.sin(ang):.1f}")
        svg.append(f'<polygon points="{" ".join(coords)}" fill="{color}" fill-opacity="0.12" '
                   f'stroke="{color}" stroke-width="2"/>')
        for (d, _x, _y, ang), item2 in zip(pts_axes, [item]):
            ratio = float((item2["dims"].get(d.key) or {}).get("ratio") or 0.0)
            rr = r * max(0.04, min(1.0, ratio))
            svg.append(f'<circle cx="{cx + rr * math.cos(ang):.1f}" cy="{cy + rr * math.sin(ang):.1f}" '
                       f'r="3.5" fill="{color}"/>')
    svg.append("</svg>")
    return "".join(svg)


def build_film(names: list[str], stamps: list, measured: list[dict]) -> list[dict]:
    """胶片带：文件名 + 真实时刻 + 该帧的本地人脸测量结果。
    帧数超过 FACE_MAX_FRAMES 时测量会抽稀，所以按时刻就近匹配而不是按下标对齐；
    匹配不上就留 None，让页面上这一帧不带任何朝向标注，避免张冠李戴。"""
    by_ts = [(float(m.get("t") or 0.0), m) for m in measured]
    out: list[dict] = []
    for i, n in enumerate(names):
        ts = float(stamps[i]) if i < len(stamps) and stamps[i] is not None else None
        m = None
        if ts is not None and by_ts:
            cand = min(by_ts, key=lambda b: abs(b[0] - ts))
            if abs(cand[0] - ts) <= 2.0:
                m = cand[1]
        state = ""
        if m is not None:
            state = "ok" if m.get("frontal") else ("off" if m.get("profile") else "miss")
        out.append({"name": n, "stamp": (fmt_ts(ts) if ts is not None else ""),
                    "face": m, "state": state})
    return out


def build_trend(history: list[sqlite3.Row]) -> str:
    pts = [(h["title"], float(h["total"]), float(h["max_total"])) for h in history]
    if len(pts) < 2:
        return ""
    # 眼神接触率来自本地逐帧测量：老视频没测过就是 None，画图上留空而不是补一个 0，
    # 否则趋势线会出现一段假的「眼神突然变差」。
    gaze = [None if r["frontal_ratio"] is None
            else max(0.0, min(1.0, float(r["frontal_ratio"]))) for r in history]
    has_gaze = any(g is not None for g in gaze)
    w, h, pad = 720.0, (262.0 if has_gaze else 240.0), 34.0
    mx = max(p[2] for p in pts) or 100.0
    xs = [pad + (w - 2 * pad) * i / max(1, len(pts) - 1) for i in range(len(pts))]
    ys = [h - pad - (h - 2 * pad) * (p[1] / mx) for p in pts]
    svg = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" class="trend" role="img" '
           f'aria-label="{"总分与眼神接触率趋势" if has_gaze else "总分趋势"}">']
    for f in (0, 0.25, 0.5, 0.75, 1.0):
        yy = h - pad - (h - 2 * pad) * f
        svg.append(f'<line x1="{pad}" y1="{yy:.1f}" x2="{w - pad}" y2="{yy:.1f}" stroke="#e7e0d1"/>')
        svg.append(f'<text x="{pad - 8}" y="{yy + 4:.1f}" text-anchor="end" class="axis">{f * mx:g}</text>')
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    svg.append(f'<polyline points="{line}" fill="none" stroke="#b3341f" stroke-width="2.5"/>')
    for i, (x, y) in enumerate(zip(xs, ys)):
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#b3341f"/>'
                   f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" class="axis">'
                   f'{pts[i][1]:g}</text>'
                   f'<text x="{x:.1f}" y="{h - pad + 18:.1f}" text-anchor="middle" class="axis">'
                   f'第 {i + 1} 次</text>')
    if has_gaze:
        known = [(x, g) for x, g in zip(xs, gaze) if g is not None]
        if len(known) >= 2:
            gline = " ".join(f"{x:.1f},{h - pad - (h - 2 * pad) * g:.1f}" for x, g in known)
            svg.append(f'<polyline points="{gline}" fill="none" stroke="#1f5f8b" '
                       f'stroke-width="2" stroke-dasharray="6 4"/>')
        for x, g in known:
            gy = h - pad - (h - 2 * pad) * g
            svg.append(f'<circle cx="{x:.1f}" cy="{gy:.1f}" r="4" fill="#f8f4eb" '
                       f'stroke="#1f5f8b" stroke-width="2"/>'
                       f'<text x="{x:.1f}" y="{gy + 17:.1f}" text-anchor="middle" class="axis">'
                       f'{g * 100:.0f}%</text>')
        svg.append(f'<line x1="{pad}" y1="{h - 8:.1f}" x2="{pad + 18}" y2="{h - 8:.1f}" '
                   f'stroke="#b3341f" stroke-width="2.5"/>'
                   f'<text x="{pad + 24}" y="{h - 4:.1f}" class="axis">总分</text>')
        lx = pad + 78
        svg.append(f'<line x1="{lx}" y1="{h - 8:.1f}" x2="{lx + 18}" y2="{h - 8:.1f}" '
                   f'stroke="#1f5f8b" stroke-width="2" stroke-dasharray="6 4"/>'
                   f'<text x="{lx + 24}" y="{h - 4:.1f}" class="axis">'
                   f'眼神接触率（本地逐帧测量，仅供质控参考，不计入分数）</text>')
    svg.append("</svg>")
    return "".join(svg)


def build_deltas(data: list[dict]) -> list[dict]:
    first, last = data[0]["dims"], data[-1]["dims"]
    out = []
    for d in DEFAULT_RUBRIC.dimensions:
        a, b = first.get(d.key), last.get(d.key)
        if not a or not b:
            continue
        out.append({"name": d.name, "max": d.max_score, "from": a["score"], "to": b["score"],
                    "delta": round(b["score"] - a["score"], 2),
                    "pct": round((b["score"] - a["score"]) / d.max_score * 100, 1)})
    return sorted(out, key=lambda x: x["delta"], reverse=True)


# ------------------------------------------------------------------ 管理员
def _admin_page(request: Request, msg: str = "", error: str = "", ok_msg: bool = False,
                account_form: dict | None = None, batch: dict | None = None, topic: str = ""):
    """后台页统一出口：闸门 + 统计 + 账号相关回执，GET 与两个 POST 共用。"""
    user = session_user(request)
    if user is None:
        return login_redirect(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可访问")
    stats = db.admin_stats()
    dstat = db.admin_dim_stats(topic)
    worst = sorted(dstat["by_dim"], key=lambda d: (d["avg_score"] / d["max_score"])
                   if d["max_score"] else 0)[:3]
    return render(request, "admin.html", nav="admin", stats=stats, dstat=dstat,
                  users=db.list_users(),
                  videos=db.list_all_videos(limit=PAGE_SIZE), worst=worst, msg=msg, error=error,
                  account_form=account_form or {}, batch=batch, ok_msg=ok_msg,
                  admin_username=settings.admin_username,
                  template_name=accounts.TEMPLATE_NAME, template_columns=accounts.HEADERS,
                  template_max_rows=accounts.MAX_ROWS,
                  template_max_mb=accounts.MAX_BYTES // 1024 // 1024,
                  site_usage=db.token_usage(per_video=8),
                  upload_limit_default=settings.user_upload_limit,
                  me_usage=db.token_usage(user["id"]))


@app.get("/admin", response_class=HTMLResponse)
def admin(request: Request, msg: str = "", error: str = "", topic: str = ""):
    """后台首页。跳转提示走 query：msg 是成功回执（绿条），error 是失败说明（红条）。

    topic 只作用于「维度均分」这张表：空表示全部主题，（未填主题）是题目留空那一桶。
    """
    return _admin_page(request, msg=msg, error=error, ok_msg=bool(msg) and not error,
                       topic=topic.strip())


@app.post("/admin/users/create")
def admin_create_user(request: Request, username: str = Form(""), password: str = Form(""),
                      display_name: str = Form(""), role: str = Form("user")):
    """单个开通账号：管理员现场填用户名 + 口令，口令只在提交这一刻存在。"""
    gate = _admin_gate(request)
    if gate is not None:
        return gate
    username = username.strip()
    err = ""
    if not valid_username(username):
        err = "用户名需 3-32 位，仅限字母、数字与 . _ -"
    elif not password_ok(password):
        err = "登录口令至少 6 位，建议用随机串"
    elif role not in {"user", "admin"}:
        err = "角色只能是普通用户或管理员"
    else:
        try:
            db.create_user(username, password, display_name.strip(), role)
        except ValueError as exc:
            err = str(exc)
    if err:
        return _admin_page(request, error=err,
                           account_form={"username": username, "display_name": display_name.strip(),
                                         "role": role if role in {"user", "admin"} else "user"})
    who = f"（{display_name.strip()}）" if display_name.strip() else ""
    return RedirectResponse("/admin?msg=" + quote(
        f"已开通账号 {username}{who}，请把口令线下转告本人；系统只存哈希，之后无法查看"),
        status_code=303)


@app.post("/admin/users/{user_id}/password")
def admin_reset_password(user_id: int, request: Request, password: str = Form("")):
    """忘记口令时由管理员重置：只改哈希，用户名、角色与历史视频都不动。"""
    gate = _admin_gate(request)
    if gate is not None:
        return gate
    row = db.get_user(user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="账号不存在")
    if not password_ok(password):
        return RedirectResponse("/admin?error=" + quote(
            f"重置给 {row['username']} 的口令至少 6 位"), status_code=303)
    db.set_password(user_id, password)
    return RedirectResponse("/admin?msg=" + quote(
        f"{row['username']} 的登录口令已重置，请转告本人尽快改密"), status_code=303)


@app.post("/admin/users/{user_id}/quota")
def admin_set_quota(user_id: int, request: Request, used: str = Form(""),
                    limit: str = Form(""), action: str = Form("save")):
    """调整上传额度：三个按钮共用一张表单，靠 action 区分意图。

    save = 按输入框写；reset = 已用次数清零（上限不动）；default = 取消单独上限、回到跟随全局
    USER_UPLOAD_LIMIT。limit 输入框留空表示不改这一列，填 0 表示禁止该账号上传。
    """
    gate = _admin_gate(request)
    if gate is not None:
        return gate
    row = db.get_user(user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="账号不存在")
    used_raw = "0" if action == "reset" else used.strip()
    # 上限框是 type=number，人不可能在里面打出 "default"，所以恢复默认只能靠按钮的 action 表达
    limit_raw = "default" if action == "default" else limit.strip()
    try:
        new_used = int(used_raw) if used_raw != "" else None
        if limit_raw == "":
            new_limit: int | None | str = ""
        elif limit_raw == "default":
            new_limit = None
        else:
            new_limit = int(limit_raw)
    except ValueError:
        return RedirectResponse("/admin?error=" + quote("上传次数与上限都要填非负整数"),
                                status_code=303)
    if (new_used is not None and new_used < 0) or (isinstance(new_limit, int) and new_limit < 0):
        return RedirectResponse("/admin?error=" + quote("上传次数与上限不能为负数"), status_code=303)
    db.set_upload_quota(user_id, used=new_used, limit=new_limit)
    now = db.upload_quota(user_id)
    left = "不限次数" if now["remaining"] is None else f"已用 {now['used']}/{now['limit']}"
    return RedirectResponse("/admin?msg=" + quote(
        f"{row['username']} 的上传额度已更新：{left}"), status_code=303)


@app.get("/admin/users/template")
def admin_user_template(request: Request):
    """下载 Excel 模板（含示例行与「填写说明」表，原样上传也不会误建示例账号）。"""
    gate = _admin_gate(request)
    if gate is not None:
        return gate
    return Response(content=accounts.build_template(), media_type=accounts.XLSX_MIME,
                    headers={"Content-Disposition":
                             f"attachment; filename*=UTF-8''{quote(accounts.TEMPLATE_NAME)}"})


def _admin_gate(request: Request):
    """非 GET 的管理员闸门：未登录 → 去登录页，非管理员 → 403；放行返回 None。"""
    user = session_user(request)
    if user is None:
        return login_redirect(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可操作")
    return None


async def _read_xlsx(file: UploadFile) -> bytes:
    """小文件大小保护：超过上限立刻停读，不把整个大文件先灌进内存。"""
    buf = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        buf += chunk
        if len(buf) > accounts.MAX_BYTES:
            raise accounts.AccountError(
                f"文件超过 {accounts.MAX_BYTES // 1024 // 1024} MB 上限，请分次导入")
    return bytes(buf)


@app.post("/admin/users/import", response_class=HTMLResponse)
async def admin_import_users(request: Request, file: UploadFile = File(None)):
    """Excel 批量导入：逐行校验后只新增账号，退回行带行号显示在回执里。"""
    gate = _admin_gate(request)
    if gate is not None:
        return gate
    if file is None or not (file.filename or "").strip():
        return _admin_page(request, error="请先选择一个 .xlsx 文件（可先下载模板）")
    try:
        table = accounts.read_table(await _read_xlsx(file), file.filename)
    except accounts.AccountError as exc:
        return _admin_page(request, error=str(exc))
    rows = accounts.plan_users(table, [u["username"] for u in db.list_users()])
    for row in rows:
        if not row.ok:
            continue
        try:
            db.create_user(row.username, row.password, row.display_name)
        except ValueError as exc:  # 并发导入撞上同一用户名，退回而不是整批失败
            row.status = "error"
            row.errors.append(str(exc))
        else:
            row.status = "created"
    batch = accounts.summarize(rows, file.filename)
    return _admin_page(request, batch=batch)


@app.post("/admin/users/{user_id}/delete")
def admin_delete_user(user_id: int, request: Request):
    user = session_user(request)
    if user is None or user["role"] != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可操作")
    if user_id == user["id"]:
        return RedirectResponse("/admin?error=" + quote("不能删除当前登录的管理员账号"),
                                status_code=303)
    db.delete_user(user_id)
    return RedirectResponse("/admin?msg=" + quote("账号及其视频、评价与抽帧产物已删除"),
                            status_code=303)


# ------------------------------------------------------------------ 系统设置（.env + 大模型提示词）
SAVE_NOTE = "改动直接写回 .env 并当场热刷新，无需重启服务；新设置用于其后的请求与分析任务。"
PROMPT_SAVE_NOTE = ("提示词写回独立的 提示词.json，只记录与内置默认不同的块；改回原样或整块清空即恢复默认，"
                    "无需重启服务，新提示词用于其后的分析任务。")


def _settings_page(request: Request, msg: str = "", error: str = "", overrides=None,
                   prompt_overrides=None):
    return render(request, "settings.html", nav="settings", groups=config.settings_overview(overrides),
                  msg=msg, error=error, save_note=SAVE_NOTE,
                  env_file=str(config.ENV_FILE), env_exists=config.ENV_FILE.exists(),
                  field_total=len(config.ENV_FIELDS),
                  prompt_groups=prompts.overview(prompt_overrides),
                  prompt_save_note=PROMPT_SAVE_NOTE,
                  prompts_file=str(prompts.PROMPTS_FILE),
                  prompts_exists=prompts.PROMPTS_FILE.exists(),
                  prompt_total=len(prompts.PROMPT_FIELDS),
                  prompt_custom=prompts.custom_count())


@app.get("/settings", response_class=HTMLResponse)
def settings_page_get(request: Request, msg: str = "", error: str = ""):
    user = session_user(request)
    if user is None:
        return login_redirect(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可访问")
    return _settings_page(request, msg=msg, error=error)


def _admin_account_issues(updates: dict[str, str]) -> list[str]:
    """管理员账号要同时改 .env 与数据库，先把明显冲突挡在写盘之前。"""
    problems: list[str] = []
    name = (updates.get("ADMIN_USERNAME") or "").strip()
    if name and not valid_username(name):
        problems.append("管理员账号需 3-32 位，仅限字母、数字与 . _ -")
    elif name:
        clash = db.get_user_by_name(name)
        if clash is not None and clash["role"] != "admin":
            problems.append(f"用户名 {name} 已被其他账号占用")
    password = updates.get("ADMIN_PASSWORD") or ""
    if password and not password_ok(password):
        problems.append("管理员口令至少 6 位")
    return problems


@app.post("/settings")
async def settings_submit(request: Request):
    user = session_user(request)
    if user is None:
        return login_redirect(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可操作")
    form = await request.form()
    raw = {key: str(form.get(key, "") or "") for key in config.ENV_BY_KEY}
    clears = tuple(key for key in config.ENV_BY_KEY if form.get(f"{key}__clear"))
    updates, errors = config.validate_env_updates(raw, clears)
    errors += _admin_account_issues(updates)
    if errors:
        return _settings_page(request, error="；".join(errors), overrides=raw)

    changed = config.save_settings(updates)
    try:
        synced = db.sync_admin_login(updates.get("ADMIN_USERNAME", ""), updates.get("ADMIN_PASSWORD", ""))
    except ValueError as exc:
        synced = f".env 已保存，但管理员账号未同步：{exc}"
    if not changed:
        flash = "没有需要保存的改动"
    else:
        keys = "、".join(changed[:8]) + ("等" if len(changed) > 8 else "")
        flash = f"已保存 {len(changed)} 项（{keys}）"
        if synced:
            flash += f"｜{synced}"
    resp = RedirectResponse(f"/settings?msg={quote(flash)}", status_code=303)
    _set_session(resp, user)  # 换签名密钥后也保持当前管理员登录
    return resp


@app.post("/settings/reload")
def settings_reload(request: Request):
    """用户在磁盘上手动改过 .env 时，用它把文件值重新灌进运行时。"""
    user = session_user(request)
    if user is None:
        return login_redirect(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可操作")
    config.refresh_runtime()
    resp = RedirectResponse("/settings?msg=" + quote("已按磁盘上的 .env 重新加载当前设置"), status_code=303)
    _set_session(resp, user)
    return resp


@app.post("/settings/prompts")
async def settings_prompts_submit(request: Request):
    """保存大模型提示词：与 .env 分开落盘，两边互不影响。"""
    user = session_user(request)
    if user is None:
        return login_redirect(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可操作")
    form = await request.form()
    raw = {key: str(form.get(key, "") or "") for key in prompts.PROMPT_BY_KEY}
    changed, errors = prompts.save(raw)
    if errors:
        # 一个字节都不写，把管理员输入的内容原样退回页面，避免白改一遍。
        return _settings_page(request, error="；".join(errors), prompt_overrides=raw)
    if not changed:
        flash = "提示词没有变化（与内置默认一致的块不写入文件）"
    else:
        flash = f"已保存提示词，当前 {prompts.custom_count()} 块为管理员自定义"
    resp = RedirectResponse(f"/settings?msg={quote(flash)}", status_code=303)
    _set_session(resp, user)
    return resp


@app.post("/settings/prompts/reset")
def settings_prompts_reset(request: Request):
    """全部恢复内置默认：清掉落盘的覆盖，不删代码里的默认文本。"""
    user = session_user(request)
    if user is None:
        return login_redirect(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可操作")
    restored = prompts.reset_all()
    flash = (f"已恢复 {len(restored)} 块内置默认提示词" if restored
             else "当前本来就是全部内置默认，无需恢复")
    resp = RedirectResponse(f"/settings?msg={quote(flash)}", status_code=303)
    _set_session(resp, user)
    return resp


@app.get("/health")
async def health():
    # 只有事件循环里才读得到 anyio 线程闸门，所以本接口保持 async：先取网关额度，
    # 再把要查库的部分交回工作线程，避免占用循环。
    tokens = anyio_thread_tokens()
    return await anyio.to_thread.run_sync(_health_payload, tokens)


def _health_payload(web_threads: int) -> dict:
    return {"ok": True, "mode": "real" if settings.real_mode else "mock",
            "key_source": settings.api_key_source, "rubric": DEFAULT_RUBRIC.version,
            "models": {"chat": settings.chat_model, "vision": settings.vlm_model,
                       "audio": settings.omni_model, "asr": settings.asr_model},
            "asr_engine": settings.asr_engine,
            "asr_engine_label": ("本地 faster-whisper" if settings.asr_engine == "whisper"
                                 else "云端 qwen3-asr-flash"),
            "whisper_model_size": settings.whisper_model_size,
            "ffmpeg": bool(shutil.which("ffmpeg")),
            "face_metrics": face_mod.available(),
            # 可选层降级时说清为什么降级：装到不兼容的 opencv 大版本是真实踩过的坑，
            # 光给一个 false 只会让人去翻 traceback。
            "face_metrics_note": face_mod.unavailable_reason(),
            "concurrency": {"analyzers": settings.max_analyzers,
                            "pool_size": pipeline.pool_size(),
                            "running": pipeline.queue_depth(),
                            "llm_requests": settings.max_llm_requests,
                            "llm_in_flight": qwen.GATE.in_flight,
                            "web_threads": web_threads},
            "tokens": db.token_usage()}
