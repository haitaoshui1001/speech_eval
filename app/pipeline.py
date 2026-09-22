"""分析流水线：媒体抽取 → 转录 → 文本/图像/语音三通道评分 → 聚合质控 → 入库。

同步入口 run_analysis(video_id) 供脚本与测试使用；
异步入口 enqueue(video_id) 供 Web 后台线程池使用。
"""
from __future__ import annotations

import hashlib
import random
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import analyze, db, face as face_mod, media, transcribe
from .analyze import Aggregated, TimingCheck
from .config import settings
from .qwen import QwenClient, QwenError
from .rubric import Rubric, rubric as DEFAULT_RUBRIC

MAX_AUDIO_SECONDS = 15 * 60

MOCK_TRANSCRIPT = (
    "Good morning everyone. Today I want to talk about whether artificial intelligence will "
    "replace English teachers. Honestly, I used to think it would. My phone corrects my grammar "
    "faster than any human, and it never gets tired of my mistakes. But last semester I watched "
    "my own classmate give up on speaking because she was afraid of being laughed at. No app "
    "noticed that. Her teacher noticed, and she stayed after class for twenty minutes just to "
    "listen. So my answer is: AI will change the job, but it will not replace the teacher. "
    "Technology gives us feedback, and feedback is not the same thing as encouragement. "
    "For the industry, I believe the demand will move from 'people who know English' to "
    "'people who can teach motivation'. That is a skill no model can copy. To sum up, machines "
    "correct our sentences, teachers correct our courage. Thank you."
)


def _artifact_dir(video_id: int) -> Path:
    p = settings.artifact_dir / str(video_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _flush_usage(client: QwenClient, video_id: int) -> dict:
    """把累计的模型用量写回视频行；分析中途失败也能留下已消耗的 token。"""
    usage = client.usage_summary()
    try:
        db.update_video(video_id, prompt_tokens=usage["prompt"],
                        completion_tokens=usage["completion"], total_tokens=usage["total"],
                        api_calls=usage["calls"], token_detail=usage)
    except Exception:  # noqa: BLE001 - 统计失败不应影响评价
        traceback.print_exc()
    return usage


# --------------------------------------------------------------- 演示模式
def _mock_channels(rubric: Rubric, duration: float, seed: str) -> list[dict]:
    """没有 API key 时也要能跑通全流程：按 seed 造一份可信的模拟评审。"""
    rng = random.Random(int(hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8], 16))
    base_pct = rng.uniform(0.62, 0.86)
    channels: list[dict] = []
    for ch_name in ("text", "vision", "audio"):
        keys = analyze.channel_dim_keys(rubric, ch_name)
        scores: dict[str, dict] = {}
        for k in keys:
            dim = rubric.by_key()[k]
            pct = max(0.4, min(0.97, base_pct + rng.uniform(-0.1, 0.1)))
            scores[k] = {
                "score": analyze._clamp(round(dim.max_score * pct * 2) / 2, dim.max_score),
                "confidence": round(rng.uniform(0.45, 0.8), 2),
                "strengths": [f"（演示数据）{dim.name}整体表现稳定，可作为基线参考"],
                "issues": [{
                    "desc": f"（演示数据）{dim.name}仍有提升空间，建议录入真实 API key 后重跑",
                    "quote": "", "at": "",
                    "fix": "在 .env 中填入 DASHSCOPE_API_KEY 后重新点击分析",
                    "example": "",
                }],
                "notes": "演示模式：分数由随机数生成，仅供界面与流程验证，不作为真实评价。",
            }
        channels.append({
            "channel": ch_name, "model": "mock", "scores": scores,
            "observations": "演示模式：未接入在线评审服务，以上内容为界面占位数据。",
        })
    return channels


# --------------------------------------------------------------- 单通道执行
def _run_channel(client: QwenClient, rubric: Rubric, name: str, *, transcript: str,
                 frames: list[Path], stamps: list[float], face, audio: Path | None,
                 topic: str, requirements: str, timing: TimingCheck,
                 note: list[str] | None = None) -> tuple[dict | None, str]:
    try:
        if name == "text":
            return analyze.run_text_channel(client, rubric, transcript, topic, requirements,
                                            timing, note=note), ""
        if name == "vision":
            if not frames:
                return None, "画面通道跳过：未抽到有效关键帧"
            return analyze.run_vision_channel(client, rubric, frames, topic, timing,
                                              transcript, note=note,
                                              stamps=stamps, face=face), ""
        if name == "audio":
            if audio is None:
                return None, "语音通道跳过：未抽到有效音频"
            return analyze.run_audio_channel(client, rubric, audio, transcript, timing,
                                             note=note), ""
    except QwenError as exc:
        return None, f"{name} 通道失败：{exc}"
    except Exception as exc:  # noqa: BLE001 - 通道降级不应中断整体分析
        return None, f"{name} 通道异常：{type(exc).__name__}: {exc}"
    return None, f"未知通道 {name}"


# --------------------------------------------------------------- 主流程
def run_analysis(video_id: int, rubric: Rubric = DEFAULT_RUBRIC) -> dict:
    video = db.get_video(video_id)
    if video is None:
        raise RuntimeError(f"视频 {video_id} 不存在")

    out_dir = _artifact_dir(video_id)
    db.set_progress(video_id, "analyzing", "抽取音视频与关键帧", 8)

    path = Path(video["path"])
    if not path.exists():
        raise RuntimeError(f"视频文件丢失：{path}")

    info = media.probe(path)
    duration = info.duration
    thumb = media.video_thumbnail(path, out_dir / "cover.jpg", at=min(1.5, duration / 3))
    audio = media.extract_audio(path, out_dir, max_seconds=MAX_AUDIO_SECONDS)
    frames: list[Path] = []
    stamps: list[float] = []
    facial = None
    if settings.use_vision_channel:
        frameset = media.extract_frames_timed(path, out_dir, duration)
        frames, stamps = frameset.paths, frameset.stamps
        if frames:
            db.set_progress(video_id, "analyzing", f"逐帧测量人脸朝向（{len(frames)} 帧）", 15)
            facial = face_mod.measure(frames, stamps)

    db.update_video(video_id, duration=duration, width=info.width, height=info.height,
                    frontal_ratio=facial.frontal_ratio if facial else None,
                    thumb=str(thumb.relative_to(settings.data_dir)).replace("\\", "/") if thumb else "")

    real = settings.real_mode
    client = QwenClient()
    note = f"{video['id']}-{video['title']}"
    notes: list[str] = []

    db.set_progress(video_id, "transcribing", "转录语音", 22)
    client.mark("转录 ASR")
    if real:
        result = transcribe.transcribe(path, audio, duration, client=client)
    else:
        result = transcribe.transcribe(path, audio, duration, mock_text=MOCK_TRANSCRIPT)
    transcript = result.text.strip()
    _flush_usage(client, video_id)

    timing = analyze.check_timing(duration, video["topic"], video["requirements"])
    db.set_progress(video_id, "analyzing", "多模态评价中", 45)

    wanted = ["text"]
    if settings.use_vision_channel and rubric and any(
            "vision" in analyze.channel_weights(d) for d in rubric.dimensions):
        wanted.append("vision")
    if settings.use_audio_channel and any(
            "audio" in analyze.channel_weights(d) for d in rubric.dimensions):
        wanted.append("audio")

    channels: list[dict] = []
    qc_extra: list[str] = []
    if not real:
        channels = _mock_channels(rubric, duration, note)
        qc_extra.append("演示模式：未检测到有效 DASHSCOPE_API_KEY，三通道结果由模拟数据生成，"
                        "评分不代表真实评价。配置 key 后重新分析即可。")
    else:
        for i, name in enumerate(wanted):
            db.set_progress(video_id, "analyzing", f"{_cn_channel(name)}通道评价中",
                            45 + int(35 * (i + 1) / max(len(wanted), 1)))
            client.mark(f"{_cn_channel(name)}通道")
            ch, warn = _run_channel(client, rubric, name, transcript=transcript, frames=frames,
                                    stamps=stamps, face=facial,
                                    audio=audio, topic=video["topic"],
                                    requirements=video["requirements"], timing=timing,
                                    note=notes)
            _flush_usage(client, video_id)
            if ch:
                channels.append(ch)
            if warn:
                qc_extra.append(warn)
        if not any(c["channel"] == "text" for c in channels):
            raise RuntimeError("文本通道未能产出评分，无法生成报告（请检查 API key 与模型额度）")
        qc_extra.extend(analyze.apply_transcript_credibility(channels, result.credible))

    db.set_progress(video_id, "analyzing", "聚合分数与证据核验", 84)
    agg: Aggregated = analyze.aggregate(rubric, channels, transcript, timing,
                                        stamps=stamps, face=facial)
    agg.qc.extend(dict.fromkeys(notes))
    agg.qc.extend(qc_extra + result.warnings)

    narr = analyze.fallback_narrative(agg)
    if real:
        db.set_progress(video_id, "analyzing", "生成优点/缺点/建议与总结", 91)
        history_note = _history_note(video["user_id"], video_id, agg)
        client.mark("评语生成")
        narr = analyze.build_narrative(client, rubric, agg, transcript, video["topic"],
                                       history_note, note=notes)
    usage = _flush_usage(client, video_id)
    if usage["calls"]:
        tail = f"（其中 {usage['estimated_calls']} 次为估算）" if usage["estimated_calls"] else ""
        agg.qc.append(f"本次分析共发起在线评审请求 {usage['calls']} 次，"
                      f"消耗 {usage['total']} token（输入 {usage['prompt']} / 输出 {usage['completion']}）{tail}")
    agg.qc = list(dict.fromkeys(agg.qc))

    report = analyze.build_report(rubric, agg, narr, channels, timing, transcript,
                                  engine=result.engine, model=client_key_model())
    report["frames"] = [str(p.name) for p in frames]
    report["stamps"] = stamps
    report["face"] = facial.to_dict() if facial else None
    report["media"] = {"duration": duration, "width": info.width, "height": info.height,
                       "has_audio": audio is not None, "mock": not real}
    report["usage"] = usage

    db.set_progress(video_id, "analyzing", "写入数据库", 97)
    db.save_evaluation(video, report, model=",".join(sorted({c.get("model", "") for c in channels})))
    db.update_video(video_id, transcript=transcript, segments=result.segments,
                    engine=result.engine, model=report["provenance"]["model"],
                    channels=[{"channel": c["channel"], "model": c.get("model", ""),
                               "keys": list(c.get("scores", {}).keys())} for c in channels],
                    qc=agg.qc, analyzed_at=db.now())
    db.set_progress(video_id, "done", "分析完成", 100)
    return report


def client_key_model() -> str:
    return "mock" if not settings.real_mode else settings.chat_model


def _cn_channel(name: str) -> str:
    return {"text": "文本", "vision": "画面", "audio": "语音"}.get(name, name)


def _history_note(user_id: int, video_id: int, agg: Aggregated) -> str:
    rows = db.history_for_user(user_id)
    if len(rows) <= 1:
        return ""
    best, worst = db.user_best_worst(user_id)
    recurring = db.recurring_issues(user_id, exclude_video=video_id)
    parts = [f"该用户已完成 {len(rows)} 次分析（本次不计入）。"]
    if best:
        parts.append(f"历史最强维度：{best['name']}，平均得分率 {best['pct']}%。")
    if worst:
        parts.append(f"历史最弱维度：{worst['name']}，平均得分率 {worst['pct']}%。")
    if recurring:
        items = "；".join(f"{r['label']}（{r['times']} 次）" for r in recurring[:5])
        parts.append(f"跨次反复出现的问题：{items}。请在建议中明确指出是否已改善。")
    prev = rows[-1]
    parts.append(f"上一次总分：{prev['total']}/{prev['max_total']}（{prev['title']}）。")
    return "\n".join(parts)


# --------------------------------------------------------------- 后台执行
_pool: ThreadPoolExecutor | None = None
_pool_size = 0
_pool_lock = threading.Lock()
_active: set[int] = set()


def _pool_instance() -> ThreadPoolExecutor:
    """取分析线程池；额度变了且当前空闲时重建。

    ThreadPoolExecutor 一旦创建就不会自己长个儿，所以在提交任务前检查一次：
    MAX_ANALYZERS 改过且没有在跑/排队的视频，就换一个新池，管理员调并发不用重启服务。
    有任务在跑时保持旧池，避免腰斩正在分析的视频。
    """
    global _pool, _pool_size
    want = max(1, settings.max_analyzers)
    with _pool_lock:
        if _pool is not None and _pool_size != want and not _active:
            old, _pool = _pool, None
            old.shutdown(wait=False)
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=want, thread_name_prefix="analyzer")
            _pool_size = want
    return _pool


def pool_size() -> int:
    """当前分析池的执行数，未创建时返回 0。"""
    return _pool_size


def queue_depth() -> int:
    """在跑 + 排队的视频数。"""
    return len(_active)


def _worker(video_id: int) -> None:
    try:
        run_analysis(video_id)
    except Exception as exc:  # noqa: BLE001 - 后台任务必须兜底
        traceback.print_exc()
        msg = f"{type(exc).__name__}: {exc}"
        try:
            db.set_progress(video_id, "failed", "分析失败", 100, error=msg[:1500])
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    finally:
        _active.discard(video_id)


def enqueue(video_id: int) -> bool:
    """提交后台分析；返回是否 newly queued。"""
    if video_id in _active:
        return False
    _active.add(video_id)
    db.set_progress(video_id, "queued", "排队中", 3)
    _pool_instance().submit(_worker, video_id)
    return True


def is_running(video_id: int) -> bool:
    return video_id in _active


def shutdown() -> None:
    global _pool, _pool_size
    if _pool is not None:
        _pool.shutdown(wait=False)
        _pool = None
        _pool_size = 0
