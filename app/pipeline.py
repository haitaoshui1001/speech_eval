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

from . import analyze, compress, db, face as face_mod, media, revise, transcribe
from . import tts as tts_mod
from . import voice as voice_mod
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
def _compress_stage(video_id: int, path: Path, notes: list[str]) -> Path:
    """体积超过压缩目标时先压再分析（进度 4–7%），返回真正要分析的文件路径。

    压缩放在流水线而不是上传请求里，是因为两遍编码在两核服务器上要跑几分钟，
    放进 HTTP 请求必然超时；这里失败会照常冒到 _worker，原始文件已被 compress 保住。
    非 MP4 原件压完会换成 .mp4 落盘，所以路径必须回传给后续抽帧抽音。
    """
    size = path.stat().st_size
    if not compress.needs_compress(size):
        return path
    db.update_video(video_id, orig_size=size)
    db.set_progress(video_id, "queued", f"压缩视频（{compress.human(size)} → 目标 {compress.target_hint()}）", 4)
    info = media.probe(path)
    last = {"p": 0}

    def report(pct: int, phase: str) -> None:
        # ffmpeg 每秒吐几十行进度，只在整数百分比变化时写库，避免刷爆 SQLite。
        if pct == last["p"]:
            return
        last["p"] = pct
        db.set_progress(video_id, "queued", f"{phase} {pct}%", 4 + int(pct * 3 / 100))

    res = compress.compress(path, info, on_progress=report)
    if res is None:
        return path
    fields = {"size": res.size, "compress_note": res.note}
    if res.renamed and res.path:
        fields["path"] = str(res.path)
        fields["filename"] = res.path.name
    db.update_video(video_id, **fields)
    notes.append(res.note)
    return res.path or path


def run_analysis(video_id: int, rubric: Rubric = DEFAULT_RUBRIC) -> dict:
    video = db.get_video(video_id)
    if video is None:
        raise RuntimeError(f"视频 {video_id} 不存在")

    path = Path(video["path"])
    if not path.exists():
        raise RuntimeError(f"视频文件丢失：{path}")

    early_notes: list[str] = []
    path = _compress_stage(video_id, path, early_notes)

    out_dir = _artifact_dir(video_id)
    db.set_progress(video_id, "analyzing", "抽取音视频与关键帧", 8)

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
    notes: list[str] = list(early_notes)

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
    client.reset_usage()
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
    try:
        db.set_progress(video_id, "analyzing", "生成导师提问", 99)
        if real:
            client.mark("导师提问")
            questions = analyze.build_questions(client, rubric, report, transcript,
                                                video["topic"], video["requirements"],
                                                note=notes)
        else:
            questions = analyze.fallback_questions(report, video["topic"])
        db.save_questions(video_id, video["user_id"], questions)
    except Exception:  # noqa: BLE001 - 出题失败不影响评价报告完成
        pass
    finally:
        db.add_usage(video_id, client.usage_summary())
    db.set_progress(video_id, "done", "分析完成", 100)
    return report


def propose_script(video_id: int, glossary: str = "",
                   client: QwenClient | None = None,
                   progress=None) -> revise.RevisionResult:
    """生成文字稿修订建议：按需一次调用，建议进 script_items，绝不碰原稿（方案 §5）。

    与导师出题同属「分析完成后的按需调用」，用量走 db.add_usage 增量记账；
    client 缺省时现建，自测注入桩即可不触网。成稿与否由学生决定，落在路由层。
    progress 是 (百分比, 阶段文案) 回调，由后台任务传入用于页面轮询显示。
    """
    video = db.get_video(video_id)
    transcript = video["transcript"] or ""
    own_client = client is None
    # 比对全稿是整条链路最慢的一次调用（实测 1600 字稿子 349 秒），沿用请求超时的
    # 180 秒会被读超时掐死在 35%，所以单独给一条更宽的预算；重试也收成 2 次——
    # 超时多半是模型真在慢慢想，再打三次只会白等半小时、白付三倍 token。
    cli = client or QwenClient(timeout=settings.revise_timeout, retries=2).mark("文字稿修订")

    def report(pct: int, phase: str) -> None:
        if progress is not None:
            progress(pct, phase)

    report(15, "通读文字稿 · 准备术语与红线约束")
    notes: list[str] = []
    try:
        report(35, "调用模型比对全稿（本步最慢，一千字上下要五六分钟，别关页面）")
        result = revise.run_revision(cli, transcript, video["topic"] or "",
                                     video["requirements"] or "",
                                     glossary=glossary.strip(), note=notes)
        report(75, "校验红线：只允许改错字与术语，越界条目自动丢弃")
    finally:
        if own_client:
            db.add_usage(video_id, cli.usage_summary())
    meta = {"source_hash": revise.source_hash(transcript), "model": result.model,
            "glossary": glossary.strip(), "dropped": result.dropped[:20],
            "warnings": result.warnings, "notes": notes, "proposed_at": db.now()}
    report(90, "写入建议清单")
    db.update_video(video_id, script_status="proposed",
                    script_items=revise.items_to_json(result.items),
                    script_text="", script_meta=meta)
    return result


def _propose_failed(video_id: int, exc: BaseException) -> None:
    """修订失败的落库口径：把英文超时报错翻译成可操作的中文。

    原样落「HTTPSConnectionPool(host=...) Read timed out. (read timeout=180)」时，
    学生看到的是一串英文栈，而里面唯一有用的线索（超时秒数）恰好是管理员该调的旋钮。
    这里既说清「稿子长属正常、可以再点一次」，也指出「修订请求超时」在哪调。
    """
    raw = f"{type(exc).__name__}: {exc}"
    if "timeout" in raw.lower() or "timed out" in raw.lower():
        raw = (f"模型比对全稿超过 {settings.revise_timeout} 秒仍未返回，本次生成中断"
               f"（原始报错：{raw[:160]}）。稿子偏长时属正常，可以重新点一次；"
               f"若反复超时，请管理员在设置页调大「修订请求超时」")
    db.update_video(video_id, script_status="failed", script_error=raw[:1500],
                    stage="修订建议生成失败", progress=100)


def run_propose(video_id: int, glossary: str = "") -> revise.RevisionResult:
    """修订建议后台任务：与朗读合成同一范式，页面靠 /status 轮询看进度。

    同步等模型是云端 504 的根因——网关先超时断连，后端却仍把结果写进了库，
    所以用户表现为「报 504，但过一会儿刷新结果在」。改成后台任务后浏览器
    立刻拿到跳转，进度与失败原因都落库，不再依赖一条长连接。
    status 全程保持 done：修订是评价完成后的按需动作，不能把页面打回进度视图。
    """
    db.update_video(video_id, script_status="running", script_error="",
                    stage="排队等待修订", progress=5)

    def report(pct: int, phase: str) -> None:
        db.update_video(video_id, stage=phase, progress=max(5, min(99, pct)))

    try:
        result = propose_script(video_id, glossary, progress=report)
        db.update_video(video_id, stage="修订建议已生成", progress=100)
        return result
    except Exception as exc:  # noqa: BLE001 - 兜底记录后原样上抛给 _job_worker
        _propose_failed(video_id, exc)
        raise


def run_propose_error(video_id: int, exc: BaseException) -> None:
    """修订任务的 _job_worker 兜底：失败只写修订侧状态。

    兜底若不指定落点，_job_worker 会按最早的调用方（朗读合成）写 tts_status，
    于是「修订建议超时」会被显示成「上次朗读合成失败」——两件事互不相干，
    学生看到一条从没跑过的朗读失败记录，只会去找根本不存在的音频问题。
    """
    _propose_failed(video_id, exc)


def run_tts(video_id: int, scene: str = "class", use_clone: bool = False) -> Path:
    """按需合成朗读示范（方案 §6.4）：闸门 → 生效稿 → 两级拼装 → 逐块合成下载 → 拼接落盘。

    videos.status 全程保持 done——合成是评价完成后的按需动作，不能把页面打回进度视图，
    所以进度只写 stage/progress 两列，结果落在 tts_status/tts_path/tts_meta 三列。
    use_clone=True 时改用本人的复刻音色（方案 §9）：音色与模型成对取出，缺一个就报
    人话错误，而不是把空音色递给接口让它回一句看不懂的 400。
    """
    video = db.get_video(video_id)
    if video is None:
        raise RuntimeError(f"视频 {video_id} 不存在")
    text = (video["script_text"] or "").strip() or (video["transcript"] or "").strip()
    ok, why = tts_mod.tts_feasibility(text)
    if not ok:
        raise tts_mod.TtsError(why)
    voice, vmodel = "", ""
    if use_clone:
        voice, vmodel = db.user_voice_for_tts(int(video["user_id"]))
        if not voice:
            raise tts_mod.TtsError("这个账号还没有可用的复刻音色，请先完成授权并创建音色")
    sc = scene if scene in tts_mod.SCENES else "class"
    out_dir = _artifact_dir(video_id) / "tts"
    db.update_video(video_id, tts_status="running", tts_error="",
                    stage="排队等待合成" if not voice else "排队等待合成（本人音色）", progress=5)
    client = QwenClient().mark("朗读合成")
    try:
        def report(pct: int, phase: str) -> None:
            db.update_video(video_id, stage=phase, progress=max(5, min(99, pct)))

        result = tts_mod.synthesize(client, text, out_dir=out_dir, scene=sc,
                                    topic=video["topic"] or "",
                                    requirements=video["requirements"] or "",
                                    dims=db.dims_for_video(video_id),
                                    duration=float(video["duration"] or 0.0),
                                    voice=voice, model=vmodel, progress=report)
        meta = result.to_meta()
        meta["scene"] = sc
        meta["warnings"] = result.warnings
        meta["synthesized_at"] = db.now()
        db.update_video(video_id, tts_status="done",
                        tts_path=str(result.path.relative_to(out_dir.parent)).replace("\\", "/"),
                        tts_meta=meta, tts_error="",
                        stage="朗读合成完成", progress=100)
        return result.path
    except Exception as exc:  # noqa: BLE001 - 兜底记录后原样上抛给 _job_worker
        db.update_video(video_id, tts_status="failed",
                        tts_error=f"{type(exc).__name__}: {exc}"[:1500],
                        stage="朗读合成失败", progress=100)
        raise
    finally:
        db.add_usage(video_id, client.usage_summary())


def _voice_dir(user_id: int) -> Path:
    """复刻样本目录按账号存，不放 artifact_dir/<video_id>：删视频会把那棵树整个 rmtree，
    正在跑的复刻任务会在读完样本之后、送出请求之前凭空丢文件。"""
    p = settings.data_dir / "voices" / str(user_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _drop_samples(user_id: int) -> None:
    """样本是学生人声的原样，用完即删；删不掉也不拦流程（下次同名覆盖）。"""
    for p in _voice_dir(user_id).glob("*.wav"):
        try:
            p.unlink()
        except OSError:
            pass


def run_voice(video_id: int) -> str:
    """账号级声音复刻（方案 §9）：从提交时所在视频挑一段连续朗读 → 建音色 → 写回账号。

    音色属于用户不属于视频，但进度借用该视频的 stage/progress 两列上报，页面就能沿用
    现成的 /videos/{id}/status 轮询，不必再造一套机制；「同一账号同时只跑一个复刻任务」
    靠 db.begin_voice_job 的行级抢锁，路由里的预检只给准信、防不住两个标签页。
    抢不到锁时只在本视频留一句说明，绝不能去写 error——锁是别人的任务持有的。
    """
    video = db.get_video(video_id)
    if video is None:
        raise RuntimeError(f"视频 {video_id} 不存在")
    user_id = int(video["user_id"])
    if not db.begin_voice_job(user_id):
        db.update_video(video_id, stage="该账号已有音色复刻任务，本次提交未执行", progress=100)
        return db.voice_status(user_id)["voice_id"]
    client = QwenClient().mark("声音复刻")
    try:
        def report(pct: int, phase: str) -> None:
            db.update_video(video_id, stage=phase, progress=max(5, min(99, pct)))

        report(10, "核对录音处理授权")
        if not db.voice_status(user_id)["consented"]:
            raise voice_mod.VoiceSampleError("授权已撤回，不能再处理这段录音，请重新勾选授权说明")
        report(25, "在原视频里挑连续朗读片段")
        sample, path = voice_mod.build_sample(
            Path(video["path"]), _voice_dir(user_id) / f"video_{video_id}.wav",
            duration=float(video["duration"] or 0.0))
        report(50, f"上传 {sample.duration:.1f} 秒样本（Base64 约 "
                   f"{voice_mod.estimate_b64_bytes(path) // 1024}KB）建音色")
        profile = client.create_voice(path, preferred_name=f"u{user_id}_{video['username']}")
        source = (f"取自视频 #{video_id}《{video['title']}》"
                  f"{sample.start:.1f}–{sample.end:.1f} 秒")
        note = ""
        if profile.fallback_mode:
            note = ("服务方以降级方式建成音色"
                    f"（{profile.fallback_reason or '未说明原因'}），"
                    "相似度可能不足，可重做音色或改用系统音色")
        db.save_user_voice(user_id, profile.voice, profile.target_model, source=source, note=note)
        db.update_video(video_id, stage="我的音色已就绪", progress=100)
        return profile.voice
    except Exception as exc:  # noqa: BLE001 - 兜底记录后原样上抛给 _job_worker
        db.mark_voice_error(user_id, f"{type(exc).__name__}: {exc}")
        db.update_video(video_id, stage="音色复刻失败", progress=100)
        raise
    finally:
        _drop_samples(user_id)
        db.add_usage(video_id, client.usage_summary())


def run_voice_error(video_id: int, exc: BaseException) -> None:
    """复刻任务的 _job_worker 兜底：崩在 run_voice 的 try 之外也要留下失败原因。"""
    video = db.get_video(video_id)
    if video is not None:
        db.mark_voice_error(int(video["user_id"]), f"{type(exc).__name__}: {exc}")
    db.update_video(video_id, stage="音色复刻失败", progress=100)


def drop_voice(user_id: int) -> str:
    """删除本人音色：先删远端、再清本地记录，返回被删除的音色标识。

    顺序很关键——本地记录是「远端还挂着这副嗓子」的唯一线索，先清本地而远端没删掉，
    就成了永远找不回的孤儿人声。远端报 404 说明本来就没有，按删除成功处理，
    否则用户点了删除却永远删不动。

    演示模式（没有密钥）压根不可能建出远端音色，这时只清本地记录：
    不然删一次就报一次网络错，学生永远撤不回授权。
    """
    st = db.voice_status(user_id)
    voice = st["voice_id"]
    client = QwenClient()
    if voice and client.enabled:
        client.mark("音色删除")
        try:
            client.delete_voice(voice, st["voice_model"] or st["target_model"])
        except QwenError as exc:
            if exc.status != 404:
                db.mark_voice_error(user_id, f"远端音色删除失败：{type(exc).__name__}: {exc}")
                raise
    db.clear_user_voice(user_id)
    _drop_samples(user_id)
    return voice


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


def _job_worker(fn, video_id: int, on_error=None) -> None:
    try:
        fn(video_id)
    except Exception as exc:  # noqa: BLE001 - 按需任务失败必须兜底留痕
        traceback.print_exc()
        try:
            # 每个任务自己已经在 except 里写过状态；这里按 on_error 指定的落点兜底，
            # 缺省仍按朗读合成处理，避免把旧的 tts 调用点一并改坏。
            if on_error is not None:
                on_error(video_id, exc)
            else:
                db.update_video(video_id, tts_status="failed",
                                tts_error=f"{type(exc).__name__}: {exc}"[:1500])
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    finally:
        _active.discard(video_id)


def submit(fn, video_id: int, on_error=None) -> bool:
    """提交按需后台任务（朗读合成、修订建议），与分析共用 _active：一个视频同时只跑一件事。"""
    if video_id in _active:
        return False
    _active.add(video_id)
    _pool_instance().submit(_job_worker, fn, video_id, on_error)
    return True


def is_running(video_id: int) -> bool:
    return video_id in _active


def shutdown() -> None:
    global _pool, _pool_size
    if _pool is not None:
        _pool.shutdown(wait=False)
        _pool = None
        _pool_size = 0
