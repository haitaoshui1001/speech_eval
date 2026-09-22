"""语音转录：优先千问 ASR（分块，块首打时间标签），可切换本地 faster-whisper。"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings
from .qwen import QwenClient, QwenError
from . import media

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]+(?:['’-][A-Za-z]+)?")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class TranscriptResult:
    text: str = ""
    segments: list[dict] = field(default_factory=list)
    engine: str = ""
    model: str = ""
    warnings: list[str] = field(default_factory=list)
    credible: bool = True


def fmt_ts(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def credibility(text: str, duration: float) -> tuple[bool, str]:
    """判断转写结果是否像“真实讲话”。

    ASR 在纯音乐/静音上会产生极短或纯数字的幻觉输出（例如把示音正弦波转成 "100.0"），
    这类文本不能作为文本通道评分依据，必须显式标记，避免整份报告被静默清零。
    """
    body = (text or "").strip()
    words = _WORD_RE.findall(body)
    cjk = _CJK_RE.findall(body)
    tokens = len(words) + len(cjk)
    if tokens < 8:
        return False, f"转写仅得到 {tokens} 个词，未检出连续人声（文本通道结论仅供参考）"
    letters = sum(len(w) for w in words) + len(cjk)
    if tokens and letters / max(tokens, 1) < 1.8:
        return False, "转写内容几乎不含实义词，疑似纯音乐或静音"
    if duration >= 20 and tokens / duration < 0.4:
        return False, f"词密度仅 {tokens / duration:.2f} 词/秒，明显低于正常语速"
    return True, ""



def _ffmpeg_cut(src: Path, dst: Path, start: float, length: float) -> bool:
    res = subprocess.run(
        [media.ffmpeg_exe(), "-y", "-v", "error",
         "-ss", f"{start:.2f}", "-i", str(src), "-t", f"{length:.2f}",
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
        capture_output=True, text=True, timeout=600)
    return res.returncode == 0 and dst.exists() and dst.stat().st_size > 5_000


def _split_wav(audio: Path, out_dir: Path, duration: float, chunk: float = 175.0) -> list[tuple[float, Path]]:
    if duration <= chunk:
        return [(0.0, audio)]
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: list[tuple[float, Path]] = []
    start = 0.0
    i = 0
    while start < duration - 1.0 and i < 24:
        dst = out_dir / f"chunk_{i:02d}.wav"
        length = min(chunk, duration - start)
        if _ffmpeg_cut(audio, dst, start, length):
            parts.append((start, dst))
        start += length
        i += 1
    return parts


def _qwen_asr(client: QwenClient, audio: Path, out_dir: Path, duration: float,
              result: TranscriptResult) -> None:
    chunks = _split_wav(audio, out_dir, duration)
    if not chunks:
        raise QwenError("分块失败：没有得到可用音频片段")
    note: list[str] = []
    pieces: list[str] = []
    for start, part in chunks:
        completion = client.asr(part, note=note)
        text = " ".join(completion.text.split()).strip()
        result.segments.append({"start": round(start, 2), "ts": fmt_ts(start), "text": text,
                                "chars": len(text)})
        pieces.append(text)
        result.model = completion.model
    result.text = " ".join(p for p in pieces if p).strip()
    result.warnings.extend(note)
    if duration > 300 and len(chunks) > 1:
        result.warnings.append(f"音频按 {fmt_ts(duration)} 分 {len(chunks)} 段转写，时间标签精度为段起始点")


def _whisper_asr(audio: Path, result: TranscriptResult) -> None:
    from faster_whisper import WhisperModel
    model = WhisperModel(settings.whisper_model_size, device="cpu", compute_type="int8")
    segs, info = model.transcribe(str(audio), language="en", vad_filter=True,
                                  word_timestamps=False, beam_size=4)
    pieces: list[str] = []
    for s in segs:
        text = (s.text or "").strip()
        if not text:
            continue
        pieces.append(text)
        result.segments.append({"start": round(s.start, 2), "ts": fmt_ts(s.start), "text": text,
                                "chars": len(text)})
    result.text = " ".join(pieces).strip()
    result.engine = f"faster-whisper:{settings.whisper_model_size}({getattr(info, 'language', '?')})"
    result.model = result.engine


def transcribe(video_path: Path, audio: Path | None, duration: float,
               client: QwenClient | None = None, mock_text: str = "") -> TranscriptResult:
    result = TranscriptResult()
    if mock_text:
        result.engine = "mock"
        result.model = "mock"
        result.text = mock_text
        result.segments = [{"start": 0.0, "ts": "00:00", "text": mock_text, "chars": len(mock_text)}]
        result.warnings.append("演示模式：转写文本为模拟生成，未调用真实模型")
        return result
    if audio is None:
        raise RuntimeError("视频没有可用音轨，或音频抽取失败（请确认视频包含人声且已安装 ffmpeg）")
    out_dir = audio.parent / "chunks"
    if settings.asr_engine == "whisper":
        try:
            _whisper_asr(audio, result)
            if result.text:
                result.credible, reason = credibility(result.text, duration)
                if not result.credible:
                    result.warnings.append(reason)
                return result
            result.warnings.append("本地转录引擎未产出文本，改用在线转录")
        except Exception as exc:
            result.warnings.append(f"本地转录引擎不可用（{type(exc).__name__}: {exc}），改用在线转录")
    if client is None or not client.enabled:
        raise QwenError("未配置转录服务密钥，无法完成语音转录")
    _qwen_asr(client, audio, out_dir, duration, result)
    if not result.engine:
        result.engine = f"qwen:{result.model or settings.asr_model}"
    if not result.text:
        raise QwenError("转写结果为空：可能是纯音乐、静音或音频过长被截断")
    result.credible, reason = credibility(result.text, duration)
    if not result.credible:
        result.warnings.append(reason)
    return result
