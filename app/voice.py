"""声音复刻样本：从原视频里挑一段最干净的连续朗读，抽成符合官方规格的音频。

官方对样本的硬约束（见平台文档「声音复刻」）：单声道、采样率 ≥24kHz、
推荐 10~20 秒、含 ≥3 秒连续清晰朗读、停顿不超过 2 秒、避免背景音乐与他人声。
转录用的 audio.wav 是 16kHz，重采样上去只会得到 16kHz 的假高频，
所以样本一律直接从原视频抽，且单独走一路 24kHz。
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import media

VC_SAMPLE_RATE = 24000          # 官方下限 24kHz，取定值而不是更高：够清晰又省体积
VC_MIN_SPEECH = 3.0             # 官方要求样本里至少 3 秒连续清晰朗读
VC_SAMPLE_MIN = 10.0            # 官方推荐区间下限
VC_SAMPLE_TARGET = 16.0         # 目标长度：留足余量又不至于把停顿也吸进来
VC_SAMPLE_MAX = 20.0            # 官方推荐区间上限（绝对上限 60 秒，没必要）
VC_MAX_PAUSE = 2.0              # 超过 2 秒的停顿视为断开，不作为样本内部
VC_NOISE_DB = "-30dB"           # silencedetect 阈值：口语录音里比这更响的都不算静默
VC_MIN_SILENCE = 0.30           # 短于 0.3 秒的空隙当语气停顿处理，别把一句话切开
VC_MAX_BYTES = 7_000_000        # 24k/单声道/16bit ≈ 48KB/s，7MB 足够 2 分钟，纯防呆
VC_MIN_BYTES = 20_000           # 太小就是抽风：不足约 0.4 秒，官方必拒
VC_QUIET_MEAN_DB = -45.0        # 平均音量低于此值，复刻出来的音色多半是噪声


class VoiceSampleError(RuntimeError):
    """挑不出/抽不出合格样本：消息必须是人能看懂的原因 + 下一步，直接进页面回执。"""


@dataclass
class Sample:
    start: float
    end: float
    speech_windows: int = 0
    pause_count: int = 0

    @property
    def duration(self) -> float:
        return round(max(0.0, self.end - self.start), 2)


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_silence_ranges(stderr: str) -> list[tuple[float, float]]:
    """从 silencedetect 的日志里抠出静默区间。

    日志成对出现：`silence_start: 12.34` 与 `silence_end: 14.56 | silence_duration: 2.22`。
    只有 start 没有 end（文件以静音收尾）时，区间右端由调用方用总时长补齐。
    """
    ranges: list[tuple[float, float]] = []
    pending: float | None = None
    for line in (stderr or "").splitlines():
        m = re.search(r"silence_start:\s*([0-9.]+)", line)
        if m:
            v = _to_float(m.group(1))
            if v is not None and pending is None:
                pending = v
            continue
        m = re.search(r"silence_end:\s*([0-9.]+)", line)
        if m and pending is not None:
            v = _to_float(m.group(1))
            if v is not None and v > pending:
                ranges.append((pending, v))
            pending = None
    return ranges


def speech_windows_from_silence(silences: list[tuple[float, float]],
                                duration: float) -> list[tuple[float, float]]:
    """静默区间的补集即说话区间（不依赖 ASR 时间戳，千问通道的块级标签粗到没法用）。"""
    if duration <= 0:
        return []
    out: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in sorted(silences):
        if s > cursor + 0.05:
            out.append((cursor, min(s, duration)))
        cursor = max(cursor, min(e, duration))
    if cursor < duration - 0.05:
        out.append((cursor, duration))
    return [(round(a, 2), round(b, 2)) for a, b in out if b - a >= VC_MIN_SPEECH]


def merge_chains(windows: list[tuple[float, float]],
                 max_pause: float = VC_MAX_PAUSE) -> list[tuple[float, float]]:
    """把间隔 ≤max_pause 的说话区间并成一条链（一条链内部停顿合规，链间才是真断）。"""
    chains: list[tuple[float, float]] = []
    for start, end in sorted(windows):
        if chains and start - chains[-1][1] <= max_pause:
            chains[-1] = (chains[-1][0], max(chains[-1][1], end))
        else:
            chains.append((start, end))
    return chains


def choose_slice(chains: list[tuple[float, float]], windows: list[tuple[float, float]],
                 *, min_len: float = VC_SAMPLE_MIN, target: float = VC_SAMPLE_TARGET,
                 max_len: float = VC_SAMPLE_MAX) -> Sample:
    """在最长的一条链里取一段，收尾尽量落在真实停顿上，避免把句子腰斩。

    优先取目标长度，但端点必须是某个说话区间的右端（即停顿开始处）；
    找不到落在 [min_len, max_len] 内的停顿才退化成定长截取。
    """
    usable = [(a, b) for a, b in chains if b - a >= min_len]
    if not usable:
        raise VoiceSampleError(
            f"最长连续朗读不足 {min_len:.0f} 秒，达不到官方 ≥{VC_MIN_SPEECH:.0f} 秒清晰朗读、"
            f"{min_len:.0f}~{max_len:.0f} 秒样本的要求；请上传一段更连贯的讲话视频后重做音色")
    a, b = max(usable, key=lambda w: w[1] - w[0])
    want = min(b, a + target)
    ends = [e for s, e in windows if s >= a - 0.01 and e <= b + 0.01]
    in_range = [e for e in ends if a + min_len <= e <= min(b, a + max_len)]
    if in_range:
        end = min(in_range, key=lambda e: abs(e - (a + target)))
    else:
        end = want
    start = a
    if end - start < min_len and b - start >= min_len:
        end = min(b, start + max_len)
    inner = [1 for s, e in windows if s >= start - 0.01 and e <= end + 0.01]
    return Sample(start=round(start, 2), end=round(end, 2),
                  speech_windows=len(inner), pause_count=max(0, len(inner) - 1))


def detect_windows(audio: Path, duration: float = 0.0) -> list[tuple[float, float]]:
    """跑一遍 ffmpeg silencedetect，返回说话区间列表。"""
    exe = media.ffmpeg_exe()
    total = duration if duration > 0 else media.probe(audio).duration
    if total <= 0:
        raise VoiceSampleError(f"读不到音频时长（{audio.name}），无法挑选复刻样本")
    res = subprocess.run([exe, "-y", "-v", "info", "-i", str(audio),
                          "-af", f"silencedetect=noise={VC_NOISE_DB}:d={VC_MIN_SILENCE}",
                          "-f", "null", "-"],
                         capture_output=True, text=True, timeout=300)
    # silencedetect 走 null muxer，ffmpeg 常以「没有输出文件」的非零码收尾，只要有日志就照用。
    logs = (res.stderr or "") + (res.stdout or "")
    if "silence_start" not in logs and res.returncode != 0:
        raise VoiceSampleError(f"静音检测失败：{logs[-200:] or 'ffmpeg 无输出'}")
    silences = [(s, min(e, total)) for s, e in parse_silence_ranges(logs)]
    # 以静音收尾时日志只有 silence_start（没有配对的 silence_end），补到文件末尾，
    # 否则最后那段静音会被算成说话区间，样本尾巴就吃掉一截空白。
    starts = re.findall(r"silence_start:\s*([0-9.]+)", logs)
    ends = re.findall(r"silence_end:\s*([0-9.]+)", logs)
    if len(starts) > len(ends):
        last = _to_float(starts[-1])
        if last is not None and total - last > VC_MIN_SILENCE:
            silences.append((last, total))
    return speech_windows_from_silence(sorted(silences), total)


def mean_volume_db(audio: Path) -> float | None:
    """volumedetect 的平均音量（dBFS）；读不出来返回 None，不拦创建流程。"""
    res = subprocess.run([media.ffmpeg_exe(), "-y", "-v", "info", "-i", str(audio),
                          "-af", "volumedetect", "-f", "null", "-"],
                         capture_output=True, text=True, timeout=300)
    m = re.search(r"mean_volume:\s*(-?[0-9.]+)\s*dB", (res.stderr or "") + (res.stdout or ""))
    return float(m.group(1)) if m else None


def extract_sample(video: Path, dest: Path, start: float, end: float,
                   duration: float = 0.0) -> Path:
    """从原视频抽出 24kHz 单声道 16bit wav 作为复刻样本。"""
    if end - start < VC_MIN_SPEECH:
        raise VoiceSampleError(f"样本长度仅 {end - start:.1f} 秒，短于官方要求的连续清晰朗读")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # 时长探测可能偏短（容器头不可靠），越界时按剩余长度截，避免 ffmpeg 报「-t 超过文件尾」。
    length = end - start
    if 0 < duration <= end:
        length = max(0.0, duration - start)
    if length < VC_MIN_SPEECH:
        raise VoiceSampleError(f"视频剩余时长不足 {VC_MIN_SPEECH:.0f} 秒，抽不出合规样本")
    cmd = [media.ffmpeg_exe(), "-y", "-v", "error", "-ss", f"{max(start, 0):.2f}",
           "-i", str(video), "-t", f"{length:.2f}",
           "-vn", "-ac", "1", "-ar", str(VC_SAMPLE_RATE), "-c:a", "pcm_s16le", str(dest)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if res.returncode != 0 or not dest.exists():
        raise VoiceSampleError(f"样本抽取失败：{(res.stderr or '')[-200:]}")
    size = dest.stat().st_size
    if size < VC_MIN_BYTES:
        dest.unlink(missing_ok=True)
        raise VoiceSampleError("抽出的样本几乎没有声音，请确认视频音轨里有本人讲话")
    if size > VC_MAX_BYTES:
        dest.unlink(missing_ok=True)
        raise VoiceSampleError(f"样本体积 {size // 1024 // 1024}MB 超过上限，请缩短片段")
    vol = mean_volume_db(dest)
    if vol is not None and vol < VC_QUIET_MEAN_DB:
        dest.unlink(missing_ok=True)
        raise VoiceSampleError(f"样本平均音量 {vol:.0f}dB 过低，容易被当成环境噪声；请靠近麦克风重录")
    return dest


def build_sample(video: Path, dest: Path, duration: float = 0.0) -> tuple[Sample, Path]:
    """一站式：先探原视频音轨，再挑段、抽取、校验。返回样本描述与落地文件。"""
    if not video.exists():
        raise VoiceSampleError(f"原视频文件已不存在：{video.name}，无法复刻音色")
    info = media.probe(video)
    total = duration or info.duration
    if not info.has_audio:
        raise VoiceSampleError("视频没有音轨，无法从讲话里挑复刻样本")
    tmp = dest.parent / "vc_probe.wav"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    src = extract_audio_full(video, tmp) or video
    try:
        windows = detect_windows(src, total)
    finally:
        if src == tmp:
            tmp.unlink(missing_ok=True)
    if not windows:
        raise VoiceSampleError(
            f"整段音频里找不到 ≥{VC_MIN_SPEECH:.0f} 秒的连续朗读"
            "（可能全是音乐、掌声或被噪声阈值判成静默），请重录一版再复刻")
    chains = merge_chains(windows)
    sample = choose_slice(chains, windows)
    path = extract_sample(video, dest, sample.start, sample.end, total)
    return sample, path


def extract_audio_full(video: Path, dest: Path) -> Path | None:
    """为静音检测抽一路原始采样率的单声道 wav（不降采样，避免把高频噪声判成静默）。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run([media.ffmpeg_exe(), "-y", "-v", "error", "-i", str(video),
                          "-vn", "-ac", "1", "-c:a", "pcm_s16le", str(dest)],
                         capture_output=True, text=True, timeout=900)
    if res.returncode != 0 or not dest.exists() or dest.stat().st_size < 10_000:
        return None
    return dest


def estimate_b64_bytes(path: Path) -> int:
    """Base64 直传的估算大小（4/3 膨胀），用于在送上传前给出人话提示。"""
    return int(path.stat().st_size * 4 / 3) + 64
