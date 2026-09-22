"""媒体处理：视频元信息、音频抽离、关键帧抽取（基于 ffmpeg）。"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings


@dataclass
class MediaInfo:
    duration: float
    width: int
    height: int
    has_audio: bool
    fps: float


def _exe(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"未找到 {name}，请安装 ffmpeg 并加入 PATH")
    return found


def ffmpeg_exe() -> str:
    return _exe("ffmpeg")


def probe(path: Path) -> MediaInfo:
    cmd = [_exe("ffprobe"), "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {out.stderr[-400:]}")
    data = json.loads(out.stdout or "{}")
    duration = 0.0
    try:
        duration = float(data.get("format", {}).get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    width = height = 0
    fps = 0.0
    has_audio = False
    for st in data.get("streams", []):
        kind = st.get("codec_type")
        if kind == "video" and not width:
            width = int(st.get("width") or 0)
            height = int(st.get("height") or 0)
            rate = st.get("avg_frame_rate") or "0/0"
            try:
                num, den = rate.split("/")
                fps = float(num) / float(den) if float(den) else 0.0
            except (ValueError, ZeroDivisionError):
                fps = 0.0
        elif kind == "audio":
            has_audio = True
            if not duration:
                try:
                    duration = float(st.get("duration") or 0)
                except (TypeError, ValueError):
                    duration = 0.0
    return MediaInfo(duration=duration, width=width, height=height,
                     has_audio=has_audio, fps=round(fps, 2))


def extract_audio(video: Path, out_dir: Path, max_seconds: int = 0) -> Path | None:
    """抽出 16kHz 单声道 wav（ASR 与声学分析共用）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "audio.wav"
    cmd = [ffmpeg_exe(), "-y", "-v", "error", "-i", str(video)]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if res.returncode != 0 or not target.exists() or target.stat().st_size < 10_000:
        return None
    return target


@dataclass
class FrameSet:
    """关键帧及其**真实采样时刻**（秒）。stamps 与 paths 一一对应。"""
    paths: list[Path] = field(default_factory=list)
    stamps: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.paths)


_MIN_JPEG_BYTES = 512
# 千问视觉单次请求图片数硬上限 100（不超过即可），这里钉在 100：再多会被服务商拒收，
# 而帧序号 frame_\d{1,3} 资产白名单与报告胶片条命名也都容忍这个上界。
_FRAME_HARD_CAP = 100


def frame_budget(duration: float) -> int:
    """按时长推导关键帧张数：每 frame_interval 秒一张，只做**上限**封顶，不做下限抬高。

    下限夹紧会让短视频被迫密采、长视频被强行拉高张数，破坏「等间隔」这个前提；
    因此 duration 为 0（探测失败）时才退回固定 FRAME_COUNT 兜底。
    """
    if duration <= 0:
        return max(1, min(settings.frame_count, _FRAME_HARD_CAP))
    step = max(1, settings.frame_interval)
    hi = max(1, min(settings.frame_count_max, _FRAME_HARD_CAP))
    return max(1, min(int(round(duration / step)), hi))


def _scale(width: int) -> str:
    """缩放到 width 宽，但不放大（min(iw,W)）：竖屏视频原始宽度常小于目标宽，
    放大只会白涨视觉 token，不会带来任何细节。"""
    w = max(160, int(width))
    return f"scale=w='min(iw,{w})':h=-2"


def _uniform_frames(video: Path, out_dir: Path, targets: list[float],
                    width: int) -> list[tuple[float, Path]]:
    """逐点 seek 抽帧，时刻即请求的 target，天然精确（不需要事后推测）。"""
    out: list[tuple[float, Path]] = []
    if not targets:
        return out
    uni = out_dir / "_uni"
    uni.mkdir(parents=True, exist_ok=True)
    for ts in targets:
        dst = uni / f"u_{int(round(ts * 100)):07d}.jpg"
        subprocess.run(
            [_exe("ffmpeg"), "-y", "-v", "error", "-ss", f"{ts:.2f}", "-i", str(video),
             "-vf", _scale(width), "-frames:v", "1", "-q:v", "3", str(dst)],
            capture_output=True, text=True, timeout=180)
        # 纯色/白底 PPT 画面压完可能不到 1KB，阈值再高会把合法帧当失败丢掉；
        # 512 字节足够排除空文件，同时放过这类"内容简单"的真实帧。
        if dst.exists() and dst.stat().st_size > _MIN_JPEG_BYTES:
            out.append((ts, dst))
    return out


def extract_frames_timed(video: Path, out_dir: Path, duration: float,
                         count: int | None = None) -> FrameSet:
    """纯均匀采样：张数按 frame_budget(duration) 推导，采样点等距铺满整段。

    刻意不做场景切换补插、也不在缺点后补空洞——那两种做法都会把采样点从等距格子上
    拽偏，模型看到的 "frame@时间点" 序列就不再均匀。个别采样点抽取失败时直接少一帧，
    时间戳仍然是真实时刻，报告与评分依据都不会错位。
    """
    count = max(1, min(count or frame_budget(duration), _FRAME_HARD_CAP))
    out_dir.mkdir(parents=True, exist_ok=True)
    width = settings.frame_width
    if duration > 0:
        step = duration / count
        targets = [step * (i + 0.5) for i in range(count)]
        picked = _uniform_frames(video, out_dir, targets, width)
    else:
        picked = []
    ordered = sorted(picked, key=lambda x: x[0])[:count]
    if ordered:
        keep = out_dir / "frames"
        if keep.exists():
            shutil.rmtree(keep, ignore_errors=True)
        keep.mkdir(parents=True, exist_ok=True)
        renamed: list[Path] = []
        stamps: list[float] = []
        for i, (ts, f) in enumerate(ordered):
            dst = keep / f"frame_{i:02d}.jpg"
            shutil.copyfile(f, dst)
            renamed.append(dst)
            stamps.append(round(ts, 2))
        shutil.rmtree(out_dir / "_uni", ignore_errors=True)
        return FrameSet(paths=renamed, stamps=stamps)
    single = out_dir / "cover.jpg"
    subprocess.run(
        [_exe("ffmpeg"), "-y", "-v", "error", "-ss", f"{(duration / 2) if duration else 1:.2f}",
         "-i", str(video), "-vf", _scale(width), "-frames:v", "1", "-q:v", "3", str(single)],
        capture_output=True, text=True, timeout=180)
    return FrameSet(paths=[single] if single.exists() else [], stamps=[duration / 2] if single.exists() else [])


def extract_frames(video: Path, out_dir: Path, duration: float,
                   count: int | None = None) -> list[Path]:
    return extract_frames_timed(video, out_dir, duration, count).paths


def video_thumbnail(video: Path, dest: Path, at: float = 1.0) -> Path | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        [_exe("ffmpeg"), "-y", "-v", "error", "-ss", f"{max(at, 0):.2f}", "-i", str(video),
         "-vf", "scale=320:-2", "-frames:v", "1", "-q:v", "4", str(dest)],
        capture_output=True, text=True, timeout=120)
    return dest if res.returncode == 0 and dest.exists() else None
