"""视频压缩：体积超过目标的上传件，在分析前用 ffmpeg 两遍编码压到目标以内。

放在流水线里而不是上传请求里执行：一段几十分钟的 1080p 视频在两核服务器上转码要几分钟，
放进 HTTP 请求必然超时；而流水线已经有排队、进度回写和失败兜底，复用即可。

画质优先的顺序是「先保分辨率、码率实在不够才降分辨率」：评审的画面通道只取 640px 宽的
关键帧，压缩对评分输入没有影响，只影响网页回放的观感，所以档位只往下掉到 480p 为止。
"""
from __future__ import annotations

import glob
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import media
from .config import settings

# 1080p / 720p / 540p / 480p。按**短边**排档位：竖屏手机视频（1080x1920）短边才是画面宽度，
# 若按长边排档，这类视频永远落在第一档降不下来。
LADDER: tuple[int, ...] = (1080, 720, 540, 480)
# 每像素每帧比特数下限。低于此值 x264 开始糊（PPT 文字边缘出块），此时留着高分辨率只是
# 浪费像素——同样的码率摊到更少的像素上，每个细节反而拿到的比特更多。
MIN_BPP = 0.03
AUDIO_KBPS = 96          # 语音评测用不到高码率音轨，96k AAC 听感已接近透明
HEADROOM = 0.95          # 只做到目标的 95%：容器开销 + 两遍编码本身有 ±3% 误差
PRESET = "veryfast"      # 后台队列跑，两核服务器上再慢一档会让排队时间明显变长
_KEYFRAMES = "expr:gte(t,n_forced*2)"   # 每 2 秒一个关键帧，保证网页回放拖得不卡
_MIN_VIDEO_KBPS = 100
_RETRY_SLACK_KBPS = 300  # 回调幅度小于这个值就别再跑第二遍了，省时间


@dataclass(frozen=True)
class Plan:
    """一次压缩的完整参数，抽出来是为了可单测、可写进质控说明。"""
    width: int
    height: int
    short_edge: int
    video_kbps: int
    audio_kbps: int
    fps: float
    duration: float
    target_bytes: int


@dataclass
class Result:
    orig_size: int
    size: int
    plan: Plan
    seconds: float
    attempts: int
    path: Path | None = None          # 压缩后文件所在位置；非 mp4 容器会改名成 .mp4
    orig_path: Path | None = None
    orig_width: int = 0
    orig_height: int = 0

    @property
    def renamed(self) -> bool:
        return bool(self.path and self.orig_path and self.path != self.orig_path)

    @property
    def saved_bytes(self) -> int:
        return max(0, self.orig_size - self.size)

    @property
    def ratio(self) -> float:
        """压缩后占原始体积的比例，0.2 表示只剩两成。"""
        return round(self.size / self.orig_size, 3) if self.orig_size else 1.0

    @property
    def note(self) -> str:
        mins, secs = divmod(int(round(self.seconds)), 60)
        cost = f"{mins} 分 {secs:02d} 秒" if mins else f"{secs} 秒"
        mb = 1024 * 1024
        extra = ""
        if (self.orig_width and self.plan.width
                and (self.orig_width, self.orig_height) != (self.plan.width, self.plan.height)):
            extra = f"，分辨率 {self.orig_width}x{self.orig_height} → {self.plan.width}x{self.plan.height}"
        if self.renamed:
            extra += "，容器统一转封为 MP4"
        return (f"原始文件 {self.orig_size / mb:.1f} MB 超过目标 "
                f"{self.plan.target_bytes / mb:.0f} MB，已两遍编码压到 {self.size / mb:.1f} MB"
                f"（视频 {self.plan.video_kbps}kbps / 音频 {self.plan.audio_kbps}kbps{extra}），"
                f"耗时 {cost}。评分用的关键帧只有 {settings.frame_width}px 宽，压缩不影响评分依据。")


def target_bytes() -> int:
    return max(0, int(settings.compress_target_bytes))


def needs_compress(size: int) -> bool:
    """是否要压。目标为 0 表示关闭自动压缩。"""
    t = target_bytes()
    return t > 0 and size > t


def human(n: int) -> str:
    """给人看的体积（进度文案用），小于 10MB 才保留一位小数。"""
    mb = (n or 0) / 1048576
    return f"{mb:.0f} MB" if mb >= 10 else f"{mb:.1f} MB"


def target_hint() -> str:
    return human(target_bytes())


def cap_short_edge(width: int, height: int, cap: int) -> tuple[int, int]:
    """把短边缩到 cap 以内，只缩不放；宽高向下取偶数（yuv420p 要求）。"""
    short = min(width, height)
    if short <= 0 or short <= cap:
        return width, height
    factor = cap / short
    w = max(16, int(width * factor) // 2 * 2)
    h = max(16, int(height * factor) // 2 * 2)
    return w, h


def pick_plan(info: media.MediaInfo, target: int) -> Plan:
    """由目标体积反推码率，再按「每像素比特数」决定要不要降分辨率。

    先用原始分辨率试，够清晰就不动像素；只有 bpp 低于 MIN_BPP 才往下一档掉，
    掉到 480p 仍然不够也就停在那里——再小网页上就没法看了，此时宁可压得狠一点。
    """
    if info.duration <= 0:
        raise RuntimeError("时长探测失败，无法计算压缩码率，请检查文件是否能正常播放")
    fps = info.fps if 0 < info.fps <= 120 else 25.0
    budget_bits = target * HEADROOM * 8 / info.duration
    audio_kbps = AUDIO_KBPS if info.has_audio else 0
    width, height = max(0, info.width), max(0, info.height)
    total_kbps = budget_bits / 1000 - audio_kbps
    video_kbps = max(_MIN_VIDEO_KBPS, int(total_kbps))
    chosen = (width, height, min(width, height) if width and height else 0)
    if width and height:
        for edge in LADDER:
            w, h = cap_short_edge(width, height, edge)
            bits_per_pixel = video_kbps * 1000 / (w * h * fps)
            if bits_per_pixel >= MIN_BPP or edge == LADDER[-1]:
                chosen = (w, h, min(w, h))
                break
    return Plan(width=chosen[0], height=chosen[1], short_edge=chosen[2],
                video_kbps=video_kbps, audio_kbps=audio_kbps, fps=fps,
                duration=round(info.duration, 2), target_bytes=target)


def _scale_filter(plan: Plan) -> str:
    if not plan.width or not plan.height:
        return ""
    return f"scale={plan.width}:{plan.height}"


def _progress_seconds(token: str) -> float | None:
    """解析 ffmpeg `-progress` 行里的 out_time_us / out_time_ms，返回已编码秒数。"""
    if not token.startswith("out_time_"):
        return None
    try:
        unit, raw = token[len("out_time_"):].split("=", 1)
        value = int(raw.strip())
    except ValueError:
        return None
    if value <= 0 or unit not in {"us", "ms"}:
        return None
    # out_time_ms 是 ffmpeg 的历史遗留命名，值同样是微秒，所以两个键统一按微秒换算。
    return value / 1_000_000


def _base_cmd(src: Path, plan: Plan, passlog: str) -> list[str]:
    cmd = [media.ffmpeg_exe(), "-nostdin", "-y", "-v", "error", "-i", str(src),
           "-map", "0:v:0", "-c:v", "libx264", "-preset", PRESET,
           "-b:v", f"{plan.video_kbps}k", "-passlogfile", passlog,
           "-pix_fmt", "yuv420p", "-force_key_frames", _KEYFRAMES]
    scale = _scale_filter(plan)
    if scale:
        cmd += ["-vf", scale]
    return cmd


def _run_pass(cmd: list[str], timeout: float, on_tick: Callable[[float], None] | None) -> None:
    """跑一遍编码；有回调时逐行读 -progress，否则等它结束。"""
    deadline = time.time() + timeout
    if on_tick is None:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg 压缩失败：{(res.stderr or '')[-400:]}")
        return
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="ignore", bufsize=1)
    buf: list[str] = []
    for line in proc.stdout or []:
        sec = _progress_seconds(line.strip())
        if sec is not None:
            on_tick(sec)
        if len(buf) < 20:
            buf.append(line.rstrip())
        if time.time() > deadline:
            proc.kill()
            raise RuntimeError("压缩超时，已保留原始文件，可在设置页调大「压缩超时」后重试")
    err = (proc.stderr.read() if proc.stderr else "") or ""
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg 压缩失败：{(err or ' '.join(buf))[-400:]}")


def _clean_logs(passlog: str) -> None:
    for p in glob.glob(passlog + "*"):
        try:
            os.remove(p)
        except OSError:
            pass


def _final_path(src: Path) -> Path:
    """压缩产物一律是 mp4 容器：非 mp4 原件要顺势改名，否则扩展名会骗浏览器导致回放失败。

    同名 .mp4 已存在时（同一秒上传了同名文件的极端情况）退回加后缀，绝不覆盖别人的视频。
    """
    if src.suffix.lower() == ".mp4":
        return src
    target = src.with_suffix(".mp4")
    return src.parent / (src.name + ".mp4") if target.exists() else target


def compress(src: Path, info: media.MediaInfo,
             on_progress: Callable[[int, str], None] | None = None) -> Result | None:
    """把 src 压到目标以内并**原地替换**，返回压缩说明；无需压缩或压完更大时返回 None。

    任何一步失败都抛异常且不动原文件，让上层把「压缩失败」显式报给用户，
    绝不静默拿超限文件继续分析——那会让磁盘悄悄涨爆，且用户以为已经压过了。

    临时文件必须带 .mp4 扩展名：ffmpeg 靠后缀猜复用器，`xxx.mp4.compressing` 这种名字
    会被它当成未知格式直接拒绝（Unable to choose an output format）。
    """
    target = target_bytes()
    orig_size = src.stat().st_size
    if target <= 0 or orig_size <= target:
        return None
    plan = pick_plan(info, target)
    tmp = src.with_name(f"{src.stem}.compressing.mp4")
    passlog = str(tmp.with_suffix("")) + ".pass"
    started = time.time()
    attempts = 0
    try:
        for _ in range(2):
            attempts += 1
            _encode(src, tmp, plan, passlog, on_progress)
            size = tmp.stat().st_size
            if size <= target or attempts >= 2:
                break
            # 两遍编码偶尔超 3%（VBV 缓冲 + 容器开销），按比例回调码率再跑一次即可收敛。
            scaled = int(plan.video_kbps * (target * HEADROOM) / size)
            if scaled >= plan.video_kbps - _RETRY_SLACK_KBPS:
                break
            plan = Plan(width=plan.width, height=plan.height, short_edge=plan.short_edge,
                        video_kbps=max(_MIN_VIDEO_KBPS, scaled), audio_kbps=plan.audio_kbps,
                        fps=plan.fps, duration=plan.duration, target_bytes=plan.target_bytes)
        size = tmp.stat().st_size
        if size >= orig_size:
            # 不划算就不换：原片码率本来就低于目标时偶尔会倒挂，保留原件比换个更大的文件强。
            return None
        if size > target:
            raise RuntimeError(f"两遍编码后仍有 {size / 1048576:.1f} MB，未能压到目标 "
                               f"{target / 1048576:.0f} MB（原文件已保留，可直接重跑或调大目标）")
        final = _final_path(src)
        os.replace(tmp, final)
        if final != src:
            src.unlink(missing_ok=True)
        return Result(orig_size=orig_size, size=size, plan=plan,
                      seconds=time.time() - started, attempts=attempts,
                      path=final, orig_path=src,
                      orig_width=info.width, orig_height=info.height)
    finally:
        if tmp.exists():
            try:
                os.remove(tmp)
            except OSError:
                pass
        _clean_logs(passlog)


def _encode(src: Path, dst: Path, plan: Plan, passlog: str,
            on_progress: Callable[[int, str], None] | None) -> None:
    """两遍编码：第一遍统计复杂度（占 25%），第二遍按分配的码率精确出片（占 75%）。

    两遍都挂 -progress pipe:1：进度要能一直动，而且它是唯一的「还活着」信号——
    超时判定靠逐行读取驱动，不收行的话卡死的那一遍没人发现。null 复用器本身
    不往 stdout 写数据，所以和 -progress 共用管道不冲突。
    """
    timeout = max(60, int(settings.compress_timeout))
    total = plan.duration or 1.0
    live = on_progress is not None
    prog = ["-nostats", "-progress", "pipe:1"] if live else []

    def report(frac: float, phase: str) -> None:
        if live:
            on_progress(max(0, min(100, int(frac * 100))), phase)  # noqa: B023 - live 已保证非空

    first = _base_cmd(src, plan, passlog) + ["-an", "-pass", "1", "-f", "null"] + prog + ["-"]
    _run_pass(first, timeout,
              (lambda sec: report(0.25 * min(sec / total, 1.0), "分析画面复杂度")) if live else None)
    second = _base_cmd(src, plan, passlog)
    if plan.audio_kbps:
        second += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", f"{plan.audio_kbps}k"]
    else:
        second += ["-an"]
    second += prog + ["-pass", "2", "-movflags", "+faststart", str(dst)]
    _run_pass(second, timeout,
              (lambda sec: report(0.25 + 0.75 * min(sec / total, 1.0), "两遍编码压缩")) if live else None)
    if not dst.exists() or dst.stat().st_size < 1024:
        raise RuntimeError("压缩产物为空文件，已保留原始视频")
    _clean_logs(passlog)
