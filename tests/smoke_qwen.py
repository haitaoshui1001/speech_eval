"""真实模式冒烟：分别验证 ASR / 文本 / 视觉 / 语音 四类调用是否可用。

用法： python tests/smoke_qwen.py [视频路径]
默认使用 data/videos 下第一个视频，或 tests/sample_speech.mp4。
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from app import media  # noqa: E402
from app.config import settings  # noqa: E402
from app.qwen import QwenError, QwenClient  # noqa: E402


def pick_video() -> Path:
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg and Path(arg).exists():
        return Path(arg)
    vids = sorted((BASE / "data" / "videos").glob("*.mp4")) if (BASE / "data" / "videos").exists() else []
    if vids:
        return vids[0]
    return BASE / "tests" / "sample_speech.mp4"


def show(name: str, fn) -> None:
    t0 = time.time()
    try:
        out = fn()
        print(f"[OK ] {name}  {time.time() - t0:.1f}s  {out}")
    except QwenError as exc:
        print(f"[ERR] {name}  {time.time() - t0:.1f}s  HTTP {exc.status}: {str(exc)[:200]}")
        print(f"       body: {exc.body[:400]}")
    except Exception:  # noqa: BLE001
        print(f"[ERR] {name}  {time.time() - t0:.1f}s")
        traceback.print_exc()


def main() -> int:
    video = pick_video()
    print(f"key source={settings.api_key_source} len={len(settings.api_key)} base={settings.base_url}")
    print(f"models: chat={settings.chat_model} vlm={settings.vlm_model} omni={settings.omni_model} asr={settings.asr_model}")
    if not video.exists():
        print("找不到视频", video)
        return 2
    out = BASE / "data" / "_smoke"
    info = media.probe(video)
    audio = media.extract_audio(video, out, max_seconds=60)
    frames = media.extract_frames(video, out, info.duration, 4) or []
    frames = list((out / "frames").glob("*.jpg"))[:4]
    print(f"video={video.name} duration={info.duration:.1f}s audio={audio} frames={len(frames)}")

    c = QwenClient()
    show("chat qwen-plus", lambda: c.chat("只回答两个字：收到", model="qwen-plus").text[:40])
    show(f"chat {settings.chat_model}", lambda: c.chat("只回答两个字：收到", model=settings.chat_model).text[:40])
    if audio:
        show(f"asr {settings.asr_model}", lambda: c.asr(audio).text[:60])
        show(f"audio {settings.omni_model}", lambda: c.audio("用一句话描述这段音频里有什么声音。", audio).text[:80])
    if frames:
        show(f"vision {settings.vlm_model}", lambda: c.vision("用一句话描述这张画面。", frames[:2]).text[:80])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
