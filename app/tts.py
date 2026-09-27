"""C 线 · 在线朗读合成：两级拼装、分块规划、下载拼接与可行性闸门。

链路（方案 §6.1/§6.2）：
  第一级 infer_style()   —— 一次廉价 chat 从主题推断情感演绎（失败落规则基线，不阻塞）；
  第二级 build_instructions() —— 纯字符串拼装：现场 / 基调 / 逐句演绎 / 相对语速 / 短板导向；
  plan_chunks()          —— 句边界切块（绝不在句中切开），每块同一指令、同一音色；
  synthesize()           —— 逐块调用 qwen.tts() → 立即下载 → media.concat_audio 整段拼接；
  tts_feasibility()      —— 发起任何请求之前的闸门：装不下 / 跑不完 / 不可用，三条判据纯函数。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

from . import media, prompts
from .config import settings
from .qwen import QwenClient, parse_json_block
from .revise import source_hash

STYLE_KEYS = ("tone", "pace_band", "structure", "avoid")

SCENES: dict[str, str] = {
    "class": "课堂展示",
    "contest": "演讲比赛",
    "defense": "毕业答辩",
    "report": "工作汇报",
}

# 推断失败/关闭时的规则基线：按现场给中性新闻腔（§6.1）。
# contest 一条是 H 样逐句演绎脚本（§3.4 人耳验收选定的基底）。
_STYLE_RULES: dict[str, dict[str, str]] = {
    "contest": {
        "tone": "庄重而激昂，情绪有层次",
        "pace_band": "整体偏快，结尾明显放慢",
        "structure": "设问句后停顿留白；短句斩钉截铁；排比层层递进、一句比一句高；结尾一字一顿放慢收束",
        "avoid": "不喊叫、不煽情",
    },
    "class": {
        "tone": "自然大方，像当面讲述",
        "pace_band": "从容适中、字字清楚",
        "structure": "句读分明；设问处微微上扬并留出停顿；重点句放慢加重；结尾干脆收束",
        "avoid": "不拖沓、不念经式平铺",
    },
    "defense": {
        "tone": "沉稳笃定，可信可靠",
        "pace_band": "从容放慢一档",
        "structure": "每句说得稳；数据和结论前稍顿、后加重；「首先/其次/最后」等路标词清晰；总结句放慢收束",
        "avoid": "不急不躁、不过度渲染",
    },
    "report": {
        "tone": "简洁利落，新闻播报感",
        "pace_band": "适中偏稳、字字清楚",
        "structure": "分条小标题提顿；并列项节奏均匀；数字逐字念清；结尾干脆收束",
        "avoid": "不拖泥带水、不加感情色彩",
    },
}

# 短板导向：命中这些关键词的维度视为「朗读可示范」维度
_FOCUS_KEYS = ("fluency", "delivery", "流利", "连贯", "表达", "台风", "语音", "语调", "节奏", "pron")
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；])|(?<=[.!?;])\s+")
_URL_EXT_RE = re.compile(r"\.(wav|mp3|opus|m4a|flac)(?=[$?])", re.I)


class TtsError(RuntimeError):
    pass


@dataclass
class TtsResult:
    path: Path
    model: str = ""
    voice: str = ""
    chunks: int = 0
    characters: int = 0
    instructions: str = ""
    style: dict[str, str] = field(default_factory=dict)
    style_source: str = "rule"
    warnings: list[str] = field(default_factory=list)
    text_hash: str = ""
    clone: bool = False

    def to_meta(self) -> dict[str, Any]:
        """落 tts_meta 的部分：绝不含签名 URL（24h 过期，落库只会漏 404）。"""
        return {
            "model": self.model, "voice": self.voice, "chunks": self.chunks,
            "characters": self.characters, "style_source": self.style_source,
            "style": self.style, "instructions_len": len(self.instructions),
            "text_hash": self.text_hash, "clone": self.clone,
        }


# ---------------------------------------------------------------- 第一级
def style_fallback(scene: str) -> dict[str, str]:
    return dict(_STYLE_RULES.get(scene) or _STYLE_RULES["class"])


def infer_style(client, *, topic: str = "", requirements: str = "", scene: str = "class",
                text: str = "", note: list[str] | None = None) -> dict[str, str]:
    """主题 → 情感演绎（一次廉价 chat）。任何失败都落规则基线，绝不阻塞合成。"""
    fallback = style_fallback(scene)
    if not (text or "").strip():
        return fallback
    try:
        prompt = prompts.render(
            "tts_style",
            topic=(topic or "").strip() or "（未填写，按稿件内容判断）",
            requirements=(requirements or "").strip() or "（无特别要求）",
            scene=SCENES.get(scene, SCENES["class"]),
            opening=text[:200],
        )
        comp = client.chat(prompt, model=settings.revise_model or settings.chat_model,
                           json_mode=True, temperature=0.3, note=note, label="朗读情感推断")
        obj = parse_json_block(comp.text)
    except Exception as exc:  # noqa: BLE001 - 推断失败必须安静降级
        if note is not None:
            note.append(f"朗读情感推断失败（{type(exc).__name__}），改用默认演绎基线")
        return fallback
    out = {k: str(obj.get(k) or "").strip()[:60] for k in STYLE_KEYS}
    if not out["tone"] or not out["structure"]:
        if note is not None:
            note.append("情感推断结果不完整，改用默认演绎基线")
        return fallback
    return out


# ---------------------------------------------------------------- 第二级
def speech_band(text: str, duration: float) -> str:
    """学生原速 → 相对档位词（§3.4 实测：绝对数值收效率低，只用相对说法）。"""
    if duration <= 10 or not (text or "").strip():
        return ""
    minutes = duration / 60.0
    cjk = len(_CJK_RE.findall(text))
    if cjk * 2 > len(text):
        rate = cjk / minutes
        slow, fast = 180.0, 260.0
    else:
        words = max(1, len(re.findall(r"[A-Za-z0-9'’-]+", text)))
        rate = words / minutes
        slow, fast = 120.0, 170.0
    if rate < slow:
        return "示范比原速稍快一点、字字饱满"
    if rate > fast:
        return "示范比原速放慢一档、句间留呼吸"
    return "示范保持与原速相当的紧凑感"


def build_focus(dims: dict[str, dict] | None) -> str:
    """从维度得分挑短板导向：fluency/delivery 类得分率弱则求稳，强则求自然。"""
    rows = [d for d in (dims or {}).values()
            if any(k in (str(d.get("dim_key", "")) + str(d.get("name", ""))).lower()
                   for k in _FOCUS_KEYS)]
    if not rows:
        return ""
    ratios = [float(d.get("ratio") or 0) for d in rows if d.get("ratio") is not None]
    if not ratios:
        return ""
    avg = sum(ratios) / len(ratios)
    if avg < 0.7:
        return "把容易含混的地方念清楚，语速平稳，句与句之间留出呼吸"
    return "比学生原演绎更松弛自然，示范「讲人话」而不是「背稿子」"


def build_instructions(style: dict[str, str], *, scene: str = "class",
                       pace_hint: str = "", focus: str = "") -> str:
    tone = (style.get("tone") or "").strip() or "清晰沉稳的新闻腔"
    structure = (style.get("structure") or "").strip() or "句读分明，重点句放慢加重"
    pace = "；".join(x for x in (pace_hint.strip(), (style.get("pace_band") or "").strip()) if x) \
        or "适中、字字清楚"
    avoid = (style.get("avoid") or "").strip()
    focus = (focus or "").strip()
    if avoid:
        focus = f"{focus}；要避免：{avoid}" if focus else f"要避免：{avoid}"
    focus = focus or "整体节奏与情绪贴合主题与现场"
    return prompts.render("tts_instructions",
                          scene=SCENES.get(scene, SCENES["class"]),
                          tone=tone, structure=structure, pace=pace, focus=focus)


# ---------------------------------------------------------------- 分块
def _sent_units(text: str) -> list[str]:
    units = [u.strip() for u in _SENT_SPLIT_RE.split(text) if u and u.strip()]
    return units or [text.strip()]


def _join_unit(head: str, unit: str) -> str:
    if not head:
        return unit
    if head[-1] in "，。！？；、：)）]」』”’" or ord(unit[0]) >= 0x2E00:
        return head + unit
    return head + " " + unit


def plan_chunks(text: str, segments: list[dict] | None = None, *,
                chunk_chars: int = 0, max_chunks: int = 0) -> list[str]:
    """按句边界累积切块（绝不在句中切开）；单句超上限时才硬切兜底。

    segments 参数保留与方案签名一致：句级切分本身就以标点为界，与逐句 segments 等价。
    """
    chunk_chars = max(200, chunk_chars or settings.tts_chunk_chars)
    units = _sent_units(text)
    chunks: list[str] = []
    cur = ""
    for u in units:
        if len(u) > chunk_chars:
            if cur:
                chunks.append(cur)
                cur = ""
            pieces = [u[i:i + chunk_chars] for i in range(0, len(u), chunk_chars)]
            chunks.extend(pieces[:-1])
            cur = pieces[-1]
            continue
        cand = _join_unit(cur, u)
        if len(cand) <= chunk_chars:
            cur = cand
        else:
            chunks.append(cur)
            cur = u
    if cur:
        chunks.append(cur)
    if max_chunks and len(chunks) > max_chunks:
        raise TtsError(f"稿件按句切成 {len(chunks)} 块，超过上限 {max_chunks} 块，请缩短稿件或调高分块上限")
    return chunks


# ---------------------------------------------------------------- 闸门
def tts_feasibility(text: str, cfg: Any = None) -> tuple[bool, str]:
    """三条判据（§7.3），全部通过才允许发起任何一次合成调用。"""
    cfg = cfg or settings
    if not getattr(cfg, "tts_enabled", True):
        return False, "朗读合成开关已关闭（TTS_ENABLED=0）"
    if not getattr(cfg, "real_mode", True):
        return False, "演示模式（mock）无法合成朗读，请切换到在线模式"
    if not (getattr(cfg, "api_key", "") or "").strip():
        return False, "未配置 API Key，无法合成朗读"
    if not (text or "").strip():
        return False, "生效稿件为空，没有可朗读的内容"
    n = len(text)
    chunk = max(200, cfg.tts_chunk_chars)
    cap = chunk * max(1, cfg.tts_max_chunks)
    if n > cap:
        return False, (f"稿件约 {n} 字符，超过上限 {cap} 字符"
                       f"（分块 {chunk} × 块数上限 {cfg.tts_max_chunks}），请缩短稿件或调高配置")
    per_chunk = max(30.0, 50.0 * chunk / 1600.0)
    if per_chunk > cfg.tts_timeout:
        want = int(per_chunk * 2) + 60
        return False, (f"单块预计最长 {per_chunk:.0f} 秒，超过 TTS_TIMEOUT={cfg.tts_timeout} 秒，"
                       f"建议将合成超时调到 {want} 秒以上")
    return True, ""


# ---------------------------------------------------------------- 下载
def download_audio(url: str, dest: Path, *, timeout: int = 120) -> Path:
    """签名 URL 只有 24h 有效期，拿到必须立即下载落盘；失败重试一次。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(2):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code != 200:
                    raise TtsError(f"音频下载失败 {r.status_code}")
                tmp = dest.with_name(dest.name + ".part")
                with open(tmp, "wb") as fh:
                    for piece in r.iter_content(64 * 1024):
                        if piece:
                            fh.write(piece)
                if tmp.stat().st_size < 1024:
                    raise TtsError("下载的音频过小，视为失败")
                tmp.replace(dest)
                return dest
        except (requests.RequestException, TtsError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    assert last is not None
    raise TtsError(f"音频下载失败：{last}") from last


# ---------------------------------------------------------------- 编排
def _chunk_suffix(url: str) -> str:
    m = _URL_EXT_RE.search(url)
    return f".{m.group(1).lower()}" if m else ".audio"


def synthesize(client, text: str, *, out_dir: Path, scene: str = "class",
               topic: str = "", requirements: str = "", dims: dict[str, dict] | None = None,
               duration: float = 0.0, voice: str = "", model: str = "",
               progress: Callable[[int, str], None] | None = None) -> TtsResult:
    """完整一次合成：推断 → 拼装 → 切块 → 逐块调用+下载 → 拼接 → 清理分块。

    任何一块失败都整次失败并删除已下载分块（绝不留「缺一截」的残段）；
    仅 ffmpeg 缺失时按 §6.5 保留首块并明确 warning「仅含前 N 字」。

    voice/model 必须成对传（复刻链路）：音色与创建时的 target_model 死绑，
    只换音色不换模型等于拿别人的嗓子读我的稿，官方直接报错。留空则回落系统音色。
    """
    report = progress or (lambda pct, phase: None)
    notes: list[str] = []
    voice = (voice or "").strip() or settings.tts_voice
    model = (model or "").strip() or settings.tts_model
    # 判定放在补默认值之后：管理员把 TTS_MODEL 配成 vc 模型时同样要按复刻口径走。
    clone = QwenClient.is_clone_model(model)
    style = style_fallback(scene)
    style_source = "rule"
    if clone:
        # 官方口径：复刻音色不接 instructions。既然注定不送，情感推断这次调用就是白烧钱，
        # 直接跳过，并把「为什么这次没有演绎指令」写进回执，别让人以为自己的稿件被忽略。
        style_source = "clone"
        notes.append(f"本次使用你的复刻音色（{voice}）按原声朗读，未附加表现力指令")
    elif client.enabled:
        style = infer_style(client, topic=topic, requirements=requirements,
                            scene=scene, text=text, note=notes)
        style_source = "model" if style != style_fallback(scene) else "rule"
    instructions = "" if clone else build_instructions(
        style, scene=scene, pace_hint=speech_band(text, duration), focus=build_focus(dims))
    chunks = plan_chunks(text, max_chunks=settings.tts_max_chunks)
    out_dir = Path(out_dir)
    work = out_dir / "_chunks"
    work.mkdir(parents=True, exist_ok=True)
    tmps: list[Path] = []
    total_chars = 0
    used_model = ""
    try:
        for i, piece in enumerate(chunks):
            report(5 + int(85 * i / len(chunks)), f"合成第 {i + 1}/{len(chunks)} 块")
            audio = client.tts(piece, voice=voice, instructions=instructions, model=model,
                               timeout=settings.tts_timeout, note=notes)
            total_chars += audio.characters
            used_model = audio.model
            dest = work / f"chunk_{i:03d}{_chunk_suffix(audio.url)}"
            if audio.url:
                download_audio(audio.url, dest)
            elif audio.data:
                dest.write_bytes(audio.data)
            else:
                raise TtsError(f"第 {i + 1} 块没有返回音频")
            tmps.append(dest)
        report(92, "拼接音频")
        final = out_dir / "speech.mp3"
        try:
            media.concat_audio(tmps, final)
        except (RuntimeError, TtsError) as exc:
            if len(tmps) == 1:
                raise
            notes.append(f"音频拼接失败（{exc}），本次仅保留第 1 块，只含前 {len(chunks[0])} 字")
            tmps[0].replace(final)
        return TtsResult(path=final, model=used_model or model,
                         voice=voice, chunks=len(chunks), characters=total_chars,
                         instructions=instructions, style=style,
                         style_source=style_source, warnings=notes,
                         text_hash=source_hash(text), clone=clone)
    finally:
        for f in tmps:
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            work.rmdir()
        except OSError:
            pass
