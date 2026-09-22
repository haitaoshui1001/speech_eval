"""评价引擎：把「文字 + 图像 + 语音」三个通道分别判分，再确定性聚合成标准分。

设计要点（与 评价标准.txt 一一对应）：
  1. 每个通道只判它有资格判的维度，避免让文本模型凭空评价眼神交流。
  2. 分数由通道加权和确定性算出，大模型不得直接改总分，杜绝「印象分」。
  3. 每条判定必须带证据（原句 / 时间点），证据在转写文本里命中不了就降级并记录质控警告。
  4. 时长是否达标由 ffprobe 的客观时长决定，不交给模型猜。
  5. 最后一步用大模型只做「优点 / 缺点 / 改进建议 / 总结」的叙述，且必须引用已核验证据。
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from . import prompts
from .qwen import Completion, QwenClient, QwenError, parse_json_block
from .rubric import Dimension, Rubric
from .transcribe import fmt_ts

if TYPE_CHECKING:  # 仅用于类型标注：face 是可选依赖层，运行期不导入 cv2
    from .face import FaceMetrics

VISION_HINT = ("态势", "舞台", "肢体", "眼神", "表情", "姿态", "台风", "手势", "着装",
               "delivery", "body", "eye contact", "gesture", "stage", "posture")
AUDIO_HINT = ("发音", "语调", "语音", "连读", "重音", "节奏", "停顿", "流畅", "流利",
              "pronunciation", "intonation", "fluency", "prosod", "rhythm", "stress", "accent")
TEXT_ONLY_HINT = ("语法", "词汇", "用词", "content", "grammar", "vocabulary", "topic",
                  "structure", "relevance", "内容", "主题", "结构", "思想")

_MIN_CONF = 0.25


def apply_transcript_credibility(channels: list[dict], credible: bool) -> list[str]:
    """转写不可信（纯音乐/静音会让 ASR 产生 "100.0" 这类幻觉）时，
    文字通道判分依据不成立：下调其可信度并留痕，但不改分数（分数口径保持确定）。"""
    if credible:
        return []
    qc: list[str] = []
    for ch in channels:
        if ch.get("channel") != "text":
            continue
        for entry in ch.get("scores", {}).values():
            old = float(entry.get("confidence", 0.5))
            entry["confidence"] = round(max(_MIN_CONF, old * 0.5), 2)
        qc.append("文字通道：转写未检出连续人声（疑似静音或纯音乐），文字判分依据不足，"
                  "可信度已下调，请核对音频内容")
    return qc


# ------------------------------------------------------------------ 通道分配
def channel_weights(dim: Dimension) -> dict[str, float]:
    hay = f"{dim.name} {dim.name_en} {' '.join(dim.checkpoints)}".lower()
    vision = any(h.lower() in hay for h in VISION_HINT)
    audio = any(h.lower() in hay for h in AUDIO_HINT)
    if vision and not audio:
        return {"vision": 1.0}
    if audio and not vision:
        if "流利" in hay or "fluency" in hay or "连贯" in hay:
            return {"audio": 0.6, "text": 0.4}
        return {"audio": 1.0}
    if vision and audio:
        return {"audio": 0.5, "vision": 0.5}
    return {"text": 1.0}


def dims_for_channel(rubric: Rubric, channel: str) -> list[Dimension]:
    return [d for d in rubric.dimensions if channel_weights(d).get(channel, 0) > 0]


def channel_dim_keys(rubric: Rubric, channel: str) -> list[str]:
    return [d.key for d in dims_for_channel(rubric, channel)]


# ------------------------------------------------------------------ 时长核查
_UNIT_MIN = r"(?:分钟|分|min(?:ute)?s?\b)"
_UNIT_SEC = r"(?:秒|sec(?:onds?)?\b)"
_RANGE_RE = re.compile(rf"(\d+(?:\.\d+)?)\s*[-~–到至]\s*(\d+(?:\.\d+)?)\s*{_UNIT_MIN}", re.I)
_RANGE_SEC_RE = re.compile(rf"(\d+)\s*[-~–到至]\s*(\d+)\s*{_UNIT_SEC}", re.I)
_ONE_RE = re.compile(rf"(\d+(?:\.\d+)?)\s*{_UNIT_MIN}", re.I)
_ONE_SEC_RE = re.compile(rf"(\d+)\s*{_UNIT_SEC}", re.I)
_MMSS_RE = re.compile(r"\b(\d{1,2}):(\d{2})\s*[-~–]\s*(\d{1,2}):(\d{2})\b")


@dataclass
class TimingCheck:
    duration: float
    target: str = ""
    low: float = 0.0
    high: float = 0.0
    ok: bool | None = None
    note: str = ""


def check_timing(duration: float, topic: str = "", requirements: str = "") -> TimingCheck:
    hay = f"{topic}\n{requirements}"
    low = high = 0.0
    m = _MMSS_RE.search(hay)
    if m:
        low = int(m.group(1)) * 60 + int(m.group(2))
        high = int(m.group(3)) * 60 + int(m.group(4))
    elif (m := _RANGE_RE.search(hay)):
        low, high = float(m.group(1)) * 60, float(m.group(2)) * 60
    elif (m := _RANGE_SEC_RE.search(hay)):
        low, high = float(m.group(1)), float(m.group(2))
    elif (m2 := _ONE_RE.search(hay)):
        target = float(m2.group(1)) * 60
        low, high = target * 0.85, target * 1.15
    elif (m2 := _ONE_SEC_RE.search(hay)):
        target = float(m2.group(1))
        low, high = target * 0.85, target * 1.15
    if not low or not high or duration <= 0:
        return TimingCheck(duration=duration, note="未从题目/要求中解析到明确时长要求，时长项不做硬性扣分")
    ok = low <= duration <= high
    gap = 0 if ok else min(abs(duration - low), abs(duration - high))
    note = (f"实测 {fmt_ts(duration)}（{duration:.0f}s），要求 {fmt_ts(low)}–{fmt_ts(high)}，"
            + ("符合要求" if ok else f"偏差 {fmt_ts(gap)}"))
    return TimingCheck(duration=duration, target=f"{fmt_ts(low)}-{fmt_ts(high)}",
                       low=low, high=high, ok=ok, note=note)


# ------------------------------------------------------------------ 证据核验
_PUNCT_RE = re.compile(r"[^0-9a-z\u4e00-\u9fff ]+")
_STOP = {"the", "a", "an", "of", "to", "and", "in", "is", "it", "that", "for", "with",
         "as", "on", "be", "are", "was", "we", "you", "i", "this", "our", "their"}


def _norm(text: str) -> str:
    text = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    text = _PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set[str]:
    return {t for t in _norm(text).split() if len(t) > 2 and t not in _STOP}


def verify_quote(quote: str, transcript: str) -> bool:
    """引文是否真的出自转写文本（子串命中或实词重叠率 ≥ 0.55）。"""
    q = _norm(quote)
    t = _norm(transcript)
    if len(q) < 3 or not t:
        return False
    if q in t:
        return True
    qt, tt = _tokens(quote), _tokens(transcript)
    if not qt:
        return False
    return len(qt & tt) / len(qt) >= 0.55


def _clamp(value: float, upper: float) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    v = max(0.0, min(float(value), upper))
    return round(v * 2) / 2


# ------------------------------------------------------------------ 通道调用
# 提示词正文全部收在 prompts 模块（内置默认 + 管理员在设置页改的覆盖），这里只管拼装。
def schema_hint(rubric: Rubric, keys: list[str]) -> str:
    dims = [d for d in rubric.dimensions if d.key in keys]
    lines = ['{', '  "observations": "对本通道材料的整体观察（中文，80字内）",', '  "scores": {']
    body = []
    for d in dims:
        pts = " / ".join(d.checkpoints)
        body.append(
            f'    "{d.key}": {{'
            f'"score": 0到{d.max_score:g}之间的数字（严格按评分要点：{pts}）, '
            f'"confidence": 0到1之间的小数（本通道判该维度的可靠度）, '
            f'"strengths": ["该维度做得好的点，须具体"], '
            f'"issues": [{{"desc": "问题是什么（具体，不要笼统形容词）", '
            f'"quote": "支撑证据：原文原句片段 或 frame@时间点", '
            f'"at": "mm:ss 时间点（如可得）", '
            f'"fix": "怎么改（可执行动作）", '
            f'"example": "改写示范（英文原句→修改后）"}}, ...], '
            f'"notes": "评分理由，指出扣分落在哪个评分要点"}}')
    lines.append(",\n".join(body))
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def _compose(*parts: str) -> str:
    """按顺序拼接非空片段，片段之间补一个空行。

    提示词块入库前会 strip 首尾空白（见 prompts.normalize），管理员也可能把某块整个清空，
    所以段与段之间的空行统一由这里补：块内文本不自带换行依赖，空块也不会留下多余空行。
    """
    return "\n\n".join(text for text in (p.strip("\n") for p in parts) if text)


def _material(block: str) -> str:
    """注入到提示词正文里的一段材料：非空则自带前后空行，空则整段消失。"""
    body = (block or "").strip("\n")
    return f"\n{body}\n" if body else ""


def _rubric_header(rubric: Rubric) -> str:
    return prompts.render("rubric_header", total=rubric.total, block=rubric.prompt_block())


def _dim_keys(keys: list[str], indent: str = "  ") -> str:
    return "\n".join(f"{indent}- {k}" for k in keys)


def run_text_channel(client: QwenClient, rubric: Rubric, transcript: str, topic: str,
                     requirements: str, timing: TimingCheck,
                     note: list[str] | None = None) -> dict:
    keys = channel_dim_keys(rubric, "text")
    prompt = _compose(
        _rubric_header(rubric),
        prompts.render(
            "text_context",
            topic=topic or "（未提供，题目契合度按内容自身判断并说明）",
            requirements=requirements or "（未提供）",
            timing=timing.note,
            transcript=transcript[:12000],
        ),
        prompts.get("common_rules"),
        prompts.render("text_tail", keys=_dim_keys(keys), schema=schema_hint(rubric, keys)),
    )
    comp = client.chat(prompt, system=prompts.get("system"), json_mode=True,
                       temperature=0.1, note=note)
    return normalize_channel("text", comp, rubric, keys)


def run_vision_channel(client: QwenClient, rubric: Rubric, frames: list[Path], topic: str,
                       timing: TimingCheck, transcript: str = "",
                       note: list[str] | None = None,
                       stamps: list[float] | None = None,
                       face: "FaceMetrics | None" = None) -> dict:
    keys = channel_dim_keys(rubric, "vision")
    n = len(frames)
    ts = [float(s) for s in (stamps or [])][:n]
    if len(ts) != n:
        # 没有真实采样时刻（兜底单帧、旧数据）才退回等距推测，措辞仍保持同一口径。
        interval = timing.duration / max(n, 1)
        ts = [i * interval for i in range(n)]
    stamp = ", ".join(f"frame@{fmt_ts(t)}" for t in ts)
    excerpt = transcript[:1500]
    prompt = _compose(
        _rubric_header(rubric),
        prompts.render("vision_frames", n=n, stamp=stamp, topic=topic or "（未提供）",
                       duration=fmt_ts(timing.duration)),
        prompts.get("common_rules"),
        prompts.render(
            "vision_tail",
            keys=_dim_keys(keys, "    "),
            micro=_material(prompts.get("micro_rules")),
            transcript=_material(
                "转写文本节选（仅供理解内容背景，不要据此给语音分）：\n<<<\n"
                + excerpt + "\n>>>") if excerpt else "",
            face=_material(face.evidence_block()) if face is not None else "",
            schema=schema_hint(rubric, keys),
        ),
    )
    comp = client.vision(prompt, frames, note=note)
    return normalize_channel("vision", comp, rubric, keys)


def run_audio_channel(client: QwenClient, rubric: Rubric, audio_path, transcript: str,
                      timing: TimingCheck, note: list[str] | None = None) -> dict:
    keys = channel_dim_keys(rubric, "audio")
    prompt = _compose(
        _rubric_header(rubric),
        prompts.render("audio_intro", transcript=transcript[:6000],
                       duration=fmt_ts(timing.duration)),
        prompts.get("common_rules"),
        prompts.render("audio_tail", keys=_dim_keys(keys), schema=schema_hint(rubric, keys)),
    )
    comp = client.audio(prompt, audio_path, note=note)
    return normalize_channel("audio", comp, rubric, keys)


def normalize_channel(name: str, comp: Completion, rubric: Rubric, keys: list[str]) -> dict:
    """把任一通道的模型输出规整成 {channel, model, observations, scores{dim: {...}}}。"""
    try:
        data = parse_json_block(comp.text or "")
    except QwenError:
        data = {}
    raw_scores = data.get("scores")
    scores = raw_scores if isinstance(raw_scores, dict) else {}
    out: dict[str, dict] = {}
    by_key = rubric.by_key()
    for k in keys:
        raw = scores.get(k)
        if isinstance(raw, (int, float)):
            raw = {"score": raw}
        if not isinstance(raw, dict):
            continue
        dim = by_key.get(k)
        if dim is None:
            continue
        out[k] = {
            "score": _clamp(float(raw.get("score") or 0), dim.max_score),
            "confidence": max(_MIN_CONF, min(1.0, float(raw.get("confidence") or 0.5))),
            "strengths": [str(s) for s in (raw.get("strengths") or []) if str(s).strip()][:6],
            "issues": _clean_issues(raw.get("issues")),
            "notes": str(raw.get("notes") or "")[:800],
        }
    return {
        "channel": name,
        "model": getattr(comp, "model", "") or "",
        "observations": str(data.get("observations") or "")[:1200],
        "scores": out,
    }


def _clean_issues(raw) -> list[dict]:
    issues: list[dict] = []
    for it in raw or []:
        if isinstance(it, str):
            issues.append({"desc": it, "quote": "", "at": "", "fix": "", "example": ""})
        elif isinstance(it, dict):
            desc = str(it.get("desc") or it.get("issue") or it.get("problem") or "").strip()
            if not desc:
                continue
            issues.append({
                "desc": desc[:400],
                "quote": str(it.get("quote") or "").strip()[:400],
                "at": str(it.get("at") or "").strip()[:12],
                "fix": str(it.get("fix") or it.get("suggestion") or "").strip()[:500],
                "example": str(it.get("example") or it.get("rewrite") or "").strip()[:500],
            })
    return issues[:8]


# ------------------------------------------------------------------ 聚合
@dataclass
class Aggregated:
    dimensions: list[dict] = field(default_factory=list)
    total: float = 0.0
    max_total: float = 0.0
    band: str = ""
    confidence: float = 0.0
    qc: list[str] = field(default_factory=list)
    pct: float = 0.0

    def to_dict(self) -> dict:
        return {"dimensions": self.dimensions, "total": self.total, "max_total": self.max_total,
                "band": self.band, "confidence": self.confidence, "qc": self.qc, "pct": self.pct}


def _signature(dim_key: str, desc: str) -> str:
    core = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", desc.lower()).strip()[:60]
    return f"{dim_key}:{hashlib_digest(core)}"


def hashlib_digest(text: str) -> str:
    import hashlib
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def aggregate(rubric: Rubric, channels: list[dict], transcript: str,
              timing: TimingCheck, stamps: list[float] | None = None,
              face: "FaceMetrics | None" = None) -> Aggregated:
    qc: list[str] = []
    dims: list[dict] = []
    total = 0.0
    if face is not None:
        qc.append(("微表情测量：" if face.available else "微表情测量未采信：") + face.headline())
    ts = [float(s) for s in (stamps or []) if s is not None]

    for dim in rubric.dimensions:
        weights = channel_weights(dim)
        parts: list[tuple[float, float, float]] = []  # (weight*conf, score, conf)
        merged_issues: list[dict] = []
        merged_strengths: list[str] = []
        notes: list[str] = []
        for ch in channels:
            entry = ch.get("scores", {}).get(dim.key)
            if not entry:
                continue
            w = weights.get(ch["channel"], 0.0)
            if w <= 0:
                qc.append(f"{dim.name}：通道 {ch['channel']} 越界判分，已忽略")
                continue
            conf = max(_MIN_CONF, min(1.0, float(entry.get("confidence", 0.5))))
            parts.append((w * conf, float(entry["score"]), conf))
            merged_strengths.extend(entry.get("strengths", []))
            notes.append(f"[{ch['channel']}] {entry.get('notes', '')}".strip())
            for iss in entry.get("issues", []):
                iss = dict(iss)
                iss["channel"] = ch["channel"]
                merged_issues.append(iss)

        if not parts:
            qc.append(f"{dim.name}：无通道给出可采信判定，标记为「未评」，"
                      f"其 {dim.max_score:g} 分权重按比例分摊到其余维度")
            dims.append(_dim_dict(dim, 0.0, 0.0, [], [], ["无可用证据（未评）"], not_applicable=True))
            continue

        wsum = sum(p[0] for p in parts)
        score = sum(p[0] * p[1] for p in parts) / wsum if wsum else 0.0
        conf = max(p[2] for p in parts)
        raw_scores = [p[1] for p in parts]
        if len(raw_scores) > 1 and (max(raw_scores) - min(raw_scores)) > dim.max_score * 0.25:
            conf = max(_MIN_CONF, conf - 0.15)
            qc.append(f"{dim.name}：通道间分歧 "
                      f"{' / '.join(f'{s:g}' for s in raw_scores)}，可信度下调至 {conf:.2f}")

        # 时长硬性核查
        if timing.ok is False and ("时间" in dim.name or "timing" in dim.name_en.lower()):
            ceiling = dim.max_score * 0.6
            if score > ceiling:
                qc.append(f"{dim.name}：时长不达标（{timing.note}），"
                          f"分数由 {score:g} 收缩至上限 {ceiling:g}")
                score = ceiling

        # 证据核验
        verified: list[dict] = []
        dropped = 0
        for iss in merged_issues:
            quote = iss.get("quote", "")
            if not quote:
                dropped += 1
                iss["evidence_ok"] = False
                verified.append(iss)
                continue
            if quote.startswith("frame@") or iss.get("channel") == "vision":
                iss["evidence_ok"] = _frame_ok(quote, timing.duration, ts)
            else:
                iss["evidence_ok"] = verify_quote(quote, transcript)
            if not iss["evidence_ok"]:
                dropped += 1
            verified.append(iss)
        if dropped:
            conf = max(_MIN_CONF, conf - 0.1 * min(dropped, 3))
            qc.append(f"{dim.name}：{dropped} 条证据未在转写中命中，标记待核并下调可信度")
        if face is not None and face.available and weights.get("vision"):
            qc.extend(_face_conflicts(dim, score, face, verified))

        evidence_ok = [i for i in verified if i.get("evidence_ok")]
        if dim.max_score and not evidence_ok and score / dim.max_score > 0.88:
            new = round(dim.max_score * 0.85 * 2) / 2
            qc.append(f"{dim.name}：无已核验证据支撑 85% 以上得分，"
                      f"由 {score:g} 收缩至 {new:g}（无证据不下判定）")
            score = new

        score = _clamp(score, dim.max_score)
        total += score
        dims.append(_dim_dict(dim, score, conf, merged_strengths, verified, notes))

    judged = [d for d in dims if not d.get("not_applicable")]
    judged_max = sum(d["max_score"] for d in judged)
    raw_total = sum(d["score"] for d in judged)
    if judged and judged_max < rubric.total:
        scaled = raw_total / judged_max * rubric.total
        qc.append(f"总分口径：已评维度 {raw_total:g}/{judged_max:g} 折算为满分 {rubric.total:g} 制 → {scaled:g}")
        total = scaled
    else:
        total = raw_total
    pct = (total / rubric.total * 100) if rubric.total else 0.0
    wsum = sum(d["max_score"] for d in judged) or 1.0
    confidence = sum(d["max_score"] * d["confidence"] for d in judged) / wsum if judged else 0.0
    return Aggregated(dimensions=dims, total=_clamp(total, rubric.total), max_total=rubric.total,
                      band=band_of(pct), confidence=round(min(1.0, max(0.0, confidence)), 2),
                      qc=qc, pct=round(pct, 1))


_FRAME_TOLERANCE = 2.0
# 与「眼神/表情」相关的判定关键词：只有这类扣分才可能和正脸率互相打脸。
_GAZE_WORDS = ("眼神", "对视", "注视", "目光", "看镜头", "表情", "面部", "微笑", "神态")


def _frame_ok(quote: str, duration: float, stamps: list[float] | None = None) -> bool:
    m = re.search(r"(\d{1,2}):(\d{2})", quote)
    if not m:
        return bool(duration)
    t = int(m.group(1)) * 60 + int(m.group(2))
    if stamps:
        # 有真实帧表时，时间点必须落在某张确实抽出来的画面附近（±2s）。只卡时长上界的话，
        # 模型编一个「视频里有、但没被抽到」的时刻照样通过，证据核验形同虚设。
        return min(abs(t - s) for s in stamps) <= _FRAME_TOLERANCE
    return 0 <= t <= (duration + 5)


def _face_conflicts(dim: Dimension, score: float, face: "FaceMetrics",
                    issues: list[dict]) -> list[str]:
    """客观朝向测量与模型判分的双向冲突提示。**只留痕、不动分数**：分数口径保持与历史
    可比，指标的作用是给人工复核指路，而不是给模型当第二个裁判。"""
    ratio = float(face.frontal_ratio or 0.0)
    pct = (score / dim.max_score) if dim.max_score else 1.0
    gaze = [i for i in issues
            if any(w in f"{i.get('desc', '')}{i.get('quote', '')}" for w in _GAZE_WORDS)]
    out: list[str] = []
    if pct < 0.7 and ratio >= 0.8 and gaze:
        out.append(f"{dim.name}：画面通道按眼神/表情扣分至 {pct:.0%}，但本地逐帧测量正脸率 "
                   f"{ratio:.0%}（{face.scanned} 帧）——两者冲突，请人工回看上述时间点；"
                   f"分数保持不调整")
    elif pct >= 0.9 and ratio < 0.35:
        out.append(f"{dim.name}：本地逐帧测量正脸率仅 {ratio:.0%}（{face.scanned} 帧），"
                   f"但该项给到 {pct:.0%}，疑似漏判回避对视；分数保持不调整，建议回看")
    return out


def _dim_dict(dim: Dimension, score: float, conf: float, strengths: list[str],
              issues: list[dict], notes: list[str], not_applicable: bool = False) -> dict:
    dedup_strengths: list[str] = []
    for s in strengths:
        s = s.strip()
        if s and s not in dedup_strengths:
            dedup_strengths.append(s)
    ratio = round(score / dim.max_score, 4) if dim.max_score else 0.0
    return {
        "key": dim.key, "index": dim.index, "name": dim.name, "name_en": dim.name_en,
        "checkpoints": dim.checkpoints, "max_score": dim.max_score,
        "score": round(score, 2), "ratio": 0.0 if not_applicable else ratio,
        "band": "未评" if not_applicable else band_of(ratio * 100),
        "not_applicable": not_applicable,
        "confidence": round(conf, 2),
        "strengths": dedup_strengths[:6],
        "issues": [_issue_out(i, dim.key) for i in issues[:12]],
        "notes": [n for n in notes if n][:6],
    }


def _issue_out(iss: dict, dim_key: str) -> dict:
    desc = iss.get("desc", "")
    return {
        "desc": desc, "quote": iss.get("quote", ""), "at": iss.get("at", ""),
        "fix": iss.get("fix", ""), "example": iss.get("example", ""),
        "channel": iss.get("channel", ""),
        "evidence_ok": bool(iss.get("evidence_ok", False)),
        "signature": _signature(dim_key, desc),
    }


def band_of(pct: float) -> str:
    if pct >= 90:
        return "A"
    if pct >= 80:
        return "B"
    if pct >= 70:
        return "C"
    if pct >= 60:
        return "D"
    return "E"


# ------------------------------------------------------------------ 叙述层
def narrative_prompt(rubric: Rubric, agg: Aggregated, transcript: str, topic: str,
                     history_note: str) -> str:
    payload = {
        "总分": f"{agg.total:g}/{agg.max_total:g}（{agg.pct:g}%，等级 {agg.band}）",
        "分项": [{
            "维度": d["name"], "得分": d["score"], "满分": d["max_score"],
            "可信度": d["confidence"], "评分理由": d["notes"],
            "优点": d["strengths"],
            "问题": [{"问题": i["desc"], "证据": i["quote"], "时间点": i["at"],
                     "证据已核验": i["evidence_ok"]} for i in d["issues"][:4]],
        } for d in agg.dimensions],
        "质控记录": agg.qc,
        "演讲题目": topic,
        "历史对比": history_note or "首次上传，无历史可比",
    }
    return prompts.render(
        "narrative_context",
        rubric=rubric.prompt_block(),
        result=json.dumps(payload, ensure_ascii=False, indent=1),
        transcript=transcript[:4000],
        schema=prompts.get("narrative_schema"),
    )


def fallback_narrative(agg: Aggregated) -> dict:
    scored = [d for d in agg.dimensions if not d.get("not_applicable")] or agg.dimensions
    ordered = sorted(scored, key=lambda d: d["ratio"])
    weakest = ordered[:2]
    strongest = list(reversed(ordered[-2:])) if len(ordered) >= 2 else ordered
    adv = [f"{d['name']}：{s}" for d in strongest for s in d["strengths"][:1]]
    if not adv:
        adv = [f"完成了一次完整演讲，全程得分 {agg.total:g}/{agg.max_total:g}"
               if agg.total else "完成了一次完整演讲，可作为改进基线"]
        adv += [f"{d['name']}：本维度相对其它维度扣分最少，优先保持"
                for d in strongest if d["ratio"] > 0][:2]
    dis = [f"{d['name']}（{d['score']:g}/{d['max_score']:g}）："
           + (d["issues"][0]["desc"] if d["issues"] else "得分偏低，缺少支撑证据") for d in weakest]
    sugg = [{"priority": i + 1, "dimension": d["name"],
             "action": (d["issues"][0]["fix"] if d["issues"] and d["issues"][0]["fix"]
                        else f"针对「{d['name']}」的评分要点逐条对照练习"),
             "example": (d["issues"][0]["example"] if d["issues"] else ""),
             "practice": "录像重讲同一段落 2 遍，只改这一个问题"} for i, d in enumerate(weakest)]
    summary = (f"本次得分 {agg.total:g}/{agg.max_total:g}（{agg.pct:g}%，等级 {agg.band}）。"
               f"相对最弱的是{'、'.join(d['name'] for d in weakest)}，"
               f"相对最好{'的' if len(strongest) > 1 else ''}是{'、'.join(d['name'] for d in strongest)}。"
               "请优先解决上面列出的第一个问题，其余保持现有水平。")
    return {"advantages": adv[:6], "disadvantages": dis[:6], "suggestions": sugg,
            "next_focus": sugg[0]["action"] if sugg else "继续按标准练习",
            "summary": summary}


def build_narrative(client: QwenClient, rubric: Rubric, agg: Aggregated,
                    transcript: str, topic: str, history_note: str = "",
                    note: list[str] | None = None) -> dict:
    """分数已定，这一步只让模型写「优点/缺点/改进建议/总结」，不允许改分。"""
    try:
        comp = client.chat(narrative_prompt(rubric, agg, transcript, topic, history_note),
                           system=prompts.get("narrative_system"), json_mode=True,
                           temperature=0.3, note=note)
        data = parse_json_block(comp.text or "")
    except Exception:
        return fallback_narrative(agg)
    if not isinstance(data, dict) or not data.get("summary"):
        return fallback_narrative(agg)
    agg_dict = agg.to_dict()
    return {
        "advantages": [str(x) for x in (data.get("advantages") or []) if str(x).strip()][:8],
        "disadvantages": [str(x) for x in (data.get("disadvantages") or []) if str(x).strip()][:8],
        "suggestions": _clean_suggestions(data.get("suggestions"), agg_dict),
        "next_focus": str(data.get("next_focus") or "")[:300],
        "summary": str(data.get("summary") or "")[:4000],
    }


def _clean_suggestions(raw, agg: dict) -> list[dict]:
    names = {d["name"] for d in agg["dimensions"]}
    out: list[dict] = []
    for i, item in enumerate(raw or []):
        if isinstance(item, str):
            out.append({"priority": i + 1, "dimension": "", "action": item,
                        "example": "", "practice": ""})
        elif isinstance(item, dict):
            dim = str(item.get("dimension") or "")
            out.append({
                "priority": int(item.get("priority") or i + 1),
                "dimension": dim if dim in names else "",
                "action": str(item.get("action") or item.get("fix") or "")[:500],
                "example": str(item.get("example") or "")[:600],
                "practice": str(item.get("practice") or "")[:400],
            })
    return out[:8]


# ------------------------------------------------------------------ 导师提问
QA_ANSWER_FALLBACK = "已收到你的作答，建议结合报告中的改进建议再补充一层论证。"
_QA_DROP_KEYS = ("frames", "stamps", "face", "transcript_excerpt")


def _ratio_key(d: dict) -> float:
    try:
        return float(d.get("ratio") or 0)
    except (TypeError, ValueError):
        return 0.0


def fallback_questions(report: dict, topic: str) -> list[str]:
    """不依赖模型的兜底出题：改进建议 + 最弱维度拼 3 道思考题，永远返回 3 条。"""
    name = (topic or "").strip() or "本次演讲"
    dims = [d for d in (report.get("dimensions") or []) if isinstance(d, dict)]
    weak = [str(d.get("name") or "").strip() for d in sorted(dims, key=_ratio_key)[:2]]
    weak = [d for d in weak if d]
    out: list[str] = []
    for s in [x for x in (report.get("suggestions") or []) if isinstance(x, dict)][:2]:
        action = str(s.get("action") or "").strip()
        dim = str(s.get("dimension") or "").strip()
        if action:
            out.append(f"关于{('「' + dim + '」') if dim else '这篇稿件'}，建议里让你「{action[:60]}」，"
                       f"你在《{name}》里打算改哪一句？为什么这样改？")
        elif dim:
            out.append(f"你在「{dim}」这一项还有差距，重讲《{name}》时你会怎么补强？请举一处具体改法。")
    for dim in weak:
        out.append(f"「{dim}」是这篇稿子相对最薄弱的一项，讲《{name}》时你哪一处最没底？打算怎么练？")
    out.append(f"如果只能改《{name}》的三段内容里的一个词，你会改哪个？请说出你的取舍理由。")
    seen: list[str] = []
    for q in out:
        q = q.strip()[:160]
        if q and q not in seen:
            seen.append(q)
    pad = [
        f"关于《{name}》，你最想让听众记住哪一句？这句现在够不够有说服力？",
        f"《{name}》的开头三句，重讲一遍你会换掉哪一句？为什么？",
        f"讲《{name}》时你自己觉得最像在读稿的是哪一段？打算怎么处理？",
    ]
    for q in pad:
        if len(seen) >= 3:
            break
        seen.append(q)
    return seen[:3]


def build_questions(client: QwenClient, rubric: Rubric, report: dict, transcript: str,
                    topic: str, requirements: str = "",
                    note: list[str] | None = None) -> list[str]:
    """评价完成后出 3 道针对本稿的思考题；模型不给足 3 条就整批改用兜底题。"""
    prompt = prompts.render(
        "qa_context",
        topic=(topic or "").strip() or "本次演讲",
        requirements=(requirements or "").strip() or "（未填写）",
        result=json.dumps({k: v for k, v in report.items() if k not in _QA_DROP_KEYS},
                          ensure_ascii=False, indent=1),
        transcript=(transcript or "")[:4000],
        schema=prompts.get("qa_schema"))
    try:
        comp = client.chat(prompt, system=prompts.get("qa_system"), json_mode=True,
                           temperature=0.5, note=note)
        data = parse_json_block(comp.text or "")
        got = [str(q).strip()[:300] for q in (data.get("questions") or []) if str(q).strip()] \
            if isinstance(data, dict) else []
    except Exception:  # noqa: BLE001 - 出题失败不应影响报告，退回本地题目
        got = []
    if len(got) < 3:
        got = fallback_questions(report, topic)
    return got[:3]


def review_answers(client: QwenClient, report: dict, pairs: list[tuple[str, str]],
                   note: list[str] | None = None) -> list[str]:
    """三题一次送评，按题序返回点评；条数对不上就整批回兜底文案。"""
    if not pairs:
        return []
    fallback = [QA_ANSWER_FALLBACK] * len(pairs)
    lines: list[str] = []
    focus = str(report.get("next_focus") or "").strip()
    if focus:
        lines.append(f"本稿下一步重点：{focus[:200]}")
    for i, (question, answer) in enumerate(pairs, 1):
        lines.append(f"问题{i}：{question}")
        lines.append(f"学生作答{i}：{(answer or '').strip() or '（未作答）'}")
    prompt = prompts.render("qa_review_context", qa="\n".join(lines),
                            schema=prompts.get("qa_review_schema"))
    try:
        comp = client.chat(prompt, system=prompts.get("qa_system"), json_mode=True,
                           temperature=0.2, note=note)
        data = parse_json_block(comp.text or "")
        got = [str(c).strip()[:300] for c in (data.get("comments") or []) if str(c).strip()] \
            if isinstance(data, dict) else []
    except Exception:  # noqa: BLE001 - 点评失败保留作答，只换文案
        return fallback
    return got if len(got) == len(pairs) else fallback


# ------------------------------------------------------------------ 最终报告
def build_report(rubric: Rubric, agg: Aggregated, narr: dict, channels: list[dict],
                 timing: TimingCheck, transcript: str, engine: str, model: str) -> dict:
    return {
        "rubric_version": rubric.version,
        "rubric_total": rubric.total,
        "total": agg.total, "max_total": agg.max_total, "band": agg.band,
        "pct": agg.pct, "confidence": agg.confidence,
        "dimensions": agg.dimensions,
        "advantages": narr.get("advantages", []),
        "disadvantages": narr.get("disadvantages", []),
        "suggestions": narr.get("suggestions", []),
        "next_focus": narr.get("next_focus", ""),
        "summary": narr.get("summary", ""),
        "qc": agg.qc,
        "timing": {"duration": timing.duration, "note": timing.note, "ok": timing.ok,
                   "target": timing.target},
        "channels": [{"channel": c["channel"], "model": c.get("model", ""),
                      "observations": c.get("observations", ""),
                      "keys": list(c.get("scores", {}).keys())} for c in channels],
        "provenance": {"asr_engine": engine, "judge_models": [c.get("model", "") for c in channels],
                       "model": model, "transcript_chars": len(transcript)},
        "transcript_excerpt": transcript[:1200],
    }
