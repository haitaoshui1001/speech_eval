"""文字稿修订：模型只回 diff，定位与应用全在代码里（方案《文字稿纠错与朗读合成》§4）。

三条不变式：
1. 模型永远不改稿 —— 建议只以 items（before/after 对）承载，成稿由 apply_revisions()
   精确替换生成，「改了什么」永远可枚举、可展示、可回滚；
2. 校验靠代码不靠恳求 —— prompt 里的规则是红线 R1~R4 的镜像说明，真正兜底的是
   validate_items()：越界条目直接丢弃并在 dropped 里留痕；
3. 绝不覆盖 transcript —— 修订产物另行落库，只有学生显式采纳（decision=accepted）
   的条目进成稿；pending 等同未采纳，不动鼠标 = 不改。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field

from . import prompts
from .config import settings
from .qwen import QwenError, parse_json_block

# kind 键名被红线校验与页面徽标共用，顺序即展示顺序；recommended 只影响 UI 默认勾选，
# 代码侧一律 pending —— 默认勾选是「建议」，采纳必须是学生的动作。
KINDS: tuple[tuple[str, str, str, bool], ...] = (
    ("homophone", "听写订正", "ASR 听错的同音/近音词（如「全面建社」应为「全面建设」）", True),
    ("term", "术语订正", "术语与专名错字，优先参考学生填写的术语提示", True),
    ("grammar", "语法订正", "明显语法口误的最小改动", True),
    ("filler", "口头禅", "口头禅与无意义重复，仅为朗读流畅服务", False),
    ("style", "小幅润色", "结合主题的小幅润色，只调词不改意", False),
)
KIND_KEYS = frozenset(k for k, _, _, _ in KINDS)
KIND_LABELS = {k: label for k, label, _, _ in KINDS}

MAX_ITEM_DELTA = 20           # R1：单条 |len(after)-len(before)| 上限（字符）
_SENT_ENDS = "。！？!?"        # R4：句末标点集合
MAX_TRANSCRIPT = 12000        # 与文字评审通道同口径的注入截断

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class RevisionItem:
    idx: int                   # 校验通过后的连续编号（页面 accept_<idx> 依赖它）
    before: str
    after: str
    kind: str
    reason: str = ""
    confidence: float = 0.5
    decision: str = "pending"  # pending | accepted | rejected
    pos: int = -1              # before 在原稿中的唯一命中位置

    @property
    def delta(self) -> int:
        return len(self.after) - len(self.before)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "RevisionItem":
        return cls(
            idx=int(raw.get("idx") or 0),
            before=str(raw.get("before") or ""),
            after=str(raw.get("after") or ""),
            kind=str(raw.get("kind") or "grammar"),
            reason=str(raw.get("reason") or ""),
            confidence=_clamp_float(raw.get("confidence"), 0.5),
            decision=str(raw.get("decision") or "pending"),
            pos=int(raw.get("pos", -1)),
        )


@dataclass
class RevisionResult:
    items: list[RevisionItem] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    model: str = ""
    warnings: list[str] = field(default_factory=list)


def _clamp_float(value: object, default: float) -> float:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, v))


def source_hash(text: str) -> str:
    """原稿指纹：重新分析后 transcript 变化即判定旧建议失效（方案 §5.2）。"""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def kinds_block() -> str:
    return "\n".join(f"{key} = {desc}" for key, _, desc, _ in KINDS)


def schema_block() -> str:
    return json.dumps({
        "items": [{
            "before": "逐字摘自原文且在全文唯一的片段",
            "after": "替换后的文本",
            "kind": "类型键名",
            "reason": "为什么这样改（40 字以内）",
            "confidence": 0.9,
        }]
    }, ensure_ascii=False, indent=2)


def build_messages(transcript: str, topic: str, requirements: str,
                   glossary: str = "", warnings: list[str] | None = None) -> list[dict]:
    text = (transcript or "").strip()
    if len(text) > MAX_TRANSCRIPT:
        text = text[:MAX_TRANSCRIPT] + "……（后文超长未送入）"
        if warnings is not None:
            warnings.append(f"转写稿超过 {MAX_TRANSCRIPT} 字，只对前段提出建议")
    prompt = prompts.render("revise_head", topic=topic or "（未填写）",
                            requirements=requirements or "（无）",
                            glossary=glossary.strip() or "（无）",
                            transcript=text)
    prompt += "\n\n" + prompts.render("revise_tail", kinds=kinds_block(),
                                      schema=schema_block())
    return [{"role": "user", "content": prompt}]


def parse_items(raw: str) -> list[dict]:
    """从模型输出里抠出 items 数组；结构不对就当作「没有建议」而不是抛错。"""
    try:
        obj = parse_json_block(raw)
    except QwenError:
        return []
    items = obj.get("items")
    return [it for it in items if isinstance(it, dict)] if isinstance(items, list) else []


def validate_items(raw_items: list[dict], transcript: str, *,
                   max_ratio: int | None = None,
                   max_style: int | None = None) -> RevisionResult:
    """五条红线逐条硬卡（方案 §4.3）：越界条目丢弃并在 dropped 留痕。

    检查顺序即优先级：先保证「对得上、改得小、事实不变、不扩句」，
    再谈配额（R5 style）与总闸门（R2 比例）；R2 一旦触发，其后条目全部不再采纳。
    """
    result = RevisionResult()
    ratio = settings.revise_max_change_ratio if max_ratio is None else max_ratio
    style_cap = settings.revise_max_style_items if max_style is None else max_style
    budget = int(len(transcript or "") * max(0, ratio) / 100)
    spent = 0
    style_count = 0
    gate_hit = False
    taken: list[tuple[int, int]] = []

    for i, raw in enumerate(raw_items, start=1):
        before = str(raw.get("before") or "").strip()
        after = str(raw.get("after") or "").strip()
        kind = str(raw.get("kind") or "").strip()
        short = before[:24] or "（空）"

        def drop(tag: str, _short: str = short, _i: int = i) -> None:
            result.dropped.append(f"#{_i} {_short}：{tag}")

        if not before or not after or before == after:
            drop("条目无效")
            continue
        if gate_hit:
            drop("全篇改幅已用尽，未采纳")
            continue
        hits = transcript.count(before)
        if hits == 0:
            drop("原文中没有这段话")
            continue
        if hits > 1:
            drop("原文出现多次，不敢定位")
            continue
        pos = transcript.find(before)
        span = (pos, pos + len(before))
        if any(s < span[1] and span[0] < e for s, e in taken):
            drop("与前一条改动区间重叠")
            continue
        if abs(len(after) - len(before)) > MAX_ITEM_DELTA:
            drop(f"单条幅度超过 {MAX_ITEM_DELTA} 字（R1）")
            continue
        if _NUMBER_RE.findall(before) != _NUMBER_RE.findall(after):
            drop("数字前后不一致（R3）")
            continue
        if _sent_ends(after) > _sent_ends(before):
            drop("新增句子，涉嫌扩写（R4）")
            continue
        if kind not in KIND_KEYS:
            drop(f"未知类型 {kind or '（空）'}")
            continue
        if kind == "style":
            if style_count >= style_cap:
                drop(f"style 条数超过上限 {style_cap}（R5）")
                continue
            style_count += 1
        cost = abs(len(after) - len(before))
        if spent + cost > budget:
            gate_hit = True
            drop(f"累计改幅将超过原文 {ratio}%（R2），本条与后续均未采纳")
            continue
        spent += cost
        taken.append(span)
        result.items.append(RevisionItem(
            idx=len(result.items) + 1,
            before=before, after=after, kind=kind,
            reason=str(raw.get("reason") or "").strip()[:120],
            confidence=_clamp_float(raw.get("confidence"), 0.5),
            pos=pos,
        ))
    return result


def _sent_ends(text: str) -> int:
    return sum(text.count(c) for c in _SENT_ENDS)


def apply_revisions(text: str, items: list[RevisionItem]) -> str:
    """只应用显式 accepted 的条目；按原稿坐标从后往前替换，天然幂等。

    幂等性来自坐标快照：第二次应用时 pos 处的文本已经是 after，与 before 对不上，
    条目被整体跳过 —— 所以 apply(apply(t)) == apply(t)，「再点一次采纳」不会越改越飞。
    """
    out = text or ""
    for it in sorted((x for x in items if x.decision == "accepted"),
                     key=lambda x: x.pos, reverse=True):
        if it.pos < 0 or not it.before:
            continue
        if out[it.pos:it.pos + len(it.before)] == it.before:
            out = out[:it.pos] + it.after + out[it.pos + len(it.before):]
    return out


def items_to_json(items: list[RevisionItem]) -> str:
    return json.dumps([it.to_dict() for it in items], ensure_ascii=False)


def items_from_json(raw: str) -> list[RevisionItem]:
    try:
        data = json.loads(raw or "[]")
    except ValueError:
        return []
    return [RevisionItem.from_dict(it) for it in data if isinstance(it, dict)]


def decisions_from_form(form: dict, items: list[RevisionItem]) -> list[RevisionItem]:
    """把「accept_<idx> 复选框」表单映射回条目：勾了的 accepted，其余 rejected。

    表单里没有的 idx 视为未勾选 —— pending 在应用时等同 rejected（§5.2 保证一）。
    """
    chosen = {str(k) for k, v in form.items() if str(v).lower() in {"1", "on", "true", "yes"}}
    for it in items:
        it.decision = "accepted" if f"accept_{it.idx}" in chosen else "rejected"
    return items


def is_stale(transcript: str, meta: dict) -> bool:
    """失效判定：meta 里存有指纹而与当前原稿不符——重新分析后旧建议作废（§5.2）。"""
    stored = str((meta or {}).get("source_hash") or "")
    return bool(stored) and stored != source_hash(transcript)


def stamp_items(items: list[RevisionItem], segments: list[dict]) -> list[dict]:
    """给条目附加时间点：报告页据此跳回视频原位，让学生听到自己原来的说法。

    transcript 由各段文本以单空格 join 而成（transcribe.py），故第 i 段在原稿中的
    起始偏移 = 前 i-1 段 (len(text)+1) 的累计；pos 落在哪段之后即取该段的 ts。
    """
    bounds: list[tuple[int, str]] = []
    acc = 0
    for seg in segments or []:
        text = str(seg.get("text") or "")
        bounds.append((acc, str(seg.get("ts") or "")))
        acc += len(text) + 1
    stamped: list[dict] = []
    for it in items:
        d = it.to_dict()
        ts = ""
        for start, label in bounds:
            if it.pos < start:
                break
            ts = label
        d["ts"] = ts
        stamped.append(d)
    return stamped


def run_revision(client, transcript: str, topic: str, requirements: str,
                 glossary: str = "", note: list[str] | None = None) -> RevisionResult:
    """一次 chat 调用拿建议（供 B 线路由 / pipeline 调用，client 可注入便于自测）。"""
    warnings: list[str] = []
    messages = build_messages(transcript, topic, requirements, glossary, warnings)
    model = settings.revise_model or settings.chat_model
    comp = client.chat(messages[0]["content"], model=model, json_mode=True,
                       temperature=0.1, note=note, label="文字稿修订")
    result = validate_items(parse_items(comp.text), transcript)
    result.model = comp.model
    result.warnings = warnings
    return result
