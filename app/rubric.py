"""把 评价标准.txt 编译成机器可用的评分契约（维度、分值、判分要点、权重）。"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .config import settings

_SCORE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*分")
_HEAD_RE = re.compile(r"^\s*(\d+)\s*[.、)]?\s*(.+)$")
_EN_RE = re.compile(r"[（(]([^）)]+)[）)]")
_SLUG_RE = re.compile(r"[^a-z0-9]+")

_SEP_MASK = {"、": "\ue001", ",": "\ue002", "，": "\ue003", "+": "\ue004",
             "＋": "\ue005", "；": "\ue006", ";": "\ue007"}
_MASK = str.maketrans(_SEP_MASK)
_UNMASK = str.maketrans({v: k for k, v in _SEP_MASK.items()})


@dataclass
class Dimension:
    index: int
    key: str
    name: str
    name_en: str
    max_score: float
    checkpoints: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def weight_pct(self) -> float:
        return round(self.max_score, 1)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "key": self.key,
            "name": self.name,
            "name_en": self.name_en,
            "max_score": self.max_score,
            "checkpoints": self.checkpoints,
            "raw": self.raw,
        }


def _slug(name: str, name_en: str, index: int, taken: set[str]) -> str:
    src = (name_en or name).lower()
    src = src.replace("&", " and ")
    slug = _SLUG_RE.sub("_", src).strip("_")[:32].rstrip("_")
    base = slug or f"dimension_{index}"
    key = base
    n = 2
    while key in taken:
        key = f"{base}_{n}"
        n += 1
    taken.add(key)
    return key


def _split_points(text: str) -> list[str]:
    # 括号内的分隔符不切开，避免「（是否现实＋对行业的看法）」被拆成两段
    protected = re.sub(r"[（(][^）)]*[）)]",
                       lambda m: m.group(0).translate(_MASK), text)
    parts = re.split(r"[；;。\n]", protected)
    out: list[str] = []
    for p in parts:
        for q in re.split(r"[、,，+＋]", p):
            q = q.translate(_UNMASK).strip()
            if len(q) >= 2:
                out.append(q)
    return out


class Rubric:
    def __init__(self, dimensions: list[Dimension], source: str, title: str = ""):
        self.dimensions = dimensions
        self.source = source
        # source 是标准文件原文，只用于算版本指纹；界面上要展示的是短标题。
        self.title = title.strip() or "评价标准"
        self.total = round(sum(d.max_score for d in dimensions), 2)
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:10]
        self.version = f"rubric-{len(dimensions)}d-{digest}"

    def keys(self) -> list[str]:
        return [d.key for d in self.dimensions]

    def by_key(self) -> dict[str, Dimension]:
        return {d.key: d for d in self.dimensions}

    def anchor(self, key: str) -> float:
        d = self.by_key().get(key)
        return d.max_score if d else 0.0

    def prompt_block(self) -> str:
        lines = [
            f"满分 {self.total:g} 分，共 {len(self.dimensions)} 个维度。每个维度必须独立给分，"
            f"得分不得超过该维度分值上限，且必须是 0 到该上限之间的数字。",
            "",
        ]
        for d in self.dimensions:
            pts = " / ".join(d.checkpoints) if d.checkpoints else d.raw
            lines.append(f"[{d.key}] {d.index}. {d.name}（{d.name_en}）— 满分 {d.max_score:g} 分")
            lines.append(f"    评分要点：{pts}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "total": self.total,
            "dimensions": [d.to_dict() for d in self.dimensions],
        }


def _parse(text: str) -> list[Dimension]:
    dims: list[Dimension] = []
    taken: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        cols = [c.strip() for c in re.split(r"\t| {3,}", line) if c.strip()]
        if len(cols) < 2:
            continue
        score = None
        for c in cols:
            m = _SCORE_RE.search(c)
            if m:
                score = float(m.group(1))
                break
        if score is None:
            continue
        head = cols[0]
        hm = _HEAD_RE.match(head)
        index = int(hm.group(1)) if hm else len(dims) + 1
        title = hm.group(2).strip() if hm else head
        en = ""
        em = _EN_RE.search(title)
        if em:
            en = em.group(1).strip()
            title = _EN_RE.sub("", title).strip(" -（(")
        points = " ".join(cols[2:]) if len(cols) > 2 else ""
        if not points and len(cols) > 1 and _SCORE_RE.search(cols[1]) is None:
            points = cols[1]
        checkpoints = _split_points(points) or _split_points(title)
        dims.append(Dimension(
            index=index,
            key=_slug(title, en, index, taken),
            name=title,
            name_en=en or title,
            max_score=score,
            checkpoints=checkpoints,
            raw=line,
        ))
    return dims


def load_rubric() -> Rubric:
    path = settings.rubric_file
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    dims = _parse(text)
    if not dims:
        raise RuntimeError(f"未能从 {path} 解析出任何评分维度，请检查评价标准文件格式（制表符分隔：维度/分值/评分要点）")
    return Rubric(dims, text, path.stem)


rubric = load_rubric()
