"""主题模糊匹配：给「历次对比」用的零依赖文本归一化 + 二元组 Dice 相似度。

为什么不能直接用等号比较主题：主题是人手打的，同一件事会有「演讲比赛 第三届」和
「演讲比赛第3届」这种写法差异——空格、标点、全角半角、中文数字与阿拉伯数字都会
让精确匹配漏掉本该归到一组的对比对象。

三条硬规矩：
1. 只匹配、不合并——本模块只回答「这两个像不像同一个主题」，从不改写库里的原文；
2. 短串不模糊——归一化后不足 MIN_FUZZY_LEN 个字符的，必须完全相等才算，
   否则「春」「秋」这种一字标题会被判成同主题；
3. 阈值可调但有下限上限，见 config.TOPIC_MATCH_THRESHOLD（默认 0.60）。
"""
from __future__ import annotations

import re
import unicodedata

MIN_FUZZY_LEN = 4            # 短于这个长度只认全等
CN_NUM = {"零": "0", "一": "1", "二": "2", "三": "3", "四": "4",
          "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}

_PUNCT = re.compile(r"[\W_]+")          # 词字符以外的全部符号（含中英文标点）


def normalize(text: str | None) -> str:
    """归一化：全角转半角、统一小写、中文数字转阿拉伯、去空格与标点。"""
    s = unicodedata.normalize("NFKC", str(text or "")).lower()
    s = "".join(CN_NUM.get(ch, ch) for ch in s)
    return _PUNCT.sub("", s)


def key_of(item: dict) -> tuple[str, str]:
    """取匹配键：主题优先，主题为空回退标题。返回 (归一化键, 来源标记)。"""
    topic = str(item.get("topic") or "").strip()
    if topic:
        return normalize(topic), "topic"
    return normalize(item.get("title") or ""), "title"


def bigrams(s: str) -> set:
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else {s} if s else set()


def similarity(a: str | None, b: str | None) -> float:
    """两段文本的相似度（0~1）。输入是原文，内部自行归一化。"""
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if len(na) < MIN_FUZZY_LEN or len(nb) < MIN_FUZZY_LEN:
        return 0.0
    ba, bb = bigrams(na), bigrams(nb)
    return 2 * len(ba & bb) / (len(ba) + len(bb))


def same_topic(a: str | None, b: str | None, threshold: float) -> bool:
    return similarity(a, b) >= threshold


def match_key(item: dict) -> str:
    return key_of(item)[0]


def raw_of(item: dict) -> str:
    """与 key_of 同一回退规则，但返回原文（similarity 内部会自行归一化）。"""
    topic = str(item.get("topic") or "").strip()
    return topic if topic else str(item.get("title") or "")


def same_topic_ids(items: list[dict], anchor_id, threshold: float) -> list[int]:
    """与锚点条目同主题（含全等）的全部 id，保持传入顺序。"""
    anchor = next((it for it in items if int(it["id"]) == int(anchor_id)), None)
    if anchor is None:
        return []
    akey = match_key(anchor)
    if not akey:
        return [int(anchor["id"])]
    araw = raw_of(anchor)
    return [int(it["id"]) for it in items
            if match_key(it) == akey or same_topic(raw_of(it), araw, threshold)]


def cluster(items: list[dict], threshold: float) -> list[dict]:
    """贪心聚类：按传入顺序，每个条目并入第一个与代表全等或相似的簇。

    返回 [{"key": 代表归一化键, "label": 代表原始主题, "members": [...]}...]，
    簇的顺序 = 各簇首次出现条目的顺序。
    """
    clusters: list[dict] = []
    for it in items:
        k = match_key(it)
        if not k:
            continue
        placed = False
        for c in clusters:
            if c["key"] == k or same_topic(k, c["key"], threshold):
                c["members"].append(it)
                placed = True
                break
        if not placed:
            clusters.append({"key": k, "label": str(it.get("topic") or it.get("title") or "").strip(),
                             "members": [it]})
    return clusters


def topic_selects(items: list[dict], threshold: float) -> list[dict]:
    """给对比页「主题下拉 + 次数下拉」的数据：列出全部主题簇（含只分析过一次的），
    按簇内次数降序、同次数按最新一次降序。

    每簇 {"label": 代表主题, "n": 次数, "anchor": 簇内最新一次的 id, "ids": 按时间升序的 id 列表}。
    下拉以 anchor 作选项值，后端据此在簇内取「最近 N 次」，天然杜绝跨主题对比。
    """
    out = []
    for c in cluster(items, threshold):
        ids = [int(m["id"]) for m in c["members"]]
        out.append({"label": c["label"] or "（未填主题）", "n": len(ids),
                    "anchor": ids[-1], "ids": ids})
    out.sort(key=lambda o: (-o["n"], -o["anchor"]))
    return out
