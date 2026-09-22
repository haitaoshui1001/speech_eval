"""大模型提示词：内置默认 + 管理员可改的落盘覆盖。

为什么不进 .env（与 config 的机制刻意分开）：
  1. .env 是 KEY=VALUE，写盘时换行会被压平（见 config._quote_value），而提示词是十几行长文本；
  2. 提示词里全是成对的花括号与引号，塞进 .env 的引号规则里极易写坏；
  3. 单独一个 JSON 文件可以整份删掉即「全部恢复内置默认」，语义干净。

存什么：只存「与内置默认不同」的块。管理员把文本改回原样，这一项就自动回到内置默认。
怎么存：每块入库前 strip 首尾空白。段与段之间的空行由模板和调用方负责，因为 textarea
        的往返本身不可靠（HTML 解析器会吞掉紧随 <textarea> 的换行，浏览器可能补尾换行），
        首尾留白会让「什么都没改的一次保存」被误判成自定义覆盖。

占位符：{name} 由系统在运行时注入（转写文本、维度键名、JSON 结构等）。保存时双向校验——
        本块声明的占位符必须都在，声明之外的 {…} 不许出现；渲染用单次正则替换，
        替换结果不再二次扫描，所以转写文本里恰好写着 {keys} 也不会被意外展开。
"""
from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .config import BASE_DIR

# 与其他配置入口一致：默认落在项目根目录，测试用环境变量指到隔离目录。
PROMPTS_FILE = Path(os.getenv("PROMPTS_FILE", "").strip() or (BASE_DIR / "提示词.json"))

_TOKEN_RE = re.compile(r"\{(\w+)\}")
_MAX_CHARS = 6000


@dataclass(frozen=True)
class PromptField:
    """一个可编辑的提示词块。"""
    key: str
    label: str
    default: str
    hint: str = ""                       # 输入框下方：这块在管什么 + 占位符含义
    note: str = ""                       # 生效方式 / 误改后果
    tokens: tuple[str, ...] = ()         # 必须原样保留的占位符
    must_contain: tuple[str, ...] = ()   # 必须出现的字面量（用于 JSON 输出的键名）
    strict_braces: bool = False          # True 时禁止出现声明之外的 {变量}


def _join(items: list[str]) -> str:
    return "\n".join(items)


def _brace(name: str) -> str:
    return "{" + name + "}"


PROMPT_GROUPS: tuple[tuple[str, str, tuple[PromptField, ...]], ...] = (
    ("共用规则", "三个评审通道都会带上的身份设定、评分标准引导语与硬性规则。", (
        PromptField("system", "评审系统提示词",
            "你是一位严格的英语演讲评分员。你必须严格按照给定的评分标准逐维度判分，"
            "不得凭整体印象给分，不得给出标准之外的维度。任何分数都必须附上来自输入材料的证据"
            "（原句或画面时间点）。没有证据就不许扣分，也不许给高分。只输出 JSON。",
            "作为 system 角色随文字通道（chat）发送，是「不凭印象打分」的总闸；"
            "画面与语音通道以 user 角色收图/收音频，同样的要求写在各自的正文规则里。",
            "保存后下一次分析生效，进行中的任务不受影响"),
        PromptField("rubric_header", "评分标准引导语",
            "以下是本次评价唯一依据的评分标准（满分 {total} 分）：\n{block}",
            "三个通道开头的标准声明。{total} = 标准总分，{block} = 评价标准.txt 渲染出的评分要点。",
            "删掉 {block} 模型就读不到评分标准，等于瞎打分"),
        PromptField("common_rules", "三通道通用硬性规则",
            _join([
                "硬性规则：",
                "1. score 只能是 0 到该维度满分之间的数字；扣分必须说明扣在哪个评分要点上。",
                "2. 锚点：0-39% 差 / 40-54% 勉强及格 / 55-69% 中等 / 70-84% 良好 / 85-100% 优秀。",
                "3. quote 必须逐字摘自我给你过的材料，不许编造、不许改写；改写在 example 字段里给。",
                "4. 口音不等于发音错误；辞藻华丽不等于准确；流利不等于有内容。不要因观点不同压分。",
                "5. 每个维度至少 1 条 strengths；确无问题可以给空 issues，不许硬凑。",
            ]),
            "拼在每个通道正文的「评分锚点 / 证据要求」段。第 2 条的锚点区间与代码里的分档一致，改法要谨慎。",
            "保存后下一次分析生效"),
    )),
    ("文字通道", "只喂转写文本，判内容、结构、语法等与声音画面无关的维度。", (
        PromptField("text_context", "文字通道 · 材料段",
            _join([
                "本次演讲题目：{topic}",
                "老师要求：{requirements}",
                "客观时长核查：{timing}",
                "",
                "演讲转写文本（唯一文字依据）：",
                "<<<",
                "{transcript}",
                ">>>",
            ]),
            "{topic} {requirements} {timing} {transcript} 由系统注入；转写超过 12000 字会被截断。",
            "「<<< >>>」是材料边界，删掉会让模型把演讲内容当成指令"),
        PromptField("text_tail", "文字通道 · 收尾指令",
            _join([
                "请只判你有依据的维度（键：",
                "{keys}",
                "）",
                "按下面结构输出 JSON：",
                "{schema}",
            ]),
            "{keys} = 本通道该判的维度键列表，{schema} = 系统按评分标准生成的 JSON 结构。",
            "占位符必须保留，否则输出无法解析，该通道会被记为失败",
            tokens=("keys", "schema"), strict_braces=True),
    )),
    ("画面通道", "喂关键帧图像，判台风、眼神、表情等只看得见的维度。", (
        PromptField("vision_frames", "画面通道 · 帧说明段",
            _join([
                "下面是从演讲视频按时间顺序抽取的 {n} 帧画面，对应时间点：{stamp}",
                "演讲题目：{topic}；视频总时长 {duration}",
            ]),
            "{n} {stamp} {topic} {duration} 由系统注入；{stamp} 是 frame@mm:ss 时间点清单。",
            "删掉 {stamp} 模型就没法给出可核验的证据时间点", strict_braces=False),
        PromptField("micro_rules", "画面通道 · 眼神与表情判读规则",
            _join([
                "眼神与表情的判读要求（逐帧看，不要只凭整体印象）：",
                "1. 眼神交流与面部表情要分别判断，各自给 frame@时间点 证据，不许合并成一句「台风欠佳」。",
                "2. 眼神：对每一帧判断视线落在 镜头/评委、稿件或屏幕、侧方/观众席、上方（回想）"
                "中的哪一类。连续多帧不朝镜头 → 记「回避对视」；朝向来回跳变 → 记「眼神飘忽」；"
                "两者都要点出具体是哪几帧。",
                "3. 表情：判断是否自然放松、与内容情绪是否匹配。皱眉、抿嘴、眼神发直、"
                "全程一个表情、表情与内容错位等，都要写明出现在哪一帧；不许只写「表情不自然」。",
                "4. 画面是按时间稀疏抽取的静态帧，眨眼频率、微表情的瞬时变化看不出来。"
                "看不出依据就别扣分，把不确定写进 notes 并调低 confidence，不要编造细节。",
                "5. 若下方给了「客观人脸测量」，它是本地逐帧测得的交叉核对材料：与你的判断一致就照常判；"
                "不一致时保留你的判断，但必须在 notes 里说明冲突，不许为了迎合数字改分。",
            ]),
            "抽帧是稀疏静态画面，能看出「朝向哪里」，看不出眨眼频率和瞬时微表情 —— "
            "不写清这条边界，模型会顺手编出「频繁眨眼、嘴角抽搐」这类无法核验的细节。",
            "整块留空只会退回内置默认；把它改写成「不必逐帧举证」等于拆掉防编造的护栏，微表情分会明显变水"),
        PromptField("vision_tail", "画面通道 · 收尾指令",
            _join([
                "你只看到画面，听不到声音。所以：",
                "  - 只判画面能支撑的维度（键：",
                "{keys}",
                "  ）",
                "  - 涉及声音的评分要点写进 notes 并说明「画面不可判」，不得凭空扣分；",
                "  - 每条证据的 quote 用 frame@时间点 的格式，例如 \"frame@01:20\"，"
                "时间点必须取自上面给出的列表，不许自己编。",
                "{micro}{transcript}{face}",
                "按下面结构输出 JSON：",
                "{schema}",
            ]),
            "{keys} {micro} {transcript} {face} {schema} 由系统注入："
            "micro = 上面的眼神与表情规则，transcript / face = 带标题和 <<< >>> 边界的整块材料"
            "（无材料时占位符位置自动消失，不留空行），schema = 系统生成的 JSON 结构。",
            "占位符缺一个就会少一整段依据",
            tokens=("keys", "micro", "transcript", "face", "schema"), strict_braces=True),
    )),
    ("语音通道", "直接听音频，对照转写判发音、语调、流畅度。", (
        PromptField("audio_intro", "语音通道 · 听审说明段",
            _join([
                "请先听音频，再对照下面的转写文本（转写只用于定位说了什么，发音与语调以音频为准）：",
                "<<<",
                "{transcript}",
                ">>>",
                "",
                "视频时长 {duration}",
            ]),
            "{transcript} {duration} 由系统注入；转写节选超过 6000 字会被截断。",
            "「以音频为准」这句是防止模型照抄转写去判发音",
            strict_braces=False),
        PromptField("audio_tail", "语音通道 · 收尾指令",
            _join([
                "请只判与声音有关的维度（键：",
                "{keys}",
                "）",
                "对发音/语调：指出具体音素或单词层面的问题（如 th、r/l、词重音位置、句末语调），"
                "并给出时间点。",
                "按下面结构输出 JSON：",
                "{schema}",
            ]),
            "{keys} {schema} 由系统注入。",
            "删掉音素层面的要求，发音维度会退化成「听着挺顺」的印象分",
            tokens=("keys", "schema"), strict_braces=True),
    )),
    ("语音转写", "把视频里的音频逐字转成文本 —— 所有文字证据都从这里来。", (
        PromptField("asr_instruction", "转写指令",
            "请把这段音频逐字转写为文本，只输出转写内容本身，不要解释、不要加引号。",
            "转写走 omni / audio 类模型兜底时，这段话随音频一起发送；主路径 qwen3-asr-flash "
            "由模型自行转写，不吃这段指令。",
            "改成「总结大意」之类会让转写文本不再是原话，逐字证据核验会大面积失败"),
    )),
    ("反馈叙述", "分数已由三通道确定性算出，这一步只让学生看懂下一步该练什么。", (
        PromptField("narrative_system", "反馈系统提示词",
            "你是大学英语演讲课的指导老师。基于已经确定的分项得分与已核验证据，"
            "写给学生看的中文反馈：先说优点，再说缺点，再给可执行改进建议，最后总结。"
            "不得改变动任何分数；不得引用没有出现在输入证据里的内容。只输出 JSON。",
            "反馈层的角色设定。「不得改动分数」是叙述与评分分离的关键一句。",
            "保存后下一次分析生效"),
        PromptField("narrative_context", "反馈 · 输入材料段",
            _join([
                "评分标准（仅供理解各维度含义）：",
                "{rubric}",
                "",
                "本次评价结果（JSON）：",
                "{result}",
                "",
                "转写文本节选：",
                "<<<",
                "{transcript}",
                ">>>",
                "",
                "请输出 JSON：",
                "{schema}",
            ]),
            "{rubric} = 评分标准，{result} = 已定的分项得分与证据，{transcript} = 转写节选，"
            "{schema} = 下面的输出结构。",
            "四个占位符缺一不可",
            tokens=("rubric", "result", "transcript", "schema"), strict_braces=True),
        PromptField("narrative_schema", "反馈 · 输出结构",
            _join([
                "{",
                '  "advantages": ["3-6 条优点，具体到维度与原句，不要空话"],',
                '  "disadvantages": ["3-6 条主要缺点，按影响大小排序"],',
                '  "suggestions": [{"priority": 1, "dimension": "对应维度名", "action": "怎么改", '
                '"example": "示范表达或练习动作", "practice": "下次练什么、怎么练"}],',
                '  "next_focus": "下一步只练一件事（1 个优先项）",',
                '  "summary": "2-3 段中文总结：整体水平、主要差距、进步路径；如需引用请只引用已核验证据"',
                "}",
            ]),
            "这份 JSON 骨架决定报告页能不能显示出优点 / 缺点 / 建议，改字段名要连同报告模板一起改。",
            "键名改了又没改模板 → 该栏空白",
            must_contain=('"advantages"', '"disadvantages"', '"suggestions"', '"summary"')),
    )),
)

PROMPT_FIELDS: tuple[PromptField, ...] = tuple(f for _, _, fs in PROMPT_GROUPS for f in fs)
PROMPT_BY_KEY: dict[str, PromptField] = {f.key: f for f in PROMPT_FIELDS}


def normalize(text: str) -> str:
    """统一换行并去掉首尾空白 / 空行，保证 textarea 往返稳定。"""
    v = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    return v.strip().strip("\n").strip()


DEFAULTS: dict[str, str] = {f.key: normalize(f.default) for f in PROMPT_FIELDS}

# 只在这一份里缓存已落盘的覆盖，用 (mtime_ns, size) 判定是否需要重读，避免每次调用都读文件。
_cache: tuple[tuple[int, int], dict[str, str]] | None = None


def clear_cache() -> None:
    global _cache
    _cache = None


def _file_stamp() -> tuple[int, int] | None:
    try:
        st = PROMPTS_FILE.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def overrides() -> dict[str, str]:
    """当前落盘的覆盖块（未知键保留，便于手工编辑时不被悄悄丢掉）。"""
    global _cache
    stamp = _file_stamp()
    if stamp is None:
        clear_cache()
        return {}
    if _cache is not None and _cache[0] == stamp:
        return dict(_cache[1])
    data: Mapping[str, object] = {}
    try:
        parsed = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        parsed = {}
    if isinstance(parsed, dict):
        inner = parsed.get("prompts")
        data = inner if isinstance(inner, Mapping) else parsed
    out: dict[str, str] = {}
    for key, value in data.items():
        if key.startswith("_") or not isinstance(value, str):
            continue
        text = normalize(value)
        if text:
            out[str(key)] = text
    _cache = (stamp, dict(out))
    return out


def get(key: str) -> str:
    """这块现在实际用的文本：有覆盖用覆盖，否则内置默认。"""
    return overrides().get(key) or DEFAULTS.get(key, "")


def is_custom(key: str) -> bool:
    stored = overrides().get(key)
    return bool(stored) and stored != DEFAULTS.get(key)


def render(key: str, **values: object) -> str:
    """把 {name} 换成运行期取值：单次扫描，替换结果不再二次扫描。"""
    text = get(key)
    if not values:
        return text
    return _TOKEN_RE.sub(lambda m: _to_text(values.get(m.group(1), m.group(0))), text)


def _to_text(value: object) -> str:
    return value if isinstance(value, str) else ("") if value is None else str(value)


def placeholder_report(key: str, text: str | None = None) -> tuple[list[str], list[str]]:
    """返回 (缺失的占位符, 多出来的占位符)，供校验与页面提示共用。"""
    f = PROMPT_BY_KEY.get(key)
    if f is None:
        return [], []
    found = set(_TOKEN_RE.findall(get(key) if text is None else text))
    declared = set(f.tokens)
    return sorted(declared - found), sorted(found - declared)


def validate(raw: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    """把表单值整理成「要落盘的覆盖表」，返回 (覆盖表, 错误列表)。

    覆盖表是完整快照：留空的块按内置默认对待（不写入），所以一次保存就能同时
    完成「改 A、清空 B、原样保留 C」。
    """
    kept: dict[str, str] = {}
    errors: list[str] = []
    for f in PROMPT_FIELDS:
        value = normalize(str(raw.get(f.key, "") or ""))
        if not value:
            continue
        if len(value) > _MAX_CHARS:
            errors.append(f"{f.label}：不得超过 {_MAX_CHARS} 字，当前 {len(value)} 字")
            continue
        missing = [t for t in f.tokens if _brace(t) not in value]
        if missing:
            need = "、".join(_brace(t) for t in missing)
            errors.append(f"{f.label}：缺少系统占位符 {need}，这些值由程序注入，删掉模型就读不到对应材料")
            continue
        if f.strict_braces:
            extra = sorted(set(_TOKEN_RE.findall(value)) - set(f.tokens))
            if extra:
                bad = "、".join(_brace(t) for t in extra)
                ok = "、".join(_brace(t) for t in f.tokens)
                errors.append(f"{f.label}：{bad} 不是本块可用的占位符，本块只认 {ok}")
                continue
        lost = [lit for lit in f.must_contain if lit not in value]
        if lost:
            keys = "、".join(lost)
            errors.append(f"{f.label}：必须保留 {keys} 这些字段名，否则输出无法写进报告")
            continue
        if value != DEFAULTS[f.key]:
            kept[f.key] = value
    return kept, errors


def save(raw: Mapping[str, str]) -> tuple[list[str], list[str]]:
    """校验并落盘，返回 (发生变化的键, 错误列表)。有错则一个字节都不写。"""
    kept, errors = validate(raw)
    if errors:
        return [], errors
    stored = overrides()
    before = {k: v for k, v in stored.items() if k in PROMPT_BY_KEY}
    changed = sorted(k for k in set(before) | set(kept) if before.get(k) != kept.get(k))
    if stored != kept:
        _write(kept)
    return changed, []


def reset_all() -> list[str]:
    """全部恢复内置默认。"""
    before = sorted(k for k in overrides() if k in PROMPT_BY_KEY)
    if before:
        _write({})
    return before


def _write(prompts: Mapping[str, str]) -> None:
    payload = {
        "_meta": {
            "version": 1,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": "管理员在「系统设置 · 大模型提示词」保存的覆盖，只记录与内置默认不同的块；"
                    "删除本文件即全部回到内置默认。",
        },
        "prompts": {k: prompts[k] for k in PROMPT_BY_KEY if k in prompts},
    }
    PROMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROMPTS_FILE.parent / (PROMPTS_FILE.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(tmp, PROMPTS_FILE)
    clear_cache()


def overview(raw: Mapping[str, str] | None = None) -> list[dict]:
    """设置页视图：分组 + 每块当前文本、是否自定义、字数与占位符状态。

    raw 是「表单回显」：管理员提交后校验没过，页面要显示他刚输入的内容而不是磁盘旧值；
    留空的块按内置默认显示，因为留空的语义就是「不改、用默认」。
    """
    groups = []
    for title, desc, fs in PROMPT_GROUPS:
        rows = []
        for f in fs:
            if raw is None:
                text = get(f.key)
            else:
                text = normalize(str(raw.get(f.key, "") or "")) or DEFAULTS[f.key]
            missing, extra = placeholder_report(f.key, text)
            rows.append({
                "field": f,
                "key": f.key,
                "text": text,
                "custom": is_custom(f.key),
                "chars": len(text),
                "lines": text.count("\n") + 1,
                "missing": missing,
                "extra": extra,
                "tokens": list(f.tokens),
            })
        groups.append({"title": title, "desc": desc, "rows": rows})
    return groups


def custom_count() -> int:
    return sum(1 for f in PROMPT_FIELDS if is_custom(f.key))


def default_of(key: str) -> str:
    return DEFAULTS.get(key, "")
