#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Markdown 文档渲染成单文件 HTML（左侧目录两栏版式）并打包。

版式约束（项目文档规范）：
  * 两竖栏：左侧可点击目录（滚动高亮 + 筛选），右侧正文
  * 标题 1 / 1.1 / 1.1.1 顶格；黑体标题 + 宋体正文；相邻两级标题差一个字号
  * 1.25 倍行距；段间距 1.25 倍行距；正文首行缩进 2 字符
  * 流程图用 mermaid，图旁有"取 mermaid 源码"按钮；无网络时降级为源码视图
  * 页面自带下载按钮；多份 HTML 另外产出 zip 打包件

环境相关值（公网 IP、发布包日期）只在**生成时注入 HTML**，不写回 Markdown 源：
仓库是 public，真实 IP 进了 .md 就会被索引。

用法：
    python tools/build_html.py --ip 203.0.113.10 --date 20260923   # IP 用你自己的
    python tools/build_html.py                 # 不注入，保留 <公网IP> 占位符
输出（默认 dist/html，dist/ 已在 .gitignore 里）：
    设计报告.html / 部署手册.html / 演讲评分系统-文档包-<日期>.zip
退出码：审计发现问题时返回 1，方便挂进发布流程。
"""

from __future__ import annotations

import argparse
import datetime as dt
import html as html_mod
import re
import sys
import zipfile
from pathlib import Path

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR_DEFAULT = ROOT / "dist" / "html"

DOCS = [
    {
        "src": Path("docs/设计报告.md"),
        "out": "设计报告.html",
        "title": "演讲视频评价系统 · 详细设计报告",
        "short": "设计报告",
        "desc": "架构 / 评分算法 / 网页化配置 / 部署运维",
    },
    {
        "src": Path("deploy/部署手册.md"),
        "out": "部署手册.html",
        "title": "演讲评分系统 · 云服务器部署手册",
        "short": "部署手册",
        "desc": "阿里云 ECS 2 vCPU / 2 GiB 单机上线与运维",
    },
    {
        "src": Path("docs/方案-版本指纹与压缩闸门.md"),
        "out": "方案-版本指纹与压缩闸门.html",
        "title": "演讲评分系统 · 版本指纹与压缩闸门方案",
        "short": "方案",
        "desc": "待决策：发布链路版本防呆 + 长视频压缩可行性闸门",
    },
]

# 生成时把 Markdown 占位符换成实际值；Markdown 源保持占位符
PLACEHOLDER_VARS = {"公网IP": "ip", "日期": "date", "新日期": "date", "服务器": "ip", "host": "ip"}

# 注入实际值后需要同步改写的说明句，否则会留下"把 IP 换成实际地址"这种自相矛盾的话
PROSE_FIXES = [
    (
        "本手册里的命令都可以直接照抄。抄之前只需要改两处：`{ip}` 和 `{date}`（发布包目录名）。",
        "本手册里的命令都可以直接照抄——这份 HTML 生成时已填入公网 IP `{ip}` 与发布包日期 `{date}`。"
        "Markdown 源（`deploy/部署手册.md`）里仍写作占位符；换服务器时用 "
        "`python tools/build_html.py --ip 新地址 --date 新日期` 重新生成本页即可。",
    ),
    (
        "把 `{ip}` 换成实际地址、`{date}` 换成当天日期（包名格式是 `speech-assist-YYYYMMDD`）就能一路贴到底。",
        "本文已填入实际公网 IP 与当天日期（包名格式 `speech-assist-YYYYMMDD`），可一路贴到底；"
        "换服务器或换包时改 `--ip` / `--date` 重新生成本页（Markdown 源里是占位符）。",
    ),
]

FENCE_RE = re.compile(
    r"^(?P<indent>[ ]{0,7})(?P<fence>`{3,}|~{3,})(?P<info>[^\n]*)\n"
    r"(?P<body>.*?)(?:\n(?P=indent)(?P=fence)[ \t]*)(?=\n|\Z)",
    re.M | re.S,
)
TOKEN_RE = re.compile(r"<p>@@B(\d+)@@</p>|@@B(\d+)@@")
STRIKE_RE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")

LANG_ALIAS = {
    "": "text",
    "text": "text",
    "txt": "text",
    "console": "bash",
    "sh": "bash",
    "shell": "bash",
    "ps1": "powershell",
    "pwsh": "powershell",
    "yml": "yaml",
}

PYG = HtmlFormatter(cssclass="codehilite", linenos=False, nowrap=False)


def slugify(value: str, separator: str = "-") -> str:
    """保留中文的锚点 id；签名需匹配 python-markdown toc 扩展（value, separator）。"""
    t = re.sub(r"<[^>]+>", "", value).strip().lower()
    t = re.sub(r"[^\w\u3001\u4e00-\u9fff\uff1a\uff08\uff09\uff0c\u2014\u00b7\s-]", "", t)
    t = re.sub(r"\s+", separator, t.strip())
    return t or "sec"


def attr(text: str) -> str:
    """toc 扩展给的 name 已经 escape_cdata 过，这里只补引号转义。"""
    return text.replace('"', "&quot;")


def plain_text(page: str) -> str:
    """还原成人眼看到的纯文本（去掉 style/script 与标签、反转义）。

    Pygments 会把一个占位符切成多个 span，直接对 HTML 找字符串会漏检，
    所以残留占位符的检查必须在这一层做；标签用空串替换以还原被切断的文本。
    """
    body = re.sub(r"<(script|style)\b.*?</\1>", " ", page, flags=re.S)
    return html_mod.unescape(re.sub(r"<[^>]+>", "", body))


def drop_in_doc_toc(source: str) -> tuple[str, int]:
    """正文里那份"## 目录"列表由 HTML 左栏承担，删掉避免重复。返回删掉的行数。"""
    lines = source.split("\n")
    out: list[str] = []
    i = 0
    dropped = 0
    while i < len(lines):
        if re.match(r"^##\s*目\s*录\s*$", lines[i]):
            j = i + 1
            while j < len(lines) and not re.match(r"^(#{1,3}\s|-{3,}\s*$)", lines[j]):
                j += 1
            while j < len(lines) and not re.match(r"^#{1,3}\s", lines[j]):
                j += 1
            dropped = j - i
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out), dropped


def inject_subs(text: str, subs: dict) -> tuple[str, list[str]]:
    notes: list[str] = []
    for key, var in PLACEHOLDER_VARS.items():
        value = subs.get(var)
        if value:
            n = text.count(f"<{key}>")
            text = text.replace(f"<{key}>", value)
            if n:
                notes.append(f"<{key}> → {value}（{n} 处）")
    leftover = text.count("<公网IP>")
    if leftover:
        notes.append(f"<公网IP> 未注入，保留 {leftover} 处占位符")
    ip = subs.get("ip") or "<公网IP>"
    date = subs.get("date") or "<日期>"
    for before, after in PROSE_FIXES:
        b = before.format(ip=ip, date=date)
        if b in text:
            text = text.replace(b, after.format(ip=ip, date=date))
            notes.append("说明句已同步改写（占位符 → 实际值）")
    return text, notes


def tokenize_fences(source: str) -> tuple[str, list[dict]]:
    """把 ``` 代码块（含 mermaid）换成占位符，避免 markdown 把它们当普通代码。"""
    blocks: list[dict] = []

    def repl(m: re.Match) -> str:
        info = m.group("info").strip()
        lang = info.split()[0].lower() if info else ""
        blocks.append({"lang": lang, "code": m.group("body")})
        return f"{m.group('indent')}@@B{len(blocks) - 1}@@"

    return FENCE_RE.sub(repl, source), blocks


def render_code_block(block: dict, number: int) -> str:
    lang = LANG_ALIAS.get(block["lang"], block["lang"])
    try:
        lexer = get_lexer_by_name(lang) if lang and lang != "text" else TextLexer()
    except Exception:
        lexer = TextLexer()
    code = block["code"].rstrip("\n")
    body = highlight(code, lexer, PYG).strip()
    return (
        '<div class="codeline">'
        '<div class="bar"><span class="chip">{lang}</span>'
        '<button type="button" class="btn cp-btn" data-target="code-{n}">复制代码</button></div>'
        '<div class="codeslot" id="code-{n}">{body}</div></div>'
    ).format(lang=html_mod.escape(block["lang"] or "text"), n=number, body=body)


def render_mermaid(source: str, number: int) -> str:
    esc = html_mod.escape(source.rstrip("\n"))
    bar = (
        '<div class="bar"><span class="chip">mermaid 流程图</span>'
        '<span class="state" data-state="wait">等待渲染…</span>'
        '<button type="button" class="btn mmd-btn" data-target="mmd-{n}">取 mermaid 源码</button>'
        '<button type="button" class="btn cp-btn" data-target="mmd-{n}">复制源码</button></div>'
    ).format(n=number)
    stage = '<div class="stage" id="stage-{n}"><pre class="pending">{s}</pre></div>'.format(n=number, s=esc)
    return '<figure class="mmd">' + bar + stage + '<pre class="mmdsrc" id="mmd-{n}" hidden>{s}</pre></figure>'.format(
        n=number, s=esc
    )


def render_toc(tokens: list[dict]) -> tuple[str, int]:
    out: list[str] = ['<ul class="toclist">']
    count = 0

    def walk(items: list[dict]) -> None:
        nonlocal count
        for it in items:
            level = it["level"]
            if level <= 1:
                walk(it["children"])
                continue
            if level <= 4:
                name = it["name"]
                count += 1
                out.append(
                    '<li class="l{lv}"><a href="#{sid}" title="{t}">{t}</a></li>'.format(
                        lv=level, sid=it["id"], t=attr(name)
                    )
                )
            walk(it["children"])

    walk(tokens)
    out.append("</ul>")
    return "\n".join(out), count


DIAGRAM_TYPES = ("flowchart", "graph", "sequenceDiagram", "classDiagram", "stateDiagram", "gantt", "pie")


def lint_mermaid(source: str, number: int) -> list[str]:
    """mermaid 语法体检：这三张图都是手写 ASCII 图转来的，容易留下渲染期才炸的坑。"""
    problems: list[str] = []
    lines = [ln for ln in source.rstrip("\n").split("\n") if ln.strip()]
    if not lines or not lines[0].strip().startswith(DIAGRAM_TYPES):
        problems.append(f"mermaid#{number}：首行不是已知图类型（{lines[0][:30] if lines else '空'}）")
    for ln in lines:
        tag = f"mermaid#{number}：{ln.strip()[:40]}"
        if "**" in ln or "{" in ln or "}" in ln:
            problems.append(f"{tag} → 含 ** 或花括号，mermaid 会解析失败")
        if ln.count('"') % 2:
            problems.append(f"{tag} → 双引号不成对")
        if ln.count("[") != ln.count("]"):
            problems.append(f"{tag} → 方括号不成对")
        if re.search(r"\[(?!\")[^\]]*[（(]", ln):
            problems.append(f"{tag} → 圆括号必须写在引号内的标签里")
    return problems


def build_doc(doc: dict, subs: dict, log: list[str]) -> tuple[str, str, dict]:
    src_file = ROOT / doc["src"]
    text = src_file.read_text(encoding="utf-8")

    text, dropped = drop_in_doc_toc(text)
    log.append(f"正文内嵌目录：{'移除 %d 行，改由左栏承担' % dropped if dropped else '源文件本就没有'}")

    text, notes = inject_subs(text, subs)
    log.extend(notes)

    text, blocks = tokenize_fences(text)
    text = STRIKE_RE.sub(r"<del>\1</del>", text)

    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
        extension_configs={"toc": {"slugify": slugify}},
    )
    content = md.convert(text)

    counters = {"code": 0, "mmd": 0}
    mmd_problems: list[str] = []

    def expand(m: re.Match) -> str:
        block = blocks[int(m.group(1) or m.group(2))]
        if block["lang"] == "mermaid":
            counters["mmd"] += 1
            mmd_problems.extend(lint_mermaid(block["code"], counters["mmd"]))
            return render_mermaid(block["code"], counters["mmd"])
        counters["code"] += 1
        return render_code_block(block, counters["code"])

    content = TOKEN_RE.sub(expand, content)
    stray = re.findall(r"@@B\d*", content)
    toc_html, toc_items = render_toc(md.toc_tokens)

    ids = set(re.findall(r"<h[1-6] id=\"([^\"]+)\"", content))
    stats = {
        "headings": len(ids),
        "toc_items": toc_items,
        "code": counters["code"],
        "mermaid": counters["mmd"],
        "stray_tokens": len(stray),
        "mmd_problems": mmd_problems,
        "ids": ids,
        "fences": len(blocks),
    }
    return content, toc_html, stats


def page_shell(doc: dict, body: str, toc: str, subs: dict, built: str) -> str:
    nav = []
    for d in DOCS:
        cls = "btn on" if d["out"] == doc["out"] else "btn ghost"
        nav.append(f'<a class="{cls}" href="{d["out"]}">{d["short"]}</a>')
    zip_name = ZIP_NAME(subs)
    meta_bits = [
        "源文件 <code>{}</code>".format(str(doc["src"]).replace("\\", "/")),
        "生成于 {}".format(built),
    ]
    if subs.get("ip"):
        meta_bits.append("公网 IP 已注入本页（Markdown 源保留占位符）")
    meta = " · ".join(meta_bits)

    return (
        TEMPLATE.replace("@@CSS@@", CSS)
        .replace("@@PYG@@", PYG.get_style_defs(".codehilite"))
        .replace("@@JS@@", JS.replace("@@PAGE_NAME@@", js_string(doc["out"])))
        .replace("@@TITLE@@", html_mod.escape(doc["title"]))
        .replace("@@HEAD@@", html_mod.escape(doc["title"]))
        .replace("@@DESC@@", html_mod.escape(doc["desc"]))
        .replace("@@META@@", meta)
        .replace("@@NAV@@", " ".join(nav))
        .replace("@@ZIP@@", html_mod.escape(zip_name))
        .replace("@@DOC_COUNT@@", str(len(DOCS)))
        .replace("@@TOC@@", toc)
        .replace("@@BODY@@", body)
    )


def ZIP_NAME(subs: dict) -> str:
    stamp = subs.get("date") or dt.date.today().strftime("%Y%m%d")
    return f"演讲评分系统-文档包-{stamp}.zip"


def js_string(s: str) -> str:
    out = '"'
    for ch in s:
        if ch in '"\\':
            out += "\\" + ch
        elif ch == "\n":
            out += "\\n"
        elif ord(ch) < 0x20:
            out += "\\u%04x" % ord(ch)
        else:
            out += ch
    return out + '"'


CSS = """
:root{
  --lh:1.25;                 /* 全文 1.25 倍行距 */
  --gap:1.5625em;            /* 段间距 = 1.25 倍行距 */
  --font-head:"SimHei","Heiti SC","Microsoft YaHei","微软雅黑",sans-serif;
  --font-body:"SimSun","Songti SC","宋体","Microsoft YaHei",serif;
  --font-mono:"Consolas","JetBrains Mono","Courier New",monospace;
  --accent:#0f766e; --accent-soft:#e8f4f2; --accent-line:#c3ded9;
  --ink:#1f2933; --ink-soft:#5b6773; --line:#e2e8ee; --bg:#f7faf9; --panel:#fff;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font-body);
  font-size:12pt;line-height:var(--lh);-webkit-text-size-adjust:100%}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}

header.top{position:sticky;top:0;z-index:30;background:rgba(255,255,255,.95);
  border-bottom:1px solid var(--line);backdrop-filter:blur(4px)}
.topin{max-width:1360px;margin:0 auto;padding:9px 24px;display:flex;flex-wrap:wrap;gap:10px;align-items:baseline}
.topin .h{font-family:var(--font-head);font-weight:700;font-size:13pt;letter-spacing:.4px}
.topin .sub{color:var(--ink-soft);font-size:10pt}
.topin .tools{margin-left:auto;display:flex;gap:8px;flex-wrap:wrap}
.btn{font-family:var(--font-body);font-size:10pt;line-height:1.5;padding:3px 9px;
  border:1px solid var(--accent-line);background:#fff;color:var(--accent);border-radius:4px;cursor:pointer}
.btn:hover{background:var(--accent-soft);text-decoration:none}
.btn.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.btn.ghost{border-color:var(--line);color:var(--ink-soft)}

.layout{max-width:1360px;margin:0 auto;padding:0 24px 60px;display:grid;
  grid-template-columns:302px minmax(0,1fr);gap:30px;align-items:start}
nav.side{position:sticky;top:50px;max-height:calc(100vh - 50px);overflow:auto;padding:18px 6px 40px}
nav.side .navtitle{font-family:var(--font-head);font-weight:700;font-size:11.5pt;margin:0 0 8px}
nav.side .navswitch{display:flex;gap:6px;margin:0 0 12px;flex-wrap:wrap}
nav.side .search{margin:0 0 10px}
nav.side input{width:100%;font-family:var(--font-body);font-size:10pt;padding:5px 8px;
  border:1px solid var(--line);border-radius:4px;background:#fff}
ul.toclist{list-style:none;margin:0;padding:0;border-left:2px solid var(--line)}
ul.toclist li{margin:0}
ul.toclist li a{display:block;padding:4px 8px;margin-left:-2px;border-left:2px solid transparent;
  color:var(--ink-soft);font-size:10pt;line-height:1.3}
ul.toclist li.l2>a{font-family:var(--font-head);font-size:10.5pt;color:var(--ink)}
ul.toclist li.l3>a{padding-left:20px}
ul.toclist li.l4>a{padding-left:36px;font-size:9.5pt}
ul.toclist li a:hover{background:var(--accent-soft);color:var(--accent);text-decoration:none}
ul.toclist li a.active{background:var(--accent-soft);color:var(--accent);border-left-color:var(--accent);font-weight:700}
ul.toclist li.hide{display:none}

main{min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:30px 40px 54px}
p.kicker{color:var(--ink-soft);font-size:9.5pt;margin:0 0 22px;text-indent:0;
  padding-bottom:10px;border-bottom:1px dashed var(--line)}

h1,h2,h3,h4,h5,h6{font-family:var(--font-head);font-weight:700;line-height:var(--lh);
  margin:0 0 .55em;text-indent:0;padding:0;color:var(--ink);scroll-margin-top:64px;text-align:left}
h1{font-size:26pt;margin-bottom:.4em;padding-bottom:.25em;border-bottom:3px solid var(--accent)}
h2{font-size:18pt;margin-top:1.9em;padding-bottom:.15em;border-bottom:1px solid var(--accent-line)}
h3{font-size:16pt;margin-top:1.5em}
h4{font-size:15pt;margin-top:1.3em}
h5{font-size:14pt;margin-top:1.2em}
h6{font-size:12pt;margin-top:1.2em}

article p{margin:0 0 var(--gap);text-indent:2em}
article li p,article td p,article th p,article blockquote p,p.kicker{text-indent:0}
article ul,article ol{margin:0 0 var(--gap);padding-left:2.4em}
article li{margin:0 0 .28em}
article blockquote{margin:0 0 var(--gap);padding:.7em 1em;background:var(--accent-soft);
  border-left:4px solid var(--accent);border-radius:0 6px 6px 0}
article hr{border:0;border-top:1px solid var(--line);margin:1.8em 0}
article strong{font-family:var(--font-head)}
del{color:#9aa5b1}
code{font-family:var(--font-mono);font-size:.92em;background:#f1f4f5;border:1px solid #e3e9eb;
  border-radius:3px;padding:.02em .28em;color:#0b5f57}
table{border-collapse:collapse;width:100%;margin:0 0 var(--gap);font-size:10.5pt;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top;line-height:1.3}
th{background:#f2f6f5;font-family:var(--font-head);white-space:nowrap}
tbody tr:nth-child(even){background:#fbfdfd}

.codeline,figure.mmd{margin:0 0 var(--gap);border:1px solid var(--line);border-radius:6px;background:#fff;overflow:hidden}
.bar{display:flex;align-items:center;gap:8px;padding:4px 10px;background:#f4f7f7;border-bottom:1px solid var(--line);flex-wrap:wrap}
.chip{font-family:var(--font-mono);font-size:8.5pt;color:var(--ink-soft);letter-spacing:.6px;text-transform:uppercase}
.state{font-size:9pt;color:var(--ink-soft)}
.state[data-state="ok"]{color:#0f766e}
.state[data-state="off"]{color:#9a5b00}
.bar .btn{margin-left:0}
.bar .cp-btn{margin-left:auto}
.codeslot{overflow:hidden}
.codehilite{background:#fcfdfd}
.codehilite pre,.pending,.mmdsrc{margin:0;padding:12px 14px;font-family:var(--font-mono);
  font-size:10pt;line-height:1.5;overflow-x:auto;white-space:pre;background:#fcfdfd}
.pending{font-family:var(--font-mono);font-size:10pt;color:var(--ink-soft);text-align:left}
figure.mmd .stage{padding:16px;overflow-x:auto;text-align:center;min-height:40px}
figure.mmd .stage svg{max-width:100%;height:auto}
figure.mmd .stage .pending{background:#fff}
figure.mmd.off .stage{display:none}
figure.mmd .mmdsrc{display:none;border-top:1px solid var(--line);background:#fbfcfc;text-align:left}
figure.mmd.off .mmdsrc,figure.mmd.showsrc .mmdsrc{display:block}

.pagefoot{max-width:1360px;margin:0 auto;padding:0 24px;color:var(--ink-soft);font-size:9.5pt}
.toast{position:fixed;right:18px;bottom:18px;z-index:60;background:var(--ink);color:#fff;font-size:10pt;
  padding:8px 14px;border-radius:6px;opacity:0;transition:opacity .18s;pointer-events:none}
.toast.show{opacity:.94}

@media (max-width:1080px){
  .layout{grid-template-columns:minmax(0,1fr);gap:0}
  nav.side{position:static;max-height:none;padding-bottom:14px;border-bottom:1px solid var(--line)}
  main{border-radius:0;border-left:0;border-right:0}
  .topin .tools{margin-left:0;width:100%}
}
@media print{
  header.top,nav.side,.pagefoot,.bar,.toast{display:none!important}
  body{background:#fff;font-size:10.5pt}
  .layout{display:block;max-width:none;padding:0}
  main{border:0;padding:0;border-radius:0}
  h2{break-after:avoid}
  figure.mmd,.codeline,table{break-inside:avoid}
}
"""

JS = r"""
(function () {
  var PAGE_NAME = @@PAGE_NAME@@;
  var seq = 0;

  function toast(msg) {
    var t = document.getElementById('toast');
    if (!t) { t = document.createElement('div'); t.id = 'toast'; t.className = 'toast'; document.body.appendChild(t); }
    t.textContent = msg;
    t.classList.add('show');
    clearTimeout(t._h);
    t._h = setTimeout(function () { t.classList.remove('show'); }, 2200);
  }

  function copyText(text, label) {
    function legacy() {
      var ta = document.createElement('textarea');
      ta.value = text; ta.readOnly = true;
      ta.style.position = 'fixed'; ta.style.left = '-9999px';
      document.body.appendChild(ta); ta.select();
      var ok = false;
      try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
      document.body.removeChild(ta);
      toast(ok ? label + '已复制' : '复制失败，请手动选中');
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { toast(label + '已复制'); }, legacy);
    } else { legacy(); }
  }

  function textOf(id) {
    var el = document.getElementById(id);
    if (!el) return '';
    var pre = el.tagName === 'PRE' ? el : (el.querySelector('pre') || el);
    return pre.innerText.replace(/\s+$/, '');
  }

  document.querySelectorAll('.cp-btn').forEach(function (b) {
    b.addEventListener('click', function () {
      copyText(textOf(b.dataset.target), b.textContent.indexOf('源码') >= 0 ? 'mermaid 源码' : '代码');
    });
  });

  document.querySelectorAll('.mmd-btn').forEach(function (b) {
    b.addEventListener('click', function () {
      var src = document.getElementById(b.dataset.target);
      if (!src) return;
      var fig = src.closest('figure.mmd');
      var open = !fig.classList.contains('showsrc');
      fig.classList.toggle('showsrc', open);
      b.classList.toggle('on', open);
      b.textContent = open ? '收起 mermaid 源码' : '取 mermaid 源码';
    });
  });

  var CDNS = [
    'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js',
    'https://unpkg.com/mermaid@11/dist/mermaid.min.js'
  ];
  function loadScript(src) {
    return new Promise(function (res, rej) {
      var s = document.createElement('script');
      s.src = src; s.onload = res; s.onerror = rej;
      document.head.appendChild(s);
    });
  }
  function setState(fig, text, kind) {
    var st = fig.querySelector('.state');
    if (st) { st.textContent = text; st.dataset.state = kind; }
  }
  (async function renderDiagrams() {
    var figs = Array.prototype.slice.call(document.querySelectorAll('figure.mmd'));
    if (!figs.length) return;
    var loaded = false;
    for (var i = 0; i < CDNS.length && !loaded; i++) {
      try { await loadScript(CDNS[i]); loaded = !!window.mermaid; } catch (e) { loaded = false; }
    }
    if (!loaded) {
      figs.forEach(function (f) { f.classList.add('off'); setState(f, '离线未渲染 · 下面直接显示 mermaid 源码，可点「复制源码」', 'off'); });
      return;
    }
    var m = window.mermaid;
    m.initialize({
      startOnLoad: false, securityLevel: 'loose', theme: 'neutral',
      fontFamily: '"Microsoft YaHei","SimHei",sans-serif',
      flowchart: { htmlLabels: true, useMaxWidth: true, curve: 'linear' }
    });
    for (var k = 0; k < figs.length; k++) {
      var fig = figs[k];
      var stage = fig.querySelector('.stage');
      var src = fig.querySelector('.mmdsrc');
      stage.innerHTML = '';
      try {
        var out = await m.render('mmdSVG' + (seq++), src.textContent.trim());
        stage.innerHTML = (typeof out === 'string') ? out : out.svg;
        setState(fig, '已渲染 · 源码可取', 'ok');
      } catch (err) {
        stage.innerHTML = '<pre class="pending">' + src.textContent.replace(/[&<>]/g, function (c) {
          return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c];
        }) + '</pre>';
        setState(fig, '渲染失败：' + ((err && err.message) || err), 'off');
      }
    }
  })();

  var links = Array.prototype.slice.call(document.querySelectorAll('#tocnav a'));
  var pairs = links.map(function (a) {
    return { a: a, h: document.getElementById(a.getAttribute('href').slice(1)) };
  }).filter(function (p) { return !!p.h; });

  var q = document.getElementById('q');
  if (q) {
    q.addEventListener('input', function () {
      var v = q.value.trim().toLowerCase();
      links.forEach(function (a) {
        a.parentNode.classList.toggle('hide', v !== '' && a.textContent.toLowerCase().indexOf(v) < 0);
      });
    });
  }

  var tops = [];
  function measure() { tops = pairs.map(function (p) { return p.h.getBoundingClientRect().top + window.pageYOffset - 80; }); }
  var cur = -1;
  function spy() {
    if (!tops.length) return;
    var y = window.pageYOffset, idx = -1;
    for (var i = 0; i < tops.length; i++) { if (tops[i] <= y) { idx = i; } else { break; } }
    if (idx === cur) return;
    cur = idx;
    links.forEach(function (a) { a.classList.remove('active'); });
    if (idx >= 0) {
      var a = pairs[idx].a;
      a.classList.add('active');
      var r = a.getBoundingClientRect();
      if (r.top < 56 || r.bottom > window.innerHeight - 12) a.scrollIntoView({ block: 'nearest' });
    }
  }
  var queued = false;
  window.addEventListener('scroll', function () {
    if (queued) return;
    queued = true;
    requestAnimationFrame(function () { queued = false; spy(); });
  }, { passive: true });
  window.addEventListener('resize', function () { measure(); spy(); });
  measure(); spy();
  window.addEventListener('load', function () { measure(); spy(); });

  var dl = document.getElementById('btn-download');
  if (dl) {
    dl.addEventListener('click', function () {
      var blob = new Blob(['<!DOCTYPE html>\n' + document.documentElement.outerHTML], { type: 'text/html;charset=utf-8' });
      var a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = PAGE_NAME;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      setTimeout(function () { URL.revokeObjectURL(a.href); }, 3000);
      toast('已下载 ' + PAGE_NAME);
    });
  }
  var pr = document.getElementById('btn-print');
  if (pr) pr.addEventListener('click', function () { window.print(); });
  var ex = document.getElementById('btn-expand');
  if (ex) {
    ex.addEventListener('click', function () {
      var on = document.body.classList.toggle('showallsrc');
      ex.classList.toggle('on', on);
      ex.textContent = on ? '收起全部源码' : '展开全部源码';
      document.querySelectorAll('figure.mmd').forEach(function (f) { f.classList.toggle('showsrc', on); });
    });
  }
})();
"""

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>@@TITLE@@</title>
<style>@@CSS@@
@@PYG@@
body.showallsrc figure.mmd .mmdsrc{display:block}
</style>
</head>
<body>
<header class="top"><div class="topin">
  <span class="h">@@HEAD@@</span><span class="sub">@@DESC@@</span>
  <span class="tools">
    <button type="button" class="btn" id="btn-download">下载本页 HTML</button>
    <button type="button" class="btn ghost" id="btn-expand">展开全部源码</button>
    <a class="btn ghost" href="@@ZIP@@" download="@@ZIP@@">打包下载（@@DOC_COUNT@@ 份 HTML）</a>
    <button type="button" class="btn ghost" id="btn-print">打印 / 存 PDF</button>
  </span>
</div></header>

<div class="layout">
  <nav class="side" aria-label="目录">
    <div class="navswitch">@@NAV@@</div>
    <p class="navtitle">目录</p>
    <div class="search"><input id="q" type="search" placeholder="筛选目录…" aria-label="筛选目录"></div>
    <div id="tocnav">@@TOC@@</div>
  </nav>
  <main><article>
    <p class="kicker">@@META@@</p>
@@BODY@@
  </article></main>
</div>

<p class="pagefoot">本页由 <code>tools/build_html.py</code> 从 Markdown 源生成。改内容请改 <code>.md</code> 后重新生成，不要直接编辑 HTML。</p>
<script>@@JS@@</script>
</body>
</html>
"""


def audit(pages: list[tuple[dict, str, dict]], subs: dict) -> list[str]:
    bad: list[str] = []

    def heading_nums(page: str) -> set[str]:
        out = set()
        for m in re.finditer(r"<h[1-6] id=\"[^\"]*\">\s*([0-9]+(?:\.[0-9]+)*)", page):
            out.add(m.group(1).rstrip("."))
        return out

    nums_by_doc = {doc["short"]: heading_nums(page) for doc, page, _ in pages}
    for doc, page, stats in pages:
        name = doc["out"]
        if stats["stray_tokens"]:
            bad.append(f"{name}: {stats['stray_tokens']} 个代码块占位符未展开")
        if stats["mermaid"] < 1:
            bad.append(f"{name}: 没有渲染出任何 mermaid 流程图")
        bad.extend(f"{name}: {p}" for p in stats["mmd_problems"])
        if stats["headings"] < 10:
            bad.append(f"{name}: 标题数异常（{stats['headings']}）")
        for href in re.findall(r'href="#([^"]+)"', page):
            if href not in stats["ids"]:
                bad.append(f"{name}: 目录项 #{href} 无对应标题")
        txt = plain_text(page)
        for ph, key in PLACEHOLDER_VARS.items():
            token = f"<{ph}>"
            n = txt.count(token)
            if not n:
                continue
            if subs.get(key):
                bad.append(f"{name}: 传了 --{key} 但页面里仍残留 {n} 处 {token}")
            else:
                bad.append(f"{name}: 保留 {n} 处 {token} 占位符（未传 --{key}）")
        nums = nums_by_doc[doc["short"]]
        for m in re.finditer(r"§([0-9]+(?:\.[0-9]+)*)", page):
            ref = m.group(1).rstrip(".")
            if ref in nums:
                continue
            # 允许显式标注了对方文档名的跨文档引用（如"部署手册 §7.2"）
            bol = page.rfind("\n", 0, m.start()) + 1
            pre = page[bol : m.start()]
            other = None
            if "设计报告" in pre:
                other = "设计报告"
            elif "部署手册" in pre or "手册" in pre:
                other = "部署手册"
            if other and other != doc["short"] and ref in nums_by_doc.get(other, set()):
                continue
            bad.append(f"{name}: §{ref} 指向不存在的章节号")
        for opener, closer in (("<div", "</div>"), ("<figure", "</figure>"), ("<pre", "</pre>"), ("<table", "</table>")):
            o = len(re.findall(opener + r"[ >]", page))
            c = page.count(closer)
            if o != c:
                bad.append(f"{name}: {opener} 开/闭不匹配（{o} vs {c}）")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Markdown → 两栏 HTML + zip 打包件")
    ap.add_argument("--ip", default=None, help="注入 HTML 的公网 IP；不传则保留 <公网IP> 占位符")
    ap.add_argument("--date", default=None, help="注入 HTML 的发布包日期，如 20260923")
    ap.add_argument("--out", default=str(OUT_DIR_DEFAULT), help="输出目录，默认 dist/html")
    ap.add_argument("--no-zip", action="store_true", help="不产出 zip 打包件")
    args = ap.parse_args(argv)

    subs = {"ip": args.ip, "date": args.date}
    built = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages: list[tuple[dict, str, dict]] = []
    for doc in DOCS:
        log: list[str] = []
        body, toc, stats = build_doc(doc, subs, log)
        page = page_shell(doc, body, toc, subs, built)
        target = out_dir / doc["out"]
        target.write_text(page, encoding="utf-8")
        pages.append((doc, page, stats))
        print(f"--- {doc['src']} → {doc['out']}")
        print(
            "    标题 {:>3} · 目录项 {:>3} · 代码块 {:>3} · mermaid {:>1} · {:>7.1f} KB".format(
                stats["headings"], stats["toc_items"], stats["code"], stats["mermaid"],
                target.stat().st_size / 1024,
            )
        )
        for line in log:
            print("    ·", line)

    problems = audit(pages, subs)
    for p in problems:
        print("[FAIL]", p)

    if not args.no_zip:
        zip_name = ZIP_NAME(subs)
        zp = out_dir / zip_name
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for doc, _, _ in pages:
                z.write(out_dir / doc["out"], doc["out"])
        print(f"--- {zip_name} · {zp.stat().st_size / 1024:.1f} KB")

    print("[out]", out_dir)
    print("[audit]", "全部通过" if not problems else f"{len(problems)} 项待修")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
