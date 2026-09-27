"""版本指纹：读取打包时由 deploy/package.ps1 写入发布包根的 BUILD_INFO.json。

存在的意义：服务器上 /opt/speech-assist 不是 git 工作树、git 远端只做代码备份，
「线上到底跑的哪一版」此前只能靠包名里的日期戳（手填、可与代码脱节）。
/health 与管理页拿到这个指纹后，升级验收可以闭环到「运行版本 = 新包指纹」。

缺文件一律降级为 unknown（开发签出、旧包、手工 rsync 都可能没有），
读坏了也不抛错——指纹是辅助信息，绝不能因为它把 /health 整个打挂。
惰性加载一次并缓存：包内容运行期不变，没必要每次请求都读盘。
"""
from __future__ import annotations

import json

from .config import BASE_DIR

BUILD_FILE = BASE_DIR / "BUILD_INFO.json"

_DATA: dict | None = None
_LOADED = False

UNKNOWN: dict = {"git_sha": "unknown", "git_short": "unknown", "git_branch": "unknown",
                 "dirty": False, "built_at": "", "package": "unknown"}


def data() -> dict:
    """BUILD_INFO.json 的解析结果；缺失/损坏时返回带 note 说明的 unknown 结构。"""
    global _DATA, _LOADED
    if not _LOADED:
        _LOADED = True
        try:
            raw = json.loads(BUILD_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                _DATA = {**UNKNOWN, **raw}
            else:
                _DATA = {**UNKNOWN, "note": "BUILD_INFO.json 不是 JSON 对象"}
        except FileNotFoundError:
            _DATA = {**UNKNOWN, "note": "包内没有 BUILD_INFO.json（旧包或开发目录）"}
        except Exception:  # noqa: BLE001
            _DATA = {**UNKNOWN, "note": "BUILD_INFO.json 损坏，无法解析"}
    return _DATA or UNKNOWN


def version() -> str:
    """一行摘要，供 /health 的 version 字段与 update.sh 比对展示。

    形如 "26b04cb1f·main"（脏工作树再挂 +dirty），取不到指纹时是 "unknown"。
    保留完整 git_short 前缀在开头，方便 grep 断言「运行版本包含新包短 sha」。
    """
    d = data()
    sha = str(d.get("git_short") or "unknown")
    if sha == "unknown":
        return "unknown"
    line = sha
    branch = str(d.get("git_branch") or "")
    if branch and branch != "unknown":
        line += f"·{branch}"
    if d.get("dirty"):
        line += "+dirty"
    return line
