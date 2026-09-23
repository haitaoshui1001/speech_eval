"""端到端 HTTP 冒烟：管理员建号 → 登录 → 上传 → 分析 → 报告 → 对比 → 批量导入。

默认 LLM_MODE=mock，不消耗 API 额度；只验证路由、模板与上下文契约。
运行： python tests/smoke_web.py        （真实模式： LLM_MODE=real python tests/smoke_web.py）
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import unquote

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "_websmoke"
shutil.rmtree(DATA, ignore_errors=True)
os.environ["DATA_DIR"] = str(DATA)
os.environ["ENV_FILE"] = str(DATA / "_websmoke.env")   # 设置页测试只写临时文件，不碰真实 .env
os.environ["PROMPTS_FILE"] = str(DATA / "_websmoke.prompts.json")   # 提示词覆盖同理
os.environ.setdefault("LLM_MODE", "mock")
sys.path.insert(0, str(BASE))

from fastapi.testclient import TestClient  # noqa: E402

from openpyxl import Workbook  # noqa: E402

from app import accounts  # noqa: E402
from app import config as cfg  # noqa: E402
from app import db, pipeline  # noqa: E402
from app import prompts as prm  # noqa: E402
from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.qwen import Completion, QwenClient, estimate_tokens  # noqa: E402
from app.rubric import rubric  # noqa: E402
from app.security import make_session_token  # noqa: E402

SAMPLE = BASE / "tests" / "sample_speech.mp4"
RESULTS: list[tuple[bool, str]] = []


def check(ok: bool, label: str, extra: str = "") -> None:
    RESULTS.append((bool(ok), label))
    print(("  PASS  " if ok else "  FAIL  ") + label + (f"  — {extra}" if extra else ""))


def page(client: TestClient, url: str, *needles: str, absent: tuple[str, ...] = (),
         label: str = "", **kw) -> str:
    r = client.get(url, **kw)
    body = r.text if r.status_code == 200 else ""
    missing = [n for n in needles if n not in body]
    leaked = [n for n in absent if n in body]
    why = "" if r.status_code == 200 else f"HTTP {r.status_code}"
    if missing:
        why = (why + " " if why else "") + f"缺关键字 {missing}"
    if leaked:
        why = (why + " " if why else "") + f"出现了不该出现的 {leaked}"
    check(r.status_code == 200 and not missing and not leaked, label or f"GET {url}", why)
    return body


def wait_done(client: TestClient, vid: int, timeout: int = 300) -> str:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        s = client.get(f"/videos/{vid}/status").json()
        last = f"{s['status']}@{s['progress']}%/{s['stage']}"
        if s["status"] in ("done", "failed"):
            return last
        time.sleep(1.0)
    return f"timeout {last}"


def upload(client: TestClient, title: str, auto: bool = True, path: Path = SAMPLE,
           topic: str = "Will AI replace English teachers?") -> int:
    with path.open("rb") as fh:
        r = client.post("/videos",
                        data={"title": title, "topic": topic,
                              "requirements": "2-3 minutes", "auto": "1" if auto else ""},
                        files={"video": (f"{title}.mp4", fh, "video/mp4")},
                        follow_redirects=False)
    assert r.status_code == 303, r.status_code
    loc = r.headers["location"]
    if loc.startswith("/videos/"):
        return int(loc.rstrip("/").split("/")[-1])
    match = [v for v in db.list_all_videos(limit=5) if v["title"] == title]
    assert match, f"上传未成功：{loc}"
    return int(match[0]["id"])


def env_form(overrides: dict | None = None, uncheck: tuple[str, ...] = ()) -> dict:
    """按页面提交规则拼一份完整表单：只回填 .env 里的值，勾上的开关才出现在表单里。"""
    form: dict[str, str] = {}
    for group in cfg.settings_overview():
        for row in group["rows"]:
            f = row["field"]
            if f.kind == "secret":
                continue
            if f.kind == "bool":
                if row["value"] in ("1", "true", "on", "yes"):
                    form[f.key] = "1"
            else:
                form[f.key] = row["raw"]
    for key in uncheck:
        form.pop(key, None)
    form.update(overrides or {})
    return form


def prompt_form(overrides: dict | None = None) -> dict:
    """按页面提交规则拼一份提示词表单：整份 19 块一起提交，未改的块回填当前生效文本。"""
    form = {f.key: prm.get(f.key) for f in prm.PROMPT_FIELDS}
    form.update(overrides or {})
    return form


def xlsx(rows: list[list]) -> bytes:
    """拼一份最小 xlsx 交给导入接口；数字单元格用于验证 `.0` 尾巴会被去掉。"""
    wb = Workbook()
    ws = wb.active
    ws.title = accounts.SHEET
    for row in rows:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


DIRTY = [
    list(accounts.HEADERS),
    ["frank", "pw-frank-123", "法兰克"],
    ["gina", "", "缺口令"],
    ["周", "pw-zhou-123", "非法用户名"],
    ["helen", "123", "口令过短"],
    ["alice", "pw-alice-123", "重名（库里已有）"],
    ["ivan", "pw-ivan-123", "伊文"],
    ["ivan", "pw-ivan-456", "重名（文件内）"],
    [20250101, 20250101, "学号当用户名"],
]


def main() -> int:
    if not SAMPLE.exists():
        print(f"缺少 {SAMPLE}")
        return 1
    print(f"模式: {'真实千问' if settings.real_mode else '演示(mock)'}   数据目录: {DATA}")

    with TestClient(app) as client:
        # ------------------------------------------------------------ 公开页
        page(client, "/", "演讲辅助评估系统", "登录进入系统", "管理员入口", "账号由管理员统一开通",
             absent=("评分要点", "关于分数", "怎么用", "千问", "Qwen", "qwen", "whisper"),
             label="落地页干净：只留主视觉与入口，且不出现模型名")
        page(client, "/guide", "评分标准", "怎么用", "关于分数", "用量统计", "谁给哪个维度打分",
             absent=("千问", "Qwen", "qwen", "faster-whisper", "DashScope"),
             label="说明页承接标准、流程与用量说明，且不出现模型名")
        page(client, "/login", "用户名", "不开放自助注册", "由管理员在后台开通", label="登录页说明账号来源")
        # 自助注册已下线：老书签与外链一律送回登录页，并说清去哪拿账号
        rg = client.get("/register", follow_redirects=False)
        check(rg.status_code == 303 and rg.headers["location"].startswith("/login?notice="),
              "GET /register 不再给注册表单", f"{rg.status_code} {rg.headers.get('location')}")
        rp = client.post("/register", data={"username": "eve", "password": "pw-eve-123",
                                           "confirm": "pw-eve-123"}, follow_redirects=False)
        check(rp.status_code == 303 and db.get_user_by_name("eve") is None,
              "POST /register 不建号同样送回登录页", f"{rp.status_code} {rp.headers.get('location')}")
        check("账号由管理员统一开通" in client.get("/register").text, "跟随后能看到账号从哪来")
        check(client.get("/dashboard", follow_redirects=False).status_code == 303, "未登录访问 /dashboard 跳登录")
        check(client.get("/admin", follow_redirects=False).status_code == 303, "未登录访问 /admin 跳登录")
        check(client.get(f"/videos/1", follow_redirects=False).status_code == 303, "未登录访问报告跳登录")
        health = client.get("/health").json()
        check(health.get("ok") is True, "/health 正常")
        check(health.get("asr_engine") in ("qwen", "whisper") and health.get("asr_engine_label"),
              "/health 说明转录走本地还是云端", f"{health.get('asr_engine')} → {health.get('asr_engine_label')}")
        check(isinstance(health.get("tokens"), dict) and "calls" in health["tokens"],
              "/health 暴露全站 token 聚合")
        check(isinstance(health.get("face_metrics"), bool),
              "/health 说明客观人脸测量层可用与否", str(health.get("face_metrics")))
        conc = health.get("concurrency")
        check(isinstance(conc, dict) and {"analyzers", "pool_size", "running", "llm_requests",
                                        "llm_in_flight", "web_threads"} <= set(conc),
              "/health 暴露并发配额", str(conc))
        check(conc["analyzers"] == settings.max_analyzers
              and conc["llm_requests"] == settings.max_llm_requests
              and conc["pool_size"] <= conc["analyzers"],
              "并发配额与设置一致且分析池不超额", f"{conc['analyzers']} 路 / {conc['llm_requests']} 请求")
        check(conc["web_threads"] >= settings.web_threads, "网页线程额度按 WEB_THREADS 放宽",
              f"{conc['web_threads']} 个（需 ≥ {settings.web_threads}）")
        check(conc["running"] == 0 and conc["llm_in_flight"] == 0, "空闲时在途计数归零", str(conc))
        check(client.post("/admin/users/create", data={"username": "mallory",
                                                      "password": "pw-mallory-1"},
                          follow_redirects=False).status_code == 303,
              "未登录不能建号")
        check(client.get("/static/style.css").status_code == 200, "/static/style.css 已挂载")
        check(client.get("/nope").status_code == 404, "未知路径 404")

        # ------------------------------------------------------------ 管理员建号 → 用户登录
        lr = client.post("/login", data={"username": settings.admin_username,
                                        "password": settings.admin_password}, follow_redirects=False)
        check(lr.status_code == 303 and client.cookies.get("sid") is not None,
              "管理员用 .env 里的口令进后台", str(lr.status_code))
        ca = client.post("/admin/users/create", data={"username": "alice", "password": "pw-alice-123",
                                                     "display_name": "小艾", "role": "user"},
                         follow_redirects=False)
        check(ca.status_code == 303 and "msg=" in ca.headers["location"], "后台逐个开通账号",
              f"{ca.status_code} {ca.headers.get('location')}")
        alice = db.get_user_by_name("alice")
        check(alice is not None and alice["display_name"] == "小艾" and alice["role"] == "user",
              "新账号按管理员填的信息入库")
        check(alice["password_hash"].startswith("pbkdf2_sha256$"), "口令只存加盐哈希",
              alice["password_hash"][:16])
        page(client, ca.headers["location"], "已开通账号 alice", "无法查看", label="跳转后后台给出开通回执")
        dup = client.post("/admin/users/create", data={"username": "alice", "password": "pw-alice-123",
                                                      "role": "user"})
        check(dup.status_code == 200 and "已被注册" in dup.text, "重名开通被拦下并留在后台")
        bad = client.post("/admin/users/create", data={"username": "a", "password": "12345678",
                                                      "role": "user"})
        check(bad.status_code == 200 and "3-32" in bad.text, "非法用户名给出表单错误")
        short = client.post("/admin/users/create", data={"username": "dave", "password": "123",
                                                        "role": "user"})
        check(short.status_code == 200 and "6 位" in short.text, "过短口令被拦下")
        root = client.post("/admin/users/create", data={"username": "dave", "password": "pw-dave-123",
                                                       "role": "root"})
        check(root.status_code == 200 and "角色" in root.text and db.get_user_by_name("dave") is None,
              "非法角色被拦下且未落库")
        check('value="dave"' in root.text, "失败时表单回填已填的用户名")
        keep = client.post("/admin/users/create", data={"username": "dave", "password": "pw-dave-123",
                                                       "display_name": "大韦", "role": "user"},
                           follow_redirects=False)
        check(keep.status_code == 303, "同一张表单改对后可直接重试", str(keep.status_code))
        dave_id = db.get_user_by_name("dave")["id"]
        client.post(f"/admin/users/{dave_id}/delete", follow_redirects=False)
        check(db.get_user_by_name("dave") is None, "管理员可删除刚建的空账号")
        client.post("/logout")
        # ------------------------------------------------------------ 登录与表单校验
        r = client.post("/login", data={"username": "alice", "password": "pw-alice-123",
                                       "next": "/dashboard"}, follow_redirects=False)
        check(r.status_code == 303 and r.headers["location"] == "/dashboard",
              "管理员开的口令用户可直接登录", str(r.status_code))
        check(client.cookies.get("sid") is not None, "会话 Cookie 已下发")
        page(client, "/dashboard", "小艾", "退出", "上传", "我的演讲", "历次对比",
             label="普通用户的个人页面入口照常保留")
        r = client.post("/login", data={"username": "alice", "password": "wrong"}, follow_redirects=False)
        check(r.status_code == 200 and "口令" in r.text, "错误口令回到表单且不建会话")

        # ------------------------------------------------------------ 上传与分析
        t0 = time.time()
        v1 = upload(client, "第一次演讲")
        s1 = wait_done(client, v1)
        check(s1.startswith("done"), f"视频 1 分析完成", s1)
        v2 = upload(client, "第二次演讲")
        s2 = wait_done(client, v2)
        check(s2.startswith("done"), "视频 2 分析完成", s2)
        print(f"     两次分析耗时 {time.time() - t0:.0f}s")
        hc2 = client.get("/health").json()["concurrency"]
        check(hc2["pool_size"] == settings.max_analyzers, "分析池按 MAX_ANALYZERS 建好",
              f"{hc2['pool_size']} 路 / 在跑 {hc2['running']}")

        # ------------------------------------------------------------ 报告页
        body = page(client, f"/videos/{v1}", "第一次演讲", "分项得分", 'id="player"',
                    "优点", "缺点", "改进建议", "可信度", "转录", "下一步只练一件事",
                    "证据", absent=("千问", "Qwen", "qwen", "whisper", "DashScope"),
                    label="报告页主区块齐全，且不出现模型名")
        check("时间" in body, "报告页含时间轴/时长核查")
        ev = db.get_evaluation(v1)
        payload = json.loads(ev["payload"] or "{}")
        check(bool(payload.get("advantages")), "优点非空")
        check(bool(payload.get("disadvantages")), "缺点非空")
        check(bool(payload.get("suggestions")), "改进建议非空")
        check(bool(str(payload.get("summary", "")).strip()), "总结非空")
        dims = db.dims_for_video(v1)
        check(len(payload.get("dimensions", [])) == len(dims) == len(rubric.dimensions),
              "维度入库数与标准一致", f"{len(payload.get('dimensions', []))}/{len(dims)}")
        check(all(0 <= d["score"] <= d["max_score"] for d in dims.values()), "各维度分数在满分区间内")
        check(abs(sum(d["score"] for d in dims.values()) - float(ev["total"])) < 0.51,
              "维度之和 ≈ 总分", f"{sum(d['score'] for d in dims.values())} vs {ev['total']}")

        # ------------------------------------------------------------ 播放与资源
        r = client.get(f"/videos/{v1}/play", headers={"Range": "bytes=0-1023"})
        check(r.status_code == 206 and len(r.content) == 1024, "Range 分片返回 206",
              f"{r.status_code}/{len(r.content)}")
        check(client.get(f"/videos/{v1}/play", headers={"Range": "bytes=999999999-"}).status_code == 416,
              "越界 Range 返回 416")
        check(client.get(f"/videos/{v1}/asset/cover.jpg").status_code == 200, "封面可读取")
        srcs = sorted(set(re.findall(r'/videos/\d+/asset/[^\s"\']+', body)))
        bad = [u for u in srcs if client.get(u).status_code != 200]
        check(bool(srcs) and not bad, "报告页内嵌资源全部可访问",
              f"{len(srcs)} 个，失败 {bad[:3]}")
        check(any("/asset/frames/" in u for u in srcs), "关键帧以子路径暴露且可访问")
        check(client.get(f"/videos/{v1}/asset/..%2f..%2fapp%2fmain.py").status_code in (400, 404),
              "资源名穿越被拒")
        check(client.get(f"/videos/{v1}/asset/nope.txt").status_code in (400, 404), "非法资源名被拒")

        # ------------------------------------------------------------ 导师提问
        qa_body = page(client, f"/videos/{v1}", "导师提问", 'name="answer_1"',
                       'name="answer_2"', 'name="answer_3"', 'id="qa"',
                       absent=("生成导师提问",), label="分析完成后报告页直接展示 3 道待作答题")
        check("等待作答" in qa_body, "未作答时区块标注等待作答")
        check('class="qa-list"' in qa_body and qa_body.count('class="qa-item"') == 3,
              "列表按题序渲染 3 条")
        dup = client.post(f"/videos/{v1}/qa/generate", follow_redirects=False)
        check(dup.status_code == 303 and "msg=" in dup.headers["location"]
              and "已存在" in unquote(dup.headers["location"]),
              "重复生成会给出已存在回执", dup.headers.get("location", ""))
        miss = client.post(f"/videos/{v1}/qa/answer",
                           data={"answer_1": "只答第一题。", "answer_2": "", "answer_3": "第三题。"},
                           follow_redirects=False)
        check(miss.status_code == 303 and "都回答" in unquote(miss.headers["location"]),
              "缺答时不入库并给出提示", miss.headers.get("location", ""))
        check(all(not q["answer"] for q in db.get_questions(v1)), "缺答提交未污染已有作答")
        full = client.post(f"/videos/{v1}/qa/answer",
                           data={"answer_1": "第一题作答。", "answer_2": "第二题作答。",
                                 "answer_3": "第三题作答。"},
                           follow_redirects=False)
        check(full.status_code == 303 and "点评" in unquote(full.headers["location"]),
              "齐答三题后跳回报告页", full.headers.get("location", ""))
        answered = page(client, f"/videos/{v1}", "已作答", 'class="qa-comment"',
                        "第一题作答。", label="提交后报告页回填作答并展示简评")
        check(answered.count('class="qa-comment"') == 3, "每题都渲染出一条简评")
        saved = db.get_questions(v1)
        check(len(saved) == 3 and all(q["status"] == "answered" and q["ai_comment"] for q in saved),
              "作答与简评一并入库", str([q["status"] for q in saved]))

        # ------------------------------------------------------------ 对比页
        page(client, "/compare", "历次对比", "维度轮廓", "逐项分值对照", "变化量", "总分趋势",
             label="对比页（默认最近三次）渲染成功")
        page(client, f"/compare?ids={v1},{v2}", "维度轮廓", label="对比页显式指定两次")
        page(client, f"/compare?ids={v1}", "", absent=("维度轮廓",), label="仅选一次时不画对比图")
        page(client, "/compare?ids=abc", "", label="非法 ids 不崩")
        page(client, "/compare?ids=999999", "", label="不存在的 id 不崩")

        # ------------------------------------------------------------ 微表情与逐帧测量
        # 样例视频是合成画面，Haar 检不出人脸，正确行为是「如实说明未采信」而不是编指标。
        check("本地逐帧测量未采信" in body, "检不出人脸时报告页如实标注未采信")
        st = [float(x) for x in payload.get("stamps") or []]
        check(bool(st) and all(b > a for a, b in zip(st, st[1:])), "落库抽帧时刻单调递增",
              f"{len(st)} 个")
        check(len(st) == len(payload.get("frames") or []), "时刻数与关键帧数一致")
        fc = payload.get("face") or {}
        check(fc.get("available") is False and bool(str(fc.get("reason", "")).strip()),
              "未采信带可解释理由", str(fc.get("reason"))[:44])
        check(fc.get("frontal_ratio") is None, "门控未通过时不输出正脸率", str(fc.get("frontal_ratio")))

        # 注入一份「测到了」的测量结果，验证渲染路径与趋势图第二折线，不依赖镜头内容。
        fake = {"available": True, "scanned": len(st), "reason": "", "face_rate": 1.0,
                "frontal_ratio": 0.62, "nonfrontal_ratio": 0.38, "down_ratio": None,
                "yaw_bias": -0.11, "head_spread": 0.08, "eye_reliability": 0.4,
                "headline": "正脸率 62%，非正脸 38%，头部平均偏左 11% 画宽",
                "notes": ["正脸率低于 35%，请人工核实画面通道对回避对视的判断"],
                "frames": [{"t": t, "frontal": i % 5 != 0, "profile": 0 if i % 5 else -1,
                            "eyes": 2 if i % 3 else 0, "offset": 0.05, "size": 0.1}
                           for i, t in enumerate(st)]}

        def patch_face(vid: int, face: dict | None, ratio: float | None) -> None:
            row = db.get_evaluation(vid)
            if face is not None:
                p = json.loads(row["payload"] or "{}")
                p["face"] = face
                with db.get_conn() as c:
                    c.execute("UPDATE evaluations SET payload = ? WHERE id = ?",
                              (json.dumps(p, ensure_ascii=False), row["id"]))
            db.update_video(vid, frontal_ratio=ratio)

        patch_face(v1, fake, 0.62)
        patch_face(v2, None, 0.9)
        fb = page(client, f"/videos/{v1}", "本地逐帧测量", "正脸率 62%", "侧脸", "质控参考，不计入分数",
                  label="报告页渲染逐帧朝向标注与测量摘要")
        check('class="off"' in fb, "非正脸帧在胶片带上被单独标出")
        check("人工核实" in fb, "测量备注随报告一并展示")
        cb = page(client, "/compare", "总分与眼神接触率趋势", "眼神接触率", "不参与加权求和",
                  label="对比页画出眼神接触率第二折线")
        check("stroke-dasharray" in cb, "眼神接触率以虚线与总分区分")
        check("62%" in cb and "90%" in cb, "折线点标注实测百分比")
        page(client, f"/videos/{v2}", "本地逐帧测量", absent=("正脸率 62%",),
             label="未测量视频不借用他人的微表情数据")
        patch_face(v1, fc, None)
        patch_face(v2, None, None)

        # ------------------------------------------------------------ 模型用量记账
        # 演示模式不发真实请求，_complete() 不会被调用，这里直接驱动记账层验证。
        qc = QwenClient()
        qc.mark("转录 ASR")._track(
            {"messages": [{"role": "user",
                           "content": [{"type": "text", "text": "请转录这段音频"}]}]},
            Completion(text="hello world 你好", model="mock", usage=None))
        qc.mark("文本通道")._track(
            {"messages": [{"role": "user", "content": "请按标准评分"}]},
            Completion(text="{}", model="mock",
                       usage={"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500}))
        usage = qc.usage_summary()
        check(usage["calls"] == 2 and usage["estimated_calls"] == 1,
              "记账区分真实 usage 与估算调用", f"{usage['calls']} 次 / 估算 {usage['estimated_calls']} 次")
        check(usage["total"] == usage["prompt"] + usage["completion"] >= 1500,
              "输入输出与总量自洽", f"入 {usage['prompt']} / 出 {usage['completion']} / 计 {usage['total']}")
        check([s["name"] for s in usage["by_stage"]] == ["文本通道", "转录 ASR"],
              "用量按阶段分组、按量倒序", str([s["name"] for s in usage["by_stage"]]))
        check(usage["by_model"][0]["calls"] == 2, "用量可按模型聚合", str(usage["by_model"]))
        check(estimate_tokens("你好，世界") == 5 and estimate_tokens("abcd") == 1
              and estimate_tokens("abcdefgh") == 2 and estimate_tokens("") == 0,
              "估算规则：汉字与全角标点 1 字、其余 4 字符 1 token")
        check(QwenClient().usage_summary()["calls"] == 0, "新建客户端用量记录互相隔离")

        pipeline._flush_usage(qc, v1)
        alice = db.get_user_by_name("alice")["id"]
        mine = db.token_usage(alice, per_video=5)
        check(mine["calls"] == usage["calls"] and mine["total"] == usage["total"],
              "落库后按用户聚合与客户端汇总一致", f"{mine['calls']} 次 / {mine['total']} token")
        check(mine["analyzed"] == 1 and mine["avg_per_video"] == mine["total"],
              "平均每次分析用量可算", str(mine["avg_per_video"]))
        check(mine["by_stage"][0]["name"] == "文本通道" and mine["estimated_calls"] == 1,
              "明细含阶段聚合与估算计数")
        check(mine["videos"] and mine["videos"][0]["id"] == v1 and mine["videos"][0]["estimated"] == 1,
              "最近消耗明细带估算标记", str(mine["videos"][:1]))
        check(db.token_usage()["total"] >= mine["total"], "全站聚合覆盖新增用量")
        site = db.token_usage(per_video=8)
        check(site["by_user"] and site["by_user"][0]["name"].endswith("（alice）"),
              "全站按用户分组带昵称与账号名", str(site["by_user"][:1]))
        check(client.get(f"/videos/{v2}").status_code == 200, "用量写入不影响他人报告读取")

        # ------------------------------------------------------------ 复发问题清理
        with db.get_conn() as conn:
            n_issue = conn.execute("SELECT COUNT(*) c FROM issues WHERE video_id = ?", (v1,)).fetchone()["c"]
        check(client.post(f"/videos/{v1}/delete", follow_redirects=False).status_code == 303, "删除视频")
        page(client, "/dashboard", "第二次演讲", absent=("第一次演讲",), label="删除后列表不再出现该视频")
        with db.get_conn() as conn:
            left = conn.execute("SELECT COUNT(*) c FROM issues WHERE video_id = ?", (v1,)).fetchone()["c"]
            orphan = conn.execute("SELECT COUNT(*) c FROM dimension_scores WHERE video_id = ?",
                                  (v1,)).fetchone()["c"]
        check(n_issue > 0 and left == 0, "删除视频同时清理 issues", f"{n_issue} → {left}")
        check(orphan == 0, "删除视频级联清理维度分", f"{orphan} 条残留")

        # ------------------------------------------------------------ 管理员
        client.post("/logout")
        check(client.cookies.get("sid") is None, "退出登录清除 Cookie")
        r = client.post("/login", data={"username": settings.admin_username,
                                       "password": settings.admin_password, "next": "/admin"},
                        follow_redirects=False)
        check(r.status_code == 303, "管理员登录", str(r.status_code))
        page(client, "/admin", "管理后台", "全部视频", "维度均分", "用户",
             "账号开通", "逐个录入", "Excel 批量导入", "下载 账号导入模板.xlsx", "重置口令",
             "上限 5000 行", label="后台页渲染成功且账号入口齐全")
        admin_body = client.get("/admin").text
        check("alice" in admin_body and "第二次演讲" in admin_body, "后台可见全站用户与视频")
        check("小艾" in admin_body, "后台显示昵称")
        page(client, f"/videos/{v2}", f"用户 alice", label="管理员查看他人报告带归属")
        check(client.get(f"/videos/{v2}/status").json()["status"] == "done", "管理员可读他人视频状态")
        page(client, f"/videos/{v2}", "管理后台", absent=("我的演讲",),
             label="管理员的报告页面包屑也指向后台")

        # ------------------------------------------------------------ 管理员侧的个人页面已撤掉
        page(client, "/admin", "系统设置", absent=("我的演讲", "历次对比"),
             label="管理员导航里不再出现两个个人页面入口")
        for route in ("/dashboard", "/compare"):
            g = client.get(route, follow_redirects=False)
            check(g.status_code == 303 and g.headers["location"] == "/admin",
                  f"管理员手敲 {route} 也一律送回后台", f"{g.status_code} {g.headers.get('location')}")

        # ------------------------------------------------------------ 维度均分：先筛主题再看人
        kc = client.post("/admin/users/create", data={"username": "carol", "password": "pw-carol-1",
                                                     "display_name": "小嘉", "role": "user"},
                         follow_redirects=False)
        check(kc.status_code == 303, "为统计再开第二个用户，凑出两个人两把尺子", str(kc.status_code))
        client.post("/logout")
        client.post("/login", data={"username": "carol", "password": "pw-carol-1",
                                   "next": "/dashboard"}, follow_redirects=False)
        v3 = upload(client, "第三次演讲", topic="How to memorize 1000 words")
        s3 = wait_done(client, v3)
        check(s3.startswith("done"), "第二个用户的视频分析完成", s3)
        client.post("/logout")
        client.post("/login", data={"username": settings.admin_username,
                                   "password": settings.admin_password, "next": "/admin"},
                    follow_redirects=False)
        allb = page(client, "/admin", "按主题筛选", "全部主题 · 2 条视频", "按人拆开",
                    "How to memorize 1000 words", "Will AI replace English teachers?",
                    label="维度均分给出主题下拉框，两个主题都能选")
        check("小嘉" in allb and "carol" in allb, "人 × 维度矩阵里两个人各占一行")
        check("全部主题 · 2 条视频" in allb and "· 1 条" in allb, "下拉项带每个主题的样本条数")
        one = page(client, "/admin", "How to memorize 1000 words · 1 条视频", "清除筛选",
                   params={"topic": "How to memorize 1000 words"}, label="按主题筛选只算该主题的视频")
        check("按人拆开" not in one, "只剩一个人时不画人 × 维度矩阵")
        check('value="How to memorize 1000 words" selected' in one, "下拉框回填当前筛选条件")
        empty = page(client, "/admin", "该主题下还没有评分记录", params={"topic": "没这个主题"},
                     label="筛到空主题时给出具体的下一步")
        check("还没有任何评分记录。用户完成" not in empty, "空态区分「筛过头」与「全站没数据」")
        dall = db.admin_dim_stats()
        check(dall["samples"] == 2 and len(dall["by_user"]) == 2,
              "不筛时两人两视频一起进统计", str([(p["username"], p["n"]) for p in dall["by_user"]]))
        d1 = db.admin_dim_stats("How to memorize 1000 words")
        check(d1["samples"] == 1 and len(d1["by_user"]) == 1
              and all(x["n"] == 1 for x in d1["by_dim"]),
              "筛选后每个维度的样本数同步收窄", f"{len(d1['by_dim'])} 个维度")
        carol_row = [p for p in dall["by_user"] if p["username"] == "carol"][0]
        check(set(carol_row["dims"]) == {d["dim_key"] for d in dall["by_dim"]},
              "矩阵列与维度表一一对应", f"{len(carol_row['dims'])} 列")

        # ------------------------------------------------------------ 账号管理：模板与批量导入
        me_id = db.get_user_by_name(settings.admin_username)["id"]
        self_del = client.post(f"/admin/users/{me_id}/delete", follow_redirects=False)
        check(self_del.status_code == 303 and "error=" in self_del.headers["location"],
              "不能删除当前登录的管理员", str(self_del.headers.get("location")))
        check("不能删除当前登录的管理员账号" in client.get(self_del.headers["location"]).text,
              "删除被拒的原因显示在红条里")
        nofile = client.post("/admin/users/import", data={})
        check(nofile.status_code == 200 and "请先选择一个 .xlsx 文件" in nofile.text, "未选文件时给出提示")
        csv = client.post("/admin/users/import", files={"file": ("名单.csv", b"a,b,c", "text/csv")})
        check("只支持 .xlsx" in csv.text, "非 xlsx 退回并说明怎么另存")
        blank = client.post("/admin/users/import", files={"file": ("名单.xlsx", b"", accounts.XLSX_MIME)})
        check("空的" in blank.text, "空文件给出说明")
        broken = client.post("/admin/users/import",
                             files={"file": ("名单.xlsx", b"not a zip" * 64, accounts.XLSX_MIME)})
        check("读不了" in broken.text, "坏文件给出可操作的说明")

        tpl = client.get("/admin/users/template")
        check(tpl.status_code == 200 and tpl.content[:2] == b"PK", "模板下载返回 xlsx 字节流",
              f"{tpl.status_code} / {len(tpl.content)} 字节")
        check(tpl.headers.get("content-type") == accounts.XLSX_MIME, "模板 MIME 正确",
              str(tpl.headers.get("content-type")))
        check("attachment" in tpl.headers.get("content-disposition", ""), "模板以附件下载（中文名已编码）",
              tpl.headers.get("content-disposition", ""))
        users_before = len(db.list_users())
        asis = client.post("/admin/users/import",
                           files={"file": (accounts.TEMPLATE_NAME, tpl.content, accounts.XLSX_MIME)})
        check(asis.status_code == 200 and "新建 0 个账号" in asis.text
              and "跳过示例与表头 2 行" in asis.text, "模板原样上传只跳过、不建号", str(asis.status_code))
        check(len(db.list_users()) == users_before, "示例行不会变成账号")

        sheet = client.post("/admin/users/import",
                            files={"file": ("名单.xlsx", xlsx(DIRTY), accounts.XLSX_MIME)})
        check(sheet.status_code == 200 and "新建 3 个账号" in sheet.text and "退回 5 行" in sheet.text,
              "脏表回执：3 个建成、5 行退回", str(sheet.status_code))
        for why in ("缺登录口令", "登录口令至少 6 位", "仅限字母、数字与 . _ -",
                    "该账号已存在", "与模板第 7 行重名"):
            check(why in sheet.text, f"回执写清退回原因：{why}")
        check("20250101" in sheet.text and db.get_user_by_name("20250101") is not None,
              "数字单元格去掉 .0 尾巴后按学号建号")
        check(db.get_user_by_name("frank") is not None and db.get_user_by_name("ivan") is not None,
              "合法行全部建成账号")
        check(db.get_user_by_name("gina") is None and db.get_user_by_name("helen") is None
              and db.get_user_by_name("周") is None, "退回行不留半截账号")
        frank = db.get_user_by_name("frank")
        check(frank["display_name"] == "法兰克" and frank["role"] == "user", "导入带上显示名且默认普通用户")
        check("已开通" in sheet.text and "退回" in sheet.text, "回执表区分已开通与退回")
        check(page(client, "/admin", absent=("名单.xlsx",), label="回执只在导入那一次显示").count("名单") == 0,
              "刷新后不留上一次的导入明细")

        # ------------------------------------------------------------ 重置口令
        rst_bad = client.post(f"/admin/users/{frank['id']}/password", data={"password": "123"},
                              follow_redirects=False)
        check(rst_bad.status_code == 303 and "error=" in rst_bad.headers["location"], "重置过短口令被拦下")
        check("至少 6 位" in client.get(rst_bad.headers["location"]).text, "红条说明口令太短")
        rst = client.post(f"/admin/users/{frank['id']}/password", data={"password": "pw-frank-new"},
                          follow_redirects=False)
        check(rst.status_code == 303 and "msg=" in rst.headers["location"], "管理员可重置口令",
              str(rst.headers.get("location")))
        after = db.get_user_by_name("frank")
        check(after["password_hash"] != frank["password_hash"] and after["username"] == "frank",
              "重置只换哈希，用户名与显示名不动")
        check("pw-frank-new" not in client.get("/admin").text and "pw-frank-123" not in client.get("/admin").text,
              "任何页面都不回显口令原文")
        client.post("/logout")
        check(client.post("/login", data={"username": "frank", "password": "pw-frank-123"},
                          follow_redirects=False).status_code == 200
              and client.cookies.get("sid") is None, "旧口令在重置后失效")
        check(client.post("/login", data={"username": "frank", "password": "pw-frank-new"},
                          follow_redirects=False).status_code == 303, "新口令可登录")
        client.post("/logout")
        check(client.post("/login", data={"username": "20250101", "password": "20250101"},
                          follow_redirects=False).status_code == 303, "学号账号可用学号登录")
        client.post("/logout")

        # ------------------------------------------------------------ 权限隔离
        client.post("/logout")
        client.post("/login", data={"username": settings.admin_username,
                                   "password": settings.admin_password}, follow_redirects=False)
        bk = client.post("/admin/users/create", data={"username": "bob", "password": "pw-bob-123",
                                                     "role": "user"}, follow_redirects=False)
        check(bk.status_code == 303, "管理员再开一个账号用于越权测试", str(bk.status_code))
        client.post("/logout")
        bb = client.post("/login", data={"username": "bob", "password": "pw-bob-123"}, follow_redirects=False)
        check(bb.status_code == 303, "新开账号可直接登录", str(bb.status_code))
        check(client.get("/admin/users/template", follow_redirects=False).status_code == 403,
              "普通用户不能下载模板")
        check(client.post("/admin/users/create", data={"username": "mallory", "password": "pw-mallory-1"},
                          follow_redirects=False).status_code == 403, "普通用户不能建号")
        check(client.post("/admin/users/import",
                          files={"file": ("名单.xlsx", tpl.content, accounts.XLSX_MIME)},
                          follow_redirects=False).status_code == 403, "普通用户不能批量导入")
        check(client.post(f"/admin/users/{me_id}/password", data={"password": "pw-x-12345"},
                          follow_redirects=False).status_code == 403, "普通用户不能重置口令")
        check(db.get_user_by_name("mallory") is None, "越权建号未落库")
        check(client.get(f"/videos/{v2}", follow_redirects=False).status_code == 403, "他人视频 403")
        check(client.get(f"/videos/{v2}/play", follow_redirects=False).status_code == 403, "他人视频不可播放")
        check(client.get(f"/videos/{v2}/status", follow_redirects=False).status_code == 403, "他人状态不可读")
        check(client.get("/admin", follow_redirects=False).status_code == 403, "普通用户访问后台 403")
        check(client.post(f"/videos/{v2}/delete", follow_redirects=False).status_code == 403, "他人视频不可删")
        check(client.post(f"/videos/{v2}/qa/generate", follow_redirects=False).status_code == 403,
              "他人视频不可生成提问")
        check(client.post(f"/videos/{v2}/qa/answer", data={"answer_1": "x"},
                          follow_redirects=False).status_code == 403,
              "他人视频不可提交作答")
        check(client.post("/admin/users/1/delete", follow_redirects=False).status_code == 403,
              "普通用户不能删账号")
        bob_id = db.get_user_by_name("bob")["id"]
        check(client.post(f"/admin/users/{bob_id}/delete", follow_redirects=False).status_code == 403,
              "普通用户不能删自己（走后台）")

        # ------------------------------------------------------------ 篡改 Cookie（独立客户端，避免污染会话 jar）
        tamper = TestClient(app)
        check(tamper.get("/dashboard", follow_redirects=False).status_code == 303, "无 Cookie 视为未登录")
        tamper.cookies.set("sid", "")
        check(tamper.get("/dashboard", follow_redirects=False).status_code == 303, "空 Cookie 视为未登录")
        tamper.cookies.set("sid", "eyJpZCI6MSwicm9sZSI6ImFkbWluIn0.forged")
        check(tamper.get("/admin", follow_redirects=False).status_code == 303, "伪造 Cookie 被拒")
        check(tamper.post(f"/videos/{v1}/qa/generate", follow_redirects=False).status_code == 303
              and tamper.post(f"/videos/{v1}/qa/generate", follow_redirects=False
                              ).headers["location"].startswith("/login"),
              "未登录点击生成跳回登录页")
        # 角色存在库里，Cookie 里写 admin 也没用
        forged = make_session_token(bob_id, "admin")
        tamper.cookies.set("sid", forged)
        check(tamper.get("/admin", follow_redirects=False).status_code == 403,
              "签名有效但角色伪装无效")
        tamper.close()

        # ------------------------------------------------------------ 损坏文件
        junk = DATA / "corrupt.mp4"
        junk.parent.mkdir(parents=True, exist_ok=True)
        junk.write_bytes(b"this is definitely not an mp4 stream " * 256)
        vid = upload(client, "坏文件", auto=False, path=junk)
        client.post(f"/videos/{vid}/analyze", follow_redirects=False)
        outcome = wait_done(client, vid, timeout=180)
        check(outcome.startswith("failed"), "损坏文件进入失败态而非卡住", outcome)
        page(client, f"/videos/{vid}", "分析中断", "重新", label="进度页展示失败与重试入口")
        check(client.get("/dashboard").text.count("失败") >= 1, "列表可见失败状态")

        # ------------------------------------------------------------ 重启自愈
        db.update_video(vid, status="analyzing", stage="分析中", progress=55)
        check(db.reset_stale_jobs() >= 1, "重启时把中断任务标记为失败")
        check(db.get_video(vid)["status"] == "failed", "自愈后状态为 failed")

        # ------------------------------------------------------------ 上传次数配额
        bob_q = db.upload_quota(bob_id)
        check(bob_q["limit"] == 5 and bob_q["used"] == 1 and bob_q["remaining"] == 4,
              "新账号默认每人 5 次，上传即计数", str(bob_q))
        page(client, "/dashboard", "已用 1/5 次 · 剩余 4 次", label="列表页顶部展示余额")
        spare = [upload(client, f"占位视频{i}", auto=False, path=junk, topic="") for i in range(1, 5)]
        files_before = len(list(settings.video_dir.iterdir()))
        over = client.post("/videos", data={"title": "第六次", "topic": "", "auto": ""},
                           files={"video": ("sixth.mp4", b"not a real stream", "video/mp4")},
                           follow_redirects=False)
        check(over.status_code == 303 and over.headers["location"].startswith("/dashboard?msg="),
              "第 6 次上传被拦下", str(over.headers.get("location")))
        check(len(list(settings.video_dir.iterdir())) == files_before,
              "额度闸门在落盘之前，被拒的上传不占磁盘")
        page(client, "/dashboard", "上传次数已用完（5/5）", "请管理员重置",
             absent=('action="/videos"', 'type="file"'), label="额度耗尽后上传表单收起并给出下一步")

        client.post("/logout")
        client.post("/login", data={"username": settings.admin_username,
                                    "password": settings.admin_password, "next": "/admin"},
                    follow_redirects=False)
        page(client, "/admin", "bob", "5/5", "已用完", "不限",
             label="后台用户表标出额度用尽的账号，管理员自身不受限")
        rs = client.post(f"/admin/users/{bob_id}/quota",
                         data={"used": "5", "limit": "", "action": "reset"}, follow_redirects=False)
        check(rs.status_code == 303 and "msg=" in rs.headers["location"],
              "管理员一键清零已用次数", str(rs.headers.get("location")))
        page(client, rs.headers["location"], "bob 的上传额度已更新：已用 0/5", label="重置后后台回显余额")
        check(db.upload_quota(bob_id)["remaining"] == 5, "清零后 5 个名额全部回来（上限列未被改动）")

        client.post("/logout")
        client.post("/login", data={"username": "bob", "password": "pw-bob-123",
                                    "next": "/dashboard"}, follow_redirects=False)
        upload(client, "重置后再传", auto=False, path=junk, topic="")
        check(db.upload_quota(bob_id)["used"] == 1, "重置后本人可以继续上传")
        client.post(f"/videos/{spare[0]}/delete", follow_redirects=False)
        check(db.upload_quota(bob_id)["used"] == 0, "删掉一条记录即返还一次名额")
        check(client.post(f"/admin/users/{bob_id}/quota", data={"action": "reset"},
                          follow_redirects=False).status_code == 403, "普通用户不能调整额度")

        client.post("/logout")
        client.post("/login", data={"username": settings.admin_username,
                                    "password": settings.admin_password, "next": "/admin"},
                    follow_redirects=False)
        solo = client.post(f"/admin/users/{bob_id}/quota",
                           data={"used": "1", "limit": "9", "action": "save"}, follow_redirects=False)
        check("bob 的上传额度已更新：已用 1/9" in client.get(solo.headers["location"]).text,
              "管理员可给单个账号另放上限", str(solo.headers.get("location")))
        check(db.upload_quota(bob_id)["limit"] == 9, "单独上限优先于全局默认")
        dft = client.post(f"/admin/users/{bob_id}/quota",
                          data={"used": "1", "limit": "", "action": "default"}, follow_redirects=False)
        check(db.get_user(bob_id)["upload_limit"] is None and db.upload_quota(bob_id)["limit"] == 5,
              "「跟随全局」按钮取消单独上限、回到默认 5 次",
              str(client.get(dft.headers["location"]).text.count("已用 1/5")))
        client.post(f"/admin/users/{bob_id}/quota", data={"used": "1", "limit": "0", "action": "save"},
                    follow_redirects=False)
        check(db.upload_quota(bob_id)["remaining"] == 0, "上限填 0 即停掉该账号上传")
        bad = client.post(f"/admin/users/{bob_id}/quota", data={"used": "-1", "limit": "", "action": "save"},
                          follow_redirects=False)
        check("error=" in bad.headers["location"] and db.upload_quota(bob_id)["used"] == 1,
              "负数被拒且不改库", str(bad.headers.get("location")))

        client.post("/logout")
        client.post("/login", data={"username": "bob", "password": "pw-bob-123",
                                    "next": "/dashboard"}, follow_redirects=False)
        banned = client.post("/videos", data={"title": "被停", "topic": "", "auto": ""},
                             files={"video": ("banned.mp4", b"nope", "video/mp4")},
                             follow_redirects=True)
        page_ok = banned.status_code == 200 and "该账号已被管理员停止上传" in banned.text
        check(page_ok, "上限为 0 时给出专门文案", f"HTTP {banned.status_code}")

        # ------------------------------------------------------------ 表单校验
        check(client.post("/videos", data={"title": ""}, files={}, follow_redirects=False
                          ).headers["location"].startswith("/dashboard"), "未选文件时回到列表提示")

        # ------------------------------------------------------------ 系统设置页
        anon = TestClient(app)
        ar = anon.get("/settings", follow_redirects=False)
        check(ar.status_code == 303 and "/login" in ar.headers["location"],
              "未登录访问设置页跳登录", f"{ar.status_code} {ar.headers.get('location')}")
        check(client.get("/settings", follow_redirects=False).status_code == 403, "普通用户访问设置页 403")
        check(client.post("/settings", data={"MAX_VIDEO_MB": "1"}, follow_redirects=False).status_code == 403,
              "普通用户不能保存设置")
        anon.close()

        adm = TestClient(app)
        adm.post("/login", data={"username": settings.admin_username, "password": settings.admin_password,
                                 "next": "/settings"}, follow_redirects=False)
        secret = settings.api_key
        body = page(adm, "/settings", "系统设置", "环境配置", "30 项", "DASHSCOPE_API_KEY",
                    "保存并生效", "api_key.txt", "来自",
                    "并发分析数", "在线请求并发数", "网页工作线程数",
                    "自动压缩目标", "压缩超时",
                    absent=((secret,) if len(secret) > 8 else ()),
                    label="设置页渲染全部配置项")
        check('type="password"' in body and "留空表示不修改" in body, "密钥框不明文回显")
        check("待保存" not in body, "干净打开时没有残留的待保存标记")

        width_before, env_before = settings.frame_width, cfg.read_env_file()
        bad = adm.post("/settings", data=env_form({"FRAME_WIDTH": "abc"}))
        check(bad.status_code == 200 and "未保存" in bad.text and "需要填数字" in bad.text
              and "待保存" in bad.text, "非法值被拦下并回显待保存", f"HTTP {bad.status_code}")
        check(cfg.read_env_file() == env_before, "校验失败时 .env 一字未改")
        check(settings.frame_width == width_before, "校验失败时运行时不变")

        ok_post = adm.post("/settings", data=env_form({"MAX_VIDEO_MB": "640", "FRAME_INTERVAL": "4"},
                                                      uncheck=("USE_AUDIO_CHANNEL", "USE_FACE_METRICS")),
                           follow_redirects=False)
        check(ok_post.status_code == 303 and "msg=" in ok_post.headers["location"],
              "合法保存后跳转回设置页", f"{ok_post.status_code} {ok_post.headers.get('location')}")
        check(settings.max_video_bytes == 640 * 1024 * 1024 and settings.frame_interval == 4
              and settings.use_audio_channel is False and settings.use_face_metrics is False,
              "保存后运行时立即生效（无需重启）",
              f"{settings.max_video_bytes // (1024 * 1024)}MB / 每 {settings.frame_interval}s")
        saved = cfg.read_env_file()
        check(saved.get("MAX_VIDEO_MB") == "640" and saved.get("FRAME_INTERVAL") == "4"
              and saved.get("USE_AUDIO_CHANNEL") == "0", "新值写回 .env 文件", str(sorted(saved)[:3]))
        page(adm, ok_post.headers["location"], "640", "已保存", label="回到页面显示保存结果与新值")
        check("来自 .env" in adm.get("/settings").text, "保存后来源徽章变成 .env")

        pwd_before = db.get_user_by_name(settings.admin_username)["password_hash"]
        adm.post("/settings", data=env_form(), follow_redirects=False)
        check(db.get_user_by_name(settings.admin_username)["password_hash"] == pwd_before,
              "密钥与口令留空时重复保存不误改")
        check(not cfg.read_env_file().get("ADMIN_PASSWORD"), "留空的密钥项没有被写进文件")

        # ------------------------------------------------ 大模型提示词（与 .env 分开的第二条链路）
        check(not prm.PROMPTS_FILE.exists(), "保存 .env 表单不会写提示词文件")
        env_snapshot = cfg.read_env_file()
        body = page(adm, "/settings", "提示词配置", "19 块 · 0 块已自定义", "保存提示词",
                    "全部恢复内置默认", 'name="common_rules"', 'name="narrative_schema"',
                    'name="qa_context"', 'name="qa_review_schema"',
                    "<textarea", "内置默认", label="设置页渲染提示词分区")
        check("文件尚不存在" in body, "无覆盖文件时页面说明全部使用内置默认")
        check(prm.PROMPTS_FILE.exists() is False, "只打开页面不会创建提示词文件")

        bad = adm.post("/settings/prompts", data=prompt_form({"text_tail": "按要求输出，不要解释。"}),
                       follow_redirects=False)
        check(bad.status_code == 200 and "未保存" in bad.text and "缺少系统占位符" in bad.text,
              "提示词缺占位符时拒绝保存", f"HTTP {bad.status_code}")
        check("按要求输出，不要解释。" in bad.text and "缺少占位符" in bad.text,
              "拒绝后草稿原样回显并标出问题")
        check(not prm.PROMPTS_FILE.exists(), "提示词校验失败时一个字节都不写")

        good = adm.post("/settings/prompts",
                        data=prompt_form({"common_rules": "自定义硬性规则：每条扣分至少两处证据。"}),
                        follow_redirects=False)
        check(good.status_code == 303 and "已保存提示词" in unquote(good.headers["location"]),
              "合法提示词保存后跳转回设置页",
              f"{good.status_code} {unquote(good.headers.get('location', ''))}")
        disk = json.loads(prm.PROMPTS_FILE.read_text(encoding="utf-8"))
        check(list(disk["prompts"]) == ["common_rules"], "提示词文件只落差异块", str(list(disk["prompts"])))
        check(prm.get("common_rules").startswith("自定义硬性规则") and prm.get("system") == prm.DEFAULTS["system"],
              "保存后分析链路立即读到新提示词")
        page(adm, "/settings", "19 块 · 1 块已自定义", "管理员自定义", "_websmoke.prompts.json",
             label="重开页面显示自定义状态")
        check(cfg.read_env_file() == env_snapshot, "保存提示词不会动 .env 文件")

        rs = adm.post("/settings/prompts/reset", follow_redirects=False)
        check(rs.status_code == 303 and "已恢复 1 块" in unquote(rs.headers["location"]),
              "一键恢复内置默认", f"{rs.status_code} {unquote(rs.headers.get('location', ''))}")
        check(prm.get("common_rules") == prm.DEFAULTS["common_rules"] and prm.custom_count() == 0,
              "恢复后回到内置默认且计数归零")
        page(adm, rs.headers["location"], "已恢复", "19 块 · 0 块已自定义", label="恢复结果提示可见")

        check(client.post("/settings/prompts", data=prompt_form(), follow_redirects=False).status_code == 403,
              "普通用户不能保存提示词")
        check(client.post("/settings/prompts/reset", follow_redirects=False).status_code == 403,
              "普通用户不能恢复默认")
        guest = TestClient(app)
        gp = guest.post("/settings/prompts", data=prompt_form(), follow_redirects=False)
        check(gp.status_code == 303 and "/login" in gp.headers["location"],
              "未登录提交提示词跳登录", f"{gp.status_code} {gp.headers.get('location')}")
        guest.close()

        rl = adm.post("/settings/reload", follow_redirects=False)
        page(adm, rl.headers["location"], "重新加载", label="手工改文件后可按磁盘重载")
        h = adm.get("/health").json()
        check(h["ok"] and h["mode"] == "mock" and h["models"]["chat"] and h["asr_engine"],
              "健康接口反映当前运行时设置", f"{h['mode']} / {h['models']['chat']}")
        adm.close()

        # ------------------------------------------------------------ 大文件自动压缩（真实 ffmpeg 两遍编码）
        keep_cmp = settings.compress_target_bytes
        raw_size = SAMPLE.stat().st_size
        adm3 = TestClient(app)
        adm3.post("/login", data={"username": settings.admin_username,
                                  "password": settings.admin_password, "next": "/admin"},
                  follow_redirects=False)
        try:
            settings.compress_target_bytes = 1024 * 1024        # 样本约 3 MB，必然触发压缩
            ce = adm3.post("/admin/users/create", data={"username": "erin", "password": "pw-erin-123",
                                                       "display_name": "小珂", "role": "user"},
                           follow_redirects=False)
            check(ce.status_code == 303, "为压缩用例再开一个普通用户", str(ce.status_code))
            cw = TestClient(app)
            cl = cw.post("/login", data={"username": "erin", "password": "pw-erin-123",
                                         "next": "/dashboard"}, follow_redirects=False)
            check(cl.status_code == 303, "压缩用例账号可登录", str(cl.status_code))
            page(cw, "/dashboard", "超过 1 MB 自动压缩", label="上传表单说明大文件会自动压缩")
            cw.close()
            t0 = time.time()
            vc = upload(adm3, "大文件自动压缩")
            sc = wait_done(adm3, vc)
            check(sc.startswith("done"), "超过压缩目标的视频可完整走完分析", f"{sc} / 耗时 {time.time() - t0:.0f}s")
            rowc = db.get_video(vc)
            check(rowc["orig_size"] == raw_size and 200 * 1024 < rowc["size"] <= 1024 * 1024,
                  "原始体积入库且落盘文件已压到 1 MB 以内",
                  f"{raw_size / 1048576:.1f} MB → {rowc['size'] / 1048576:.1f} MB")
            saved = Path(rowc["path"])
            check(saved.exists() and saved.stat().st_size == rowc["size"]
                  and saved.suffix.lower() == ".mp4", "path 指向压缩后的新 mp4", saved.name)
            check("原始文件" in rowc["compress_note"] and "两遍编码" in rowc["compress_note"],
                  "压缩说明记录前后体积与编码方式", rowc["compress_note"][:46] + "…")
            check(not list(settings.video_dir.glob("*compressing*")),
                  "两遍编码的中间文件未残留", "、".join(p.name for p in settings.video_dir.iterdir())[:90])
            page(adm3, f"/videos/{vc}", "压缩前", "两遍编码", "分项得分",
                 label="报告页标注压缩前体积并把压缩写进质控记录")
            db.update_video(vc, status="analyzing", stage="两遍编码压缩 5%", progress=5)
            page(adm3, f"/videos/{vc}", "自动压缩到目标体积", "已自动压缩", "两遍编码压缩",
                 label="进度页含压缩步骤并显示原始体积")
            db.update_video(vc, status="done", stage="完成", progress=100)
            page(adm3, "/admin", "大文件自动压缩", "压缩前", label="后台列表标出被压缩过的视频")
            rc = adm3.get(f"/videos/{vc}/play", headers={"Range": "bytes=0-1023"})
            check(rc.status_code == 206 and len(rc.content) == 1024, "压缩后仍可分段播放", str(rc.status_code))
        finally:
            settings.compress_target_bytes = keep_cmp
            adm3.close()

        # ------------------------------------------------------------ 管理员删号（放在最后，避免级联删掉越权目标）
        client.post("/logout")
        lr = client.post("/login", data={"username": settings.admin_username,
                                         "password": settings.admin_password}, follow_redirects=False)
        check(lr.status_code == 303, "管理员重新登录", str(lr.status_code))
        alice_id = db.get_user_by_name("alice")["id"]
        art = settings.artifact_dir / str(v2)
        check(art.exists(), "抽帧产物目录已就绪", str(art))
        dr = client.post(f"/admin/users/{alice_id}/delete", follow_redirects=False)
        check(dr.status_code == 303, "管理员删除用户", f"{dr.status_code} {dr.headers.get('location')}")
        page(client, "/admin", "bob", absent=("alice",), label="被删用户从后台消失")
        check(client.get(f"/videos/{v2}").status_code == 404, "级联删除后视频 404")
        check(not art.exists(), "抽帧产物目录已清理", str(art))
        check(db.get_user_by_name("alice") is None, "被删用户从库中消失")
        client.post("/logout")
        r = client.post("/login", data={"username": "alice", "password": "pw-alice-123"},
                        follow_redirects=False)
        check(r.status_code == 200 and "alice" not in (client.cookies.get("sid") or ""),
              "被删用户无法再登录")
        with db.get_conn() as conn:
            check(conn.execute("SELECT COUNT(*) c FROM issues WHERE user_id = ?", (alice_id,)).fetchone()["c"] == 0
                  and conn.execute("SELECT COUNT(*) c FROM evaluations WHERE user_id = ?",
                                   (alice_id,)).fetchone()["c"] == 0, "用户评价数据全部清除")

    ok = sum(1 for r, _ in RESULTS if r)
    bad = [label for r, label in RESULTS if not r]
    print(f"\n{ok}/{len(RESULTS)} 通过")
    for label in bad:
        print(f"  ✗ {label}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
