"""端到端自检：不启动 Web，直接跑 建号 → 上传 → 分析 → 报告 → 对比 → 管理员视角。

用法：
    python tests/selfcheck.py            # 使用 .env 中的模式（无 key 时自动 mock）
    LLM_MODE=mock python tests/selfcheck.py

数据落在 data/_selfcheck，不会污染正式库。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.environ.setdefault("DATA_DIR", str(BASE / "data" / "_selfcheck"))
os.environ.setdefault("PROMPTS_FILE", str(BASE / "data" / "_selfcheck" / "_selfcheck.prompts.json"))
shutil.rmtree(os.environ["DATA_DIR"], ignore_errors=True)

from app import analyze, db, face, media, pipeline  # noqa: E402
from app import prompts as prm  # noqa: E402
from app.config import settings  # noqa: E402
from app.qwen import Completion, QwenClient  # noqa: E402
from app.rubric import rubric  # noqa: E402
from app.security import verify_password  # noqa: E402

SAMPLE = BASE / "tests" / "sample_speech.mp4"
SEED = {
    "alice": {"topic": "Will AI replace English teachers?", "requirements": "2-3 minutes"},
    "bob": {"topic": "The value of failure", "requirements": "about 2 minutes"},
}


def ok(label: str, cond: bool, extra: str = "") -> bool:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + extra) if extra else ''}")
    return cond


def main() -> int:
    print(f"模式: {'真实千问' if settings.real_mode else '演示(mock)'}   API key 来源: {settings.api_key_source or '（无）'}")
    print(f"评分标准: {rubric.version}  满分 {rubric.total:g}  维度 {len(rubric.dimensions)} 个")
    if not SAMPLE.exists():
        print(f"缺少测试视频 {SAMPLE}，先运行: ffmpeg -f lavfi -i testsrc2=... 生成")
        return 2

    db.init_db()
    results: list[bool] = []
    print("\n== 1. 建号与登录凭据 ==")
    for name in SEED:
        try:
            db.create_user(name, "pass1234", display_name=name.upper())
        except ValueError:
            pass
    for name in SEED:
        u = db.get_user_by_name(name)
        results.append(ok(f"用户 {name} 已建", u is not None and verify_password("pass1234", u["password_hash"])))
    admin = db.get_user_by_name(settings.admin_username)
    results.append(ok("管理员账号存在", admin is not None and admin["role"] == "admin"))

    print("\n== 2. 上传 + 分析（每人 2 次，用于对比） ==")
    video_ids: dict[str, list[int]] = {}
    for name, meta in SEED.items():
        u = db.get_user_by_name(name)
        ids = []
        for round_no in (1, 2):
            dest = settings.video_dir / f"{u['id']}-r{round_no}-sample.mp4"
            shutil.copyfile(SAMPLE, dest)
            vid = db.create_video(u["id"], f"{name} 第{round_no}次演练", dest.name, str(dest),
                                  dest.stat().st_size, topic=meta["topic"], requirements=meta["requirements"])
            report = pipeline.run_analysis(vid)
            ids.append(vid)
            print(f"  {name} 第{round_no}次 → 总分 {report['total']}/{report['max_total']} "
                  f"等级 {report['band']} 可信度 {report['confidence']} 通道 {len(report['channels'])}")
        video_ids[name] = ids

    last = video_ids["alice"][-1]
    row = db.get_video(last)
    ev = db.get_evaluation(last)
    dims = db.get_dimension_rows(last)
    results.append(ok("状态 done", row["status"] == "done"))
    results.append(ok("有转写文本", bool(row["transcript"].strip()), f"{len(row['transcript'])} 字"))
    results.append(ok("评价记录入库", ev is not None))
    results.append(ok(f"7 维度分项得分齐全", len(dims) == len(rubric.dimensions), f"{len(dims)} 行"))
    report = __import__("json").loads(ev["payload"])
    results.append(ok("优点/缺点/建议/总结非空",
                      bool(report["advantages"]) and bool(report["disadvantages"])
                      and bool(report["suggestions"]) and bool(report["summary"])))
    results.append(ok("满分口径 = 100", abs(report["max_total"] - rubric.total) < 0.01, f"{report['max_total']}"))
    results.append(ok("总分不超过满分", report["total"] <= report["max_total"] + 0.01))
    results.append(ok("时间控制有客观核查", bool(report["timing"]["note"]), report["timing"]["note"]))
    results.append(ok("质控记录可见", bool(report["qc"]), f"{len(report['qc'])} 条"))

    print("\n== 3. 多次上传对比 ==")
    hist = db.history_for_user(db.get_user_by_name("alice")["id"])
    results.append(ok("历史记录 2 条", len(hist) == 2))
    d1 = db.dims_for_video(video_ids["alice"][0])
    d2 = db.dims_for_video(video_ids["alice"][1])
    results.append(ok("两次维度可对齐比较", set(d1) == set(d2) and len(d1) == len(rubric.dimensions)))
    for key in sorted(d1):
        print(f"    {d1[key]['name']:<12} {d1[key]['score']:>6} → {d2[key]['score']:>6} "
              f"(Δ {d2[key]['score'] - d1[key]['score']:+g})")
    issues = db.recurring_issues(db.get_user_by_name("alice")["id"])
    results.append(ok("复发问题查询可用", isinstance(issues, list), f"{len(issues)} 项"))

    print("\n== 4. 管理员视角 ==")
    allv = db.list_all_videos()
    results.append(ok("可见全部用户视频", len(allv) >= 4, f"{len(allv)} 条"))
    stats = db.admin_stats()
    results.append(ok("统计卡片可用", stats["videos"] >= 4 and stats["by_dim"],
                      f"用户 {stats['users']} / 视频 {stats['videos']} / 均分 {stats['avg']}"))

    print("\n== 5. 模型用量（token）记账 ==")
    blank = db.token_usage()
    results.append(ok("演示模式不消耗真实额度", blank["calls"] == 0 or settings.real_mode,
                      f"全站 {blank['calls']} 次调用"))
    qc = QwenClient()
    qc.mark("转录 ASR")._track(
        {"messages": [{"role": "user", "content": [{"type": "input_audio",
                                                    "input_audio": {"data": "A" * 1800}}]}]},
        Completion(text="good morning 各位老师", model="mock", usage=None))
    qc.mark("文本通道")._track(
        {"messages": [{"role": "user", "content": "请按标准评分"}]},
        Completion(text="{}", model="mock",
                   usage={"prompt_tokens": 2000, "completion_tokens": 500, "total_tokens": 2500}))
    usage = qc.usage_summary()
    results.append(ok("响应缺 usage 时估算并打标记", usage["estimated_calls"] == 1,
                      f"估算 {usage['estimated_calls']}/{usage['calls']} 次"))
    pipeline._flush_usage(qc, video_ids["alice"][0])
    mine = db.token_usage(db.get_user_by_name("alice")["id"], per_video=3)
    results.append(ok("落库后可按用户聚合",
                      mine["total"] == usage["total"] and mine["calls"] == usage["calls"],
                      f"{mine['total']} token / {len(mine['by_stage'])} 个阶段"))
    results.append(ok("全站聚合带最近消耗明细",
                      bool(db.token_usage(per_video=5)["videos"]), f"均次 {db.token_usage()['avg_per_video']}"))

    probe = video_ids["alice"][0]
    qa_client = QwenClient()
    qa_client.mark("导师提问")._track(
        {"messages": [{"role": "user", "content": "请针对本稿出 3 道思考题"}]},
        Completion(text='{"questions": ["一", "二", "三"]}', model="mock",
                   usage={"prompt_tokens": 900, "completion_tokens": 100, "total_tokens": 1000}))
    merged = db.add_usage(probe, qa_client.usage_summary())
    row = db.get_video(probe)
    results.append(ok("按需调用累加用量且不覆盖分析统计",
                      merged["total"] == usage["total"] + 1000
                      and merged["calls"] == usage["calls"] + 1
                      and row["total_tokens"] == merged["total"]
                      and row["api_calls"] == merged["calls"],
                      f"{usage['total']} → {merged['total']} token"))
    results.append(ok("累加后阶段明细既留分析又加导师提问",
                      {s["name"] for s in merged["by_stage"]} == {"转录 ASR", "文本通道", "导师提问"},
                      " / ".join(s["name"] for s in merged["by_stage"])))
    results.append(ok("估算次数跨批次累加",
                      merged["estimated_calls"] == usage["estimated_calls"],
                      f"{merged['estimated_calls']} 次估算"))
    again = db.add_usage(probe, QwenClient().usage_summary())
    results.append(ok("零调用累加直接跳过不改写",
                      again == {} and db.get_video(probe)["total_tokens"] == merged["total"]))
    results.append(ok("聚合视图把累加后的用量算进用户口径",
                      db.token_usage(db.get_user_by_name("alice")["id"])["calls"] == merged["calls"],
                      f"{db.token_usage(db.get_user_by_name('alice')['id'])['calls']} 次"))

    print("\n== 6. 失败任务兜底 ==")
    u = db.get_user_by_name("bob")
    bad = db.create_video(u["id"], "坏文件", "missing.mp4", str(settings.video_dir / "missing.mp4"), 0)
    db.set_progress(bad, "queued", "排队中", 3)
    try:
        pipeline.run_analysis(bad)
        results.append(ok("坏文件应抛错", False))
    except Exception as exc:  # noqa: BLE001
        results.append(ok("坏文件抛出可读错误", True, f"{type(exc).__name__}: {exc}"))
    stale = db.reset_stale_jobs()
    results.append(ok("中断任务可清理", stale >= 1, f"{stale} 条"))

    print("\n== 7. 关键帧抽取与 ffmpeg 兼容 ==")
    probe_dir = settings.data_dir / "_frames_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    clip = probe_dir / "cuts.mp4"
    # 三段纯色 + 一段测试图。颜色沿用历史上的白/黑/红：硬切处的帧最可能被抽成空白或被体积门槛
    # 吃掉，等亮切换（红↔绿）将来若重新用于内容感知采样也不会误报故障。
    gen = subprocess.run(
        [media.ffmpeg_exe(), "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=white:s=640x360:d=3",
         "-f", "lavfi", "-i", "color=c=black:s=640x360:d=3",
         "-f", "lavfi", "-i", "color=c=red:s=640x360:d=3",
         "-f", "lavfi", "-i", "testsrc=s=640x360:d=3:r=25",
         "-filter_complex", "[0][1][2][3]concat=n=4:v=1:a=0", "-r", "25", str(clip)],
        capture_output=True, text=True, timeout=300)
    info = media.probe(clip) if gen.returncode == 0 and clip.exists() else None
    results.append(ok("探针视频（3 次硬切）可生成", info is not None and info.duration > 10,
                      f"{info.duration:.1f}s" if info else " ".join(gen.stderr.split())[:120]))
    if info:
        budget = media.frame_budget(info.duration)
        step = info.duration / budget
        frames = media.extract_frames_timed(clip, probe_dir / "uni", info.duration)
        results.append(ok("硬切夹具不被误删（抽满预算张数）",
                          len(frames.paths) == budget, f"{len(frames.paths)}/{budget} 张"))
        results.append(ok("帧文件按 frame_NN 连续编号",
                          [p.name for p in frames.paths] == [f"frame_{i:02d}.jpg" for i in range(budget)],
                          frames.paths[0].name + " … " + frames.paths[-1].name))
        deltas = [b - a for a, b in zip(frames.stamps, frames.stamps[1:])]
        results.append(ok("相邻帧间隔恒定（纯均匀，不做场景补插）",
                          bool(deltas) and max(deltas) - min(deltas) <= 0.05,
                          f"{step:.2f}s，抖动 {max(deltas) - min(deltas):.3f}s" if deltas else "单帧"))
        results.append(ok("采样点落在格子中点而非片头", abs(frames.stamps[0] - step / 2) <= 0.05,
                          f"首帧 {frames.stamps[0]:.2f}s / 期望 {step / 2:.2f}s"))
        results.append(ok("时间戳与帧一一对应且单调递增",
                          len(frames.stamps) == len(frames.paths)
                          and all(b > a for a, b in zip(frames.stamps, frames.stamps[1:])),
                          " · ".join(f"{t:g}s" for t in frames.stamps)))
    shutil.rmtree(probe_dir, ignore_errors=True)

    print("\n== 8. 抽帧张数自适应 + 微表情客观测量门控 ==")
    cap = media._FRAME_HARD_CAP
    curve = [(0, settings.frame_count), (1, 1), (8, 2), (42, 8), (145, 29),
             (3600, min(settings.frame_count_max, cap))]
    bad = [d for d, want in curve if media.frame_budget(d) != want]
    results.append(ok(f"张数随时长推导（每 {settings.frame_interval}s 一张，只封上限不抬下限）",
                      not bad, " ".join(f"{d}s→{media.frame_budget(d)}" for d, _ in curve)))
    results.append(ok("时长未知时退回 FRAME_COUNT",
                      media.frame_budget(0) == settings.frame_count,
                      f"FRAME_COUNT={settings.frame_count}"))
    results.append(ok("再长的视频也不越过视觉单请求张数硬顶",
                      media.frame_budget(86400) == cap and media.frame_budget(123456) == cap,
                      f"硬顶 {cap} 张"))

    probe_dir = settings.data_dir / "_face_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    got = media.extract_frames_timed(SAMPLE, probe_dir / "sample", media.probe(SAMPLE).duration)
    results.append(ok("样例视频按自适应预算抽帧", len(got.paths) == media.frame_budget(media.probe(SAMPLE).duration),
                      f"{len(got.paths)} 张 / {media.probe(SAMPLE).duration:.0f}s"))
    # 合成画面里没有人脸，正确行为是「如实说没测准」，而不是编一个正脸率出来。
    m = face.measure(got.paths, got.stamps)
    results.append(ok("人脸测量层可用（缺依赖会让下面一条失去意义）", face.available(),
                      "cv2 + 级联就绪" if face.available() else face.unavailable_reason()))
    results.append(ok("检出率低于门限时不输出朝向指标",
                      m.available is False and m.frontal_ratio is None and m.down_ratio is None,
                      f"检出率 {m.face_rate if m.face_rate is not None else '—'}"))
    results.append(ok("降级带可解释理由", bool(m.reason.strip()), m.reason[:64]))
    results.append(ok("降级文案不夹带虚构数字", "正脸率" not in m.headline()))
    results.append(ok("降级时提示模型不要引用未测量的数字",
                      "不要引用任何未经测量的数字" in m.evidence_block()))
    stored = __import__("json").loads(db.get_evaluation(video_ids["alice"][0])["payload"])
    results.append(ok("分析报告落库带回真实时刻与测量结果",
                      len(stored.get("stamps") or []) == len(stored.get("frames") or []) > 0
                      and bool(stored.get("face")),
                      f"{len(stored.get('stamps') or [])} 个时刻"))
    results.append(ok("门控未通过时 videos.frontal_ratio 留空而不是 0",
                      db.get_video(video_ids["alice"][0])["frontal_ratio"] is None))
    shutil.rmtree(probe_dir, ignore_errors=True)

    print("\n== 9. 网页化配置：.env 读写、校验与热刷新 ==")
    from app import config as cfg
    scratch = settings.data_dir / "_env_probe.env"
    keep_file, env_before = cfg.ENV_FILE, dict(os.environ)
    scratch.write_text("# 头注释\nLLM_MODE=mock\n\n# 抽帧\nFRAME_INTERVAL=5\n"
                       "USE_AUDIO_CHANNEL=1\nMAX_VIDEO_MB=500\n", encoding="utf-8")
    cfg.ENV_FILE = scratch
    try:
        results.append(ok("设置项声明表覆盖 28 个键", len(cfg.ENV_FIELDS) == 28, f"{len(cfg.ENV_FIELDS)} 项"))
        fields = {f.key: f for f in cfg.ENV_FIELDS}
        results.append(ok("并发三键已声明且默认值够 10 人同时用",
                          fields["MAX_ANALYZERS"].default == "10"
                          and int(fields["MAX_LLM_REQUESTS"].default) >= 20
                          and int(fields["WEB_THREADS"].default) >= 8,
                          f"{fields['MAX_ANALYZERS'].default} 路 / "
                          f"{fields['MAX_LLM_REQUESTS'].default} 请求 / "
                          f"{fields['WEB_THREADS'].default} 线程"))
        results.append(ok("线程数标明需重启才生效，其余并发项即时生效",
                          "重启" in fields["WEB_THREADS"].note
                          and "重启" not in fields["MAX_ANALYZERS"].note, fields["WEB_THREADS"].note))
        parsed = cfg.read_env_file()
        results.append(ok("解析 .env 时跳过注释与空行",
                          set(parsed) == {"LLM_MODE", "FRAME_INTERVAL", "USE_AUDIO_CHANNEL", "MAX_VIDEO_MB"},
                          " ".join(sorted(parsed))))
        n = cfg.write_env_file({"FRAME_INTERVAL": "3", "NEW_KEY": "x"})
        text = scratch.read_text(encoding="utf-8")
        results.append(ok("写回保留注释与原有键顺序，新键追加末尾",
                          text.startswith("# 头注释") and "# 抽帧" in text
                          and text.index("LLM_MODE") < text.index("USE_AUDIO_CHANNEL") < text.index("NEW_KEY"),
                          f"{n} 项"))
        results.append(ok("写回只替换命中的键", "FRAME_INTERVAL=3\n" in text and "FRAME_INTERVAL=5" not in text))

        crlf_file = settings.data_dir / "_env_crlf.env"
        crlf_file.write_bytes(b"# CRLF \xe7\x89\x88\nLLM_MODE=mock\r\nFRAME_INTERVAL=5\r\n\r\n")
        cfg.ENV_FILE = crlf_file
        cfg.write_env_file({"FRAME_INTERVAL": "4"})
        raw = crlf_file.read_bytes()
        results.append(ok("CRLF 文件写回后仍是 CRLF，不整文件换行",
                          raw.count(b"\r\n") == 3 and raw.count(b"\n") == 3, repr(raw[:40])))
        lf_file = settings.data_dir / "_env_lf.env"
        lf_file.write_bytes(b"# LF\nLLM_MODE=mock\nFRAME_INTERVAL=5\n")
        cfg.ENV_FILE = lf_file
        cfg.write_env_file({"FRAME_INTERVAL": "6"})
        lf_raw = lf_file.read_bytes()
        results.append(ok("LF 文件写回后不会变成 CRLF",
                          b"\r\n" not in lf_raw and b"FRAME_INTERVAL=6\n" in lf_raw, repr(lf_raw[-40:])))
        cfg.ENV_FILE = scratch

        upd, errs = cfg.validate_env_updates({
            "FRAME_COUNT_MAX": "200", "FRAME_WIDTH": "abc",
            "FACE_MIN_DETECT": "1.4", "LLM_MODE": "nope", "ASR_ENGINE": "whisper",
            "DASHSCOPE_API_KEY": "", "SECRET_KEY": "   ", "USE_AUDIO_CHANNEL": ""})
        results.append(ok("帧数超过视觉单请求上限被拦下",
                          any("不得大于 100" in e for e in errs), "；".join(errs)))
        results.append(ok("非数字与越界分别报错",
                          any("需要填数字" in e for e in errs) and any("不得大于 1" in e for e in errs),
                          "；".join(errs)))
        results.append(ok("非法枚举值报错", any("取值只能是" in e for e in errs), "；".join(errs)))
        results.append(ok("密钥留空即不改，空白等同留空",
                          "DASHSCOPE_API_KEY" not in upd and "SECRET_KEY" not in upd))
        results.append(ok("表单里缺失或未勾选的开关按关闭处理",
                          upd["USE_AUDIO_CHANNEL"] == "0" and upd["USE_FACE_METRICS"] == "0"))
        results.append(ok("选 whisper 引擎时补默认 whisper 规模",
                          upd["ASR_ENGINE"] == "whisper" and upd["WHISPER_MODEL_SIZE"] == "small"))
        cleared, _ = cfg.validate_env_updates({"DASHSCOPE_API_KEY": ""}, ("DASHSCOPE_API_KEY",))
        results.append(ok("勾选「清除覆盖」才写入空值", cleared.get("SECRET_KEY", "·") == "·"
                          and cleared.get("DASHSCOPE_API_KEY") == ""))
        blank, _ = cfg.validate_env_updates({"LLM_MODE": "", "ASR_ENGINE": ""})
        results.append(ok("下拉框选「不覆盖」时写入空值而不是默认值",
                          blank["LLM_MODE"] == "" and blank["ASR_ENGINE"] == ""))
        results.append(ok("密钥掩码只留首尾", cfg.mask_secret("sk-1234567890abcdef1234") == "sk-1" + "*" * 12 + "1234"
                          and cfg.mask_secret("short") == "*****", cfg.mask_secret("sk-1234567890abcdef1234")))

        cfg.save_settings({"MAX_VIDEO_MB": "123", "FRAME_INTERVAL": "9", "USE_AUDIO_CHANNEL": "0"})
        results.append(ok("保存后运行时立即反映新值（无需重启）",
                          settings.max_video_bytes == 123 * 1024 * 1024
                          and settings.frame_interval == 9 and settings.use_audio_channel is False,
                          f"{settings.max_video_bytes // (1024 * 1024)}MB / 每 {settings.frame_interval}s"))
        results.append(ok("值未变化时不重复写盘", cfg.save_settings({"MAX_VIDEO_MB": "123"}) == []))
        results.append(ok("热刷新只重算环境变量字段，静态字段与单例不变",
                          settings.base_dir == cfg.BASE_DIR and settings.rubric_file.exists()
                          and cfg.settings is settings))
        results.append(ok("save_settings 只回报真正变化的键",
                          cfg.save_settings({"MAX_VIDEO_MB": "456", "LLM_MODE": "mock"}) == ["MAX_VIDEO_MB"]))
        flat = {r["field"].key: r for g in cfg.settings_overview() for r in g["rows"]}
        results.append(ok("来源徽章区分 .env 与上游默认",
                          flat["MAX_VIDEO_MB"]["source"] == ".env"
                          and flat["CHAT_MODEL"]["source"] != ".env",
                          f"MAX_VIDEO_MB←{flat['MAX_VIDEO_MB']['source']} / CHAT_MODEL←{flat['CHAT_MODEL']['source']}"))
        results.append(ok("输入框只回填真正写在 .env 里的值",
                          all((r["raw"] == "") == (r["source"] != ".env")
                              for k, r in flat.items() if r["field"].kind not in ("secret", "bool"))))
        results.append(ok("密钥生效值不回填也不明文显示",
                          flat["DASHSCOPE_API_KEY"]["value"] == ""
                          and "*" in (flat["DASHSCOPE_API_KEY"]["shown"] or "*")))
        results.append(ok("字节存储按 scale 换算成 MB 显示",
                          flat["MAX_VIDEO_MB"]["value"] == "456" and flat["MAX_VIDEO_MB"]["field"].scale > 1))
        pend = {r["field"].key: r for g in cfg.settings_overview({"FRAME_INTERVAL": "7"}) for r in g["rows"]}
        results.append(ok("校验失败回显标记为待保存",
                          pend["FRAME_INTERVAL"]["pending"] and pend["FRAME_INTERVAL"]["value"] == "7"
                          and not pend["LLM_MODE"]["pending"]))

        cfg.write_env_file({"QWEN_BASE_URL": "https://example.com/v1/"})
        cfg.refresh_runtime()
        results.append(ok("网关地址按原样写回，运行时自动去掉尾斜杠",
                          cfg.read_env_file()["QWEN_BASE_URL"] == "https://example.com/v1/"
                          and settings.base_url == "https://example.com/v1", settings.base_url))
    finally:
        cfg.ENV_FILE = keep_file
        for key in set(os.environ) - set(env_before):
            del os.environ[key]
        os.environ.update(env_before)
        cfg.refresh_runtime()
    results.append(ok("测试后运行时恢复原值", settings.max_video_bytes != 456 * 1024 * 1024
                      and cfg.ENV_FILE == keep_file, str(cfg.ENV_FILE)))

    print("\n== 10. 账号录入：模板、解析与逐行判定 ==")
    from io import BytesIO
    from openpyxl import Workbook
    from app import accounts

    def to_xlsx(rows: list) -> bytes:
        wb = Workbook()
        ws = wb.active
        ws.title = accounts.SHEET
        for r in rows:
            ws.append(r)
        buf = BytesIO()
        wb.save(buf)
        return buf.getvalue()

    def rejected(fn, want: str) -> bool:
        try:
            fn()
        except accounts.AccountError as exc:
            return want in str(exc)
        return False

    tpl = accounts.build_template()
    results.append(ok("模板生成的是 xlsx（ZIP 魔数 + 双表）",
                      tpl[:2] == b"PK" and len(tpl) > 2048
                      and all(h in " ".join(accounts.NOTES) for h in accounts.HEADERS), f"{len(tpl)} 字节"))
    tplan = accounts.plan_users(accounts.read_table(tpl, accounts.TEMPLATE_NAME))
    results.append(ok("模板原样上传只会跳过示例行，不会建号",
                      len(tplan) == len(accounts.EXAMPLES)
                      and all(r.status == "skip" for r in tplan)))
    tsum = accounts.summarize(tplan, accounts.TEMPLATE_NAME)
    results.append(ok("回执可区分跳过与退回（空表不算出错）",
                      tsum["skipped"] == len(accounts.EXAMPLES) and tsum["created"] == []
                      and tsum["failed"] == [] and tsum["ok"] is True, f"跳过 {tsum['skipped']} 行"))

    results.append(ok("数字单元格去掉 .0 尾巴，文本去首尾空格",
                      accounts.cell_text(20250101) == "20250101"
                      and accounts.cell_text(20250101.0) == "20250101"
                      and accounts.cell_text("  zhang3 ") == "zhang3"
                      and accounts.cell_text(None) == "" and accounts.cell_text(True) == "1",
                      accounts.cell_text(20250101.0)))

    dirty = [
        list(accounts.HEADERS),
        ["ok-01", "pw-ok-01", "小赵"],
        ["", "pw-no-name", "缺用户名"],
        ["短", "pw-short-name", "中文名不合法"],
        ["kid2", "123", "口令过短"],
        ["dup", "pw-dup-123", "库里已有"],
        ["multi", "pw-multi-1", "多余列应被忽略", "第四列", "第五列"],
        ["multi", "pw-multi-2", "文件内第二次出现"],
        [20250101, 20250101, "学号当用户名"],
        ["", "", ""],
        list(accounts.HEADERS),
    ]
    dplan = accounts.plan_users(dirty, existing=["dup"])
    by = {r.line: r for r in dplan}
    results.append(ok("合法行判为可导入，多余列与数字用户名被纠正",
                      by[2].status == "ready" and by[7].status == "ready"
                      and by[7].display_name == "多余列应被忽略"
                      and by[9].status == "ready" and by[9].username == "20250101",
                      f"第 9 行 {by[9].username}"))
    results.append(ok("三列全空的行不进入任何统计", 10 not in by, f"{len(dplan)} 行有判定"))
    results.append(ok("表头若在中间再次出现按跳过处理", by[11].status == "skip"))
    results.append(ok("缺项与非法名逐条退回并给出原因",
                      "缺用户名" in by[3].reason and "字母" in by[4].reason
                      and "至少 6 位" in by[5].reason and "已存在" in by[6].reason,
                      by[4].reason))
    results.append(ok("文件内重名指向首次出现的行号", "第 7 行" in by[8].reason, by[8].reason))
    dsum = accounts.summarize(dplan, "名单.xlsx")
    results.append(ok("回执按行号列出退回清单并标记整表有错",
                      dsum["ok"] is False and len(dsum["failed"]) == 5
                      and [f["line"] for f in dsum["failed"]] == [3, 4, 5, 6, 8],
                      f"退回 {len(dsum['failed'])} 行"))
    for r in dplan:
        if r.status == "ready":
            r.status = "created"
    csum = accounts.summarize(dplan, "名单.xlsx")
    results.append(ok("导入后回执新建清单与用户名一致",
                      csum["created"] == ["ok-01", "multi", "20250101"], "、".join(csum["created"])))

    results.append(ok("非 .xlsx 后缀直接拒收并给出另存指引",
                      rejected(lambda: accounts.read_table(b"ab,cd", "名单.csv"), ".xlsx")))
    results.append(ok("空文件拒收", rejected(lambda: accounts.read_table(b"", "名单.xlsx"), "空的")))
    results.append(ok("伪装成 xlsx 的坏文件拒收",
                      rejected(lambda: accounts.read_table(b"not a zip" * 32, "名单.xlsx"), "读不了")))
    results.append(ok("超过体积上限拒收",
                      rejected(lambda: accounts.read_table(b"x" * (accounts.MAX_BYTES + 1), "名单.xlsx"), "MB 上限")))
    empty = Workbook()
    empty.active.title = accounts.SHEET
    buf = BytesIO()
    empty.save(buf)
    results.append(ok("有格式无内容的表拒收",
                      rejected(lambda: accounts.read_table(buf.getvalue(), "名单.xlsx"), "没有内容")))
    over = [list(accounts.HEADERS)] + [[f"stu{i:05d}", "pw-pass-123", ""] for i in range(accounts.MAX_ROWS + 1)]
    results.append(ok("超过行数上限拒收并建议分次导入",
                      rejected(lambda: accounts.read_table(to_xlsx(over), "名单.xlsx"), "上限")))

    print("\n== 11. 大模型提示词：声明、差量落盘、校验与生效 ==")
    prm.clear_cache()
    results.append(ok("提示词声明表覆盖 19 块 / 7 组",
                      len(prm.PROMPT_FIELDS) == 19 and len(prm.PROMPT_GROUPS) == 7,
                      f"{len(prm.PROMPT_FIELDS)} 块 / {len(prm.PROMPT_GROUPS)} 组"))
    results.append(ok("内置默认自带所声明的占位符",
                      not (bad := [f.key for f in prm.PROMPT_FIELDS
                                   if not all(("{" + t + "}") in prm.DEFAULTS[f.key] for t in f.tokens)]),
                      "、".join(bad)))
    results.append(ok("内置默认保留必须存在的输出字段名",
                      not (bad := [f.key for f in prm.PROMPT_FIELDS
                                   if not all(lit in prm.DEFAULTS[f.key] for lit in f.must_contain)]),
                      "、".join(bad)))
    results.append(ok("每块默认非空、无回车、normalize 幂等",
                      all(v and "\r" not in v and v == v.strip() == prm.normalize(v)
                          for v in prm.DEFAULTS.values())))
    results.append(ok("normalize 统一换行并剥掉首尾空行", prm.normalize("  \r\na\r\nb\n\n  ") == "a\nb"))
    results.append(ok("textarea 往返稳定：保存过的文本再读一次不变",
                      prm.normalize(prm.DEFAULTS["narrative_context"]) == prm.DEFAULTS["narrative_context"]))
    rendered = prm.render("text_tail", keys="K1", schema="里面写着 {keys} 的转写")
    results.append(ok("render 单次替换，注入内容不会被二次展开",
                      "里面写着 {keys} 的转写" in rendered and "K1" in rendered, rendered[:60]))

    changed, errs = prm.save({f.key: prm.DEFAULTS[f.key] for f in prm.PROMPT_FIELDS})
    results.append(ok("全部原样提交不产生覆盖文件",
                      changed == [] and errs == [] and not prm.PROMPTS_FILE.exists()))
    changed, errs = prm.save({"common_rules": "自定义硬性规则：一条证据即可。"})
    results.append(ok("改一块即落盘，且当场对读生效",
                      changed == ["common_rules"] and errs == []
                      and prm.get("common_rules").startswith("自定义硬性规则")))
    results.append(ok("未改的块仍走内置默认", prm.get("system") == prm.DEFAULTS["system"]))
    results.append(ok("自定义计数只算真正不同的块", prm.custom_count() == 1))
    disk = json.loads(prm.PROMPTS_FILE.read_text(encoding="utf-8"))
    results.append(ok("文件里只存差异块，不整份复制默认",
                      list(disk["prompts"]) == ["common_rules"] and disk["_meta"]["version"] == 1,
                      str(list(disk["prompts"]))))
    changed, _ = prm.save({"common_rules": prm.DEFAULTS["common_rules"]})
    results.append(ok("把文本改回原样即自动回到内置默认",
                      changed == ["common_rules"] and prm.custom_count() == 0
                      and prm.get("common_rules") == prm.DEFAULTS["common_rules"]))

    _, errs = prm.validate({"text_tail": "按下面结构输出 JSON："})
    results.append(ok("删掉占位符被拒绝并点名缺哪几个", any("缺少系统占位符" in e for e in errs), "；".join(errs)))
    _, errs = prm.validate({"text_tail": prm.DEFAULTS["text_tail"] + "\n另需 {bogus}"})
    results.append(ok("严格块里的未知占位符被拒绝", any("不是本块可用的占位符" in e for e in errs), "；".join(errs)))
    results.append(ok("非严格块允许字面花括号",
                      prm.validate({"common_rules": prm.DEFAULTS["common_rules"] + " {whatever}"})[1] == []))
    _, errs = prm.validate({"narrative_schema": '{\n  "advantages": [],\n  "other": 1\n}'})
    results.append(ok("改掉反馈输出结构里的键名会被拦下", any("必须保留" in e for e in errs), "；".join(errs)))
    _, errs = prm.validate({"common_rules": "字" * 6001})
    results.append(ok("超出单块字数上限被拦下", any("不得超过" in e for e in errs)))
    before = prm.overrides()
    ch, errs = prm.save({"micro_rules": "只看整体印象。", "text_tail": "缺占位符的草稿"})
    results.append(ok("任一块非法则整份不写盘",
                      ch == [] and bool(errs) and prm.overrides() == before))

    prm.save({"micro_rules": "只看整体印象，不必逐帧举证。"})
    restored = prm.reset_all()
    results.append(ok("一键恢复内置默认",
                      restored == ["micro_rules"] and prm.get("micro_rules") == prm.DEFAULTS["micro_rules"]
                      and prm.custom_count() == 0, str(restored)))
    prm.PROMPTS_FILE.unlink(missing_ok=True)
    prm.clear_cache()
    results.append(ok("删掉覆盖文件等于全部回默认", prm.get("text_tail") == prm.DEFAULTS["text_tail"]))

    rows = {r["key"]: r for g in prm.overview() for r in g["rows"]}
    results.append(ok("设置页视图列出全部块并带字数/行数",
                      len(rows) == len(prm.PROMPT_FIELDS)
                      and all(r["chars"] == len(r["text"]) and r["lines"] >= 1 for r in rows.values())))
    prm.save({"asr_instruction": "只输出转写内容。"})
    rows = {r["key"]: r for g in prm.overview() for r in g["rows"]}
    results.append(ok("视图标记出自定义块", rows["asr_instruction"]["custom"]
                      and not rows["system"]["custom"]))
    echo = {r["key"]: r for g in prm.overview({"text_tail": "只写 {keys}"}) for r in g["rows"]}
    results.append(ok("校验失败时按草稿回显并标出缺失占位符",
                      echo["text_tail"]["text"] == "只写 {keys}" and echo["text_tail"]["missing"] == ["schema"],
                      str(echo["text_tail"]["missing"])))
    results.append(ok("草稿留空按内置默认回显", echo["system"]["text"] == prm.DEFAULTS["system"]))
    prm.reset_all()
    prm.PROMPTS_FILE.unlink(missing_ok=True)
    prm.clear_cache()

    class _Spy:
        """记录送进模型的正文，然后停下：只验拼装，不需要模型返回。"""

        def __init__(self) -> None:
            self.calls: dict[str, str] = {}

        def _stop(self, kind: str, prompt: str, system: str = "") -> None:
            self.calls[kind] = prompt
            self.calls[kind + ":system"] = system
            raise RuntimeError("captured")

        def chat(self, prompt, system="", **kw):
            self._stop("chat", prompt, system)

        def vision(self, prompt, images, system="", **kw):
            self._stop("vision", prompt, system)

        def audio(self, prompt, audio, system="", **kw):
            self._stop("audio", prompt, system)

    spy = _Spy()
    timing = analyze.check_timing(120, "Will AI replace teachers?", "2-3 minutes")
    for label, call in (
        ("text", lambda: analyze.run_text_channel(spy, rubric, "逐字转写的演讲内容", "题目", "2-3 minutes", timing)),
        ("vision", lambda: analyze.run_vision_channel(spy, rubric, [Path("f01.jpg"), Path("f02.jpg")],
                                                      "题目", timing, transcript="供理解背景的转写",
                                                      stamps=[12.0, 96.0])),
        ("audio", lambda: analyze.run_audio_channel(spy, rubric, Path("a.wav"), "供对照的转写", timing)),
    ):
        try:
            call()
        except RuntimeError:
            pass
    text_p, vision_p, audio_p = spy.calls.get("chat", ""), spy.calls.get("vision", ""), spy.calls.get("audio", "")
    results.append(ok("文字通道正文含标准声明、通用规则、转写与输出结构",
                      all(s in text_p for s in ("评分标准", "硬性规则", "逐字转写的演讲内容", "输出 JSON"))))
    results.append(ok("文字通道以 system 角色发送总则",
                      spy.calls.get("chat:system") == prm.DEFAULTS["system"]))
    results.append(ok("画面/语音通道不重复挂 system 角色",
                      spy.calls.get("vision:system", "") == "" and spy.calls.get("audio:system", "") == ""))
    results.append(ok("画面通道正文含逐帧判读规则、时间点与转写节选",
                      all(s in vision_p for s in ("眼神与表情", "frame@01:36", "供理解背景的转写"))))
    results.append(ok("语音通道正文含「以音频为准」与音素层面要求",
                      "以音频为准" in audio_p and "供对照的转写" in audio_p and "音素" in audio_p))
    results.append(ok("拼装结果不留双空行或空块残迹",
                      "\n\n\n\n" not in text_p and "\n\n\n\n" not in vision_p))
    prm.save({"common_rules": "自定义硬性规则甲：只写一条。"})
    try:
        analyze.run_text_channel(spy, rubric, "逐字转写的演讲内容", "题目", "2-3 minutes", timing)
    except RuntimeError:
        pass
    results.append(ok("管理员改写的提示词直接进入下一次请求正文",
                      "自定义硬性规则甲" in spy.calls["chat"] and "硬性规则：" not in spy.calls["chat"]))
    prm.reset_all()
    prm.PROMPTS_FILE.unlink(missing_ok=True)
    prm.clear_cache()

    print("\n== 12. 导师提问：整组重建、作答回写与兜底出题 ==")
    vid_qa = video_ids["alice"][0]
    uid_qa = db.get_video(vid_qa)["user_id"]
    report_qa = json.loads(db.get_evaluation(vid_qa)["payload"])
    auto_qa = db.get_questions(vid_qa)
    results.append(ok("分析完成后自动带出 3 道待作答的提问",
                      len(auto_qa) == 3 and [q["idx"] for q in auto_qa] == [1, 2, 3]
                      and all(q["status"] == "open" and q["question"].strip() for q in auto_qa),
                      f"{len(auto_qa)} 条"))
    written = db.save_questions(vid_qa, uid_qa, ["第一题", "   ", "第二题", "第三题", "第四题"])
    rows_qa = db.get_questions(vid_qa)
    results.append(ok("保存提问：空文本跳过、只取前 3 条、idx 从 1 连号",
                      written == 3 and [q["idx"] for q in rows_qa] == [1, 2, 3]
                      and [q["question"] for q in rows_qa] == ["第一题", "第二题", "第三题"],
                      f"{written} 条"))
    results.append(ok("新提问初始状态是 open 且无作答",
                      all(q["status"] == "open" and not q["answer"] and not q["ai_comment"]
                          for q in rows_qa)))
    results.append(ok("回写作答成功", db.save_qa_answer(vid_qa, 2, "我的作答", "导师点评") is True))
    answered = db.get_questions(vid_qa)[1]
    results.append(ok("作答/点评/状态一起落库",
                      answered["answer"] == "我的作答" and answered["ai_comment"] == "导师点评"
                      and answered["status"] == "answered"))
    results.append(ok("序号不存在时回写作答返回 False",
                      db.save_qa_answer(vid_qa, 9, "x", "y") is False))
    again = db.save_questions(vid_qa, uid_qa, ["新第一题", "新第二题", "新第三题"])
    rows_qa = db.get_questions(vid_qa)
    results.append(ok("重复出题整组重建，不累积旧题也不留旧作答",
                      again == 3 and len(rows_qa) == 3
                      and all(q["status"] == "open" and not q["answer"] for q in rows_qa)))
    db.save_questions(vid_qa, uid_qa, ["问" * 600])
    results.append(ok("超长提问截断到 500 字",
                      len(db.get_questions(vid_qa)[0]["question"]) == 500))
    db.save_questions(vid_qa, uid_qa, ["第一题", "第二题", "第三题"])
    db.save_qa_answer(vid_qa, 1, "我的作答", "导师点评")
    db.save_evaluation(db.get_video(vid_qa), report_qa, model="selfcheck")
    results.append(ok("重新分析（save_evaluation）清空提问与作答", db.get_questions(vid_qa) == []))

    probe_vid = db.create_video(uid_qa, "外键探针", "qa-probe.mp4", "qa-probe.mp4", 1)
    db.save_questions(probe_vid, uid_qa, ["探针一", "探针二", "探针三"])
    db.delete_video(probe_vid)
    with db.get_conn() as conn:
        left = conn.execute("SELECT COUNT(*) FROM qa_turns WHERE video_id = ?",
                            (probe_vid,)).fetchone()[0]
    results.append(ok("删除视频级联清掉导师提问（ON DELETE CASCADE）", left == 0, f"残留 {left} 条"))
    probe_uid = db.create_user("qa_probe", "pass1234", display_name="探针")
    db.save_questions(db.create_video(probe_uid, "探针视频", "p.mp4", "p.mp4", 1),
                      probe_uid, ["探针一", "探针二", "探针三"])
    db.delete_user(probe_uid)
    with db.get_conn() as conn:
        left = conn.execute("SELECT COUNT(*) FROM qa_turns WHERE user_id = ?", (probe_uid,)).fetchone()[0]
    results.append(ok("删除用户跨两级级联清掉导师提问", left == 0, f"残留 {left} 条"))

    fb = analyze.fallback_questions(report_qa, report_qa.get("topic") or "")
    results.append(ok("兜底出题永远 3 条且非空", len(fb) == 3 and all(q.strip() for q in fb),
                      f"{len(fb)} 条"))
    fb_empty = analyze.fallback_questions({"dimensions": [], "suggestions": []}, "")
    results.append(ok("报告缺字段时仍 3 条并回退到「本次演讲」",
                      len(fb_empty) == 3 and all("本次演讲" in q for q in fb_empty)))
    fb_topic = analyze.fallback_questions({}, "My Topic")
    results.append(ok("题目写进兜底问题里", len(fb_topic) == 3 and all("My Topic" in q for q in fb_topic)))

    class _QAStub:
        def __init__(self, text: str) -> None:
            self.text = text
            self.prompts: list[str] = []

        def chat(self, prompt, system="", **kw):
            self.prompts.append(prompt)
            return Completion(text=self.text, model="mock", usage=None)

    good = json.dumps({"questions": ["甲题？", " 乙题？ ", "丙题？", "丁题？"]}, ensure_ascii=False)
    qs = analyze.build_questions(_QAStub(good), rubric, report_qa, "转写", "题目", "2-3 分钟")
    results.append(ok("模型给足 3 条即采用并清洗顺序、截掉多余",
                      qs == ["甲题？", "乙题？", "丙题？"], "、".join(qs)))
    short = analyze.build_questions(_QAStub('{"questions": ["只有一题"]}'), rubric, report_qa,
                                    "转写", "题目")
    results.append(ok("模型条数不足整批改用兜底题",
                      len(short) == 3 and "只有一题" not in short
                      and all(q.strip() for q in short), "、".join(short)))
    broken = analyze.build_questions(_QAStub("这里没有 JSON"), rubric, report_qa, "转写", "题目")
    results.append(ok("模型返回无法解析时仍返回 3 条", len(broken) == 3))
    spy_q = _QAStub(good)
    analyze.build_questions(spy_q, rubric, report_qa, "转写正文", "题目", "2-3 分钟")
    results.append(ok("出题正文带主题、要求与本稿评价结果",
                      all(s in spy_q.prompts[0] for s in ("题目", "2-3 分钟", "转写正文"))))
    results.append(ok("出题正文不重复塞关键帧与逐帧测量",
                      '"frames"' not in spy_q.prompts[0] and '"stamps"' not in spy_q.prompts[0]))

    spy_r = _QAStub(json.dumps({"comments": ["点评甲。", " 点评乙。 "]}, ensure_ascii=False))
    cm = analyze.review_answers(spy_r, report_qa, [("问题一", "作答一"), ("问题二", "作答二")])
    results.append(ok("点评按题序返回且与作答等长", cm == ["点评甲。", "点评乙。"], "、".join(cm)))
    results.append(ok("点评正文带上学生作答",
                      "问题一" in spy_r.prompts[0] and "作答一" in spy_r.prompts[0]))
    few = analyze.review_answers(_QAStub('{"comments": ["只有一条"]}'), report_qa,
                                 [("问题一", "作答一"), ("问题二", "作答二")])
    results.append(ok("点评条数不符整批回兜底文案",
                      len(few) == 2 and all(c == analyze.QA_ANSWER_FALLBACK for c in few)))
    results.append(ok("模型异常时点评仍与作答等长",
                      len(analyze.review_answers(_QAStub(""), report_qa, [("问题一", "作答一")])) == 1))
    results.append(ok("无作答时点评返回空表", analyze.review_answers(_QAStub(good), report_qa, []) == []))

    qa_keys = ("qa_system", "qa_context", "qa_schema", "qa_review_context", "qa_review_schema")
    results.append(ok("导师提问提示词组已声明 5 块", all(k in prm.PROMPT_BY_KEY for k in qa_keys)))
    results.append(ok("导师提问自成一组且带说明",
                      any(g[0] == "导师提问" and len(g[2]) == 5 for g in prm.PROMPT_GROUPS)))
    q_render = prm.render("qa_context", topic="题目", requirements="要求", result='{"total": 1}',
                          transcript="讲稿全文", schema=prm.get("qa_schema"))
    results.append(ok("出题块渲染后不残留占位符",
                      not any(("{" + t + "}") in q_render
                              for t in ("topic", "requirements", "result", "transcript", "schema"))))
    r_render = prm.render("qa_review_context", qa="问题一：Q\n学生作答一：A",
                          schema=prm.get("qa_review_schema"))
    results.append(ok("点评块渲染后不残留占位符", "{qa}" not in r_render and "{schema}" not in r_render))
    db.save_questions(vid_qa, uid_qa, ["最终一题", "最终二题", "最终三题"])

    print("\n" + "=" * 56)
    passed = sum(1 for r in results if r)
    print(f"通过 {passed}/{len(results)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
