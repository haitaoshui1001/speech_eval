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
from app.qwen import Completion, QwenClient, QwenError  # noqa: E402
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

    print("\n== 5. 模型用量（token + 字符）记账 ==")
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

    alice = db.get_user_by_name("alice")["id"]
    mix = QwenClient()
    mix.mark("文字稿修订")._track(
        {"messages": [{"role": "user", "content": "请找出稿中错字与术语误用"}]},
        Completion(text='{"items": []}', model="mock",
                   usage={"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500}))
    mix.mark("朗读合成").track_characters("qwen3-tts-flash", 620)
    mix.mark("朗读合成").track_characters("qwen3-tts-flash", 480, estimated=True)
    mix_usage = mix.usage_summary()
    results.append(ok("字符与 token 双轴并存且不互相污染",
                      mix_usage["characters"] == 1100 and mix_usage["total"] == 1500,
                      f"{mix_usage['total']} token + {mix_usage['characters']} 字符"))
    results.append(ok("合成未回 usage 时折算并打估算标记",
                      mix_usage["estimated_calls"] == 1
                      and {s["name"]: s["estimated"] for s in mix_usage["by_stage"]}["朗读合成"] == 1,
                      f"{mix_usage['estimated_calls']} 次估算"))
    st = {s["name"]: s for s in mix_usage["by_stage"]}
    results.append(ok("合成行只记字符、评判行只记 token",
                      st["朗读合成"]["characters"] == 1100 and st["朗读合成"]["total"] == 0
                      and st["文字稿修订"]["characters"] == 0,
                      " / ".join(f"{k}:{v['characters']}字符" for k, v in st.items())))
    results.append(ok("阶段排序先按 token 再按字符",
                      [s["name"] for s in mix_usage["by_stage"]] == ["文字稿修订", "朗读合成"],
                      " → ".join(s["name"] for s in mix_usage["by_stage"])))
    bumped = db.add_usage(probe, mix_usage)
    results.append(ok("按需累加同时保住两条计费轴",
                      bumped["characters"] == 1100
                      and bumped["total"] == merged["total"] + 1500
                      and json.loads(db.get_video(probe)["token_detail"])["characters"] == 1100,
                      f"{merged['total']} → {bumped['total']} token / 1100 字符"))
    twice = db.add_usage(probe, mix_usage)
    results.append(ok("二次合成字符累加不被冲掉",
                      twice["characters"] == 2200
                      and {s["name"]: s["characters"] for s in twice["by_stage"]}["朗读合成"] == 2200,
                      f"累计 {twice['characters']} 字符"))
    site = db.token_usage(per_video=8)
    results.append(ok("全站聚合带字符口径",
                      site["characters"] >= 2200
                      and db.token_usage(alice)["characters"] >= 2200,
                      f"全站 {site['characters']} 字符"))
    results.append(ok("聚合阶段明细与视频行都带上字符",
                      any(s["characters"] >= 2200 for s in site["by_stage"])
                      and any(v["characters"] >= 2200 for v in site["videos"]),
                      f"{len(site['by_stage'])} 个阶段 / {len(site['videos'])} 条明细"))

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
        results.append(ok("设置项声明表覆盖 47 个键", len(cfg.ENV_FIELDS) == 47, f"{len(cfg.ENV_FIELDS)} 项"))
        fields = {f.key: f for f in cfg.ENV_FIELDS}
        results.append(ok("同主题阈值已声明：默认 0.6 且限定 0.3~0.95",
                          fields["TOPIC_MATCH_THRESHOLD"].default == "0.6"
                          and float(fields["TOPIC_MATCH_THRESHOLD"].low) == 0.3
                          and float(fields["TOPIC_MATCH_THRESHOLD"].high) == 0.95))
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
    results.append(ok("提示词声明表覆盖 23 块 / 9 组",
                      len(prm.PROMPT_FIELDS) == 23 and len(prm.PROMPT_GROUPS) == 9,
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
    results.append(ok("分析完成后自动带出 2 道待作答的英文提问",
                      len(auto_qa) == analyze.QA_QUESTION_COUNT
                      and [q["idx"] for q in auto_qa] == list(range(1, analyze.QA_QUESTION_COUNT + 1))
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
    results.append(ok("兜底出题永远 2 条且都是英文问句",
                      len(fb) == 2 and all(q.strip() and q.endswith("?") for q in fb),
                      f"{len(fb)} 条"))
    fb_empty = analyze.fallback_questions({"dimensions": [], "suggestions": []}, "")
    results.append(ok("报告缺字段时仍 2 条并回退到 this speech",
                      len(fb_empty) == 2 and all("this speech" in q for q in fb_empty)))
    fb_topic = analyze.fallback_questions({}, "My Topic")
    results.append(ok("题目写进兜底问题里", len(fb_topic) == 2 and all("My Topic" in q for q in fb_topic)))
    results.append(ok("兜底题不含评判措辞",
                      not any(w in " ".join(fb).lower() for w in
                              ("you failed", "weak", "the problem with", "poor", "bad"))))

    class _QAStub:
        def __init__(self, text: str) -> None:
            self.text = text
            self.prompts: list[str] = []

        def chat(self, prompt, system="", **kw):
            self.prompts.append(prompt)
            return Completion(text=self.text, model="mock", usage=None)

    good = json.dumps({"questions": ["Question one?", " Question two? ", "Question three?",
                                     "Question four?"]}, ensure_ascii=False)
    qs = analyze.build_questions(_QAStub(good), rubric, report_qa, "转写", "题目", "2-3 分钟")
    results.append(ok("模型给足 2 条即采用并清洗顺序、截掉多余",
                      qs == ["Question one?", "Question two?"], "、".join(qs)))
    short = analyze.build_questions(_QAStub('{"questions": ["Only one?"]}'), rubric, report_qa,
                                    "转写", "题目")
    results.append(ok("模型条数不足整批改用兜底题",
                      len(short) == 2 and "Only one?" not in short
                      and all(q.strip() for q in short), "、".join(short)))
    broken = analyze.build_questions(_QAStub("这里没有 JSON"), rubric, report_qa, "转写", "题目")
    results.append(ok("模型返回无法解析时仍返回 2 条", len(broken) == 2))
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
    results.append(ok("内置出题提示词锁定英文两题且不评判",
                      "exactly 2 questions, written in English" in prm.DEFAULTS["qa_system"]
                      and "Never judge" in prm.DEFAULTS["qa_system"]
                      and "exactly 2 items" in prm.DEFAULTS["qa_schema"],
                      prm.DEFAULTS["qa_schema"][:60]))
    q_render = prm.render("qa_context", topic="题目", requirements="要求", result='{"total": 1}',
                          transcript="讲稿全文", schema=prm.get("qa_schema"))
    results.append(ok("出题块渲染后不残留占位符",
                      not any(("{" + t + "}") in q_render
                              for t in ("topic", "requirements", "result", "transcript", "schema"))))
    r_render = prm.render("qa_review_context", qa="问题一：Q\n学生作答一：A",
                          schema=prm.get("qa_review_schema"))
    results.append(ok("点评块渲染后不残留占位符", "{qa}" not in r_render and "{schema}" not in r_render))
    db.save_questions(vid_qa, uid_qa, ["最终一题", "最终二题", "最终三题"])

    print("\n== 13. 视频自动压缩：档位选择、进度解析与落盘 ==")
    from app import compress as zc
    keep_cmp = (settings.compress_target_bytes, settings.compress_timeout)
    settings.compress_timeout = 600
    mb = 1024 * 1024
    zc_dir = settings.data_dir / "_compress_probe"
    zc_dir.mkdir(parents=True, exist_ok=True)
    try:
        settings.compress_target_bytes = 0
        results.append(ok("压缩目标填 0 即整段关闭自动压缩", not zc.needs_compress(10 ** 10)))
        settings.compress_target_bytes = 50 * mb
        results.append(ok("恰好等于目标不压，超出 1 字节才压",
                          not zc.needs_compress(50 * mb) and zc.needs_compress(50 * mb + 1)))
        results.append(ok("目标提示按 MB 显示", zc.target_hint() == "50 MB", zc.target_hint()))
        results.append(ok("横屏按短边降档", zc.cap_short_edge(1920, 1080, 720) == (1280, 720),
                          str(zc.cap_short_edge(1920, 1080, 720))))
        results.append(ok("竖屏同样按短边降档（长边排档会永远降不下来）",
                          zc.cap_short_edge(1080, 1920, 720) == (720, 1280),
                          str(zc.cap_short_edge(1080, 1920, 720))))
        results.append(ok("小分辨率原样保留（只缩不放）", zc.cap_short_edge(640, 360, 1080) == (640, 360)))
        results.append(ok("宽高向下取偶数满足 yuv420p",
                          all(v % 2 == 0 for v in zc.cap_short_edge(1079, 1921, 540)),
                          str(zc.cap_short_edge(1079, 1921, 540))))

        def _mi(duration, width, height, audio=True, fps=25.0):
            return media.MediaInfo(duration=duration, width=width, height=height,
                                   has_audio=audio, fps=fps)

        easy = zc.pick_plan(_mi(300.0, 1280, 720), 50 * mb)
        results.append(ok("码率宽裕时保住原始分辨率", (easy.width, easy.height) == (1280, 720),
                          f"{easy.width}x{easy.height} / {easy.video_kbps}kbps"))
        keep_gate = settings.compress_feasibility
        settings.compress_feasibility = False  # 1 小时 50MB 现在会被闸门先拦下，这里单测阶梯本身
        try:
            tight = zc.pick_plan(_mi(3600.0, 1920, 1080, fps=30.0), 50 * mb)
        finally:
            settings.compress_feasibility = keep_gate
        results.append(ok("码率不够才降分辨率（1 小时 1080p 压到 50MB）",
                          tight.short_edge < 1080 and tight.video_kbps >= zc._MIN_VIDEO_KBPS,
                          f"降到 {tight.width}x{tight.height} / {tight.video_kbps}kbps"))
        results.append(ok("档位最低到 480p 为止", tight.short_edge >= zc.LADDER[-1], f"短边 {tight.short_edge}"))
        budget_kbps = 50 * mb * zc.HEADROOM * 8 / 300 / 1000
        results.append(ok("码率预算先扣音频再分给视频",
                          easy.audio_kbps == zc.AUDIO_KBPS
                          and 0 <= budget_kbps - (easy.video_kbps + easy.audio_kbps) < 1,
                          f"预算 {budget_kbps:.0f}k = 视频 {easy.video_kbps}k + 音频 {easy.audio_kbps}k"))
        results.append(ok("无声视频不预留音频码率",
                          zc.pick_plan(_mi(60.0, 640, 360, audio=False), 5 * mb).audio_kbps == 0))
        results.append(ok("异常帧率回落到 25fps（避免除零与极端 bpp）",
                          zc.pick_plan(_mi(60.0, 640, 360, fps=0.0), 5 * mb).fps == 25.0))
        try:
            zc.pick_plan(_mi(0.0, 640, 360), 5 * mb)
            zero_bad = "没报错"
        except RuntimeError as exc:
            zero_bad = "" if "时长探测失败" in str(exc) else str(exc)[:40]
        results.append(ok("时长探测失败时报错而不是算出无穷码率", not zero_bad, zero_bad))

        # 本段上面把 compress_timeout 调成了 600（给编码回调用），闸门断言一律显式传 timeout，
        # 免得纯算术用例被环境的超时值串扰。
        gate_40 = zc.feasibility(_mi(2400.0, 1280, 720), 50 * mb, timeout=8000)
        results.append(ok("闸门：目标装不下时长时先报「至少需要多少 MB」",
                          bool(gate_40.error) and "至少需要" in gate_40.error
                          and gate_40.min_target_bytes == int(2400 * 196 * 1000 / 8),
                          f"min={gate_40.min_target_bytes / mb:.2f}MB · " + gate_40.error[:40]))
        keep_to, settings.compress_timeout = settings.compress_timeout, 8000
        try:
            zc.pick_plan(_mi(2400.0, 1280, 720), 50 * mb)
            gate_raise = "没拦截"
        except RuntimeError as exc:
            gate_raise = "" if "至少需要" in str(exc) else str(exc)[:40]
        finally:
            settings.compress_timeout = keep_to
        results.append(ok("闸门拦截发生在 pick_plan：没开始编码就失败", not gate_raise, gate_raise))
        settings.compress_feasibility = False
        try:
            results.append(ok("闸门关闭时退回旧行为：pick_plan 照常出计划",
                              zc.pick_plan(_mi(2400.0, 1280, 720), 50 * mb).video_kbps
                              == zc._MIN_VIDEO_KBPS))
        finally:
            settings.compress_feasibility = keep_gate
        g60 = zc.feasibility(_mi(2400.0, 1280, 720, audio=False), 50 * mb, timeout=8000)
        g80 = zc.feasibility(_mi(4800.0, 1280, 720), 50 * mb, timeout=8000)
        g15 = zc.feasibility(_mi(900.0, 1280, 720), 50 * mb, timeout=8000)
        silent_min = int(2400 * zc._MIN_VIDEO_KBPS * 1000 / 8)
        results.append(ok("闸门算术与 pick_plan 同源：无声口径更小、15 分钟放行而 80 分钟拦",
                          g60.min_target_bytes == silent_min
                          and g60.min_target_bytes < gate_40.min_target_bytes
                          and g80.error and "剪辑分段" in g80.error
                          and not g15.error,
                          f"rel={g60.min_target_bytes}=={silent_min} "
                          f"smaller={g60.min_target_bytes < gate_40.min_target_bytes} "
                          f"g80={bool(g80.error) and ('剪辑分段' in g80.error)} "
                          f"g15err={g15.error[:30]} g15note={g15.quality_note[:20]}"))
        gt = zc.feasibility(_mi(1500.0, 1280, 720), 300 * mb, timeout=1800)
        gl = zc.feasibility(_mi(2700.0, 1280, 720), 500 * mb, timeout=600)
        results.append(ok("闸门：按 80 秒/分钟折算必超时的拦下，并给出调超时或分段出路",
                          "必然被中断" in gt.error and "压缩超时" in gt.error
                          and "上限" in gl.error))
        gq = zc.feasibility(_mi(1320.0, 1920, 1080, fps=30.0), 50 * mb, timeout=8000)
        results.append(ok("画质破线只留痕不拦截：告警写清 bpp 与关键帧不受影响",
                          not gq.error and str(zc.MIN_BPP) in gq.quality_note and "480p" in gq.quality_note
                          and "每像素比特数" in gq.quality_note and "关键帧" in gq.quality_note,
                          f"err={gq.error[:20]} note={gq.quality_note[:60]}"))

        results.append(ok("-progress 行按微秒换算已编码秒数",
                          abs((zc._progress_seconds("out_time_us=12500000") or 0) - 12.5) < 1e-6))
        results.append(ok("out_time_ms 也按微秒读（ffmpeg 历史遗留命名）",
                          abs((zc._progress_seconds("out_time_ms=12500000") or 0) - 12.5) < 1e-6))
        results.append(ok("无关行、非数字、负值、未知单位一律忽略",
                          all(zc._progress_seconds(t) is None for t in
                              ("frame=  123", "out_time_us=abc", "out_time_us=-1", "out_time_code=5"))))

        fake = zc_dir / "fake.mp4"
        fake_info = _mi(8.0, 640, 360)

        zclip = zc_dir / "gen.mp4"
        zgen = subprocess.run(
            [media.ffmpeg_exe(), "-y", "-v", "error",
             "-f", "lavfi", "-i", "testsrc2=s=640x360:d=8:r=25",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
             "-c:v", "libx264", "-b:v", "4000k", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-shortest", str(zclip)],
            capture_output=True, text=True, timeout=300)
        raw_size = zclip.stat().st_size if zgen.returncode == 0 and zclip.exists() else 0
        results.append(ok("压缩夹具（4Mbps 高码率 8 秒片）可生成", raw_size > 1_500_000,
                          f"{raw_size / mb:.1f} MB" if raw_size else " ".join(zgen.stderr.split())[:120]))
        if raw_size:
            settings.compress_target_bytes = raw_size // 3
            zinfo = media.probe(zclip)
            ticks: list[tuple[int, str]] = []
            res = zc.compress(zclip, zinfo, on_progress=lambda p, phase: ticks.append((p, phase)))
            after = media.probe(zclip)
            results.append(ok("超过目标的视频真实压到线内并原地替换",
                              res is not None and res.size <= raw_size // 3
                              and zclip.stat().st_size == res.size,
                              f"{raw_size / mb:.1f} MB → {(res.size / mb if res else 0):.1f} MB"))
            results.append(ok("压完仍可正常探测（时长与音轨都在）",
                              res is not None and after.duration > 6 and after.has_audio,
                              f"{after.duration:.1f}s / 音频 {after.has_audio}"))
            results.append(ok("码率够用时不牺牲分辨率",
                              res is not None and (after.width, after.height) == (zinfo.width, zinfo.height),
                              f"{zinfo.width}x{zinfo.height} → {after.width}x{after.height}"))
            phases = {p for _, p in ticks}
            results.append(ok("进度覆盖分析一遍与编码一遍并走到 100%",
                              bool(ticks) and max(t for t, _ in ticks) == 100
                              and len(phases) == 2, f"{len(ticks)} 次回调 · " + "、".join(sorted(phases))))
            results.append(ok("压缩说明写清前后体积与码率", res is not None
                              and "原始文件" in res.note and "两遍编码" in res.note
                              and f"{res.plan.video_kbps}kbps" in res.note, (res.note[:60] + "…") if res else "")
                          )
            results.append(ok("临时文件与两遍日志全部清理",
                              not list(zc_dir.glob("*compressing*")) and not list(zc_dir.glob("*.pass*")),
                              " ".join(p.name for p in zc_dir.iterdir())))

            mov = zc_dir / "phone.MOV"
            mov.write_bytes(zclip.read_bytes())
            settings.compress_target_bytes = max(1, mov.stat().st_size // 2)
            res_mov = zc.compress(mov, media.probe(mov))
            results.append(ok("非 MP4 原件压完转封并改名成 .mp4",
                              res_mov is not None and res_mov.renamed
                              and res_mov.path.name == "phone.mp4" and res_mov.path.exists()
                              and not mov.exists(),
                              res_mov.path.name if res_mov else "未改名"))

        settings.compress_target_bytes = 1_000_000
        real_encode, zc._encode = zc._encode, None
        try:
            def _writes(n: int):
                def _inner(src, dst, plan, passlog, on_progress=None):
                    dst.write_bytes(b"\0" * n)
                return _inner

            zc._encode = _writes(1_500_000)
            fake.write_bytes(b"\0" * 2_000_000)
            try:
                zc.compress(fake, fake_info)
                msg = "没报错"
            except RuntimeError as exc:
                msg = "" if "未能压到目标" in str(exc) else str(exc)[:40]
            results.append(ok("两遍后仍超标：明确报错且原件一字未动",
                              not msg and fake.stat().st_size == 2_000_000, msg))

            zc._encode = _writes(2_500_000)
            res_worse = zc.compress(fake, fake_info)
            results.append(ok("压完反而更大时放弃替换（返回 None 而非报错）",
                              res_worse is None and fake.stat().st_size == 2_000_000))
            results.append(ok("失败路径同样不留半成品与日志",
                              not list(zc_dir.glob("*compressing*")) and not list(zc_dir.glob("*.pass*"))))

            hour = _mi(3600.0, 1920, 1080, fps=30.0)
            fake.write_bytes(b"\0" * 20_000_000)
            settings.compress_feasibility = False  # 模拟闸门上线前的旧包：只能编码后才失败
            try:
                zc._encode = _writes(2_000_000)
                zc.compress(fake, hour)
                floor_msg = "没报错"
            except RuntimeError as exc:
                floor_msg = "" if "至少需要 85 MB" in str(exc) else str(exc)[:60]
            finally:
                settings.compress_feasibility = keep_gate
            results.append(ok("码率触底导致的失败会说「至少需要多少 MB」而不是含糊报错",
                              not floor_msg and fake.stat().st_size == 20_000_000, floor_msg))
        finally:
            zc._encode = real_encode

        uid_z = db.get_user_by_name("alice")["id"]
        vid_z = db.create_video(uid_z, "压缩落库", "z.mp4", "z.mp4", 2_000_000)
        db.update_video(vid_z, orig_size=2_000_000, compress_note="已压到 1 MB")
        row_z = db.get_video(vid_z)
        results.append(ok("原始体积与压缩说明可落库读回",
                          row_z["orig_size"] == 2_000_000 and row_z["compress_note"] == "已压到 1 MB"))
        results.append(ok("没压过的视频 orig_size 为空（区别于 0）",
                          db.get_video(db.create_video(uid_z, "未压缩", "n.mp4", "n.mp4", 1))["orig_size"] is None))
        results.append(ok("压缩目标、超时与闸门开关已在设置页声明",
                          {"COMPRESS_TARGET_MB", "COMPRESS_TIMEOUT", "COMPRESS_FEASIBILITY"}
                          <= {f.key for f in cfg.ENV_FIELDS}))
    finally:
        settings.compress_target_bytes, settings.compress_timeout = keep_cmp
        shutil.rmtree(zc_dir, ignore_errors=True)

    print("\n== 14. 版本指纹（BUILD_INFO.json）==")
    from app import buildinfo

    bd = buildinfo.data()
    if buildinfo.BUILD_FILE.exists():
        results.append(ok("有指纹包：version 以短 sha 开头且 sha 对得上文件",
                          buildinfo.version().startswith(str(bd.get("git_short")))
                          and len(str(bd.get("git_sha"))) == 40, buildinfo.version()))
    else:
        results.append(ok("缺指纹：降级 unknown 并给出可解释原因，绝不抛错",
                          buildinfo.version() == "unknown"
                          and bd.get("git_sha") == "unknown" and "BUILD_INFO.json" in bd.get("note", ""),
                          bd.get("note", "")))

    print("\n== 15. 文字稿修订：五条红线与幂等应用 ==")
    from app import revise

    script = ("同学们好，青春是用来奋斗的，幸福也是奋斗出来的。"
              "我们已达成 2020 年目标。我的汇报完毕，谢谢大家。")

    def _v(items, text=script, **kw):
        return revise.validate_items(items, text, **kw)

    r_miss = _v([{"before": "天空", "after": "大地", "kind": "term"}])
    results.append(ok("原文找不到的建议被丢弃并留痕",
                      not r_miss.items and any("原文中没有" in d for d in r_miss.dropped),
                      str(r_miss.dropped)))
    r_dup = _v([{"before": "奋斗", "after": "拼搏", "kind": "homophone"}])
    results.append(ok("原文命中多次的建议不敢定位，丢弃",
                      not r_dup.items and any("多次" in d for d in r_dup.dropped),
                      str(r_dup.dropped)))
    r_ovl = _v([{"before": "青春是用来奋斗的", "after": "青春因奋斗而精彩", "kind": "homophone"},
                {"before": "是用来奋斗的", "after": "是拼出来的", "kind": "grammar"}])
    results.append(ok("与前一条改动区间重叠的建议丢弃",
                      len(r_ovl.items) == 1 and any("重叠" in d for d in r_ovl.dropped),
                      str(r_ovl.dropped)))
    r_r1 = _v([{"before": "同学们好", "after": "同学们好，非常荣幸今天能站在这里，和大家聊一聊青春",
                "kind": "grammar"}])
    results.append(ok("R1：单条改幅超过 20 字丢弃",
                      not r_r1.items and any("R1" in d for d in r_r1.dropped), str(r_r1.dropped)))
    r_r3 = _v([{"before": "2020", "after": "二零二零", "kind": "homophone"}])
    results.append(ok("R3：数字前后不一致（阿拉伯改汉字）丢弃",
                      not r_r3.items and any("数字" in d for d in r_r3.dropped), str(r_r3.dropped)))
    r_r4 = _v([{"before": "谢谢大家", "after": "谢谢大家！明天见。", "kind": "style"}])
    results.append(ok("R4：新增句末标点涉嫌扩写丢弃",
                      not r_r4.items and any("扩写" in d for d in r_r4.dropped), str(r_r4.dropped)))
    t6 = "甲乙丙丁戊己"
    r_r5 = _v([{"before": "甲", "after": "啊甲", "kind": "style"},
               {"before": "乙", "after": "哎乙", "kind": "style"},
               {"before": "丙", "after": "哦丙", "kind": "style"}], t6, max_ratio=50, max_style=2)
    results.append(ok("R5：style 条数超上限后只丢润色不动订正",
                      len(r_r5.items) == 2 and any("R5" in d for d in r_r5.dropped),
                      str(r_r5.dropped)))
    r_r2 = _v([{"before": "甲", "after": "啊甲", "kind": "homophone"},
               {"before": "乙", "after": "哦哦乙", "kind": "term"},
               {"before": "丙", "after": "咦丙", "kind": "grammar"}], t6, max_ratio=30)
    results.append(ok("R2：累计改幅触顶后本条与后续全部连坐留痕",
                      len(r_r2.items) == 1 and any("R2" in d for d in r_r2.dropped)
                      and any("用尽" in d for d in r_r2.dropped), str(r_r2.dropped)))
    t4 = "甲乙丙丁"
    pack = _v([{"before": "甲", "after": "啊甲", "kind": "homophone"},
               {"before": "丙", "after": "哦哦丙", "kind": "term"}], t4, max_ratio=100)
    for it in pack.items:
        it.decision = "accepted"
    once = revise.apply_revisions(t4, pack.items)
    twice = revise.apply_revisions(once, pack.items)
    results.append(ok("apply 只改 accepted 且幂等：再应用一次纹丝不动",
                      once == "啊甲乙哦哦丙丁" and twice == once, once))
    form = revise.decisions_from_form({"accept_1": "1"}, pack.items)
    results.append(ok("表单映射：勾了才 accepted，未勾（含 pending）一律 rejected",
                      form[0].decision == "accepted" and form[1].decision == "rejected"))
    back = revise.items_from_json(revise.items_to_json(pack.items))
    results.append(ok("建议清单可 JSON 落库读回，坐标与决策不丢",
                      back == pack.items and back[0].pos == pack.items[0].pos))
    results.append(ok("原稿指纹稳定且改一字即变（旧建议失效判定用）",
                      revise.source_hash(script) == revise.source_hash(script)
                      and revise.source_hash(script) != revise.source_hash(script + "。")))

    class _ReviseStub:
        def __init__(self, text):
            self._text = text
            self.prompt = ""

        def chat(self, prompt, **kw):
            self.prompt = prompt
            return Completion(text=self._text, model="stub-revise")

    good = _ReviseStub('```json\n{"items":[{"before":"奋斗出来","after":"拼搏出来",'
                       '"kind":"homophone","reason":"近音误识","confidence":0.9}]}\n```')
    res_good = revise.run_revision(good, script, "青春奋斗", "无特别要求", glossary="奋斗")
    results.append(ok("run_revision 端到端：桩模型回 diff，经校验成建议",
                      len(res_good.items) == 1 and res_good.items[0].kind == "homophone"
                      and res_good.model == "stub-revise", str(res_good.dropped)))
    results.append(ok("注入检查：转写稿与主题真的渲染进了提示词",
                      "青春是用来奋斗的" in good.prompt and "青春奋斗" in good.prompt))
    res_junk = revise.run_revision(_ReviseStub("模型今天不说人话"), script, "青春", "")
    results.append(ok("模型输出不是 JSON 时按「没有建议」处理，绝不抛错",
                      isinstance(res_junk, revise.RevisionResult) and not res_junk.items))

    print("\n== 15b. 确认门：propose / decide / 失效（方案 §5.2） ==")
    meta_gate = {"source_hash": revise.source_hash(script)}
    results.append(ok("失效判定：指纹一致放行、稿变即失效、无指纹不误伤",
                      not revise.is_stale(script, meta_gate)
                      and revise.is_stale(script + "。", meta_gate)
                      and not revise.is_stale(script, {})
                      and not revise.is_stale(script, {"source_hash": ""})))

    segs = [{"start": 0, "ts": "00:00", "text": "同学们好，青春是用来奋斗的，"},
            {"start": 6, "ts": "00:07", "text": "幸福也是奋斗出来的。"},
            {"start": 14, "ts": "00:15", "text": "我的汇报完毕，谢谢大家。"}]
    t_seg = " ".join(s["text"] for s in segs)  # 与 transcribe.py 的组装口径一致
    pack_s = _v([{"before": "同学们好", "after": "老师们同学好", "kind": "grammar"},
                 {"before": "我的汇报完毕", "after": "汇报到此结束", "kind": "homophone"}],
                t_seg, max_ratio=100)
    stamped = revise.stamp_items(pack_s.items, segs)
    results.append(ok("时间点回填：以空格拼接的偏移映射回所属段的 ts",
                      len(stamped) == 2 and stamped[0]["ts"] == "00:00" and stamped[1]["ts"] == "00:15",
                      f'{stamped[0]["ts"]}/{stamped[1]["ts"]}'))

    gvid = db.create_video(db.get_user_by_name("alice")["id"], "确认门夹具", "gate.mp4",
                           str(settings.video_dir / "gate.mp4"), 1,
                           topic="青春奋斗", requirements="5 分钟")
    db.update_video(gvid, transcript=t_seg, segments=segs, status="done")
    pipeline.propose_script(gvid, glossary="奋斗", client=_ReviseStub(
        '```json\n{"items":[{"before":"奋斗出来","after":"拼搏出来",'
        '"kind":"homophone","reason":"近音误识","confidence":0.9}]}\n```'))
    gate = db.get_video(gvid)
    results.append(ok("propose 落库：proposed + pending，成稿留空且原稿一字未动",
                      gate["script_status"] == "proposed"
                      and json.loads(gate["script_items"])[0]["decision"] == "pending"
                      and gate["script_text"] == "" and gate["transcript"] == t_seg))
    gate_items = revise.items_from_json(gate["script_items"])
    dec_none = revise.decisions_from_form({}, gate_items)
    results.append(ok("空表单提交：pending 等同未采纳，成稿保持空串（不动鼠标 = 不改）",
                      not any(d.decision == "accepted" for d in dec_none)
                      and revise.apply_revisions(t_seg, dec_none) == t_seg))
    dec_yes = revise.decisions_from_form({f"accept_{gate_items[0].idx}": "1"}, gate_items)
    final1 = revise.apply_revisions(t_seg, dec_yes)
    db.update_video(gvid, script_status="decided", script_text=final1,
                    script_items=revise.items_to_json(dec_yes))
    mid = db.get_video(gvid)
    db.update_video(gvid, script_text="", script_status="decided")
    gone = db.get_video(gvid)
    results.append(ok("采纳即局部替换、清空成稿即全量回原：transcript 从未被覆盖",
                      final1 == t_seg.replace("奋斗出来", "拼搏出来") != t_seg
                      and mid["script_text"] == final1
                      and gone["script_text"] == "" and gone["transcript"] == t_seg))

    # 35% 卡住的真实成因是「比对全稿」被 180 秒请求超时掐死（实测 1600 字要 349 秒），
    # 失败又因 _job_worker 的缺省兜底写进了朗读侧状态，两条都要钉住。
    built_rev: dict = {}

    class _ReviseClientProbe:
        def __init__(self, *a, **kw):
            built_rev.update(kw)
            self.stage = ""

        def mark(self, stage):
            self.stage = stage
            return self

        def chat(self, prompt, **kw):
            return Completion(text='{"items":[]}', model="stub-revise")

        def usage_summary(self):
            return {"calls": 1, "total": 0}

    tvid = db.create_video(db.get_user_by_name("alice")["id"], "修订超时夹具", "revto.mp4",
                           str(settings.video_dir / "revto.mp4"), 1, topic="青春奋斗")
    db.update_video(tvid, transcript=t_seg, status="done", tts_status="done", tts_error="")
    orig_qc_rev = pipeline.QwenClient
    pipeline.QwenClient = _ReviseClientProbe
    try:
        pipeline.propose_script(tvid, glossary="")
    finally:
        pipeline.QwenClient = orig_qc_rev
    results.append(ok("修订调用单吃「修订请求超时」，不再跟全局 180 秒同一条命",
                      built_rev.get("timeout") == settings.revise_timeout
                      and built_rev.get("retries") == 2
                      and settings.revise_timeout >= 300,
                      f'timeout={built_rev.get("timeout")} retries={built_rev.get("retries")}'))

    pipeline.run_propose_error(tvid, RuntimeError(
        "评审请求异常: HTTPSConnectionPool(host='dashscope.aliyuncs.com')"
        ": Read timed out. (read timeout=180)"))
    bad_rev = db.get_video(tvid)
    results.append(ok("修订失败只写修订侧：不再冒充「上次朗读合成失败」",
                      bad_rev["script_status"] == "failed"
                      and bad_rev["tts_status"] == "done" and bad_rev["tts_error"] == ""))
    results.append(ok("超时失败落库成人话：说清能重开、并指出该调哪个旋钮",
                      "修订请求超时" in (bad_rev["script_error"] or "")
                      and str(settings.revise_timeout) in (bad_rev["script_error"] or ""),
                      (bad_rev["script_error"] or "")[:60]))

    print("\n== 15c. 朗读合成：拼装、切块、闸门与拼接（方案 §6） ==")
    import re as _re
    import tempfile as _tempfile
    from types import SimpleNamespace as _NS
    from app import tts as tts_mod

    s15 = "第一句话讲完了。第二句话也讲完了。" * 40
    ch15 = tts_mod.plan_chunks(s15, chunk_chars=240)
    results.append(ok("切块绝不在句中切开：块块以句末标点收尾且一句不丢",
                      all(c.endswith(("。", "！", "？", "；")) for c in ch15)
                      and sum(c.count("。") for c in ch15) == 80
                      and all(len(c) <= 240 for c in ch15), f"{len(ch15)} 块"))

    cfg_full = _NS(tts_enabled=True, real_mode=True, api_key="sk-test",
                   tts_chunk_chars=1600, tts_max_chunks=8, tts_timeout=300)
    g1, w1 = tts_mod.tts_feasibility("字" * 13000, cfg=cfg_full)
    g2, w2 = tts_mod.tts_feasibility("字" * 2000, cfg=_NS(**{**cfg_full.__dict__, "tts_timeout": 40}))
    g3, _ = tts_mod.tts_feasibility("字" * 2000, cfg=cfg_full)
    over = False
    try:
        tts_mod.plan_chunks(s15, chunk_chars=240, max_chunks=len(ch15) - 1)
    except tts_mod.TtsError:
        over = True
    results.append(ok("闸门判据：装不下 / 跑不完 / 切超块数都拦下，正常稿放行",
                      not g1 and "超过上限" in w1 and not g2 and "TTS_TIMEOUT" in w2
                      and g3 and over, w2[:44]))

    ins15 = tts_mod.build_instructions(
        tts_mod.style_fallback("contest"), scene="contest",
        pace_hint="示范比原速放慢一档、句间留呼吸",
        focus=tts_mod.build_focus({"fluency": {"dim_key": "fluency", "name": "流利度",
                                               "ratio": 0.55}}))
    results.append(ok("指令拼装：H 逐句演绎脚本 + 相对语速 + 短板导向全部在文内",
                      "演讲比赛" in ins15 and "设问句后停顿留白" in ins15
                      and "斩钉截铁" in ins15 and "示范比原速放慢一档" in ins15
                      and "念清楚" in ins15 and "不喊叫" in ins15, ins15[:48]))

    class _StyleBoom:
        def chat(self, prompt, **kw):
            raise RuntimeError("网络断了")

    note15: list[str] = []
    st15 = tts_mod.infer_style(_StyleBoom(), topic="青春", scene="contest",
                               text=s15[:100], note=note15)
    st_ok = tts_mod.infer_style(
        _ReviseStub('{"tone":"温暖坚定","pace_band":"偏快",'
                    '"structure":"重点句放慢","avoid":"不喊"}'),
        topic="青春", scene="class", text=s15[:100])
    results.append(ok("情感推断：chat 失败安静落规则基线并留痕，成功则收齐四键",
                      st15 == tts_mod.style_fallback("contest")
                      and any("推断失败" in n for n in note15)
                      and st_ok["tone"] == "温暖坚定" and st_ok["avoid"] == "不喊"))

    d15 = Path(_tempfile.mkdtemp(prefix="tts_join_"))
    try:
        blocks = []
        for i, hz in enumerate((440, 660)):
            b = d15 / f"b{i}.wav"
            subprocess.run([media.ffmpeg_exe(), "-y", "-v", "error", "-f", "lavfi",
                            "-i", f"sine=frequency={hz}:duration=1",
                            "-ar", "44100", str(b)], check=True)
            blocks.append(b)
        joined = media.concat_audio(blocks, d15 / "all.mp3", gap_ms=250)
        probe = subprocess.run([media.ffmpeg_exe(), "-i", str(joined)],
                               capture_output=True, text=True)
        m15 = _re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d\d)", probe.stderr)
        secs = (int(m15.group(1)) * 3600 + int(m15.group(2)) * 60 + int(m15.group(3))
                + int(m15.group(4)) / 100) if m15 else -1.0
        results.append(ok("两块音频真拼接：时长≈内容之和 + 250ms 垫片",
                          2.2 <= secs <= 2.9 and joined.stat().st_size > 1024, f"{secs:.2f}s"))
    finally:
        shutil.rmtree(d15, ignore_errors=True)

    gm, wm = tts_mod.tts_feasibility("有稿子", cfg=_NS(**{**cfg_full.__dict__, "real_mode": False}))
    keep_mode = settings.llm_mode
    settings.llm_mode = "mock"  # 第 9 节 finally 的 refresh_runtime 会把 .env 的 LLM_MODE 冲回环境，这里显式钉住
    try:
        raised = ""
        try:
            pipeline.run_tts(gvid)
        except tts_mod.TtsError as exc:
            raised = str(exc)
        clean_tts = db.get_video(gvid)["tts_status"] == ""
    finally:
        settings.llm_mode = keep_mode
    results.append(ok("mock 下闸门拦在任何调用之前：TtsError 上抛且不碰 tts 列",
                      gm is False and "演示模式" in wm and "演示模式" in raised
                      and clean_tts, raised[:44]))

    print("\n== 16. 模型名口径统一与降级提示归因 ==")

    def _boom(model: str) -> QwenError:
        return QwenError(f"The model {model} does not exist", status=404)

    class _Resp:
        def json(self):
            return {"output": {"audio": {"url": "https://tmp/x.wav"}},
                    "usage": {"characters": 12}}

    class _ChatProbe(QwenClient):
        """把网络层换成「只认白名单模型」的桩，观察降级链的真实走法。"""

        def __init__(self, alive):
            super().__init__()
            self.alive, self.tried = set(alive), []

        def _complete(self, payload):
            m = payload["model"]
            self.tried.append(m)
            if m not in self.alive:
                raise _boom(m)
            return Completion(text="{}", model=m, usage={"total_tokens": 3})

    class _TtsProbe(QwenClient):
        def __init__(self, alive):
            super().__init__()
            self.alive, self.tried, self.sent = set(alive), [], []

        def _post_native(self, payload, timeout=None):
            m = payload["model"]
            self.tried.append(m)
            self.sent.append(payload["input"])
            if m not in self.alive:
                raise _boom(m)
            return _Resp()

    keep_resolved = dict(QwenClient._RESOLVED)
    keep_instr = settings.tts_instructions
    keep_env16 = {k: os.environ.get(k) for k in ("CHAT_MODEL", "TTS_MODEL", "REVISE_MODEL")}
    keep_files16 = dict(cfg.FILE_VALUES)
    kf16 = settings.data_dir / "_keyfile_probe.txt"
    QwenClient._RESOLVED.clear()
    try:
        results.append(ok("模型名归一化：转小写并去掉首尾空白，避免先吃一个 400 再试错",
                          cfg.normalize_model(" Qwen3-TTS-Flash ") == "qwen3-tts-flash"
                          and cfg.normalize_model("QWEN-PLUS") == "qwen-plus"
                          and cfg.normalize_model("") == ""))
        kf16.write_text("api key: sk-selfcheck16placeholder\nsynthesis model: Qwen3-TTS-Flash\n"
                        "revision model: Qwen-Plus\n", encoding="utf-8")
        slots16 = cfg._read_key_file(kf16)
        results.append(ok("api_key.txt 认得 synthesis/revision model 两种口语写法，合成与修订也能文件指定",
                          slots16.get("tts_model") == "Qwen3-TTS-Flash"
                          and slots16.get("revise_model") == "Qwen-Plus"
                          and fields["TTS_MODEL"].file_slot == ("tts_model",)
                          and fields["REVISE_MODEL"].file_slot == ("revise_model",),
                          json.dumps(slots16, ensure_ascii=False)))
        cfg.FILE_VALUES.update({"model": "Qwen-Plus", "tts_model": "Qwen3-TTS-Instruct-Flash"})
        os.environ.pop("TTS_MODEL", None)
        results.append(ok("模型三级优先：.env 留空回落 api_key.txt，取值一律小写化",
                          cfg.resolve_model("TTS_MODEL", ("tts_model",), "qwen3-tts-flash")
                          == "qwen3-tts-instruct-flash"))
        os.environ["TTS_MODEL"] = "  QWEN3-TTS-FLASH  "
        results.append(ok("模型三级优先：.env 填了就压过 api_key.txt",
                          cfg.resolve_model("TTS_MODEL", ("tts_model",), "other") == "qwen3-tts-flash"))
        _, err_sp = cfg.validate_env_updates({"TTS_MODEL": "qwen3 tts flash"})
        upd_lc, err_lc = cfg.validate_env_updates({"TTS_MODEL": "Qwen3-TTS-Flash"})
        results.append(ok("设置页保存模型名：大小写自动归一，含空格报错并给出例式",
                          upd_lc["TTS_MODEL"] == "qwen3-tts-flash"
                          and not [e for e in err_lc if "模型名" in e]
                          and any("不能含空格" in e for e in err_sp),
                          (err_sp or [""])[0][:36]))

        chain16 = _ChatProbe(["qwen-plus", "qwen-turbo"])._model_chain("qwen-max", "chat")
        results.append(ok("降级链：首选在前、同族替补在后，且不重复不丢序",
                          chain16[0] == "qwen-max" and chain16[1] == "qwen-plus"
                          and len(set(chain16)) == len(chain16), " ".join(chain16)))

        p16a = _ChatProbe(["qwen-plus"])
        n16a: list[str] = []
        p16a.chat("x", model="qwen-not-here", note=n16a)
        again16 = QwenClient._RESOLVED.get("qwen-not-here")
        p16b = _ChatProbe(["qwen-plus"])
        n16b: list[str] = []
        p16b.chat("x", model="qwen-not-here", note=n16b)
        results.append(ok("降级一次即记账：后续不再白吃 404，但报告仍标明在用替补模型",
                          p16a.tried == ["qwen-not-here", "qwen-plus"] and again16 == "qwen-plus"
                          and p16b.tried == ["qwen-plus"] and n16b == n16a,
                          f"首次 {p16a.tried} / 复用 {p16b.tried}"))
        results.append(ok("降级提示带通道标签与后果说明",
                          n16a[0].startswith("[文本评审] 模型 qwen-not-here 不可用，已改用 qwen-plus")
                          and "由替补模型产出" in n16a[0], n16a[0][:52]))
        QwenClient._RESOLVED.clear()

        p16c = _ChatProbe(["qwen-plus", "qwen-vl-max"])
        n16c: list[str] = []
        p16c.chat("x", model="qwen-bad-rev", note=n16c, label="文字稿修订")
        p16c.chat("x", model="qwen-bad-rev", note=n16c, label="文字稿修订")
        p16c.vision("x", [], model="qwen-vl-bad", note=n16c)
        results.append(ok("同族不同用途分得开，同一条事实不重复刷屏",
                          n16c[0].startswith("[文字稿修订]") and n16c[1].startswith("[画面评审]")
                          and len(n16c) == 2, " | ".join(n16c)[:60]))
        QwenClient._RESOLVED.clear()

        t16a = _TtsProbe(["qwen3-tts-flash"])
        settings.tts_instructions = True
        QwenClient._RESOLVED.clear()
        n16t: list[str] = []
        aud16 = t16a.tts("测试文本", instructions="放慢语速", note=n16t)
        results.append(ok("instruct 不可用：降级 flash 保朗读，并说明丢了表现力指令",
                          aud16.model == "qwen3-tts-flash" and not aud16.instructions_used
                          and "instructions" not in t16a.sent[-1]
                          and "该模型不支持表现力指令" in n16t[0], n16t[0][:60]))
        QwenClient._RESOLVED["qwen3-tts-instruct-flash"] = "qwen3-tts-flash"
        t16z = _TtsProbe(["qwen3-tts-instruct-flash"])
        t16z.tts("测试文本", model="qwen3-tts-instruct-flash", instructions="放慢语速")
        results.append(ok("已记住的替补名一旦失效即放弃，回到原始探测",
                          QwenClient._RESOLVED.get("qwen3-tts-instruct-flash") is None
                          and t16z.tried == ["qwen3-tts-flash", "qwen3-tts-instruct-flash"],
                          " ".join(t16z.tried)))
        QwenClient._RESOLVED.clear()
        t16b = _TtsProbe(["qwen3-tts-flash"])
        n16t2: list[str] = []
        t16b.tts("测试文本", model="qwen3-tts-flash", instructions="放慢语速", note=n16t2)
        QwenClient._RESOLVED.clear()
        t16c = _TtsProbe(["qwen3-tts-instruct-flash"])
        settings.tts_instructions = False
        t16c.tts("测试文本", model="qwen3-tts-instruct-flash", instructions="放慢语速", note=n16t2)
        settings.tts_instructions = True
        QwenClient._RESOLVED.clear()
        t16d = _TtsProbe(["qwen3-tts-instruct-flash"])
        t16d.tts("测试文本", model="qwen3-tts-instruct-flash", instructions="放慢语速", note=n16t2)
        results.append(ok("丢指令的三种成因各说各话：非 instruct 通道 / 开关关闭 / 正常带指令",
                          any("不属于 instruct 通道" in x for x in n16t2)
                          and any("TTS_INSTRUCTIONS）已关闭" in x for x in n16t2)
                          and "instructions" in t16d.sent[0] and len(n16t2) == 2
                          and len(t16d.tried) == 1 and len(t16c.tried) == 1,
                          " | ".join(n16t2)[:56]))
        t16e = _TtsProbe([])
        n16e: list[str] = []
        raised16 = ""
        try:
            t16e.tts("测试文本", model="cosyvoice-v2", note=n16e)
        except QwenError as exc:
            raised16 = str(exc)
        results.append(ok("cosyvoice 走另一套协议，绝不静默换成 qwen3-tts",
                          t16e.tried == ["cosyvoice-v2"] and "cosyvoice" in raised16
                          and not n16e, f"{t16e.tried} / {raised16[:30]}"))
    finally:
        QwenClient._RESOLVED.clear()
        QwenClient._RESOLVED.update(keep_resolved)
        settings.tts_instructions = keep_instr
        cfg.FILE_VALUES.clear()
        cfg.FILE_VALUES.update(keep_files16)
        for k, v in keep_env16.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        kf16.unlink(missing_ok=True)

    print("\n== 17. 声音复刻：样本挑选、建音色与合规闸门（方案 §9） ==")
    from app import tts as tts_mod
    from app import voice as voice_mod
    from app.voice import VoiceSampleError

    target17 = settings.tts_vc_model
    keep_mode17 = settings.llm_mode
    keep_key17 = settings.api_key
    keep_vcmodel17 = settings.tts_vc_model

    class _VcResp:
        def __init__(self, payload, voice, fallback, reason):
            self.p, self.v, self.fb, self.rs = payload, voice, fallback, reason

        def json(self):
            if self.p["input"]["action"] == "create":
                return {"output": {"voice": self.v,
                                   "target_model": self.p["input"]["target_model"],
                                   "fallback_mode": self.fb, "fallback_reason": self.rs},
                        "usage": {"count": 1}}
            return {"output": {}}

    class _VcProbe(QwenClient):
        """只桩住复刻这一个网络叶子：create_voice 与 delete_voice 都走 _post_vc。"""

        def __init__(self, voice="probe-voice-1", fallback=False, reason=""):
            super().__init__(api_key="sk-probe")
            self.sent: list[dict] = []
            self.tried_tts: list[tuple] = []
            self.chats = 0
            self.v, self.fb, self.rs = voice, fallback, reason

        def _post_vc(self, payload, timeout=None):
            self.sent.append(payload)
            return _VcResp(payload, self.v, self.fb, self.rs)

        def chat(self, *a, **k):
            self.chats += 1
            raise QwenError("probe: 复刻链路不该发起 chat 调用")

    class _CloneTts(_VcProbe):
        def tts(self, text, voice=None, instructions="", model=None, timeout=None,
                note=None, label=""):
            self.tried_tts.append((voice, instructions, model))
            raise QwenError("probe: 停在下载与拼接之前，不让测试真跑 ffmpeg")

    class _GoneVc(_VcProbe):
        def _post_vc(self, payload, timeout=None):
            self.sent.append(payload)
            if payload["input"]["action"] == "delete":
                raise QwenError("音色不存在", status=404, body="")
            return _VcResp(payload, self.v, self.fb, self.rs)

    sil17 = voice_mod.parse_silence_ranges(
        "[silencedetect] silence_start: 4\nsilence_end: 5 | silence_duration: 1\n"
        "silence_start: 8\nsilence_end: 9 | silence_duration: 1\n")
    results.append(ok("silencedetect 日志按 start/end 成对解析",
                      sil17 == [(4.0, 5.0), (8.0, 9.0)], str(sil17)))
    results.append(ok("重复的 silence_start 被吞掉，不会造出零宽静默",
                      voice_mod.parse_silence_ranges(
                          "silence_start: 4\nsilence_start: 6\nsilence_end: 7\n") == [(4.0, 7.0)]))
    results.append(ok("落单的 silence_end 忽略；以静音收尾时右端留给调用方补齐",
                      voice_mod.parse_silence_ranges("silence_end: 3\nsilence_start: 10\n") == []))
    results.append(ok("逆序的 silence_end 丢弃但不卡住后面的配对",
                      voice_mod.parse_silence_ranges(
                          "silence_start: 9\nsilence_end: 8\nsilence_start: 12\nsilence_end: 13\n"
                      ) == [(12.0, 13.0)]))
    w17 = voice_mod.speech_windows_from_silence([(4.0, 5.0), (8.0, 9.0)], 12.0)
    results.append(ok("说话区间取静默补集，短于 3 秒的碎段直接丢掉",
                      w17 == [(0.0, 4.0), (5.0, 8.0), (9.0, 12.0)], str(w17)))
    results.append(ok("时长非正不编造区间，静音铺到结尾也不补零宽尾块",
                      voice_mod.speech_windows_from_silence([(1.0, 2.0)], 0.0) == []
                      and voice_mod.speech_windows_from_silence([(4.0, 12.0)], 12.0) == [(0.0, 4.0)]))
    ends17 = [(0.0, 4.0), (5.0, 8.0), (9.0, 13.0), (14.0, 17.0), (18.0, 22.0)]
    ch17 = voice_mod.merge_chains(ends17)
    results.append(ok("间隔 ≤2 秒的区间粘成一条链，超过 2 秒才算真断",
                      ch17 == [(0.0, 22.0)]
                      and voice_mod.merge_chains([(0.0, 4.0), (8.0, 12.0)]) == [(0.0, 4.0), (8.0, 12.0)],
                      str(ch17)))
    cut17 = voice_mod.choose_slice(ch17, ends17)
    results.append(ok("切片收尾落在目标长度后最近的真实停顿上，不把句子腰斩",
                      cut17.start == 0.0 and cut17.end == 17.0
                      and cut17.speech_windows == 4 and cut17.pause_count == 3,
                      f"{cut17.start}-{cut17.end} / {cut17.speech_windows} 段"))
    flat17 = voice_mod.choose_slice([(0.0, 30.0)], [(0.0, 30.0)])
    results.append(ok("链内找不到合规停顿时退化成定长截取，不硬凑停顿",
                      flat17.start == 0.0 and flat17.end == voice_mod.VC_SAMPLE_TARGET,
                      f"{flat17.start}-{flat17.end}"))
    try:
        voice_mod.choose_slice([(0.0, 8.0)], [(0.0, 8.0)])
        short17 = ""
    except VoiceSampleError as exc:
        short17 = str(exc)
    results.append(ok("最长连续朗读不足 10 秒时给出「换一段更连贯视频」的人话原因",
                      "更连贯" in short17 and "10" in short17, short17[:40]))
    results.append(ok("备注名洗成官方接受的字符集：纯中文退回 student 并截到 16 字符",
                      QwenClient.preferred_name_of("u7_alice") == "u7_alice"
                      and QwenClient.preferred_name_of("张三的同学") == "student"
                      and len(QwenClient.preferred_name_of("abcdefghijklmnopqrstuvwx")) == 16))

    vdir17 = settings.data_dir / "voices" / "_probe17"
    vdir17.mkdir(parents=True, exist_ok=True)
    wav17 = voice_mod.extract_sample(SAMPLE, vdir17 / "probe17.wav", 0.0, 12.0, 42.0)
    results.append(ok("样本按 24kHz 单声道落盘，大小过防呆下限",
                      wav17.exists() and wav17.stat().st_size >= voice_mod.VC_MIN_BYTES,
                      f"{wav17.stat().st_size} 字节"))

    orig_qc17 = pipeline.QwenClient
    probe17 = _VcProbe(fallback=True, reason="噪声过高")
    try:
        settings.llm_mode = "real"
        settings.api_key = "sk-probe"
        p17 = _VcProbe()
        prof17 = p17.create_voice(wav17, "u3_alice")
        pl17 = p17.sent[0]
        results.append(ok("建音色走官方 enrollment 模型，样本以 data:audio/wav;base64 直传",
                          pl17["model"] == "qwen-voice-enrollment"
                          and pl17["input"]["action"] == "create"
                          and pl17["input"]["audio"]["data"].startswith("data:audio/wav;base64,"),
                          f"{pl17['model']} / {pl17['input']['audio']['data'][:22]}"))
        results.append(ok("载荷里的 target_model 取 TTS_VC_MODEL，备注名走清洗后的形式",
                          pl17["input"]["target_model"] == target17
                          and pl17["input"]["preferred_name"] == "u3_alice"
                          and prof17.voice == "probe-voice-1"))
        results.append(ok("可选的 text/language 一致性校验宁缺毋滥：没传就不附",
                          "text" not in pl17["input"] and "language" not in pl17["input"]))
        up17 = p17.create_voice(wav17, "x", target_model="QWEN3-TTS-VC-2026-01-22")
        results.append(ok("target_model 一律归一小写，大写配置不会造出对不上的音色",
                          p17.sent[1]["input"]["target_model"] == "qwen3-tts-vc-2026-01-22"
                          and up17.target_model == "qwen3-tts-vc-2026-01-22",
                          up17.target_model))
        try:
            _VcProbe(voice="").create_voice(wav17, "x")
            n17 = ""
        except QwenError as exc:
            n17 = str(exc)
        results.append(ok("服务方没回 voice 标识时报人话错误，绝不落一条空音色",
                          "voice 标识" in n17, n17[:40]))
        settings.llm_mode = "mock"
        try:
            _VcProbe().create_voice(wav17, "x")
            off17 = ""
        except QwenError as exc:
            off17 = str(exc)
        finally:
            settings.llm_mode = "real"
        results.append(ok("演示模式（mock）压根不许建音色，避免静默烧额度",
                          "未启用" in off17 or "密钥缺失" in off17, off17[:40]))
        sum17 = p17.usage_summary()
        results.append(ok("建音色按「次」计费：creations 进汇总，不吃 token 也不记字符",
                          sum17["creations"] >= 1 and sum17["total"] == 0
                          and sum17["characters"] == 0, str(sum17["creations"])))

        uid17 = db.create_user("vc17user", "pass1234", display_name="复刻夹具")
        vvid = db.create_video(uid17, "复刻夹具演讲", "sample_speech.mp4", str(SAMPLE),
                               SAMPLE.stat().st_size, topic="青春奋斗", requirements="3 分钟")
        db.update_video(vvid, status="done",
                        transcript="大家好，今天我想聊聊坚持的意义。" * 3, duration=42.0)
        pipeline.QwenClient = lambda *a, **k: probe17
        try:
            pipeline.run_voice(vvid)
            gate17 = ""
        except voice_mod.VoiceSampleError as exc:
            gate17 = str(exc)
        st17 = db.voice_status(uid17)
        results.append(ok("没授权时第一步就拦下，不抽样本也不送出人声",
                          "授权已撤回" in gate17 and probe17.sent == []
                          and st17["status"] == "failed" and "VoiceSampleError" in st17["error"],
                          f"{gate17[:24]} / {st17['status']}"))
        db.set_voice_consent(uid17, True)
        built17 = pipeline.run_voice(vvid)
        st17 = db.voice_status(uid17)
        results.append(ok("授权后从本段视频挑样建成音色，写回账号而不是写回视频",
                          built17 == "probe-voice-1" and st17["voice_id"] == "probe-voice-1"
                          and st17["usable"] and st17["status"] == "done", st17["status"]))
        results.append(ok("音色绑定的模型与 TTS_VC_MODEL 一致，页面据此判要不要重做",
                          st17["voice_model"] == target17 and not st17["stale"], st17["voice_model"]))
        results.append(ok("样本来源写进记录，学生知道这副嗓子取自哪一段",
                          f"取自视频 #{vvid}" in st17["source"], st17["source"][:48]))
        results.append(ok("降级建成的音色如实提醒相似度可能不足，不当成干净成功隐瞒",
                          "降级方式建成音色" in st17["error"] and "噪声过高" in st17["error"],
                          st17["error"][:44]))
        results.append(ok("样本用完即删：账号目录里不留下学生的人声",
                          list(pipeline._voice_dir(uid17).glob("*.wav")) == []
                          and db.get_video(vvid)["stage"] == "我的音色已就绪",
                          db.get_video(vvid)["stage"]))
        use17 = db.video_usage(db.get_video(vvid))
        results.append(ok("建音色这笔计入该次任务的用量，creations 在统计里看得见",
                          use17["creations"] >= 1 and use17["total"] == 0, str(use17["creations"])))
        results.append(ok("同一账号的复刻任务互斥：抢不到锁只说明，不动别人持有的记录",
                          db.begin_voice_job(uid17)
                          and pipeline.run_voice(vvid) == "probe-voice-1"
                          and "本次提交未执行" in db.get_video(vvid)["stage"],
                          db.get_video(vvid)["stage"]))
        db.save_user_voice(uid17, "probe-voice-1", target17, source="锁测试", note="")
        settings.tts_vc_model = "qwen3-tts-vd-2026-01-22"
        st17s = db.voice_status(uid17)
        results.append(ok("管理员改了 TTS_VC_MODEL：老音色判作废，宁可用系统音色也不硬合成",
                          st17s["stale"] and not st17s["usable"]
                          and db.user_voice_for_tts(uid17) == ("", ""), st17s["status"]))
        settings.tts_vc_model = keep_vcmodel17

        settings.llm_mode = "mock"
        dropped17 = pipeline.drop_voice(uid17)
        st17d = db.voice_status(uid17)
        results.append(ok("演示模式删音色只清本地记录：不报网络错，授权留痕仍然在",
                          dropped17 == "probe-voice-1" and not st17d["has_voice"]
                          and st17d["consented"] and st17d["consent_at"]
                          and all(x["input"]["action"] != "delete" for x in probe17.sent),
                          st17d["status"]))
        settings.llm_mode = "real"
        pipeline.run_voice(vvid)
        pipeline.drop_voice(uid17)
        del17 = [x for x in probe17.sent if x["input"]["action"] == "delete"]
        results.append(ok("在线态删音色先删远端再清本地，载荷带 voice 与绑定模型",
                          len(del17) == 1 and del17[0]["model"] == "qwen-voice-enrollment"
                          and del17[0]["input"]["voice"] == "probe-voice-1"
                          and not db.voice_status(uid17)["has_voice"], str(len(del17))))
        pipeline.run_voice(vvid)
        gone17 = _GoneVc()
        pipeline.QwenClient = lambda *a, **k: gone17
        pipeline.drop_voice(uid17)
        results.append(ok("远端本来就没了（404）按删除成功处理，不让人永远删不动",
                          not db.voice_status(uid17)["has_voice"]
                          and any(x["input"]["action"] == "delete" for x in gone17.sent)))

        cli17 = _CloneTts()
        out17 = vdir17 / "ttsout"
        try:
            tts_mod.synthesize(cli17, "这是一段用来验证复刻分支的稿子。", out_dir=out17,
                               voice="probe-voice-1", model=target17)
        except QwenError:
            pass
        results.append(ok("用复刻音色合成时不送表现力指令（官方口径：vc 音色不接 instructions）",
                          len(cli17.tried_tts) == 1 and cli17.tried_tts[0][1] == "",
                          str(cli17.tried_tts)[:60]))
        results.append(ok("音色与模型成对下传，绝不出现换了嗓子没换模型",
                          bool(cli17.tried_tts) and cli17.tried_tts[0][0] == "probe-voice-1"
                          and cli17.tried_tts[0][2] == target17, str(cli17.tried_tts)[:60]))
        results.append(ok("复刻分支跳过情感推断，不白烧一次 chat", cli17.chats == 0))
        cli17b = _CloneTts()
        try:
            tts_mod.synthesize(cli17b, "另一段用来做对照的稿子。", out_dir=out17,
                               voice="Cherry", model="qwen3-tts-flash")
        except QwenError:
            pass
        results.append(ok("对照组：系统音色才走情感推断，差别是复刻刻意省掉的",
                          cli17b.chats == 1 and len(cli17b.tried_tts) == 1
                          and cli17b.tried_tts[0][0] == "Cherry", f"{cli17b.chats} chat"))

        try:
            pipeline.run_tts(vvid, scene="class", use_clone=True)
            t17f = ""
        except tts_mod.TtsError as exc:
            t17f = str(exc)
        results.append(ok("没音色却选复刻通道：给人话错误，且在动笔前就挡住不碰 tts_status",
                          "还没有可用的复刻音色" in t17f
                          and db.get_video(vvid)["tts_status"] == "", t17f[:40]))
    finally:
        pipeline.QwenClient = orig_qc17
        settings.llm_mode = keep_mode17
        settings.api_key = keep_key17
        settings.tts_vc_model = keep_vcmodel17
        shutil.rmtree(vdir17, ignore_errors=True)

    print("\n== 18. 同主题模糊匹配：归一化、阈值、聚类与主题下拉 ==")
    from app import topicmatch

    thr = settings.topic_match_threshold
    results.append(ok("写法差异归一：全半角、标点空格、中文数字都能对上",
                      topicmatch.normalize("第三届「环保」演讲") == topicmatch.normalize("第3届环保演讲"),
                      topicmatch.normalize("第３届 环保—演讲")))
    rows18 = [
        {"id": 1, "topic": "春天", "title": "A1"},
        {"id": 2, "topic": "环保宣传", "title": "A2"},
        {"id": 3, "topic": "", "title": "第3届演讲比赛"},
        {"id": 4, "topic": "演讲比赛第三届", "title": "A4"},
        {"id": 5, "topic": "演讲比赛 第三届", "title": "A5"},
    ]
    ids18 = topicmatch.same_topic_ids(rows18, 5, thr)
    results.append(ok("同主题圈选：变体写法与空主题回退标题都进组，异主题和短词不进",
                      ids18 == [3, 4, 5], str(ids18)))
    results.append(ok("同系列不同届次算同一主题（第二届 vs 第三届）",
                      topicmatch.same_topic("第二届人工智能大会", "第三届人工智能大会", thr)))
    results.append(ok("短于 4 字的主题只认完全相同：春天 vs 秋天相似度 0",
                      topicmatch.similarity("春天", "秋天") == 0.0
                      and topicmatch.similarity("春天", "春天") == 1.0))
    results.append(ok("语义无关的主题不误伤",
                      not topicmatch.same_topic("读书分享", "篮球比赛", thr)))
    clus18 = topicmatch.cluster([{"id": 1, "topic": "校园环保宣讲会", "title": ""},
                                 {"id": 2, "topic": "环保宣讲会", "title": ""},
                                 {"id": 3, "topic": "运动会", "title": ""}], thr)
    results.append(ok("聚类：相近写法并成一簇，无关主题各自成簇",
                      [(len(c["members"])) for c in clus18] == [2, 1],
                      str([(c["label"], len(c["members"])) for c in clus18])))
    results.append(ok("阈值边界：环保宣传 vs 环保宣讲会相似度 0.571，低于 0.6 不并簇",
                      not topicmatch.same_topic("环保宣传", "环保宣讲会", thr)))
    sel18 = topicmatch.topic_selects(rows18, thr)
    results.append(ok("主题下拉：全部簇都列出（含仅 1 次的），次数多的在前，锚点取簇内最新 id",
                      [(s["label"], s["n"], s["anchor"]) for s in sel18] ==
                      [("第3届演讲比赛", 3, 5), ("环保宣传", 1, 2), ("春天", 1, 1)],
                      str([(s["label"], s["n"], s["anchor"]) for s in sel18])))

    print("\n== 19. 改进建议清洗：键名漂移兼容、空行丢弃与整组兜底 ==")
    agg19 = {"dimensions": [{"name": "内容结构"}, {"name": "舞台表现"}]}
    s19a = analyze._clean_suggestions(
        [{"priority": 1, "dimension": "内容结构", "action": "列三点式展开",
          "example": "第一…", "practice": "重讲一遍"},
         {"priority": 2, "dimension": "不存在的维度", "suggestion": "开头用提问钩住主题"}], agg19)
    results.append(ok("建议清洗：标准 schema 全字段取齐；异名键 suggestion 也当正文；越界维度名清空",
                      s19a[0]["action"] == "列三点式展开" and s19a[0]["dimension"] == "内容结构"
                      and s19a[1]["action"] == "开头用提问钩住主题" and s19a[1]["dimension"] == "",
                      str(s19a)))
    s19b = analyze._clean_suggestions(
        [{"priority": "二", "建议": "手势别插兜", "示例": "把手放在中线"},
         {"point": "把结尾从口号改成回扣主题的一句行动号召"}], agg19)
    results.append(ok("建议清洗：中文键名与陌生键（取最长字符串值）都兜得住，priority 非法回落序号",
                      s19b[0]["action"] == "手势别插兜" and s19b[0]["example"] == "把手放在中线"
                      and s19b[0]["priority"] == 1 and s19b[1]["priority"] == 2
                      and s19b[1]["action"].startswith("把结尾"), str(s19b)))
    s19c = analyze._clean_suggestions(
        [{"priority": 1}, {"priority": 2, "example": "只有示例也要能看"}, "纯字符串建议", "  "], agg19)
    results.append(ok("建议清洗：只有示例的提为正文、字符串项保留、全空行与空串丢弃（不再出现空徽标行）",
                      [x["action"] for x in s19c] == ["只有示例也要能看", "纯字符串建议"], str(s19c)))

    class _NarrStub:
        def chat(self, prompt, system="", **kw):
            return Completion(text=self.payload, model="mock", usage=None)

    narr_stub = _NarrStub()
    narr_stub.payload = json.dumps(
        {"summary": "整体尚可。", "advantages": ["流畅"], "disadvantages": ["结构散"],
         "suggestions": [{"priority": 1}, {"priority": 2}]}, ensure_ascii=False)
    agg19obj = analyze.Aggregated(
        dimensions=[{"key": "content", "name": "内容结构", "score": 4, "max_score": 25,
                     "ratio": 0.16, "confidence": 0.8, "notes": "", "strengths": [],
                     "issues": [{"desc": "结构散", "fix": "", "example": "", "quote": "",
                                 "at": "", "evidence_ok": False}]},
                    {"key": "delivery", "name": "舞台表现", "score": 20, "max_score": 25,
                     "ratio": 0.8, "confidence": 0.9, "notes": "", "strengths": ["流畅"],
                     "issues": []}],
        total=24, max_total=50, band="C", pct=48.0)
    narr19 = analyze.build_narrative(narr_stub, rubric, agg19obj, "转写正文", "环保宣讲")
    results.append(ok("建议清洗：模型给的全是抠不出正文的空壳时整组回落维度兜底，报告页不出现空建议段",
                      len(narr19["suggestions"]) >= 1
                      and all(str(s.get("action") or "").strip() for s in narr19["suggestions"])
                      and narr19["summary"] == "整体尚可。", str(narr19["suggestions"])))

    print("\n" + "=" * 56)
    passed = sum(1 for r in results if r)
    print(f"通过 {passed}/{len(results)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
