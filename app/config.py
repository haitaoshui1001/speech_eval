from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
# 允许用进程环境变量改指向，测试据此隔离，避免写坏用户的 .env
ENV_FILE = Path(os.getenv("ENV_FILE", "").strip() or (BASE_DIR / ".env"))
load_dotenv(ENV_FILE)

PLACEHOLDER_KEYS = {"", "sk-", "your-api-key", "sk-请替换成你的apikey", "none"}

# 用户可能把 key 直接放在目录下的文本文件里，这里按顺序探测。
KEY_FILES = ("api_key.txt", "API_KEY.txt", "apikey.txt", "api-key.txt",
             "key.txt", "dashscope_key.txt", "千问api_key.txt", "APIKEY.txt")


def key_is_valid(value: str) -> bool:
    v = (value or "").strip().strip('"').strip("'")
    return len(v) >= 12 and v.lower() not in PLACEHOLDER_KEYS


def _read_key_file(p: Path) -> dict[str, str]:
    """解析 api_key.txt，支持：
        api key: sk-xxx / DASHSCOPE_API_KEY=sk-xxx / 裸 key
        base url: https://.../compatible-mode/v1
        model: qwen3.8-flash          （文本/图像评价模型）
        vision model / audio model / asr model  （可选，分通道指定）
    """
    try:
        raw_lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return {}
    labels = {
        "api_key": "api_key", "apikey": "api_key", "dashscope_api_key": "api_key",
        "key": "api_key", "token": "api_key",
        "base_url": "base_url", "baseurl": "base_url", "qwen_base_url": "base_url",
        "endpoint": "base_url", "url": "base_url",
        "model": "model", "chat_model": "model", "text_model": "model",
        "vision_model": "vision_model", "vlm_model": "vision_model",
        "image_model": "vision_model",
        "audio_model": "audio_model", "omni_model": "audio_model",
        "asr_model": "asr_model", "transcribe_model": "asr_model",
    }
    found: dict[str, str] = {}
    for line in raw_lines:
        ln = line.strip()
        if not ln or ln.startswith("#"):
            continue
        for sep in (":", "="):
            if sep in ln:
                label, value = ln.split(sep, 1)
                label = label.strip().lower().replace("-", "_").replace(" ", "")
                value = value.strip().strip('"').strip("'")
                slot = labels.get(label)
                if slot and value:
                    found.setdefault(slot, value)
                break
        if "api_key" not in found and not any(s in ln for s in (":", "=")):
            if key_is_valid(ln):
                found["api_key"] = ln
    return found


def _candidates() -> list[tuple[str, str]]:
    """返回 (来源说明, 原始值) 列表，优先级从高到低。"""
    out: list[tuple[str, str]] = [("环境变量 DASHSCOPE_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))]
    for name in KEY_FILES:
        p = BASE_DIR / name
        if not p.exists():
            continue
        key = _read_key_file(p).get("api_key", "")
        if key:
            out.append((name, key))
    return out


def resolve_api_key() -> tuple[str, str]:
    for source, raw in _candidates():
        v = (raw or "").strip().strip('"').strip("'")
        if key_is_valid(v):
            return v, source
    return "", ""


def _file_values() -> dict[str, str]:
    """从第一个存在的 key 文件里读 base_url / model（与 key 同源，避免混用）。"""
    for name, _ in _candidates():
        p = BASE_DIR / name
        if p.exists():
            return _read_key_file(p)
    for name in KEY_FILES:
        p = BASE_DIR / name
        if p.exists():
            return _read_key_file(p)
    return {}


FILE_VALUES = _file_values()
FILE_MODEL = FILE_VALUES.get("model", "")
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def resolve_base_url() -> str:
    """网关地址优先级：.env / 环境变量填写 > api_key.txt 的 base url > 官方默认。"""
    from_env = os.getenv("QWEN_BASE_URL", "").strip()
    if from_env:
        return from_env.rstrip("/")
    return (FILE_VALUES.get("base_url") or DEFAULT_BASE_URL).strip().rstrip("/")


def resolve_model(env_name: str, file_slots: tuple[str, ...], default: str) -> str:
    """模型优先级：.env / 环境变量填写 > api_key.txt 的 model / vision model / … > 内置默认。

    只要 .env 里填了非空值就视为显式指定（网页设置页正是这样写回的）；
    留空则沿用 api_key.txt，那是用户指定的“唯一配置入口”。
    """
    from_env = os.getenv(env_name, "").strip()
    if from_env:
        return from_env
    for slot in file_slots:
        val = (FILE_VALUES.get(slot) or "").strip()
        if val:
            return val
    return default.strip()


def _flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class EnvField:
    """设置页的字段声明：一项对应 .env 里的一个键。"""
    key: str
    label: str
    kind: str                      # secret / text / select / bool / int / float
    attr: str = ""                 # 对应的 settings 属性名（用于回显生效值）
    hint: str = ""                 # 输入框下方说明
    note: str = ""                 # 生效方式 / 副作用
    default: str = ""
    choices: tuple[tuple[str, str], ...] = ()
    file_slot: tuple[str, ...] = ()  # api_key.txt 里的对应槽位
    low: float | None = None
    high: float | None = None
    step: float | None = None
    scale: float = 1.0               # 生效值与 .env 数值的换算（如 MB ↔ 字节）
    clearable: bool = False          # 密钥类是否允许“清除覆盖”


ENV_GROUPS: tuple[tuple[str, str, tuple[EnvField, ...]], ...] = (
    ("服务接入", "密钥与网关地址优先读 api_key.txt，这里留空即沿用；填了才覆盖。", (
        EnvField("DASHSCOPE_API_KEY", "API Key", "secret", "api_key",
                 "填写后优先于 api_key.txt；保存时留空表示不修改。",
                 "写入 .env 后立即用于新的分析任务",
                 file_slot=("api_key",), clearable=True),
        EnvField("QWEN_BASE_URL", "网关地址", "text", "base_url",
                 "兼容模式接口地址，留空则用 api_key.txt 里的地址。",
                 "留空默认使用官方网关地址", default=DEFAULT_BASE_URL,
                 file_slot=("base_url",)),
        EnvField("LLM_MODE", "运行模式", "select", "llm_mode",
                 "决定分析是否真的调用在线评审服务。", "切换后立即生效",
                 choices=(("auto", "auto · 有密钥走真实，否则演示"),
                          ("real", "real · 强制真实调用"),
                          ("mock", "mock · 演示模式，不联网")), default="auto"),
    )),
    ("评审通道", "三通道评审（文本 / 画面 / 语音）所使用的服务与参数，留空即沿用 api_key.txt。", (
        EnvField("CHAT_MODEL", "文本评审通道", "text", "chat_model",
                 "按评分标准逐维度打分的主通道。", "影响新任务",
                 default="qwen-plus", file_slot=("model",)),
        EnvField("VLM_MODEL", "画面评审通道", "text", "vlm_model",
                 "观看关键帧、判断台风与目光。", "影响新任务",
                 default="qwen-vl-max", file_slot=("vision_model", "model")),
        EnvField("OMNI_MODEL", "语音评审通道", "text", "omni_model",
                 "听语速、停顿与响度。", "影响新任务",
                 default="qwen-omni-turbo-latest", file_slot=("audio_model",)),
        EnvField("ASR_MODEL", "在线转录通道", "text", "asr_model",
                 "语音识别服务标识（转录引擎为「在线」时使用）。", "影响新任务",
                 default="qwen3-asr-flash", file_slot=("asr_model",)),
        EnvField("ASR_ENGINE", "转录引擎", "select", "asr_engine",
                 "「本地」为离线转录，需已下载模型权重。", "影响新任务",
                 choices=(("qwen", "在线 · 云端语音识别"),
                          ("whisper", "本地 · 离线语音识别")), default="qwen"),
        EnvField("WHISPER_MODEL_SIZE", "本地转录精度", "text", "whisper_model_size",
                 "可选 tiny / base / small / medium / large-v3，越大越准也越慢。",
                 "仅在转录引擎为「本地」时使用", default="small"),
        EnvField("USE_VISION_CHANNEL", "启用画面通道", "bool", "use_vision_channel",
                 "关闭后仅用文本 + 语音评价，速度更快。", "影响新任务", default="1"),
        EnvField("USE_AUDIO_CHANNEL", "启用语音通道", "bool", "use_audio_channel",
                 "关闭后不再进行音频听审。", "影响新任务", default="1"),
    )),
    ("关键帧与微表情", "抽帧密度与人脸质控指标，只影响证据采集，不改评分标准。", (
        EnvField("FRAME_INTERVAL", "抽帧间隔（秒）", "int", "frame_interval",
                 "每隔多少秒取一张关键帧，等间隔均匀铺满整段。", "影响新任务",
                 default="5", low=1, high=60),
        EnvField("FRAME_COUNT_MAX", "最多帧数", "int", "frame_count_max",
                 "长视频封顶，控制 token 消耗；短视频不抬底，按时长如实等距采样。",
                 "影响新任务", default="100", low=1, high=100),
        EnvField("FRAME_COUNT", "兜底帧数", "int", "frame_count",
                 "仅在时长探测失败时使用。", "影响新任务", default="12", low=1, high=100),
        EnvField("FRAME_WIDTH", "帧宽（像素）", "int", "frame_width",
                 "长边缩放宽度，不要放大大于原始分辨率。", "影响新任务",
                 default="640", low=160, high=1920),
        EnvField("USE_FACE_METRICS", "启用人脸质控", "bool", "use_face_metrics",
                 "统计正脸率 / 低头率 / 目光偏向，缺 opencv 时自动降级。",
                 "影响新任务", default="1"),
        EnvField("FACE_MIN_DETECT", "人脸检测阈值", "float", "face_min_detect",
                 "0-1，越低越容易检到侧脸，也越容易误检。", "影响新任务",
                 default="0.5", low=0, high=1, step=0.05),
        EnvField("FACE_MAX_FRAMES", "人脸分析帧上限", "int", "face_max_frames",
                 "抽帧多时按此上限等间隔采样。", "影响新任务", default="120", low=1, high=600),
    )),
    ("站点与安全", "会话、上传与分析并发；管理员账号保存后会同步到数据库。", (
        EnvField("SECRET_KEY", "会话签名密钥", "secret", "secret_key",
                 "留空表示不修改；建议换成长随机串。", "修改后所有人需重新登录"),
        EnvField("ADMIN_USERNAME", "管理员账号", "text", "admin_username",
                 "首次启动用它创建管理员，保存后同步改名。", "同步数据库 users 表"),
        EnvField("ADMIN_PASSWORD", "管理员口令", "secret", "admin_password",
                 "留空表示不修改；至少 6 位。", "同步数据库，普通用户不受影响"),
        EnvField("MAX_VIDEO_MB", "单个视频上限（MB）", "int", "max_video_bytes",
                 "上传时校验。", "影响下一次上传", default="500", low=1, high=102400,
                 scale=1024 * 1024),
        EnvField("USER_UPLOAD_LIMIT", "每人上传次数", "int", "user_upload_limit",
                 "新账号默认可上传的视频个数，用完即止；管理员可在后台单独重置或放行。",
                 "影响新账号与未单独设置的账号", default="5", low=1, high=10000),
        EnvField("SESSION_DAYS", "登录保持（天）", "int", "session_days",
                 "会话 cookie 有效期。", "影响新的登录", default="7", low=1, high=365),
        EnvField("MAX_ANALYZERS", "并发分析数", "int", "max_analyzers",
                 "同时分析的视频个数，超出排队的用户会看到「等待中」。",
                 "影响新任务", default="10", low=1, high=64),
        EnvField("MAX_LLM_REQUESTS", "在线请求并发数", "int", "max_llm_requests",
                 "全局同时在途的在线评审请求个数，防止多路分析同时打满网关被限流。",
                 "影响新任务", default="20", low=1, high=200),
        EnvField("WEB_THREADS", "网页工作线程数", "int", "web_threads",
                 "同时处理网页请求（打开页面、上传、查进度）的线程数。",
                 "重启服务后生效", default="64", low=8, high=512),
        EnvField("REQUEST_TIMEOUT", "请求超时（秒）", "int", "request_timeout",
                 "单次在线评审请求超时。", "影响新任务", default="180", low=10, high=1800),
    )),
)

ENV_FIELDS: tuple[EnvField, ...] = tuple(f for _, _, fs in ENV_GROUPS for f in fs)
ENV_BY_KEY: dict[str, EnvField] = {f.key: f for f in ENV_FIELDS}


@dataclass
class Settings:
    base_dir: Path = BASE_DIR
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "").strip() or (BASE_DIR / "data")))
    rubric_file: Path = BASE_DIR / "评价标准.txt"

    api_key: str = field(default_factory=lambda: resolve_api_key()[0])
    api_key_source: str = field(default_factory=lambda: resolve_api_key()[1])
    base_url: str = field(default_factory=lambda: resolve_base_url())
    llm_mode: str = field(default_factory=lambda: os.getenv("LLM_MODE", "auto").strip().lower())

    asr_model: str = field(default_factory=lambda: resolve_model("ASR_MODEL", ("asr_model",), "qwen3-asr-flash"))
    chat_model: str = field(default_factory=lambda: resolve_model("CHAT_MODEL", ("model",), "qwen-plus"))
    vlm_model: str = field(default_factory=lambda: resolve_model("VLM_MODEL", ("vision_model", "model"), "qwen-vl-max"))
    omni_model: str = field(default_factory=lambda: resolve_model("OMNI_MODEL", ("audio_model",), "qwen-omni-turbo-latest"))

    asr_engine: str = field(default_factory=lambda: os.getenv("ASR_ENGINE", "qwen").strip().lower())
    whisper_model_size: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL_SIZE", "small"))
    use_vision_channel: bool = field(default_factory=lambda: _flag("USE_VISION_CHANNEL"))
    use_audio_channel: bool = field(default_factory=lambda: _flag("USE_AUDIO_CHANNEL"))

    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "insecure-dev-secret"))
    admin_username: str = field(default_factory=lambda: os.getenv("ADMIN_USERNAME", "admin"))
    admin_password: str = field(default_factory=lambda: os.getenv("ADMIN_PASSWORD", "admin123"))
    max_video_bytes: int = field(default_factory=lambda: _int("MAX_VIDEO_MB", 500) * 1024 * 1024)
    user_upload_limit: int = field(default_factory=lambda: _int("USER_UPLOAD_LIMIT", 5))
    session_days: int = field(default_factory=lambda: _int("SESSION_DAYS", 7))

    frame_count: int = field(default_factory=lambda: _int("FRAME_COUNT", 12))
    frame_interval: int = field(default_factory=lambda: _int("FRAME_INTERVAL", 5))
    frame_count_max: int = field(default_factory=lambda: _int("FRAME_COUNT_MAX", 100))
    frame_width: int = field(default_factory=lambda: _int("FRAME_WIDTH", 640))
    use_face_metrics: bool = field(default_factory=lambda: _flag("USE_FACE_METRICS"))
    face_min_detect: float = field(default_factory=lambda: _float("FACE_MIN_DETECT", 0.5))
    face_max_frames: int = field(default_factory=lambda: _int("FACE_MAX_FRAMES", 120))
    max_analyzers: int = field(default_factory=lambda: _int("MAX_ANALYZERS", 10))
    max_llm_requests: int = field(default_factory=lambda: _int("MAX_LLM_REQUESTS", 20))
    web_threads: int = field(default_factory=lambda: _int("WEB_THREADS", 64))
    request_timeout: int = field(default_factory=lambda: _int("REQUEST_TIMEOUT", 180))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "speech.db"

    @property
    def video_dir(self) -> Path:
        return self.data_dir / "videos"

    @property
    def artifact_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def has_api_key(self) -> bool:
        return key_is_valid(self.api_key)

    @property
    def real_mode(self) -> bool:
        if self.llm_mode == "mock":
            return False
        if self.llm_mode == "real":
            return True
        return self.has_api_key

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.video_dir, self.artifact_dir):
            p.mkdir(parents=True, exist_ok=True)

    def reload(self) -> None:
        """原地刷新所有由环境变量推导的字段，保持 settings 单例不换对象。

        全项目都是运行时读 settings.xxx，所以刷新后立即对之后的请求生效，无需重启。
        """
        for f in fields(self):
            if callable(f.default_factory):
                setattr(self, f.name, f.default_factory())
        self.ensure_dirs()


settings = Settings()
settings.ensure_dirs()


# ------------------------------------------------------------------ .env 读写
def _unquote(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in {'"', "'"}:
        return v[1:-1]
    return v


def read_env_file() -> dict[str, str]:
    """按 KEY=VALUE 解析 .env（忽略注释），用于区分“文件里写了什么”。"""
    out: dict[str, str] = {}
    try:
        lines = ENV_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return out
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = _unquote(v)
    return out


def _quote_value(v: str) -> str:
    v = (v or "").replace("\r", " ").replace("\n", " ").strip()
    if not v:
        return ""
    if any(c in v for c in (' ', '"', "'", "#")) or v != v.strip():
        return '"' + v.replace('"', "'") + '"'
    return v


def write_env_file(updates: Mapping[str, str]) -> int:
    """就地改写 .env：保留原有注释、空行、键顺序与换行风格，新键追加在末尾。原子替换。"""
    raw = b""
    try:
        raw = ENV_FILE.read_bytes()
    except OSError:
        pass
    lines = raw.decode("utf-8", "ignore").splitlines()
    crlf = raw.count(b"\r\n")
    eol = "\r\n" if crlf > raw.count(b"\n") - crlf else "\n"
    pending = dict(updates)
    out: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#") or "=" not in s:
            out.append(ln)
            continue
        name = s.split("=", 1)[0].strip()
        if name in pending:
            out.append(f"{name}={_quote_value(pending.pop(name))}")
        else:
            out.append(ln)
    while out and not out[-1].strip():
        out.pop()
    for name, value in pending.items():
        out.append(f"{name}={_quote_value(value)}")
    text = eol.join(out) + eol if out else ""
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ENV_FILE.parent / (ENV_FILE.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    os.replace(tmp, ENV_FILE)
    return len(updates)


def refresh_runtime() -> None:
    """把 .env 重新灌进 os.environ、重算 api_key.txt 取值，并热刷新 settings。"""
    global FILE_VALUES, FILE_MODEL
    load_dotenv(ENV_FILE, override=True)
    FILE_VALUES = _file_values()
    FILE_MODEL = FILE_VALUES.get("model", "")
    settings.reload()


def mask_secret(value: str) -> str:
    v = (value or "").strip()
    if not v:
        return ""
    if len(v) <= 8:
        return "*" * len(v)
    return f"{v[:4]}{'*' * min(len(v) - 8, 12)}{v[-4:]}"


def value_source(f: EnvField, file_values: dict[str, str] | None = None) -> str:
    """生效值来自哪里：.env / 环境变量 / api_key.txt / 内置默认。"""
    if ((file_values if file_values is not None else read_env_file()).get(f.key) or "").strip():
        return ".env"
    if (os.environ.get(f.key) or "").strip():
        return "环境变量"
    for slot in f.file_slot:
        if (FILE_VALUES.get(slot) or "").strip():
            return "api_key.txt"
    return "内置默认"


def effective_text(f: EnvField) -> str:
    raw = getattr(settings, f.attr) if f.attr and hasattr(settings, f.attr) else ""
    if isinstance(raw, bool):
        return "1" if raw else "0"
    if f.scale != 1.0 and isinstance(raw, (int, float)):
        return _num(raw / f.scale)
    return str(raw)


_SECRET_ATTR = {"DASHSCOPE_API_KEY": "api_key", "SECRET_KEY": "secret_key",
                "ADMIN_PASSWORD": "admin_password"}


def settings_overview(overrides: Mapping[str, str] | None = None) -> list[dict]:
    """给设置页用的分组视图：每行含 EnvField、当前生效值、来源与掩码摘要。

    overrides 用于校验失败时回显用户刚填的值（密钥类不回显）。
    value 是“生效值”，raw 是“输入框里该预填什么”：只有真的写在 .env 里的值才回填，
    来自 api_key.txt 或内置默认的一律留空，避免一次无意的保存把上游值钉死进 .env。
    """
    file_values = read_env_file()
    groups = []
    for title, desc, fs in ENV_GROUPS:
        rows = []
        for f in fs:
            attr = _SECRET_ATTR.get(f.key)
            pending = bool(overrides) and f.kind != "secret" and f.key in overrides
            src = value_source(f, file_values)
            if pending:
                value = raw = str(overrides[f.key])
            else:
                value = "" if f.kind == "secret" else effective_text(f)
                raw = ((file_values.get(f.key) or "") if src == ".env" else "") if f.kind != "bool" else value
            rows.append({
                "field": f,
                "value": value,
                "raw": raw,
                "pending": pending,
                "shown": mask_secret(str(getattr(settings, attr))) if attr else "",
                "source": src,
            })
        groups.append({"title": title, "desc": desc, "rows": rows})
    return groups


def _num(value: float) -> str:
    return f"{value:g}"


def validate_env_updates(raw: Mapping[str, str],
                         clears: tuple[str, ...] = ()) -> tuple[dict[str, str], list[str]]:
    """把表单值整理成可写回 .env 的字典，返回 (值, 错误列表)。"""
    updates: dict[str, str] = {}
    errors: list[str] = []
    for f in ENV_FIELDS:
        if f.kind == "bool":
            updates[f.key] = "1" if str(raw.get(f.key, "")).strip() in {"1", "on", "true", "yes"} else "0"
            continue
        value = str(raw.get(f.key, "")).replace("\r", " ").replace("\n", " ").strip()
        if f.kind == "secret":
            if value:
                updates[f.key] = value
            elif f.clearable and f.key in clears:
                updates[f.key] = ""
            continue
        if f.kind == "select":
            allowed = {c[0] for c in f.choices}
            if not value:
                updates[f.key] = ""       # 空选项 = 不覆盖，沿用 api_key.txt / 内置默认
                continue
            if value not in allowed:
                errors.append(f"{f.label}：取值只能是 {' / '.join(sorted(allowed))}")
                continue
            updates[f.key] = value
            continue
        if f.kind in {"int", "float"}:
            if not value:
                updates[f.key] = ""
                continue
            try:
                num = float(value)
            except ValueError:
                errors.append(f"{f.label}：需要填数字")
                continue
            if f.kind == "int" and not num.is_integer():
                errors.append(f"{f.label}：需要填整数")
                continue
            if f.low is not None and num < f.low:
                errors.append(f"{f.label}：不得小于 {_num(f.low)}")
                continue
            if f.high is not None and num > f.high:
                errors.append(f"{f.label}：不得大于 {_num(f.high)}")
                continue
            updates[f.key] = str(int(num)) if f.kind == "int" else _num(num)
            continue
        updates[f.key] = value
    if updates.get("ASR_ENGINE") == "whisper" and not (updates.get("WHISPER_MODEL_SIZE") or "").strip():
        updates["WHISPER_MODEL_SIZE"] = "small"
    if updates.get("QWEN_BASE_URL"):
        updates["QWEN_BASE_URL"] = updates["QWEN_BASE_URL"].rstrip("/")
    return updates, errors


def save_settings(updates: Mapping[str, str]) -> list[str]:
    """写 .env 并热刷新，返回实际发生变化的键。"""
    before = read_env_file()
    changed = [k for k, v in updates.items() if (before.get(k) or "") != v]
    if changed:
        write_env_file({k: updates[k] for k in changed})
    refresh_runtime()
    return changed
