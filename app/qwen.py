"""千问（DashScope OpenAI 兼容模式）客户端：文本 / 图像 / 音频多模态调用。

只依赖 requests，走 /compatible-mode/v1/chat/completions：
  - chat()        普通文本（qwen-plus 等）
  - vision()      图像序列（qwen-vl-*）
  - audio()       音频（qwen-omni-*，强制流式聚合）
  - asr()         语音转文字（qwen3-asr-flash）
  - tts()         朗读合成（qwen3-tts-*，走原生 multimodal-generation 端点）
  - create_voice() 声音复刻（qwen-voice-enrollment，走原生 customization 端点）
所有调用带重试与降级（流式回退、错误信息透传）。
"""
from __future__ import annotations

import base64
import json
import mimetypes
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

import requests

from . import prompts
from .config import settings

MIME_BY_EXT = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
    ".aac": "audio/aac", ".ogg": "audio/ogg", ".flac": "audio/flac",
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp",
}


class QwenError(RuntimeError):
    def __init__(self, message: str, status: int = 0, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class RequestGate:
    """进程内「同时在途千问请求」闸门。

    多路分析共用一个网关额度：不设限的话，10 路分析会在同一瞬间并发几十个请求，
    轻则排队重则被服务端 429 打回，重试退避又放大拥塞。这里用一个信号量把发起阶段
    的执行数钉在 MAX_LLM_REQUESTS 以内，超出的调用先安静等待，而不是把错误抛给用户。

    设置页可以热改额度，所以数量变化时按需重建信号量；已放出去的票仍归旧信号量管，
    不会互相干扰。等待超过 timeout 仍拿不到额度则报错，让任务失败可见而不是无限挂起。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sem = threading.BoundedSemaphore(1)
        self._size = 0
        self._active = 0

    def _current(self) -> threading.BoundedSemaphore:
        want = max(1, settings.max_llm_requests)
        with self._lock:
            if want != self._size:
                self._size = want
                self._sem = threading.BoundedSemaphore(want)
            return self._sem

    @contextmanager
    def slot(self, timeout: float | None = None) -> Iterator[None]:
        sem = self._current()
        t0 = time.monotonic()
        if not sem.acquire(timeout=timeout):
            raise QwenError(f"评审请求排队超过 {int((timeout or 0) / 60)} 分钟，请稍后重试分析")
        queued = time.monotonic() - t0
        self._active += 1
        try:
            yield
        finally:
            self._active -= 1
            sem.release()
            # 排队时长要在 acquire 之后取，若在 finally 里用同一个起点量，
            # 打出来的其实是「排队 + 整次请求」，会把人引去查网关限流而真正慢的是模型。
            if queued > 5:
                print(f"[qwen] 网关排队 {queued:.0f}s 后才发出请求")

    @property
    def in_flight(self) -> int:
        return self._active

    @property
    def limit(self) -> int:
        self._current()
        return self._size


GATE = RequestGate()


def data_url(path: Path) -> str:
    mime = MIME_BY_EXT.get(path.suffix.lower()) or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def image_url_item(path: Path) -> dict:
    return {"type": "image_url", "image_url": {"url": data_url(path)}}


def input_audio_item(path: Path) -> dict:
    return {"type": "input_audio", "input_audio": {"data": data_url(path)}}


def text_item(t: str) -> dict:
    return {"type": "text", "text": t}


# TTS 不在兼容模式路径上：与 compatible-mode 同 host，走原生 multimodal-generation。
_NATIVE_TTS_PATH = "/api/v1/services/aigc/multimodal-generation/generation"
# 声音复刻（创建/删除定制音色）是另一个原生端点，同样只借 compatible-mode 的 host。
_NATIVE_VC_PATH = "/api/v1/services/audio/tts/customization"
# 复刻接口的固定入口模型：它只负责「造音色」，产出的音色归 target_model 使用。
VOICE_ENROLLMENT_MODEL = "qwen-voice-enrollment"
# 官方限制 Base64 直传的样本编码后 <10MB；按编码串长度卡线，避免传上去才被拒。
_VC_MAX_DATA_URL_CHARS = 10 * 1024 * 1024


def _native_url(base_url: str, path: str = _NATIVE_TTS_PATH) -> str:
    """从 base_url（…/compatible-mode/v1）推导原生端点（scheme://host + 原生路径）。"""
    u = urlsplit(base_url)
    if u.scheme and u.netloc:
        return f"{u.scheme}://{u.netloc}{path}"
    return base_url


@dataclass
class Completion:
    text: str
    model: str
    usage: dict | None = None


@dataclass
class TtsAudio:
    """合成回执：url 为临时签名地址（优先），data 为 base64 兜底；url 绝不落库。"""
    model: str
    url: str = ""
    data: bytes | None = None
    characters: int = 0
    instructions_used: bool = False


@dataclass
class VoiceProfile:
    """复刻音色回执：voice 就是合成时的 voice 参数，与 target_model 死绑、不可跨模型使用。

    fallback_mode=True 表示服务方没拿到理想样本、按降级配置建了音色，
    此时音色能用但效果未必像，必须让学生知情（原因见 fallback_reason）。
    """
    voice: str
    target_model: str = ""
    fallback_mode: bool = False
    fallback_reason: str = ""


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

# ------------------------------------------------------------ token 估算
# 流式响应不一定回传 usage（部分 omni 模型就不回），此时按下面的粗估规则记账，
# 并在记录里打 estimated 标记，界面上会注明「估算」。
_CJK_RE = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")
IMAGE_TOKENS = 1000          # 一张关键帧的粗估开销
AUDIO_B64_CHARS_PER_TOKEN = 900   # 16k 单声道 wav ≈ 47 token/秒


def estimate_tokens(text: str) -> int:
    """中日韩字符按 1 token，其余按 4 字符 1 token 粗估。"""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    return max(1, cjk + int((len(text) - cjk) / 4))


def prompt_tokens_estimate(payload: dict) -> int:
    total = 0
    for msg in payload.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
            continue
        for part in content or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" or "text" in part:
                total += estimate_tokens(part.get("text") or "")
            elif "image_url" in part:
                total += IMAGE_TOKENS
            elif "input_audio" in part:
                b64 = len(((part.get("input_audio") or {}).get("data")) or "")
                total += max(1, b64 // AUDIO_B64_CHARS_PER_TOKEN)
    return total


def parse_json_block(raw: str) -> dict:
    """从模型输出里稳健地抠出 JSON 对象。"""
    if not raw:
        raise QwenError("评审服务返回为空")
    candidates = [raw.strip()]
    candidates += [m.strip() for m in _FENCE_RE.findall(raw)]
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start:end + 1])
    for c in candidates:
        if not c.startswith("{"):
            continue
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise QwenError("评审结果不是合法 JSON", body=raw[:600])


class QwenClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 timeout: int | None = None, retries: int = 3):
        self.api_key = api_key if api_key is not None else settings.api_key
        self.base_url = (base_url or settings.base_url).rstrip("/")
        self.timeout = timeout or settings.request_timeout
        self.retries = retries
        self.stage = "未标注"
        self.usage_records: list[dict] = []

    def mark(self, stage: str) -> "QwenClient":
        """标注后续调用的用途，token 统计按此分组。"""
        self.stage = stage
        return self

    def reset_usage(self) -> None:
        self.usage_records = []

    def _track(self, payload: dict, comp: Completion) -> None:
        usage = comp.usage or {}
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total = int(usage.get("total_tokens") or 0) or prompt + completion
        estimated = not (prompt or completion)
        if estimated:
            prompt = prompt_tokens_estimate(payload)
            completion = estimate_tokens(comp.text)
            total = prompt + completion
        self.usage_records.append({
            "stage": self.stage, "model": comp.model, "prompt": prompt,
            "completion": completion, "total": total, "estimated": estimated,
        })

    def track_characters(self, model: str, characters: int, estimated: bool = False) -> None:
        """TTS 按字符计费：token 恒为 0，characters 键进 usage_summary 聚合。

        estimated=True 表示服务方这次没回 usage，字符数按送合成的文本长度折算，
        只用于观察量级；不记这一笔的话「朗读合成」在用量表里就是一条空账。
        """
        self.usage_records.append({
            "stage": self.stage, "model": model, "prompt": 0, "completion": 0,
            "total": 0, "estimated": False, "characters_estimated": bool(estimated),
            "characters": int(characters or 0),
        })

    def track_voice_creation(self, model: str, count: int = 1) -> None:
        """声音复刻按「次」计费（官方 usage.count），既不吃 token 也不按字符，单记一条。

        不记这一笔的话，建音色这笔真实开销在用量表里完全隐身；
        记成 characters 又会把一次 $0.01 的定制混进按千字计费的朗读里，口径全乱。
        """
        self.usage_records.append({
            "stage": self.stage, "model": model, "prompt": 0, "completion": 0,
            "total": 0, "estimated": False, "characters": 0,
            "creations": int(count or 0),
        })

    def usage_summary(self) -> dict:
        """汇总本次会话累计的模型用量：总量 + 按阶段 + 按模型。

        token 与 characters 是两条独立计费轴：评判类调用记 token，朗读合成记字符，
        音色定制记 creations（次）。
        服务方未回 usage 时按文本长度折算，与 token 估算共用「估算」标记。
        """
        buckets: dict[str, dict[str, dict]] = {
            "stage": {}, "model": {}}
        out = {"calls": len(self.usage_records), "prompt": 0, "completion": 0,
               "total": 0, "estimated_calls": 0, "characters": 0, "creations": 0,
               "by_stage": [], "by_model": []}
        for r in self.usage_records:
            chars = int(r.get("characters") or 0)
            creations = int(r.get("creations") or 0)
            est = bool(r["estimated"] or r.get("characters_estimated"))
            out["prompt"] += r["prompt"]
            out["completion"] += r["completion"]
            out["total"] += r["total"]
            out["characters"] += chars
            out["creations"] += creations
            if est:
                out["estimated_calls"] += 1
            for axis, key in (("stage", r["stage"]), ("model", r["model"])):
                row = buckets[axis].setdefault(
                    key, {"name": key, "calls": 0, "prompt": 0, "completion": 0,
                          "total": 0, "estimated": 0, "characters": 0, "creations": 0})
                row["calls"] += 1
                row["prompt"] += r["prompt"]
                row["completion"] += r["completion"]
                row["total"] += r["total"]
                row["estimated"] += 1 if est else 0
                row["characters"] += chars
                row["creations"] += creations
        out["by_stage"] = sorted(buckets["stage"].values(),
                                 key=lambda x: (-x["total"], -x["characters"]))
        out["by_model"] = sorted(buckets["model"].values(),
                                 key=lambda x: (-x["total"], -x["characters"]))
        return out

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and settings.real_mode

    # ---------- 底层 ----------
    def _post(self, payload: dict) -> requests.Response:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                stream = bool(payload.get("stream"))
                with GATE.slot(timeout=self.timeout * 3.0):
                    r = requests.post(url, json=payload, headers=headers,
                                      timeout=self.timeout, stream=stream)
            except requests.RequestException as exc:
                last = exc
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 200:
                return r
            body = r.text[:800]
            if r.status_code in (408, 409, 429) or r.status_code >= 500:
                last = QwenError(f"评审请求失败 {r.status_code}", r.status_code, body)
                time.sleep(2.0 * (attempt + 1))
                continue
            raise QwenError(f"评审请求失败 {r.status_code}", r.status_code, body)
        raise last if isinstance(last, QwenError) else QwenError(f"评审请求异常: {last}")

    def _post_native(self, payload: dict, timeout: int | None = None) -> requests.Response:
        """原生 multimodal-generation 端点（TTS 用），退避口径与 _post 一致。"""
        url = settings.tts_endpoint or _native_url(self.base_url)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        budget = timeout or self.timeout
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                with GATE.slot(timeout=budget * 3.0):
                    r = requests.post(url, json=payload, headers=headers, timeout=budget)
            except requests.RequestException as exc:
                last = exc
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 200:
                return r
            body = r.text[:800]
            if r.status_code in (408, 409, 429) or r.status_code >= 500:
                last = QwenError(f"合成请求失败 {r.status_code}", r.status_code, body)
                time.sleep(2.0 * (attempt + 1))
                continue
            raise QwenError(f"合成请求失败 {r.status_code}", r.status_code, body)
        raise last if isinstance(last, QwenError) else QwenError(f"合成请求异常: {last}")

    def _post_vc(self, payload: dict, timeout: int | None = None) -> requests.Response:
        """音色定制端点（复刻音色的创建/删除），退避口径与 _post_native 一致。

        样本是整段 Base64，服务端还要做预处理，比一次合成的耗时更长，
        因此默认预算不低于 120 秒——按普通 30 秒超时会把正常等待误判成失败。
        """
        url = _native_url(self.base_url, _NATIVE_VC_PATH)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        budget = timeout or max(self.timeout, 120)
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                with GATE.slot(timeout=budget * 3.0):
                    r = requests.post(url, json=payload, headers=headers, timeout=budget)
            except requests.RequestException as exc:
                last = exc
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 200:
                return r
            body = r.text[:800]
            if r.status_code in (408, 409, 429) or r.status_code >= 500:
                last = QwenError(f"音色复刻请求失败 {r.status_code}", r.status_code, body)
                time.sleep(2.0 * (attempt + 1))
                continue
            raise QwenError(f"音色复刻请求失败 {r.status_code}", r.status_code, body)
        raise last if isinstance(last, QwenError) else QwenError(f"音色复刻请求异常: {last}")

    @staticmethod
    def _unwrap(data: dict) -> str:
        try:
            choice = data["choices"][0]
            msg = choice.get("message") or {}
            content = msg.get("content")
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            if not content:
                content = choice.get("text") or ""
            return (content or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise QwenError(f"响应结构异常: {exc}", body=json.dumps(data)[:600])

    def _complete(self, payload: dict) -> Completion:
        model = payload["model"]
        r = self._post(payload)
        if payload.get("stream"):
            parts: list[str] = []
            usage: dict | None = None
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk in ("[DONE]", ""):
                    continue
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj.get("usage"), dict):
                    usage = obj["usage"]
                try:
                    delta = obj["choices"][0].get("delta") or {}
                    piece = delta.get("content")
                    if isinstance(piece, list):
                        piece = "".join(p.get("text", "") for p in piece if isinstance(p, dict))
                    if piece:
                        parts.append(piece)
                except (KeyError, IndexError):
                    continue
            text = "".join(parts).strip()
            if not text:
                raise QwenError("流式响应为空")
            comp = Completion(text=text, model=model, usage=usage)
            self._track(payload, comp)
            return comp
        data = r.json()
        comp = Completion(text=self._unwrap(data), model=model, usage=data.get("usage"))
        self._track(payload, comp)
        return comp

    # ---------- 模型降级 ----------
    # 注意：*-latest 变体在部分账号上是 access_denied（403），因此降级链优先落到稳定版。
    FALLBACKS = {
        "chat": ("qwen-plus", "qwen-turbo", "qwen-max"),
        "vision": ("qwen-vl-max", "qwen3-vl-plus", "qwen-vl-plus", "qwen-vl-ocr"),
        "audio": ("qwen-omni-turbo-latest", "qwen-omni-turbo"),
        "asr": ("qwen3-asr-flash", "qwen-omni-turbo-latest"),
        # 同族降级只是丢掉 instructions、保留朗读（方案 §3.3）
        "tts": ("qwen3-tts-instruct-flash", "qwen3-tts-flash"),
    }

    @classmethod
    def _model_family(cls, model: str) -> str:
        m = (model or "").lower()
        if "tts" in m:
            return "tts"
        if "vl" in m:
            return "vision"
        if "omni" in m:
            return "audio"
        if "asr" in m:
            return "asr"
        return "chat"

    _MODEL_ERROR_HINTS = (
        "model", "not exist", "does not exist", "not_found", "invalidparameter",
        "unsupported", "not supported", "access_denied", "access denied",
        "forbidden", "no access", "not authorized", "arrearage", "not activated",
        "does not exist or you do not have access",
    )

    @classmethod
    def is_model_error(cls, exc: QwenError) -> bool:
        """4xx 中的“模型名/模型权限”问题可以换模型重试；额度、参数、鉴权类不重试。"""
        if exc.status not in (400, 403, 404, 422):
            return False
        blob = (exc.body + str(exc)).lower()
        if any(k in blob for k in ("invalid_api_key", "incorrect api key", "authentication",
                                   "insufficient", "quota", "free allocated")):
            return False
        return any(k in blob for k in cls._MODEL_ERROR_HINTS)

    # 降级提示的通道标签：评审精度受影响与朗读效果受影响必须在同一份质控记录里分得开。
    CHANNEL_LABELS = {
        "chat": "文本评审", "vision": "画面评审", "audio": "语音评审",
        "asr": "在线转录", "tts": "朗读合成",
    }

    # 复刻/定制音色专用模型：这类音色与创建时锁定的 target_model 死绑，换模型即换人。
    CLONE_MARKERS = ("-vc-", "-vd-")

    @classmethod
    def is_clone_model(cls, model: str) -> bool:
        m = (model or "").lower()
        return any(k in m for k in cls.CLONE_MARKERS)

    @staticmethod
    def preferred_name_of(name: str) -> str:
        """把学生填的备注名洗成官方接受的形式：仅字母数字下划线、最长 16 字符。

        官方不接受中文（会 400），而这里只是服务侧的内部标识，展示用的仍是本库里的记录，
        所以清洗而不是报错——纯中文会被洗空，退回 student 占位，不影响使用。
        """
        cleaned = re.sub(r"[^A-Za-z0-9_]", "", (name or "").strip())
        return (cleaned or "student")[:16]

    # 换了模型意味着什么：只写确实会变的后果，不下结论。
    _CONSEQUENCES = {
        "chat": "本次文本内容由替补模型产出，口径可能略有差异",
        "vision": "画面证据由替补模型产出",
        "audio": "听审结论由替补模型产出",
    }

    _RESOLVED: dict[str, str] = {}

    @classmethod
    def _model_note(cls, note: list[str] | None, family: str, label: str, text: str) -> None:
        """把提示落到 [通道] 前缀的句子里，并按内容去重（分块合成会重复触发同一条）。"""
        if note is None:
            return
        line = f"[{label or cls.CHANNEL_LABELS.get(family, '模型')}] {text}"
        if line not in note:
            note.append(line)

    def _model_chain(self, primary: str, family: str,
                     fallbacks: tuple[str, ...] | None = None) -> list[str]:
        """首选 → 已验证的替代名 → 小写变体 → 同族降级链，去重保序。

        三条通道（chat / asr / tts）共用一份组链逻辑，避免各写各的导致行为漂移。
        配置层的模型名已在 config.normalize_model 里小写化，小写变体只是给
        直接传 model= 的调用兜底。
        """
        chain: list[str] = []
        known = QwenClient._RESOLVED.get(primary)
        if known and known != primary:
            chain.append(known)
        chain.append(primary)
        lower = primary.lower()
        if lower != primary:
            chain.append(lower)
        for alt in self.FALLBACKS.get(family, ()) if fallbacks is None else fallbacks:
            if alt not in chain:
                chain.append(alt)
        return chain

    def _try_models(self, payload: dict, note: list[str] | None = None,
                    family: str | None = None, label: str = "") -> Completion:
        """按“同族模型降级链”重试：模型名不存在 / 无权限（403/404）时换下一个。"""
        primary = payload["model"]
        fam = family or self._model_family(primary)
        known = QwenClient._RESOLVED.get(primary)
        tail = self._CONSEQUENCES.get(fam, "")
        last: QwenError | None = None
        for model in self._model_chain(primary, fam):
            payload["model"] = model
            try:
                comp = self._complete(payload)
            except QwenError as exc:
                if not self.is_model_error(exc):
                    raise
                if model == known:
                    QwenClient._RESOLVED.pop(primary, None)   # 记住的替代名也失效了，回到原始探测
                last = exc
                continue
            if model != primary:
                QwenClient._RESOLVED[primary] = model
                if model.lower() == primary.lower():
                    self._model_note(note, fam, label,
                                     f"模型名 {primary} 大小写不被接受，已改用 {model}")
                else:
                    self._model_note(note, fam, label, f"模型 {primary} 不可用，已改用 {model}"
                                     + (f"：{tail}" if tail else ""))
            return comp
        assert last is not None
        raise last

    # ---------- 模态封装 ----------
    def chat(self, prompt: str, system: str = "", model: str | None = None,
             history: list[dict] | None = None, temperature: float = 0.2,
             json_mode: bool = False, max_tokens: int = 4096,
             note: list[str] | None = None, label: str = "") -> Completion:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(history or [])
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": model or settings.chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
            try:
                return self._try_models(payload, note, label=label)
            except QwenError as exc:
                if exc.status not in (400, 404, 422):
                    raise
                payload.pop("response_format", None)
        return self._try_models(payload, note, label=label)

    def chat_content(self, content: list[dict], system: str = "", model: str | None = None,
                     temperature: float = 0.2, json_mode: bool = False,
                     modalities: list[str] | None = None, max_tokens: int = 4096,
                     note: list[str] | None = None, family: str = "chat",
                     label: str = "") -> Completion:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        payload: dict[str, Any] = {
            "model": model or settings.chat_model,
            "messages": messages,
            "temperature": temperature,
        }
        if modalities:
            # omni 系列：必须流式，且不接受 max_tokens / response_format（传了会静默返回空流）
            payload["modalities"] = modalities
            payload["stream"] = True
        else:
            payload["max_tokens"] = max_tokens
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
                try:
                    return self._try_models(payload, note, family, label)
                except QwenError as exc:
                    if exc.status not in (400, 404, 422):
                        raise
                    payload.pop("response_format", None)
        return self._try_models(payload, note, family, label)

    def vision(self, prompt: str, images: list[Path], system: str = "",
               model: str | None = None, json_mode: bool = True,
               temperature: float = 0.2, note: list[str] | None = None,
               label: str = "") -> Completion:
        content = [text_item(prompt)] + [image_url_item(p) for p in images]
        return self.chat_content(content, system=system, model=model or settings.vlm_model,
                                 json_mode=json_mode, temperature=temperature, note=note,
                                 family="vision", label=label)

    def audio(self, prompt: str, audio: Path, system: str = "",
              model: str | None = None, json_mode: bool = True,
              note: list[str] | None = None, label: str = "") -> Completion:
        content = [input_audio_item(audio), text_item(prompt)]
        return self.chat_content(content, system=system, model=model or settings.omni_model,
                                 json_mode=json_mode, modalities=["text"], note=note,
                                 family="audio", label=label)

    def asr(self, audio: Path, language: str = "en", model: str | None = None,
            note: list[str] | None = None, label: str = "") -> Completion:
        """转写：qwen3-asr-flash 优先，不可用时用 omni 逐字转写兜底。"""
        instruction = prompts.get("asr_instruction")
        primary = model or settings.asr_model
        known = QwenClient._RESOLVED.get(primary)
        last: QwenError | None = None
        for m in self._model_chain(primary, "asr"):
            if "omni" in m.lower() or "audio" in m.lower():
                payload: dict[str, Any] = {
                    "model": m,
                    "messages": [{"role": "user", "content": [input_audio_item(audio), text_item(instruction)]}],
                    "modalities": ["text"],
                    "stream": True,
                }
            else:
                payload = {
                    "model": m,
                    "messages": [{"role": "user", "content": [input_audio_item(audio)]}],
                    "asr_options": {"language": language, "enable_itn": False},
                }
            try:
                comp = self._complete(payload)
            except QwenError as exc:
                last = exc
                if not self.is_model_error(exc):
                    raise
                if m == known:
                    QwenClient._RESOLVED.pop(primary, None)   # 替补名也失效了，回到原始探测
                continue
            if m != primary:
                QwenClient._RESOLVED[primary] = m
                tail = "改由通用语音模型逐字转写，精度低于专用转录模型" if "omni" in m.lower() else ""
                self._model_note(note, "asr", label, f"模型 {primary} 不可用，已改用 {m}"
                                 + (f"：{tail}" if tail else ""))
            return comp
        assert last is not None
        raise last

    def tts(self, text: str, voice: str | None = None, instructions: str = "",
            model: str | None = None, timeout: int | None = None,
            note: list[str] | None = None, label: str = "") -> TtsAudio:
        """朗读合成：instruct 版优先，模型不可用时降级 flash 并丢表现力指令。

        走原生 multimodal-generation 端点。非流式响应里 output.audio.url 是
        临时签名地址（audio.data 恒为空，仅作兜底解析）；计费看 usage.characters。
        传入复刻音色时（model 含 -vc-/-vd-）不降级、不送指令，见下方说明。
        """
        primary = model or settings.tts_model
        # cosyvoice 等其它通道协议不同（数值参数 + 异步端点），不能拿 qwen3-tts 降级链去换引擎。
        # 复刻音色更狠：音色与创建时的 target_model 死绑，换模型等于悄悄把学生的声音换成 Neil，
        # 所以 vc/vd 通道锁死单模型，宁可报错让人重做音色，也不产出「听着不像我」的音频。
        clone = self.is_clone_model(primary)
        alt = () if clone or not primary.lower().startswith("qwen3-tts") else self.FALLBACKS["tts"]
        chain = self._model_chain(primary, "tts", alt)
        voice = voice or settings.tts_voice
        want_instr = bool((instructions or "").strip())
        known = QwenClient._RESOLVED.get(primary)
        last: QwenError | None = None
        for m in chain:
            use_instr = bool(settings.tts_instructions and "instruct" in m.lower() and want_instr)
            inp: dict[str, Any] = {"text": text, "voice": voice, "language_type": "Auto"}
            if use_instr:
                inp["instructions"] = instructions
                inp["optimize_instructions"] = True
            payload = {"model": m, "input": inp}
            try:
                data = self._post_native(payload, timeout).json()
            except QwenError as exc:
                last = exc
                if not self.is_model_error(exc):
                    raise
                if m == known:
                    QwenClient._RESOLVED.pop(primary, None)   # 替补名也失效了，回到原始探测
                continue
            except json.JSONDecodeError as exc:
                raise QwenError(f"合成响应解析失败: {exc}") from exc
            output = data.get("output") or {}
            audio = output.get("audio") or {}
            url = (audio.get("url") or "").strip()
            b64 = audio.get("data") or ""
            if not url and not b64:
                raise QwenError("合成响应未包含音频", body=json.dumps(data)[:600])
            chars = int((data.get("usage") or {}).get("characters") or 0)
            # 服务方偶尔不回 usage：按送合成的文本长度折算，合成量不能记成 0。
            self.track_characters(m, chars or len(text), estimated=not chars)
            if m != primary:
                QwenClient._RESOLVED[primary] = m
                lost = want_instr and not use_instr
                self._model_note(note, "tts", label, f"模型 {primary} 不可用，已改用 {m}"
                                 + ("；该模型不支持表现力指令，本次按普通音色合成" if lost else ""))
            elif want_instr and not use_instr:
                if clone:
                    # 官方口径：复刻音色不吃 instructions。静默丢掉会让人以为「我写的演绎没生效」，
                    # 必须说明这次用的是他本人的原声、本来就不该有指令。
                    self._model_note(note, "tts", label,
                                     f"复刻音色（{m}）不支持表现力指令，本次按你的原声朗读，未附加指令")
                else:
                    why = ("表现力指令开关（TTS_INSTRUCTIONS）已关闭" if not settings.tts_instructions
                           else f"模型 {m} 不属于 instruct 通道")
                    self._model_note(note, "tts", label, f"{why}，本次按普通音色合成")
            return TtsAudio(model=m, url=url,
                            data=base64.b64decode(b64) if b64 else None,
                            characters=chars, instructions_used=use_instr)
        assert last is not None
        if clone:
            # 复刻音色长期不用会被服务方清理，直接抛原始 404 学生看不懂，指明「重做我的音色」这条出路。
            raise QwenError(f"复刻音色在模型 {primary} 上不可用（可能已被服务方清理），"
                            f"请重做我的音色后重试：{last}", last.status, last.body) from last
        raise last

    # ---------- 声音复刻 ----------
    def create_voice(self, audio_path: Path, preferred_name: str,
                     target_model: str = "", text: str = "", language: str = "",
                     timeout: int | None = None) -> VoiceProfile:
        """用一段本人朗读样本创建复刻音色。

        样本以 Base64 Data URL 直传（官方支持，编码后须 <10MB），不需要公网地址，
        也不会像上传到第三方存储那样把学生的人声留在别处。
        官方约束：单声道、采样率 ≥24kHz、10~20 秒最佳、含 ≥3 秒连续清晰朗读、无背景音乐与他人声。
        """
        target = (target_model or settings.tts_vc_model).strip().lower()
        if not target:
            raise QwenError("未配置复刻合成模型（TTS_VC_MODEL），无法创建音色")
        if not self.enabled:
            raise QwenError("真实评审模式未启用或接口密钥缺失，无法创建复刻音色")
        if not audio_path.exists():
            raise QwenError(f"复刻样本文件不存在：{audio_path.name}")
        sample = data_url(audio_path)
        if len(sample) >= _VC_MAX_DATA_URL_CHARS:
            raise QwenError(f"复刻样本 Base64 后 {len(sample) // 1024 // 1024}MB，超过官方 10MB 上限，请缩短片段")
        inp: dict[str, Any] = {
            "action": "create",
            "target_model": target,
            "preferred_name": self.preferred_name_of(preferred_name),
            "audio": {"data": sample},
        }
        # text/language 是可选校验项：填了服务方会做一致性检查，样本与文本对不上会报
        # Audio.PreprocessError。我们传的是从转录里挑出的片段，文本本就来自同一段音频，
        # 因此只在确实拿到对应文本时才附上，宁可少校验也不误伤。
        if text.strip():
            inp["text"] = text.strip()[:1000]
        if language.strip():
            inp["language"] = language.strip().lower()
        payload = {"model": VOICE_ENROLLMENT_MODEL, "input": inp}
        try:
            data = self._post_vc(payload, timeout).json()
        except json.JSONDecodeError as exc:
            raise QwenError(f"音色复刻响应解析失败: {exc}") from exc
        output = data.get("output") or {}
        voice = (output.get("voice") or "").strip()
        if not voice:
            raise QwenError("音色复刻未返回 voice 标识，样本可能不合格",
                            body=json.dumps(data)[:600])
        usage = data.get("usage") or {}
        self.track_voice_creation(target, int(usage.get("count") or 1))
        return VoiceProfile(
            voice=voice,
            target_model=(output.get("target_model") or target).strip().lower(),
            fallback_mode=bool(output.get("fallback_mode")),
            fallback_reason=(output.get("fallback_reason") or "").strip(),
        )

    def delete_voice(self, voice: str, target_model: str = "") -> None:
        """删除远端复刻音色：本地记录删了而远端还留着，等于学生的人声一直挂在服务方。"""
        if not voice:
            return
        target = (target_model or settings.tts_vc_model).strip().lower()
        payload = {"model": VOICE_ENROLLMENT_MODEL,
                   "input": {"action": "delete", "voice": voice, "target_model": target}}
        self._post_vc(payload)
