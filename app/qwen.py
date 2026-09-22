"""千问（DashScope OpenAI 兼容模式）客户端：文本 / 图像 / 音频多模态调用。

只依赖 requests，走 /compatible-mode/v1/chat/completions：
  - chat()        普通文本（qwen-plus 等）
  - vision()      图像序列（qwen-vl-*）
  - audio()       音频（qwen-omni-*，强制流式聚合）
  - asr()         语音转文字（qwen3-asr-flash）
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
        waited = time.monotonic()
        if not sem.acquire(timeout=timeout):
            raise QwenError(f"评审请求排队超过 {int((timeout or 0) / 60)} 分钟，请稍后重试分析")
        self._active += 1
        try:
            yield
        finally:
            self._active -= 1
            sem.release()
            if time.monotonic() - waited > 5:
                print(f"[qwen] 网关排队 {time.monotonic() - waited:.0f}s 后才发出请求")

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


@dataclass
class Completion:
    text: str
    model: str
    usage: dict | None = None


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

    def usage_summary(self) -> dict:
        """汇总本次会话累计的模型用量：总量 + 按阶段 + 按模型。"""
        buckets: dict[str, dict[str, dict]] = {
            "stage": {}, "model": {}}
        out = {"calls": len(self.usage_records), "prompt": 0, "completion": 0,
               "total": 0, "estimated_calls": 0, "by_stage": [], "by_model": []}
        for r in self.usage_records:
            out["prompt"] += r["prompt"]
            out["completion"] += r["completion"]
            out["total"] += r["total"]
            if r["estimated"]:
                out["estimated_calls"] += 1
            for axis, key in (("stage", r["stage"]), ("model", r["model"])):
                row = buckets[axis].setdefault(
                    key, {"name": key, "calls": 0, "prompt": 0, "completion": 0,
                          "total": 0, "estimated": 0})
                row["calls"] += 1
                row["prompt"] += r["prompt"]
                row["completion"] += r["completion"]
                row["total"] += r["total"]
                row["estimated"] += 1 if r["estimated"] else 0
        out["by_stage"] = sorted(buckets["stage"].values(), key=lambda x: -x["total"])
        out["by_model"] = sorted(buckets["model"].values(), key=lambda x: -x["total"])
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
    }

    @classmethod
    def _model_family(cls, model: str) -> str:
        m = (model or "").lower()
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

    _RESOLVED: dict[str, str] = {}

    def _try_models(self, payload: dict, note: list[str] | None = None,
                    family: str | None = None) -> Completion:
        """按“同族模型降级链”重试：模型名不存在 / 无权限（403/404）时换下一个。

        千问模型名区分大小写，api_key.txt 里写 `Qwen3.8-flash` 会被判为不存在，
        因此链条里自动补一个小写变体；一旦验证可用就在进程内记住，后续调用不再先吃一个 400。
        """
        primary = payload["model"]
        known = QwenClient._RESOLVED.get(primary)
        chain: list[str] = []
        if known and known != primary:
            chain.append(known)
        chain.append(primary)
        lower = primary.lower()
        if lower != primary and lower not in chain:
            chain.append(lower)
        for alt in self.FALLBACKS.get(family or self._model_family(primary), ()):
            if alt not in chain:
                chain.append(alt)
        last: QwenError | None = None
        for model in chain:
            payload["model"] = model
            try:
                comp = self._complete(payload)
            except QwenError as exc:
                last = exc
                if not self.is_model_error(exc):
                    raise
                if model == known:
                    QwenClient._RESOLVED.pop(primary, None)
                continue
            if model != primary:
                QwenClient._RESOLVED[primary] = model
                if note is not None:
                    if model.lower() == primary.lower():
                        note.append(f"模型名 {primary} 大小写不被接受，已使用 {model}")
                    else:
                        note.append(f"模型 {primary} 不可用，已降级为 {model}")
            return comp
        assert last is not None
        raise last

    # ---------- 模态封装 ----------
    def chat(self, prompt: str, system: str = "", model: str | None = None,
             history: list[dict] | None = None, temperature: float = 0.2,
             json_mode: bool = False, max_tokens: int = 4096,
             note: list[str] | None = None) -> Completion:
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
                return self._try_models(payload, note)
            except QwenError as exc:
                if exc.status not in (400, 404, 422):
                    raise
                payload.pop("response_format", None)
        return self._try_models(payload, note)

    def chat_content(self, content: list[dict], system: str = "", model: str | None = None,
                     temperature: float = 0.2, json_mode: bool = False,
                     modalities: list[str] | None = None, max_tokens: int = 4096,
                     note: list[str] | None = None, family: str = "chat") -> Completion:
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
                    return self._try_models(payload, note, family)
                except QwenError as exc:
                    if exc.status not in (400, 404, 422):
                        raise
                    payload.pop("response_format", None)
        return self._try_models(payload, note, family)

    def vision(self, prompt: str, images: list[Path], system: str = "",
               model: str | None = None, json_mode: bool = True,
               temperature: float = 0.2, note: list[str] | None = None) -> Completion:
        content = [text_item(prompt)] + [image_url_item(p) for p in images]
        return self.chat_content(content, system=system, model=model or settings.vlm_model,
                                 json_mode=json_mode, temperature=temperature, note=note,
                                 family="vision")

    def audio(self, prompt: str, audio: Path, system: str = "",
              model: str | None = None, json_mode: bool = True,
              note: list[str] | None = None) -> Completion:
        content = [input_audio_item(audio), text_item(prompt)]
        return self.chat_content(content, system=system, model=model or settings.omni_model,
                                 json_mode=json_mode, modalities=["text"], note=note,
                                 family="audio")

    def asr(self, audio: Path, language: str = "en", model: str | None = None,
            note: list[str] | None = None) -> Completion:
        """转写：qwen3-asr-flash 优先，不可用时用 omni 逐字转写兜底。"""
        instruction = prompts.get("asr_instruction")
        primary = model or settings.asr_model
        chain = [primary]
        if primary.lower() != primary:
            chain.append(primary.lower())
        chain += [m for m in self.FALLBACKS["asr"] if m not in chain]
        last: QwenError | None = None
        for m in chain:
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
                continue
            if m != primary and note is not None:
                note.append(f"转写模型 {primary} 不可用，已改用 {m}")
            return comp
        assert last is not None
        raise last
