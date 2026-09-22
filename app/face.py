"""微表情的客观指标层（可选能力，缺 opencv 时整块降级为「不可用」）。

设计约束（与用户确认的规格一一对应）：
  · 不新增大模型请求、不参与加权求和。指标只作为**质控证据**注入视觉通道提示词，
    并在聚合层与模型判分做双向冲突提示；分数本身仍由模型 + 确定性聚合产出。
  · Haar 级联可靠区分的只有「正脸 / 侧脸 / 无脸」和「脸在画面里的位置」。真实视线
    方向（眼睛乱瞟）在本机实测里做不出来：双眼级联在同一批真实帧上极少同时命中，
    所以这里只报「头部朝向偏离」和「疑似低头」，不假装能测注视角。
  · 检出率低于 FACE_MIN_DETECT 时一个指标都不输出。低质/远景素材的正脸检出率可以
    低到两成，把「拍不清」当成「不看镜头」是直接扣分级别的误判。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings

_CV2 = "unset"
_CASCADES: dict | None = None
_NOTE = ""  # 降级原因：available() 只说能不能用，这里说为什么不能用


def _backend():
    """惰性加载 cv2 与级联文件；任何一步失败都返回 None，调用侧整块降级。

    「失败」不止 import 失败一种：OpenCV 5.0 起 Haar 级联移出主 wheel，
    `import cv2` 照样成功、`cv2.CascadeClassifier` 却不存在。所以属性探测和
    级联加载也必须在保护范围内，否则一个可选层能把 /health 和整条分析链路抛异常。
    """
    global _CV2, _CASCADES, _NOTE
    if not settings.use_face_metrics:
        return None, None
    if _CASCADES is not None:
        return _CV2, _CASCADES
    if _CV2 == "unset":
        try:
            import cv2  # noqa: PLC0415  可选依赖，缺失时客观指标整块停用
            _CV2 = cv2
        except Exception:  # noqa: BLE001
            _CV2 = None
            _NOTE = "未安装 opencv（requirements 里的 opencv-python-headless）"
    if not _CV2:
        return None, None
    cv2 = _CV2
    if not hasattr(cv2, "CascadeClassifier"):
        _CASCADES = {}
        _NOTE = (f"opencv {getattr(cv2, '__version__', '?')} 不含 CascadeClassifier"
                 f"（5.x 起 Haar 级联移出主包），需装 opencv-python-headless<5")
        return None, None
    try:
        base = cv2.data.haarcascades

        def load(name: str):
            clf = cv2.CascadeClassifier()
            return clf if clf.load(base + name) else None

        cascades = {
            "frontal": load("haarcascade_frontalface_default.xml"),
            "frontal_alt": load("haarcascade_frontalface_alt2.xml"),
            "profile": load("haarcascade_profileface.xml"),
            "eye": load("haarcascade_eye.xml"),
        }
    except Exception:  # noqa: BLE001
        _CASCADES = {}
        _NOTE = "opencv 级联文件加载失败，客观指标停用"
        return None, None
    if not cascades["frontal"]:
        _CASCADES = {}
        _NOTE = "正脸级联文件缺失，客观指标停用"
        return cv2, None
    _CASCADES = {k: v for k, v in cascades.items() if v is not None}
    return cv2, _CASCADES


def unavailable_reason() -> str:
    """可用时返回空串；不可用时返回一句人话，供报告与 /health 直接引用。"""
    cv2, cas = _backend()
    if cv2 and cas:
        return ""
    if not settings.use_face_metrics:
        return "USE_FACE_METRICS=0 已关闭"
    return _NOTE or "客观指标层未就绪"


def available() -> bool:
    """客观指标这层当前到底跑不跑得起来（供报告与自检直接问一句）。"""
    cv2, cas = _backend()
    return bool(cv2 and cas)


@dataclass
class FrameFace:
    """单帧测量结果。profile 取 -1/1 表示头朝向画面左/右，2 表示两侧都命中（歧义）。"""
    t: float
    frontal: bool = False
    profile: int = 0
    eyes: int = 0
    offset: float | None = None
    size: float = 0.0

    def to_dict(self) -> dict:
        return {"t": self.t, "frontal": self.frontal, "profile": self.profile,
                "eyes": self.eyes, "offset": self.offset, "size": round(self.size, 4)}


@dataclass
class FaceMetrics:
    scanned: int = 0
    available: bool = False
    reason: str = ""
    face_rate: float | None = None
    frontal_ratio: float | None = None
    nonfrontal_ratio: float | None = None
    down_ratio: float | None = None
    yaw_bias: float | None = None
    head_spread: float | None = None
    eye_reliability: float = 0.0
    frames: list[FrameFace] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def headline(self) -> str:
        if not self.available:
            return self.reason or "客观人脸指标不可用"
        bits = [f"正脸率 {_pct(self.frontal_ratio)}"]
        if self.nonfrontal_ratio is not None:
            bits.append(f"非正脸 {_pct(self.nonfrontal_ratio)}")
        if self.down_ratio is not None:
            bits.append(f"疑似低头/遮脸 {_pct(self.down_ratio)}")
        if self.yaw_bias is not None and abs(self.yaw_bias) >= 0.04:
            side = "右" if self.yaw_bias > 0 else "左"
            bits.append(f"头部平均偏{side} {abs(self.yaw_bias) * 100:.0f}% 画宽")
        return "，".join(bits)

    def evidence_block(self) -> str:
        """给视觉通道提示词的「客观核查块」。不可用时只留一行说明，不误导模型。"""
        if not self.available:
            return ("【客观人脸测量】未运行（" + (self.reason or "指标不可用") +
                    "）。眼神与表情的判断只能依据画面本身，不要引用任何未经测量的数字。")
        lines = ["【客观人脸测量｜由本地 Haar 级联在关键帧上逐帧测得，仅供交叉核对，不是最终结论】"]
        lines.append(f"- 采样帧数 {self.scanned}，检出人脸 {_pct(self.face_rate)}，"
                     f"正脸朝向镜头 {_pct(self.frontal_ratio)}")
        if self.nonfrontal_ratio is not None:
            lines.append(f"- 检出人脸中非正脸（侧脸/背向）占比 {_pct(self.nonfrontal_ratio)}")
        if self.down_ratio is not None:
            lines.append(f"- 正脸帧中检测不到眼睛（疑似低头读稿/遮脸/闭眼）占比 {_pct(self.down_ratio)}")
        if self.yaw_bias is not None:
            side = "右" if self.yaw_bias > 0 else "左"
            lines.append(f"- 人脸中心相对画面中线平均偏{side} {abs(self.yaw_bias) * 100:.1f}% 画宽"
                         f"，左右散布 {abs(self.head_spread or 0) * 100:.1f}%")
        lines.append("- 该测量不含视线方向（眼睛级联不可靠），「乱瞟/躲闪」请以画面为准；"
                     "若你的判断与上述数字冲突，请在证据或备注里写明理由，不要为了迎合数字改分。")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"available": self.available, "scanned": self.scanned, "reason": self.reason,
                "face_rate": self.face_rate, "frontal_ratio": self.frontal_ratio,
                "nonfrontal_ratio": self.nonfrontal_ratio, "down_ratio": self.down_ratio,
                "yaw_bias": self.yaw_bias, "head_spread": self.head_spread,
                "eye_reliability": self.eye_reliability, "headline": self.headline(),
                "notes": self.notes, "frames": [f.to_dict() for f in self.frames]}


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.0f}%"


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


def _sample(n: int, limit: int) -> list[int]:
    if n <= limit:
        return list(range(n))
    return [int(round(i * (n - 1) / (limit - 1))) for i in range(limit)] if limit > 1 else [0]


def _detect(clf, img, min_side: int, neighbors: int = 5):
    if clf is None:
        return []
    return clf.detectMultiScale(img, scaleFactor=1.12, minNeighbors=neighbors,
                               minSize=(min_side, min_side))


def _scan_frame(cv2, cas, path: Path, t: float) -> FrameFace | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    side = min(w, h)
    # 短边 12% 以下的框多半是背景纹理/观众席，宁可漏检也不要误报成正脸。
    min_face = max(32, int(side * 0.12))
    eq = cv2.equalizeHist(img)
    boxes = list(_detect(cas.get("frontal"), eq, min_face))
    if cas.get("frontal_alt"):
        boxes += list(_detect(cas["frontal_alt"], eq, min_face))
    frontal = bool(boxes)
    # 侧脸双向都要跑（只在正脸扑空时）：单向检到才是「朝某侧」，双向同时检到多半是误报。
    profile = 0
    prof_l: list = []
    prof_r: list = []
    prof_boxes: list = []
    if not frontal and cas.get("profile"):
        prof_l = list(_detect(cas["profile"], eq, min_face, 6))
        prof_r = list(_detect(cas["profile"], cv2.flip(eq, 1), min_face, 6))
        # 镜像图上的框要换算回原图坐标，否则侧脸帧的脸位置是左右颠倒的。
        prof_boxes = list(prof_l) + [(w - x - bw, y, bw, bh) for x, y, bw, bh in prof_r]
        if prof_l and prof_r:
            profile = 2
        elif prof_l:
            profile = -1
        elif prof_r:
            profile = 1
    eyes = 0
    if frontal and cas.get("eye"):
        x, y, bw, bh = max(boxes, key=lambda b: b[2] * b[3])
        band = eq[y + int(bh * 0.12): y + int(bh * 0.58), x: x + bw]
        if band.size:
            eyes = len(_detect(cas["eye"], band, max(12, int(bw * 0.10)), 6))
    box = max(boxes, key=lambda b: b[2] * b[3]) if frontal else (
        max(prof_boxes, key=lambda b: b[2] * b[3]) if prof_boxes else None)
    offset = size = None
    if box is not None:
        x, y, bw, bh = box
        offset = ((x + bw / 2) - w / 2) / w
        size = (bw * bh) / (w * h)
    return FrameFace(t=t, frontal=frontal, profile=profile, eyes=eyes,
                     offset=offset, size=size or 0.0)


def measure(paths: list[Path], stamps: list[float]) -> FaceMetrics:
    """在关键帧上跑一遍人脸朝向测量，并按门控决定这些数字能不能用。"""
    m = FaceMetrics()
    cv2, cas = _backend()
    if cv2 is None or not cas:
        m.reason = (_NOTE or "客观指标层不可用"
                    if settings.use_face_metrics else "USE_FACE_METRICS=0 已关闭")
        m.notes.append("客观微表情指标未参与质控")
        return m
    if cas.get("profile") is None:
        m.notes.append("侧脸级联缺失，非正脸占比只能由「未检出正脸」粗估")
    idx = _sample(len(paths), max(3, settings.face_max_frames))
    for i in idx:
        rec = _scan_frame(cv2, cas, paths[i], stamps[i] if i < len(stamps) else float(i))
        if rec is not None:
            m.frames.append(rec)
    m.scanned = len(m.frames)
    if m.scanned < 3:
        m.reason = f"有效采样帧过少（{m.scanned} 帧），不足以判断朝向习惯"
        return m
    face_frames = [f for f in m.frames if f.frontal or f.profile]
    frontal_frames = [f for f in m.frames if f.frontal]
    m.face_rate = round(len(face_frames) / m.scanned, 4)
    if m.face_rate < settings.face_min_detect:
        m.reason = (f"人脸检出率 {_pct(m.face_rate)} 低于门限 {_pct(settings.face_min_detect)}，"
                    f"镜头过远或过糊，朝向类指标不可信")
        m.notes.append("检出率门控未通过，未输出任何朝向指标")
        return m
    n = m.scanned
    m.frontal_ratio = round(len(frontal_frames) / n, 4)
    m.nonfrontal_ratio = round((len(face_frames) - len(frontal_frames)) / len(face_frames), 4)
    offs = [f.offset for f in face_frames if f.offset is not None]
    m.yaw_bias = round(_mean(offs), 4) if offs else None
    m.head_spread = round(_stdev(offs), 4) if len(offs) >= 3 else 0.0
    if cas.get("eye") and frontal_frames:
        m.eye_reliability = round(sum(1 for f in frontal_frames if f.eyes > 0) / len(frontal_frames), 4)
        # 眼睛级联本身不稳（远景、侧转、眼镜都会漏），可靠度过低时"检不到眼"≠低头，
        # 与其给一个假指标不如不给。
        if m.eye_reliability >= 0.5:
            m.down_ratio = round(sum(1 for f in frontal_frames if f.eyes == 0) / len(frontal_frames), 4)
        else:
            m.notes.append(f"眼睛级联可靠度仅 {_pct(m.eye_reliability)}，未输出低头率")
    if m.frontal_ratio < 0.35:
        m.notes.append("正脸率偏低：请核实是否长期侧对镜头/评委，而非镜头本身拍不清")
    m.available = True
    return m
