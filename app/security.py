"""口令散列与会话签名（全部使用标准库，避免额外依赖）。"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time

from .config import settings

_ALGO = "pbkdf2_sha256"
_ITERATIONS = 180_000


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _ITERATIONS)
    return f"{_ALGO}${_ITERATIONS}${salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt, digest = stored.split("$")
    except ValueError:
        return False
    if algo != _ALGO:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iters))
    return hmac.compare_digest(dk.hex(), digest)


def _sign(payload: str) -> str:
    return hmac.new(settings.secret_key.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]


def make_session_token(user_id: int, role: str) -> str:
    exp = int(time.time()) + settings.session_days * 86400
    raw = f"{user_id}.{role}.{exp}"
    sig = _sign(raw)
    token = base64.urlsafe_b64encode(f"{raw}.{sig}".encode()).decode().rstrip("=")
    return token


def read_session_token(token: str) -> tuple[int, str] | None:
    if not token:
        return None
    try:
        padded = token + "=" * (-len(token) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode()).decode()
        user_id, role, exp, sig = decoded.split(".")
    except (ValueError, UnicodeDecodeError):
        return None
    raw = f"{user_id}.{role}.{exp}"
    if not hmac.compare_digest(_sign(raw), sig):
        return None
    if int(exp) < time.time():
        return None
    try:
        return int(user_id), role
    except ValueError:
        return None


def valid_username(name: str) -> bool:
    return bool(name) and 3 <= len(name) <= 32 and all(c.isalnum() or c in "._-" for c in name)


def password_ok(pw: str) -> bool:
    return bool(pw) and len(pw) >= 6
