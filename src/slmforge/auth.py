"""Sign-in sessions for the web app.

"Sign in with Hugging Face" is an OAuth 2.0 / OpenID Connect flow (see
`integrations/huggingface_oauth.py` for the Hub calls). This module owns the two
things the flow needs on our side:

* **Session state** — the signed, HttpOnly cookie that remembers who is signed
  in. It is a keyed HMAC over a small JSON payload, not a database row: the
  prototype has no migration tooling and the cookie only ever carries the public
  Hub profile.
* **CSRF/handshake state** — the short-lived cookie holding the OAuth `state`
  and the PKCE `code_verifier` between the redirect out and the redirect back.

The Hub **access token is never stored**: it is used once to read `userinfo` and
then discarded, so the `hf` CLI remains the only component that holds Hub
credentials (see AGENTS.md, guardrails).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
import time
from typing import Any
from urllib.parse import urlsplit

from .config import settings
from .log import get_logger

logger = get_logger("auth")

SESSION_COOKIE = "slmforge_session"
STATE_COOKIE = "slmforge_oauth_state"
#: How long the user has to complete the Hub consent screen before the handshake
#: cookie expires. Generous enough for a slow login, short enough to be useless.
STATE_TTL_S = 600


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def sign_payload(payload: dict[str, Any]) -> str:
    """Serialize `payload` and append an HMAC-SHA256 signature over it."""
    body = _b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    digest = hmac.new(settings.session_signing_key.encode("utf-8"), body.encode("ascii"), hashlib.sha256)
    return f"{body}.{_b64encode(digest.digest())}"


def unsign_payload(token: str | None) -> dict[str, Any] | None:
    """Verify a token from `sign_payload`, returning its payload or None.

    Any tampering, truncation or key mismatch yields None rather than raising, so
    a bad cookie is simply "not signed in".
    """
    if not token or "." not in token:
        return None
    body, _, signature = token.rpartition(".")
    expected = hmac.new(
        settings.session_signing_key.encode("utf-8"), body.encode("ascii"), hashlib.sha256
    ).digest()
    try:
        provided = _b64decode(signature)
    except (ValueError, binascii.Error):
        return None
    if not hmac.compare_digest(expected, provided):
        return None
    try:
        decoded = json.loads(_b64decode(body))
    except (ValueError, binascii.Error):
        return None
    return decoded if isinstance(decoded, dict) else None


def create_session(user: dict[str, Any], *, ttl_s: int | None = None) -> str:
    """Build a signed session cookie value carrying the public user profile."""
    ttl = settings.session_ttl_s if ttl_s is None else ttl_s
    return sign_payload({"user": user, "exp": int(time.time()) + ttl})


def read_session(token: str | None) -> dict[str, Any] | None:
    """Return the user profile from a session cookie, or None if absent/expired."""
    payload = unsign_payload(token)
    if payload is None:
        return None
    try:
        expires_at = int(payload.get("exp", 0))
    except (TypeError, ValueError):
        return None
    if expires_at < time.time():
        return None
    user = payload.get("user")
    return user if isinstance(user, dict) else None


def new_pkce_pair() -> tuple[str, str]:
    """Return a `(code_verifier, code_challenge)` pair for PKCE S256."""
    verifier = _b64encode(secrets.token_bytes(32))
    challenge = _b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def new_state() -> str:
    """An unguessable value echoed back by the Hub to bind the callback."""
    return secrets.token_urlsafe(24)


def _origin(url: str) -> str:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return ""
    return f"{parts.scheme}://{parts.netloc}"


def safe_next(candidate: str | None) -> str:
    """Choose where to land after sign-in, refusing anything off-site.

    An attacker must not be able to use our callback as an open redirect, so the
    target has to share an origin with the API or with a configured CORS origin
    (which in development is the Vite dev server).
    """
    default = f"{settings.public_base_url.rstrip('/')}/"
    if not candidate:
        return default
    allowed = {_origin(url) for url in settings.cors_origins}
    allowed.add(_origin(settings.public_base_url))
    origin = _origin(candidate)
    if origin and origin in allowed:
        return candidate
    logger.warning("Refused an off-site post-login redirect to %r", candidate)
    return default
