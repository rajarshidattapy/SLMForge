"""Sign in with Hugging Face — OAuth 2.0 / OpenID Connect.

Docs: https://huggingface.co/docs/hub/oauth

The endpoints below are the ones the Hub advertises in its OpenID metadata
(https://huggingface.co/.well-known/openid-configuration). We use the
authorization-code flow with PKCE; because our app is a confidential client it
presents the client secret with HTTP Basic auth on the token exchange, as the
Hub documents.

Every call goes through `net.retry_call_async` for the same reason the rest of
the codebase does: the reference network drops TLS handshakes to
`huggingface.co` under load (see AGENTS.md).
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

import httpx

from ..config import settings
from ..log import get_logger
from ..net import retry_call_async

logger = get_logger("integrations.huggingface_oauth")

AUTHORIZE_URL = "https://huggingface.co/oauth/authorize"
TOKEN_URL = "https://huggingface.co/oauth/token"
USERINFO_URL = "https://huggingface.co/oauth/userinfo"

_TIMEOUT_S = 30.0


class OAuthError(RuntimeError):
    """The Hub rejected a step of the flow (bad code, bad client, network down)."""


def authorize_url(*, state: str, code_challenge: str) -> str:
    """Build the URL the browser is sent to for the Hub consent screen."""
    params = {
        "client_id": settings.hf_client_id,
        "redirect_uri": settings.oauth_redirect_uri,
        "response_type": "code",
        "scope": " ".join(settings.oauth_scopes),
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


async def exchange_code(code: str, code_verifier: str) -> dict[str, Any]:
    """Trade the authorization `code` for an access token.

    The token is returned to the caller, used to read `userinfo`, and then
    dropped — SLMForge does not persist Hub credentials.
    """

    async def _once() -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            response = await client.post(
                TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": settings.oauth_redirect_uri,
                    "code_verifier": code_verifier,
                },
                # Confidential client: the secret is sent, never logged.
                auth=(settings.hf_client_id, settings.hf_client_secret),
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            return response.json()

    try:
        return await retry_call_async(
            _once,
            attempts=settings.network_attempts,
            base_delay=settings.network_base_delay_s,
            max_delay=settings.network_max_delay_s,
            description="HF OAuth token exchange",
        )
    except httpx.HTTPStatusError as exc:
        raise OAuthError("The Hub rejected the sign-in code.") from exc
    except httpx.HTTPError as exc:
        raise OAuthError("Could not reach Hugging Face to complete sign-in.") from exc


async def fetch_userinfo(access_token: str) -> dict[str, Any]:
    """Read the signed-in user's profile (OpenID `userinfo`)."""

    async def _once() -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            response = await client.get(
                USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
            )
            response.raise_for_status()
            return response.json()

    try:
        return await retry_call_async(
            _once,
            attempts=settings.network_attempts,
            base_delay=settings.network_base_delay_s,
            max_delay=settings.network_max_delay_s,
            description="HF OAuth userinfo",
        )
    except httpx.HTTPError as exc:
        raise OAuthError("Could not read your Hugging Face profile.") from exc


def normalize_user(info: dict[str, Any]) -> dict[str, Any]:
    """Map the Hub `userinfo` payload to the small profile we keep in the session."""
    username = info.get("preferred_username") or info.get("name") or "user"
    return {
        "sub": str(info.get("sub") or username),
        "username": str(username),
        "name": str(info.get("name") or username),
        "avatar_url": info.get("picture") or None,
        "email": info.get("email") or None,
    }
