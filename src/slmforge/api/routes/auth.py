"""Sign in with Hugging Face.

Four small endpoints implement the OAuth flow:

``GET  /api/auth/me``                        who (if anyone) is signed in
``GET  /api/auth/huggingface/login``         redirect to the Hub consent screen
``GET  /api/auth/huggingface/callback``      finish the flow and set the cookie
``POST /api/auth/logout``                    clear the session cookie

Sign-in is *optional*: it identifies the person using the build and gives the
dashboard a real account chip. Endpoints that cost compute or learn private data
already have their own approval gates; gating them behind login is a product
decision for later, so nothing here blocks an anonymous visitor.
"""

from __future__ import annotations

from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Cookie, HTTPException, Query, Response, status
from fastapi.responses import RedirectResponse

from ...auth import (
    SESSION_COOKIE,
    STATE_COOKIE,
    STATE_TTL_S,
    create_session,
    new_pkce_pair,
    new_state,
    read_session,
    safe_next,
    sign_payload,
    unsign_payload,
)
from ...config import settings
from ...integrations.huggingface_oauth import (
    OAuthError,
    authorize_url,
    exchange_code,
    fetch_userinfo,
    normalize_user,
)
from ...log import get_logger
from ..payloads import AuthStatusOut, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])
logger = get_logger("api.auth")


def _error_redirect(target: str, code: str) -> RedirectResponse:
    """Send the browser back to the app with a machine-readable failure code."""
    separator = "&" if "?" in target else "?"
    return RedirectResponse(
        f"{target}{separator}{urlencode({'auth_error': code})}", status_code=302
    )


@router.get("/me", response_model=AuthStatusOut)
def whoami(
    session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
) -> AuthStatusOut:
    """Report sign-in availability and the current session, if any."""
    user = read_session(session)
    return AuthStatusOut(
        enabled=settings.hf_oauth_enabled,
        authenticated=user is not None,
        user=UserOut(**user) if user else None,
    )


@router.get("/huggingface/login")
def login(
    next_url: Annotated[str | None, Query(alias="next")] = None,
) -> RedirectResponse:
    """Start the flow: stash the handshake in a cookie and go to the Hub.

    `next` is where the user lands afterwards (the Vite dev server in a dev
    setup); it is only honoured when it shares an origin we already trust.
    """
    if not settings.hf_oauth_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Sign in with Hugging Face is not configured. Set HF_CLIENT_ID and "
                "HF_CLIENT_SECRET, and register the redirect URI on the HF OAuth app."
            ),
        )

    verifier, challenge = new_pkce_pair()
    state = new_state()
    response = RedirectResponse(
        authorize_url(state=state, code_challenge=challenge), status_code=302
    )
    response.set_cookie(
        STATE_COOKIE,
        sign_payload({"state": state, "verifier": verifier, "next": safe_next(next_url)}),
        max_age=STATE_TTL_S,
        httponly=True,
        samesite="lax",
        secure=settings.session_cookie_secure,
        path="/",
    )
    return response


@router.get("/huggingface/callback")
async def callback(
    code: Annotated[str | None, Query()] = None,
    state: Annotated[str | None, Query()] = None,
    error: Annotated[str | None, Query()] = None,
    state_cookie: Annotated[str | None, Cookie(alias=STATE_COOKIE)] = None,
) -> RedirectResponse:
    """Finish the flow: verify the handshake, read the profile, set the session."""
    if not settings.hf_oauth_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sign in with Hugging Face is not configured.",
        )

    handshake = unsign_payload(state_cookie) or {}
    target = safe_next(handshake.get("next"))

    # The state cookie is single-use: clear it whatever the outcome.
    def finish(response: RedirectResponse) -> RedirectResponse:
        response.delete_cookie(STATE_COOKIE, path="/")
        return response

    if error:
        # The user declined, or the Hub refused. Either way this is not our bug.
        logger.info("HF sign-in returned an error: %s", error)
        return finish(_error_redirect(target, error))

    expected_state = handshake.get("state")
    code_verifier = handshake.get("verifier")
    if not state or not expected_state or state != expected_state or not code_verifier:
        # Missing/forged handshake — the CSRF check failed. Do not proceed.
        logger.warning("HF sign-in callback failed the state check")
        return finish(_error_redirect(target, "invalid_state"))

    if not code:
        return finish(_error_redirect(target, "missing_code"))

    try:
        token = await exchange_code(code, code_verifier)
        access_token = token.get("access_token")
        if not access_token:
            raise OAuthError("The Hub did not return an access token.")
        profile = normalize_user(await fetch_userinfo(access_token))
    except OAuthError as exc:
        logger.warning("HF sign-in failed: %s", exc)
        return finish(_error_redirect(target, "sign_in_failed"))

    logger.info("Signed in Hugging Face user %s", profile.get("username"))
    response = RedirectResponse(target, status_code=302)
    response.set_cookie(
        SESSION_COOKIE,
        create_session(profile),
        max_age=settings.session_ttl_s,
        httponly=True,
        samesite="lax",
        secure=settings.session_cookie_secure,
        path="/",
    )
    return finish(response)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout() -> Response:
    """Clear the session cookie."""
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response
