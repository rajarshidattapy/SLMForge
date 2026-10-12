"""Sign in with Hugging Face: handshake, session cookie and redirect safety.

The Hub is mocked with `respx`, so these tests never touch the network and never
need a real client id — they assert on the *contract* the Hub documents.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from slmforge.auth import (
    SESSION_COOKIE,
    create_session,
    read_session,
    safe_next,
    sign_payload,
    unsign_payload,
)
from slmforge.config import settings
from slmforge.integrations.huggingface_oauth import TOKEN_URL, USERINFO_URL

PROFILE = {
    "sub": "12345",
    "preferred_username": "sodium16",
    "name": "Sodium Turing",
    "picture": "https://cdn.example/avatar.png",
    "email": "sodium@example.com",
}


@pytest.fixture()
def hf_oauth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure a fake OAuth app pointing at the TestClient's own origin."""
    monkeypatch.setattr(settings, "hf_client_id", "test-client-id")
    monkeypatch.setattr(settings, "hf_client_secret", "test-client-secret")
    monkeypatch.setattr(settings, "session_secret", "test-session-secret")
    monkeypatch.setattr(settings, "public_base_url", "http://testserver")


@pytest.fixture(autouse=True)
def _isolated_cookies(client: TestClient) -> None:
    """The client fixture is session-scoped, so sessions must not leak between tests."""
    client.cookies.clear()
    yield
    client.cookies.clear()


def test_sign_in_is_off_until_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deployment with no OAuth app must say so rather than start a broken flow."""
    monkeypatch.setattr(settings, "hf_client_id", "")
    monkeypatch.setattr(settings, "hf_client_secret", "")

    body = client.get("/api/auth/me").json()
    assert body["enabled"] is False
    assert body["authenticated"] is False

    login = client.get("/api/auth/huggingface/login", follow_redirects=False)
    assert login.status_code == 503


def test_me_reports_anonymous_when_configured(client: TestClient, hf_oauth: None) -> None:
    body = client.get("/api/auth/me").json()
    assert body["enabled"] is True
    assert body["provider"] == "huggingface"
    assert body["authenticated"] is False
    assert body["user"] is None


def test_login_redirects_to_the_hub_with_pkce(client: TestClient, hf_oauth: None) -> None:
    response = client.get("/api/auth/huggingface/login", follow_redirects=False)
    assert response.status_code == 302

    location = response.headers["location"]
    assert location.startswith("https://huggingface.co/oauth/authorize?")
    query = parse_qs(urlsplit(location).query)
    assert query["client_id"] == ["test-client-id"]
    assert query["response_type"] == ["code"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["scope"] == ["openid profile email"]
    assert query["redirect_uri"] == ["http://testserver/api/auth/huggingface/callback"]
    assert query["state"][0]
    assert query["code_challenge"][0]


@respx.mock
def test_callback_completes_sign_in_without_storing_the_token(
    client: TestClient, hf_oauth: None
) -> None:
    respx.post(TOKEN_URL).mock(
        return_value=httpx.Response(
            200, json={"access_token": "hf_oauth_secret", "token_type": "bearer"}
        )
    )
    respx.get(USERINFO_URL).mock(return_value=httpx.Response(200, json=PROFILE))

    login = client.get("/api/auth/huggingface/login", follow_redirects=False)
    state = parse_qs(urlsplit(login.headers["location"]).query)["state"][0]

    callback = client.get(
        f"/api/auth/huggingface/callback?code=the-code&state={state}",
        follow_redirects=False,
    )
    assert callback.status_code == 302
    assert callback.headers["location"] == "http://testserver/"

    me = client.get("/api/auth/me").json()
    assert me["authenticated"] is True
    assert me["user"]["username"] == "sodium16"
    assert me["user"]["name"] == "Sodium Turing"
    assert me["user"]["avatar_url"] == PROFILE["picture"]

    # The access token is spent on `userinfo` and dropped — it must not be in the
    # session cookie the browser now holds.
    assert "hf_oauth_secret" not in client.cookies[SESSION_COOKIE]


def test_callback_rejects_a_forged_state(client: TestClient, hf_oauth: None) -> None:
    client.get("/api/auth/huggingface/login", follow_redirects=False)

    response = client.get(
        "/api/auth/huggingface/callback?code=the-code&state=forged",
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert "auth_error=invalid_state" in response.headers["location"]
    assert client.get("/api/auth/me").json()["authenticated"] is False


def test_callback_reports_a_hub_error_back_to_the_app(client: TestClient, hf_oauth: None) -> None:
    client.get("/api/auth/huggingface/login", follow_redirects=False)

    response = client.get(
        "/api/auth/huggingface/callback?error=access_denied",
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert "auth_error=access_denied" in response.headers["location"]


@respx.mock
def test_logout_clears_the_session(client: TestClient, hf_oauth: None) -> None:
    respx.post(TOKEN_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "hf_oauth_secret"})
    )
    respx.get(USERINFO_URL).mock(return_value=httpx.Response(200, json=PROFILE))

    login = client.get("/api/auth/huggingface/login", follow_redirects=False)
    state = parse_qs(urlsplit(login.headers["location"]).query)["state"][0]
    client.get(
        f"/api/auth/huggingface/callback?code=the-code&state={state}",
        follow_redirects=False,
    )
    assert client.get("/api/auth/me").json()["authenticated"] is True

    assert client.post("/api/auth/logout").status_code == 204
    assert client.get("/api/auth/me").json()["authenticated"] is False
    assert SESSION_COOKIE not in client.cookies


def test_session_cookie_is_signed_and_expires() -> None:
    user = {"sub": "1", "username": "u", "name": "U", "avatar_url": None, "email": None}
    token = create_session(user, ttl_s=60)
    assert read_session(token) == user

    # Expiry is enforced, not merely recorded.
    assert read_session(create_session(user, ttl_s=-1)) is None

    # A flipped character invalidates the signature.
    tampered = f"{token[:-1]}{'a' if not token.endswith('a') else 'b'}"
    assert read_session(tampered) is None

    # Our own keyed signature verifies; a payload with no signature does not.
    assert unsign_payload(sign_payload({"a": 1})) == {"a": 1}
    assert unsign_payload("not-a-token") is None


def test_safe_next_refuses_off_site_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    """The callback must not become an open redirect."""
    monkeypatch.setattr(settings, "public_base_url", "http://testserver")

    assert safe_next(None) == "http://testserver/"
    assert safe_next("http://evil.example/steal") == "http://testserver/"
    assert safe_next("//evil.example") == "http://testserver/"
    # The Vite dev server is an allowed CORS origin and stays allowed.
    assert safe_next("http://localhost:5173/") == "http://localhost:5173/"
