"""Guards for friendly_error: backend failures must map to generic, client-safe
messages — never a raw provider error or stack trace reaching the user.

No API/Streamlit needed; the mapping lives in src.rag.errors precisely so it can
be tested in isolation.
"""

import pytest

from src.rag.errors import GENERIC, MISCONFIGURED, RATE_LIMITED, friendly_error


# --- rate-limit / quota → "service is busy" ----------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "429 Too Many Requests",
        "RESOURCE_EXHAUSTED: quota exceeded",
        "You have exceeded your quota.",
    ],
)
def test_rate_limit_errors_map_to_rate_limited(message):
    assert friendly_error(Exception(message)) == RATE_LIMITED


# --- auth / key / permissions → "isn't configured correctly" -----------------


@pytest.mark.parametrize(
    "message",
    [
        "API key not valid. Please pass a valid API key.",
        "API_KEY_INVALID",
        "PERMISSION_DENIED: caller does not have permission",
        "401 Unauthorized",
        "403 Forbidden",
        "Request had invalid authentication credentials (UNAUTHENTICATED)",
    ],
)
def test_auth_errors_map_to_misconfigured(message):
    assert friendly_error(Exception(message)) == MISCONFIGURED


# --- the real bug: the deployed GoogleGenerativeAIError must not leak ---------


def test_invalid_key_does_not_leak_provider_details():
    # The exact failure class from the Streamlit Cloud traceback.
    ex = Exception("GoogleGenerativeAIError: API key not valid (API_KEY_INVALID)")
    result = friendly_error(ex)
    assert result == MISCONFIGURED
    # Nothing provider-specific or key-related escapes to the client.
    for leak in (
        "GoogleGenerativeAI",
        "API_KEY_INVALID",
        "Traceback",
        "api key not valid",
    ):
        assert leak not in result


# --- anything else → generic fallback ----------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "Connection reset by peer",
        "some unexpected internal failure",
        "",
    ],
)
def test_unknown_errors_map_to_generic(message):
    assert friendly_error(Exception(message)) == GENERIC


def test_matching_is_case_insensitive():
    assert friendly_error(Exception("Api Key Not Valid")) == MISCONFIGURED
    assert friendly_error(Exception("Resource_Exhausted")) == RATE_LIMITED
