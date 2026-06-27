"""Map backend failures to generic, user-safe messages.

Kept separate from app.py so it can be unit-tested without importing the
Streamlit UI (which executes on import) or making any API call. The caller is
responsible for logging the original exception server-side; this function
returns only what is safe to show a client — no stack trace, no provider
internals, no secrets.
"""

RATE_LIMITED = "⚠️ The service is busy (rate limit reached). Please try again in a minute."
MISCONFIGURED = "⚠️ The card service isn't configured correctly right now. Please try again later."
GENERIC = "⚠️ Something went wrong answering that. Please try again."

_RATE_LIMIT_MARKERS = ("resource_exhausted", "429", "quota")
_AUTH_MARKERS = (
    "api key",
    "api_key_invalid",
    "permission_denied",
    "unauthenticated",
    "401",
    "403",
)


def friendly_error(ex: Exception) -> str:
    """Return a generic client-safe message for a backend failure.

    - rate-limit / quota  → RATE_LIMITED
    - auth / key / perms  → MISCONFIGURED
    - anything else       → GENERIC
    """
    text = str(ex).lower()
    if any(marker in text for marker in _RATE_LIMIT_MARKERS):
        return RATE_LIMITED
    if any(marker in text for marker in _AUTH_MARKERS):
        return MISCONFIGURED
    return GENERIC
