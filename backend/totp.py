"""TOTP enrollment and verification (replaces the SMS-style OTP).

Each enrolled user gets a base32 secret and a provisioning QR for Google
Authenticator. Verification accepts the adjacent time steps for clock skew but
refuses a step that was already used, so a judge replaying a code is rejected.
"""

import base64
import hmac
import io
import logging
import time
from typing import Optional, Tuple

import pyotp
import qrcode

import config

log = logging.getLogger("totp")


def new_secret() -> str:
    return pyotp.random_base32()


def _totp(secret: str) -> pyotp.TOTP:
    return pyotp.TOTP(secret, digits=config.TOTP_DIGITS, interval=config.TOTP_PERIOD_S)


def provisioning_uri(name: str, secret: str) -> str:
    return _totp(secret).provisioning_uri(name=name, issuer_name=config.TOTP_ISSUER)


def qr_data_url(uri: str) -> str:
    """Provisioning URI -> PNG data URL the kiosk can drop straight into an <img>."""
    img = qrcode.make(uri)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def enroll_secret(name: str) -> Tuple[str, str]:
    """Returns (secret, qr_data_url)."""
    secret = new_secret()
    return secret, qr_data_url(provisioning_uri(name, secret))


def current_code(secret: str) -> str:
    """Only for the demo panel and tests -- never shown to a real enrollee."""
    return _totp(secret).now()


def verify(secret: str, code: str,
           last_step: Optional[int] = None) -> Tuple[bool, Optional[int], Optional[str]]:
    """Check a code. Returns (ok, step_used, error_reason).

    `last_step` is the most recent step this user already consumed; any step at
    or below it is a replay and is refused even though the code is arithmetically
    correct.
    """
    code = (code or "").strip().replace(" ", "")
    if not code.isdigit() or len(code) != config.TOTP_DIGITS:
        return False, None, "Malformed code"

    totp = _totp(secret)
    now = int(time.time())
    window = config.TOTP_WINDOW
    for delta in range(-window, window + 1):
        at = now + delta * config.TOTP_PERIOD_S
        if hmac.compare_digest(totp.at(at), code):
            step = at // config.TOTP_PERIOD_S
            if last_step is not None and step <= last_step:
                return False, step, "Code already used"
            return True, step, None
    return False, None, "Invalid code"
