"""User-facing copy: exactly one title + one message per outcome.

The kiosk renders only `title` and `message`. Everything else in a response --
`reasons`, `scores` -- is diagnostic and belongs in the debug panel, never on
the screen of the person standing at the door.

Each entry is (verdict, title, message). The verdict is stored so a response can
check that a reason code still matches the decision it ships with (the dev
/force override can change the verdict after the code was chosen).
"""

from typing import Optional, Tuple

import decision as dec

# -- reason codes -------------------------------------------------------------
ALLOW = "ALLOW"
DENY_SPOOF = "DENY_SPOOF"
DENY_DEVICE = "DENY_DEVICE"
DENY_UNKNOWN_FACE = "DENY_UNKNOWN_FACE"
DENY_VERIFICATION_FAILED = "DENY_VERIFICATION_FAILED"
STEP_UP_CHALLENGE = "STEP_UP_CHALLENGE"
STEP_UP_TOTP = "STEP_UP_TOTP"
RECAPTURE_DARK = "RECAPTURE_DARK"
RECAPTURE_STILL = "RECAPTURE_STILL"
RECAPTURE_CLOSER = "RECAPTURE_CLOSER"
RECAPTURE_BRIGHT = "RECAPTURE_BRIGHT"

COPY = {
    ALLOW:                    (dec.ALLOW, "Welcome, {name}", "Identity verified."),
    DENY_SPOOF:               (dec.DENY, "Access denied",
                               "A photo or screen was detected. "
                               "Please present your face directly."),
    DENY_DEVICE:              (dec.DENY, "Access denied",
                               "Please remove any phone or screen from view."),
    DENY_UNKNOWN_FACE:        (dec.DENY, "Access denied", "Face not recognised."),
    DENY_VERIFICATION_FAILED: (dec.DENY, "Access denied", "Verification failed."),
    STEP_UP_CHALLENGE:        (dec.STEP_UP, "One more step", "{challenge}"),
    STEP_UP_TOTP:             (dec.STEP_UP, "Enter your code",
                               "Open Google Authenticator and enter the 6-digit code."),
    RECAPTURE_DARK:           (dec.RECAPTURE, "Let's try again", "Too dark"),
    RECAPTURE_STILL:          (dec.RECAPTURE, "Let's try again", "Hold still"),
    RECAPTURE_CLOSER:         (dec.RECAPTURE, "Let's try again", "Move closer"),
    RECAPTURE_BRIGHT:         (dec.RECAPTURE, "Let's try again", "Too bright"),
}

# Fallback per verdict, used when no specific code fits (or a /dev/force
# override replaced the verdict the code was chosen for).
DEFAULT_CODE = {
    dec.ALLOW: ALLOW,
    dec.STEP_UP: STEP_UP_CHALLENGE,
    dec.DENY: DENY_VERIFICATION_FAILED,
    dec.RECAPTURE: RECAPTURE_STILL,
}


def code_for(verdict: str, reason_code: Optional[str] = None) -> str:
    """The reason code to ship: `reason_code` if it belongs to `verdict`."""
    entry = COPY.get(reason_code or "")
    if entry and entry[0] == verdict:
        return reason_code
    return DEFAULT_CODE.get(verdict, DENY_VERIFICATION_FAILED)


def render(reason_code: str, name: Optional[str] = None,
           challenge: Optional[str] = None) -> Tuple[str, str]:
    """(title, message) for a reason code, with the name/challenge filled in."""
    _, title, message = COPY[reason_code]
    if "{name}" in title:
        title = title.format(name=name) if name else "Welcome"
    if "{challenge}" in message:
        instruction = (challenge or "Please follow the on-screen instruction").strip()
        message = instruction if instruction.endswith((".", "!", "?")) else instruction + "."
    return title, message


def recapture_code(reason: Optional[str]) -> str:
    """Map an internal quality/detection reason onto a retake code."""
    text = (reason or "").lower()
    if "dark" in text:
        return RECAPTURE_DARK
    if "bright" in text or "backlight" in text:
        return RECAPTURE_BRIGHT
    if "closer" in text or "too far" in text or "no face" in text:
        return RECAPTURE_CLOSER
    return RECAPTURE_STILL
