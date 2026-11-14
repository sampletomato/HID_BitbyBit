"""Biometric verdict: ALLOW / STEP_UP / DENY / RECAPTURE.

This is the *biometric* half of the decision only. From step 3 onward
policy.py combines this verdict with the campus access schedule to produce the
final authorisation. Keep it score-only so it never needs changing when the
models behind the scores do.
"""

from typing import List, Optional, Tuple

import config

ALLOW = "ALLOW"
STEP_UP = "STEP_UP"
DENY = "DENY"
RECAPTURE = "RECAPTURE"

ALL = (ALLOW, STEP_UP, DENY, RECAPTURE)


def decide(liveness: float, match: float, name: Optional[str],
           match_allow: float, match_deny: float,
           device_seen: bool = False,
           spoof_detected: bool = False) -> Tuple[str, List[str], str]:
    """Map scores to (verdict, diagnostic reasons, reason code).

    The reason code is the stable identifier the API ships so the kiosk can
    show one clear sentence; `reasons` stays diagnostic (debug panel only).

    Match thresholds are passed in because they depend on which recognition
    backend is loaded -- a cosine score from SFace is not comparable to one
    from buffalo_l.

    Order matters: a detected spoof outranks a match, and a match failure
    outranks a merely low-confidence liveness score.
    """
    reasons: List[str] = []

    if device_seen:
        return DENY, ["Replay attack suspected: a screen or phone was in frame"],             "DENY_DEVICE"

    if spoof_detected:
        return DENY, ["Spoof detected: print/screen texture"], "DENY_SPOOF"

    if liveness < config.LIVENESS_DENY:
        return DENY, [f"Presentation attack detected (liveness {liveness:.2f})"],             "DENY_SPOOF"

    if name is None:
        return DENY, ["Face does not match any enrolled user"], "DENY_UNKNOWN_FACE"

    if match < match_deny:
        return DENY, [f"Identity mismatch (match {match:.2f})"], "DENY_UNKNOWN_FACE"

    if liveness >= config.LIVENESS_ALLOW and match >= match_allow:
        reasons.append(f"Live face confirmed ({liveness:.2f})")
        reasons.append(f"Identity confirmed as {name} ({match:.2f})")
        return ALLOW, reasons, "ALLOW"

    if liveness < config.LIVENESS_ALLOW:
        reasons.append(f"Liveness confidence low ({liveness:.2f})")
    if match < match_allow:
        reasons.append(f"Identity confidence low for {name} ({match:.2f})")
    reasons.append("Additional verification required")
    return STEP_UP, reasons, "STEP_UP_CHALLENGE"
