"""Smoke test for Surprise Challenge 1: biometric result + campus schedule.

Covers the six required cases without a camera: it checks the schedule layer
(policy.evaluate) and the combination layer (app._apply_schedule) with the
biometric verdict injected, so the rules can be verified in a second.

    python tools/smoke_schedule.py
"""

import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config            # noqa: E402
import decision as dec   # noqa: E402
import messages as msg   # noqa: E402
import policy            # noqa: E402

MON = "2026-11-16"       # a Monday
SAT = "2026-11-21"       # a Saturday


def at(day: str, clock: str) -> str:
    return f"{day}T{clock}"


def combine(biometric: str, name, sim_time: str):
    """The real rule from app.py, not a copy of it.

    Importing app is cheap: the face/liveness models load in the FastAPI
    lifespan, not at import. A biometric DENY/RECAPTURE never reaches
    _apply_schedule in app.py, so it short-circuits the same way here.
    """
    import app
    if biometric in (dec.DENY, dec.RECAPTURE):
        sched = policy.evaluate(name, policy.parse_sim_time(sim_time))
        return biometric, msg.code_for(biometric, None), sched.reason, False
    verdict, code, reason, totp, _reasons = app._apply_schedule(
        biometric, name, sim_time)
    return verdict, code, reason, totp


CASES = [
    # (label, biometric verdict, name, sim_time, expected verdict, expected code)
    ("Divya 10:00 Mon        ", dec.ALLOW, "Divya", at(MON, "10:00"),
     dec.ALLOW, None),
    ("Divya 23:00 Mon        ", dec.ALLOW, "Divya", at(MON, "23:00"),
     dec.DENY, msg.SCHEDULE_DENY),
    ("Divya 18:20 Mon (grace)", dec.ALLOW, "Divya", at(MON, "18:20"),
     dec.STEP_UP, msg.SCHEDULE_GRACE),
    ("soham 10:00 Mon        ", dec.ALLOW, "soham", at(MON, "10:00"),
     dec.DENY, msg.SCHEDULE_DENY),
    ("soham 19:00 Mon        ", dec.ALLOW, "soham", at(MON, "19:00"),
     dec.ALLOW, None),
    # Extra guards on the rules around the six required cases.
    ("Divya 08:40 Mon (grace)", dec.ALLOW, "Divya", at(MON, "08:40"),
     dec.STEP_UP, msg.SCHEDULE_GRACE),
    ("Divya 10:00 Sat        ", dec.ALLOW, "Divya", at(SAT, "10:00"),
     dec.DENY, msg.SCHEDULE_DENY),
    ("Unknown user 10:00 Mon ", dec.ALLOW, "nobody", at(MON, "10:00"),
     dec.DENY, msg.SCHEDULE_DENY),
    ("Divya biometric STEP_UP", dec.STEP_UP, "Divya", at(MON, "10:00"),
     dec.STEP_UP, None),
    ("Divya STEP_UP at 23:00 ", dec.STEP_UP, "Divya", at(MON, "23:00"),
     dec.DENY, msg.SCHEDULE_DENY),
    ("Spoof inside window    ", dec.DENY, "Divya", at(MON, "10:00"),
     dec.DENY, None),
]


def main() -> int:
    print(f"zone={config.KIOSK_ZONE}  grace={config.SCHEDULE_GRACE_MIN} min  "
          f"source={policy.store.load() and policy.store.source}")
    print("-" * 94)
    failures = 0
    for label, biometric, name, sim, want_verdict, want_code in CASES:
        verdict, code, reason, totp = combine(biometric, name, sim)
        ok = verdict == want_verdict and (want_code is None or code == want_code)
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {label}  {biometric:9} -> "
              f"{verdict:8} {'code+' if totp else '    '} "
              f"{(code or '-'):16} {reason}")
        if not ok:
            print(f"      expected {want_verdict} / {want_code or 'any'}")

    print("-" * 94)
    # The line the kiosk shows must read as plain words.
    divya = policy.evaluate("Divya", policy.parse_sim_time(at(MON, "10:00")))
    print(f"schedule_reason sample: {divya.reason!r}")
    print(f"{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
