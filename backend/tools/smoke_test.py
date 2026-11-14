"""End-to-end API test: enroll -> verify -> challenge -> TOTP.

Uses the bundled sample faces, so it needs no webcam. Checks the live contract
including the STEP_UP rename and the mode / schedule_reason / totp_required
fields.

    python tools/smoke_test.py [base_url]
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sample_faces  # noqa: E402

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"
CONTRACT_FIELDS = {"decision", "name", "liveness", "match", "mode", "schedule_reason",
                   "totp_required", "reasons", "time_ms", "session_id", "challenge"}
DECISIONS = {"ALLOW", "STEP_UP", "DENY", "RECAPTURE"}

failures = []


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=30) as resp:
        return json.loads(resp.read())


def check(cond: bool, label: str):
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}")
    if not cond:
        failures.append(label)


def frames(base, n, offset=0):
    return [sample_faces.to_data_url(f)
            for f in sample_faces.burst(base, n, offset=offset)]


def main():
    ids = sample_faces.identities()
    labels = sorted(ids)
    me, impostor = labels[0], labels[1]

    print(f"\n[health] {BASE}")
    h = get("/health")
    print(f"  {h}")
    check(h.get("ok") is True, "health ok")

    print(f"\n[enroll] {me}")
    e = post("/enroll", {"name": me, "frames": frames(ids[me], 5)})
    print(f"  faces_used={e.get('faces_used')} ok={e.get('ok')}")
    check(e.get("ok") is True, "enroll succeeded")
    check(e.get("faces_used", 0) >= 3, "at least 3 faces used")
    check(str(e.get("totp_qr", "")).startswith("data:image/png;base64,"),
          "enroll returned a TOTP QR data URL")
    secret = e.get("totp_secret")
    check(bool(secret), "enroll returned the TOTP secret (DEV_MODE)")

    print(f"\n[verify] genuine ({me})")
    v = post("/verify", {"frames": frames(ids[me], 10, offset=500)})
    print(f"  decision={v['decision']} match={v['match']} liveness={v['liveness']} "
          f"time={v['time_ms']}ms")
    check(set(v) >= CONTRACT_FIELDS, "response has every contract field")
    check(v["decision"] in DECISIONS, "decision is in the allowed set")
    check(v["mode"] in ("online", "offline"), "mode is online|offline")
    check(isinstance(v["totp_required"], bool), "totp_required is a bool")
    check(v["name"] == me, f"identified as {me}")
    check(v["match"] >= 0.45, "genuine match above the allow threshold")
    check(v["time_ms"] < 2000, f"within the 2 s budget ({v['time_ms']} ms)")

    print(f"\n[verify] impostor ({impostor})")
    i = post("/verify", {"frames": frames(ids[impostor], 10, offset=900)})
    print(f"  decision={i['decision']} name={i['name']} match={i['match']}")
    check(i["decision"] == "DENY", "impostor denied")
    check(i["match"] < 0.25, "impostor match below the deny threshold")

    print("\n[verify] unusable frames")
    r = post("/verify", {"frames": ["not-an-image"] * 10})
    print(f"  decision={r['decision']} reason={r['reasons'][0]}")
    check(r["decision"] == "RECAPTURE", "unusable frames give RECAPTURE")

    print("\n[challenge] forcing the step-up path")
    get("/dev/force?decision=STEP_UP")
    v2 = post("/verify", {"frames": frames(ids[me], 10, offset=500)})
    sid = v2["session_id"]
    check(v2["decision"] == "STEP_UP", "forced STEP_UP")
    get("/dev/force?decision=off")

    totp_needed = False
    for attempt in range(1, 4):
        c = post("/challenge", {"session_id": sid, "frames": frames(ids[me], 20, offset=attempt * 50)})
        print(f"  attempt {attempt}: decision={c['decision']} "
              f"totp_required={c['totp_required']} :: {c['reasons'][0]}")
        if c["totp_required"]:
            totp_needed = True
            break
        if c["decision"] == "ALLOW":
            break

    if totp_needed:
        print("\n[totp]")
        code = get(f"/dev/totp/{me}")["code"]
        bad = post("/otp", {"session_id": sid, "code": "000000"})
        check(bad["decision"] == "DENY", "wrong TOTP code denied")
        good = post("/otp", {"session_id": sid, "code": code})
        print(f"  correct code -> {good['decision']}")
        check(good["decision"] == "ALLOW", "correct TOTP code allowed")
        replay = post("/otp", {"session_id": sid, "code": code})
        print(f"  replayed code -> {replay['decision']} ({replay.get('reason')})")
        check(replay["decision"] == "DENY", "replayed TOTP code rejected")
    else:
        print("\n[totp] skipped: challenge passed before the TOTP fallback")

    print("\n" + ("smoke test OK" if not failures
                  else f"smoke test FAILED ({len(failures)}): " + "; ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.URLError as exc:
        sys.exit(f"cannot reach {BASE}: {exc}")
