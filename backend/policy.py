"""Campus access schedule: the authorisation half of the decision.

A live, recognised person is not automatically authorised. decision.py answers
"is this really Divya, live, at the kiosk?"; this module answers "is Divya
allowed into THIS zone at THIS moment?". The final verdict is the combination:

    biometric ALLOW + inside window                -> ALLOW
    biometric ALLOW + within SCHEDULE_GRACE_MIN    -> STEP_UP (authenticator code)
    outside window / wrong zone / expired / none   -> DENY
    biometric STEP_UP                              -> challenge, then re-check
    biometric DENY                                 -> DENY (never upgraded)

Schedule rows come from the Supabase table `schedules`, read over its REST API
with httpx (no extra SDK). When Supabase is not configured or not reachable we
fall back to backend/schedule.json, which has the same shape -- so the demo
works offline. Rows are cached for CACHE_TTL_S so a /verify never pays for a
network round trip twice in a row.
"""

import datetime as dt
import json
import logging
import os
import threading
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import config

log = logging.getLogger("policy")

# Schedule outcomes. These are NOT API verdicts -- app.py maps them onto
# ALLOW / STEP_UP / DENY together with the biometric verdict.
IN_WINDOW = "IN_WINDOW"
GRACE = "GRACE"
OUT_OF_WINDOW = "OUT_OF_WINDOW"
WRONG_ZONE = "WRONG_ZONE"
EXPIRED = "EXPIRED"
NO_SCHEDULE = "NO_SCHEDULE"

SCHEDULE_PATH = config.BASE_DIR / "schedule.json"
CACHE_TTL_S = 60.0

_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_DAY_INDEX = {name.lower(): i for i, name in enumerate(_DAY_NAMES)}

DASH = "–"      # en dash, for 09:00-18:00 rendered as a range
DOT = "·"       # middle dot separator


# ------------------------------------------------------------------- row model

@dataclass
class Schedule:
    user_name: str
    zone: str
    days: str
    start_time: dt.time
    end_time: dt.time
    valid_until: Optional[dt.date]
    days_set: frozenset           # weekday numbers, Mon=0
    days_label: str               # "Mon-Fri" rendered for a human

    def window_label(self) -> str:
        """Plain words for the kiosk, e.g. "Lab B / Mon-Fri 09:00-18:00"."""
        return (f"{zone_label(self.zone)} {DOT} {self.days_label} "
                f"{self.start_time:%H:%M}{DASH}{self.end_time:%H:%M}")


@dataclass
class ScheduleResult:
    status: str
    reason: str                   # one plain-words line for the result card
    schedule: Optional[Schedule] = None

    @property
    def allowed(self) -> bool:
        return self.status == IN_WINDOW

    @property
    def grace(self) -> bool:
        return self.status == GRACE


def zone_label(zone: str) -> str:
    """lab_b -> "Lab B"."""
    return " ".join(part.upper() if len(part) == 1 else part.capitalize()
                    for part in str(zone).replace("-", "_").split("_") if part)


# -------------------------------------------------------------------- parsing

def _parse_time(value) -> Optional[dt.time]:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%H:%M:%S", "%H:%M", "%H%M"):
        try:
            return dt.datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    log.warning("unparseable time in schedule row: %r", value)
    return None


def _parse_date(value) -> Optional[dt.date]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text[:10])
    except ValueError:
        log.warning("unparseable valid_until in schedule row: %r", value)
        return None


def _parse_days(spec) -> Tuple[frozenset, str]:
    """Day spec -> (weekday numbers, pretty label).

    Accepts "Mon-Fri", "Mon,Wed,Fri", "daily". Unparseable input yields an
    empty set, which makes the row match nothing -- a typo in the table denies
    rather than silently opening the door.
    """
    text = str(spec or "").strip()
    if not text:
        return frozenset(), ""
    if text.lower() in ("daily", "all", "everyday", "every day", "mon-sun"):
        return frozenset(range(7)), "Every day"

    days = set()
    labels = []
    for chunk in text.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk or DASH in chunk:
            lo, _, hi = chunk.replace(DASH, "-").partition("-")
            a = _DAY_INDEX.get(lo.strip()[:3].lower())
            b = _DAY_INDEX.get(hi.strip()[:3].lower())
            if a is None or b is None:
                log.warning("unparseable day range in schedule row: %r", chunk)
                continue
            # Wraps across the weekend (e.g. Sat-Mon) when b < a.
            days.update((a + i) % 7 for i in range((b - a) % 7 + 1))
            labels.append(f"{_DAY_NAMES[a]}{DASH}{_DAY_NAMES[b]}")
        else:
            one = _DAY_INDEX.get(chunk[:3].lower())
            if one is None:
                log.warning("unparseable day in schedule row: %r", chunk)
                continue
            days.add(one)
            labels.append(_DAY_NAMES[one])
    return frozenset(days), ", ".join(labels)


def _row_to_schedule(row: dict) -> Optional[Schedule]:
    start = _parse_time(row.get("start_time"))
    end = _parse_time(row.get("end_time"))
    name = str(row.get("user_name") or "").strip()
    if not name or start is None or end is None:
        log.warning("skipping incomplete schedule row for %r", row.get("user_name"))
        return None
    days_set, days_label = _parse_days(row.get("days"))
    return Schedule(
        user_name=name,
        zone=str(row.get("zone") or "").strip(),
        days=str(row.get("days") or "").strip(),
        start_time=start,
        end_time=end,
        valid_until=_parse_date(row.get("valid_until")),
        days_set=days_set,
        days_label=days_label or str(row.get("days") or "").strip(),
    )


# --------------------------------------------------------------------- sources

def _env(key: str) -> str:
    return (os.getenv(key) or "").strip()


def _supabase_rows() -> Optional[List[dict]]:
    """Read the `schedules` table over the Supabase REST API.

    Returns None when Supabase is not configured or the read failed, which is
    the caller's signal to fall back to schedule.json.
    """
    url, key = _env("SUPABASE_URL"), _env("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return None
    try:
        import httpx
        response = httpx.get(
            url.rstrip("/") + "/rest/v1/schedules",
            params={"select": "user_name,zone,days,start_time,end_time,valid_until"},
            headers={"apikey": key, "Authorization": f"Bearer {key}"},
            timeout=3.0,
        )
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list):
            raise ValueError("expected a JSON array of rows")
        return rows
    except Exception as exc:
        # Log the class of failure only -- never the URL or the key.
        log.warning("Supabase schedule read failed (%s); using schedule.json",
                    type(exc).__name__)
        return None


def _file_rows() -> List[dict]:
    try:
        raw = json.loads(Path(SCHEDULE_PATH).read_text(encoding="utf-8"))
    except FileNotFoundError:
        log.warning("no schedule.json at %s -- every user will be denied",
                    SCHEDULE_PATH)
        return []
    except Exception as exc:
        log.error("schedule.json unreadable (%s) -- every user will be denied", exc)
        return []
    rows = raw.get("schedules") if isinstance(raw, dict) else raw
    return rows if isinstance(rows, list) else []


class ScheduleStore:
    """Cached schedule rows plus the name of the source they came from."""

    def __init__(self):
        self._lock = threading.Lock()
        self._rows: List[Schedule] = []
        self._loaded_at = 0.0
        self.source = "none"

    def load(self, force: bool = False) -> List[Schedule]:
        with self._lock:
            fresh = (_time.time() - self._loaded_at) < CACHE_TTL_S
            if not force and fresh and self._loaded_at:
                return self._rows
            rows = _supabase_rows()
            source = "supabase"
            if rows is None:
                rows = _file_rows()
                source = "schedule.json"
            parsed = [s for s in (_row_to_schedule(r) for r in rows) if s]
            self._rows, self._loaded_at, self.source = parsed, _time.time(), source
            log.info("loaded %d schedule row(s) from %s", len(parsed), source)
            return self._rows

    def for_user(self, name: str) -> List[Schedule]:
        wanted = (name or "").strip().lower()
        return [s for s in self.load() if s.user_name.strip().lower() == wanted]


store = ScheduleStore()


# ------------------------------------------------------------------ evaluation

def _minutes_outside(now: dt.datetime, sched: Schedule) -> Optional[float]:
    """Minutes `now` sits outside the nearest instance of this window.

    0.0 means inside the window. None means `now` is further than the grace
    band from any scheduled day, so the grace rule cannot apply.
    """
    grace = dt.timedelta(minutes=config.SCHEDULE_GRACE_MIN)
    best: Optional[float] = None
    # Yesterday/today/tomorrow, so a grace band may cross midnight.
    for offset in (-1, 0, 1):
        day = now.date() + dt.timedelta(days=offset)
        if day.weekday() not in sched.days_set:
            continue
        start = dt.datetime.combine(day, sched.start_time)
        end = dt.datetime.combine(day, sched.end_time)
        if end <= start:                      # window spans midnight
            end += dt.timedelta(days=1)
        if start <= now <= end:
            return 0.0
        gap = (start - now) if now < start else (now - end)
        if gap <= grace:
            minutes = gap.total_seconds() / 60.0
            best = minutes if best is None else min(best, minutes)
    return best


def evaluate(name: Optional[str], now: Optional[dt.datetime] = None,
             zone: Optional[str] = None) -> ScheduleResult:
    """Does the campus schedule authorise `name` into `zone` at `now`?"""
    now = now or dt.datetime.now()
    zone = (zone or config.KIOSK_ZONE).strip()

    if not name:
        return ScheduleResult(NO_SCHEDULE,
                              f"{zone_label(zone)} {DOT} no schedule on file")

    rows = store.for_user(name)
    if not rows:
        return ScheduleResult(NO_SCHEDULE,
                              f"No access schedule on file for {name}")

    in_zone = [s for s in rows if s.zone.strip().lower() == zone.lower()]
    if not in_zone:
        other = ", ".join(sorted({zone_label(s.zone) for s in rows if s.zone}))
        return ScheduleResult(
            WRONG_ZONE,
            f"Not cleared for {zone_label(zone)}"
            + (f" {DOT} cleared for {other}" if other else ""),
            rows[0])

    # Prefer a row that authorises outright, then the nearest row in its grace
    # band, so a user with several windows gets the best of them.
    best_grace: Optional[Tuple[float, Schedule]] = None
    expired: Optional[Schedule] = None
    live: List[Schedule] = []
    for sched in in_zone:
        if sched.valid_until and now.date() > sched.valid_until:
            expired = expired or sched
            continue
        live.append(sched)
        gap = _minutes_outside(now, sched)
        if gap == 0.0:
            return ScheduleResult(IN_WINDOW, sched.window_label(), sched)
        if gap is not None and (best_grace is None or gap < best_grace[0]):
            best_grace = (gap, sched)

    if best_grace:
        return ScheduleResult(GRACE, best_grace[1].window_label(), best_grace[1])

    if not live and expired:
        return ScheduleResult(
            EXPIRED,
            f"{zone_label(expired.zone)} access expired on "
            f"{expired.valid_until:%d %b %Y}",
            expired)

    return ScheduleResult(OUT_OF_WINDOW,
                          f"Outside your hours {DOT} {live[0].window_label()}",
                          live[0])


def parse_sim_time(value: Optional[str]) -> Optional[dt.datetime]:
    """Demo helper: an ISO datetime from the request, or None for "now".

    Accepts "2026-11-16T10:00" and a trailing "Z". A tz-aware value is reduced
    to naive local time, because schedule windows are wall-clock times.
    """
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = dt.datetime.strptime(text, "%Y-%m-%d %H:%M")
        except ValueError:
            log.warning("ignoring unparseable sim_time %r", text)
            return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed
