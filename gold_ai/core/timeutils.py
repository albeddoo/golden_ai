"""UTC-normalized time helpers.

Rule: all timestamps are stored/compared internally in UTC. The UI may display
the user's selected timezone; conversion happens only at presentation time.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

UTC = timezone.utc


def utcnow() -> datetime:
    """Timezone-aware current UTC time."""
    return datetime.now(tz=UTC)


def to_utc(dt: datetime) -> datetime:
    """Normalize any datetime to UTC-aware. Naive datetimes are assumed UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def epoch_ms(dt: datetime) -> int:
    return int(to_utc(dt).timestamp() * 1000)


def from_epoch_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC)


def iso(dt: datetime) -> str:
    return to_utc(dt).isoformat()


def age_seconds(dt: datetime, now: datetime | None = None) -> float:
    """Seconds elapsed since ``dt`` (both normalized to UTC)."""
    now = now or utcnow()
    return max(0.0, (to_utc(now) - to_utc(dt)).total_seconds())


def fmt_age(seconds: float) -> str:
    """Human readable freshness string used across the UI."""
    if seconds < 2:
        return "live"
    if seconds < 120:
        return f"{seconds:.0f}s ago"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m ago"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h ago"
    return f"{seconds / 86400:.1f}d ago"


def countdown(seconds: float) -> str:
    """Format seconds remaining as e.g. '2d 04:11:09' / '04:11:09'."""
    if seconds <= 0:
        return "NOW"
    d, rem = divmod(int(seconds), 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d {h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def floor_to(dt: datetime, delta: timedelta) -> datetime:
    """Floor a UTC timestamp to a multiple of ``delta`` since epoch."""
    ts = to_utc(dt).timestamp()
    step = delta.total_seconds()
    return datetime.fromtimestamp(ts - (ts % step), tz=UTC)
