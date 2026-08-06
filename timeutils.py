"""
RegistraX Solar — Indian Standard Time helpers.

The app is used inside India only, so every timestamp that gets written to
the database or shown on screen (attendance check-in/out, OTP expiry,
"today" for reports) must be in IST — not the server's local time and not
naive UTC. A cloud VPS is very often provisioned with its clock in UTC, so
relying on plain `datetime.now()` would silently show/store times ~5.5
hours off from what happened in real life.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def now_ist():
    """Current time as a timezone-aware datetime in IST."""
    return datetime.now(IST)


def now_ist_str():
    """Current IST time formatted for storage in the DB (naive string, IST wall-clock)."""
    return now_ist().strftime("%Y-%m-%d %H:%M:%S")


def today_ist_str():
    """Today's date in IST, 'YYYY-MM-DD' — use this instead of date.today()."""
    return now_ist().date().isoformat()
