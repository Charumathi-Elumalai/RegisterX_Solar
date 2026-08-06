"""
RegistraX Solar — salary calculation.

Pairs up each day's 'in'/'out' attendance rows into worked hours, then
applies the employee's pay_type:
  - 'daily'   : pay_rate per day the employee was present (any valid in+out pair)
  - 'hourly'  : pay_rate * total hours worked in the period
  - 'monthly' : pay_rate / expected working days in the period * days present
                (a simple pro-rata; swap in your own payroll formula here if
                the client's rules are more specific — e.g. Sundays off,
                fixed monthly regardless of attendance, etc.)

This is intentionally a straightforward, inspectable reference
implementation — real payroll rules (paid leave, holidays, overtime
multipliers, PF/ESI deductions) vary a lot between businesses, so treat this
as the starting point to customize with the client's actual rules rather
than a finished payroll engine.
"""
from datetime import datetime, timedelta
from collections import defaultdict


def _parse_ts(ts):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(ts)


def _daterange(start_date, end_date):
    d = start_date
    while d <= end_date:
        yield d
        d += timedelta(days=1)


def compute_salary(conn, employee_id, start_date, end_date):
    """
    start_date, end_date: 'YYYY-MM-DD' strings, inclusive.
    Returns a dict with a day-by-day breakdown and a total.
    """
    employee = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
    if employee is None:
        return None

    rows = conn.execute(
        """SELECT check_type, timestamp FROM attendance
           WHERE employee_id = ? AND date(timestamp) BETWEEN ? AND ?
           ORDER BY timestamp ASC""",
        (employee_id, start_date, end_date),
    ).fetchall()

    by_day = defaultdict(list)
    for r in rows:
        day = r["timestamp"][:10]
        by_day[day].append((r["check_type"], _parse_ts(r["timestamp"])))

    start_d = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_d = datetime.strptime(end_date, "%Y-%m-%d").date()

    pay_type = employee["pay_type"]
    pay_rate = float(employee["pay_rate"] or 0)
    expected_hours = float(employee["expected_hours_per_day"] or 8)

    daily_breakdown = []
    total_hours = 0.0
    days_present = 0

    for d in _daterange(start_d, end_d):
        key = d.isoformat()
        events = sorted(by_day.get(key, []), key=lambda e: e[1])

        hours_today = 0.0
        pending_in = None
        for check_type, ts in events:
            if check_type == "in":
                pending_in = ts
            elif check_type == "out" and pending_in is not None:
                hours_today += max(0.0, (ts - pending_in).total_seconds() / 3600.0)
                pending_in = None

        present = hours_today > 0 or any(c == "in" for c, _ in events)
        if present:
            days_present += 1
        total_hours += hours_today

        day_pay = 0.0
        if pay_type == "hourly":
            day_pay = hours_today * pay_rate
        elif pay_type == "daily":
            day_pay = pay_rate if present else 0.0
        # 'monthly' is pro-rated at the end from days_present, not per day

        daily_breakdown.append({
            "date": key,
            "hours": round(hours_today, 2),
            "present": present,
            "pay": round(day_pay, 2),
        })

    total_days_in_range = (end_d - start_d).days + 1

    if pay_type == "hourly":
        total_pay = round(total_hours * pay_rate, 2)
    elif pay_type == "daily":
        total_pay = round(days_present * pay_rate, 2)
    elif pay_type == "monthly":
        per_day = (pay_rate / total_days_in_range) if total_days_in_range else 0
        total_pay = round(per_day * days_present, 2)
    else:
        total_pay = 0.0

    return {
        "employee_id": employee_id,
        "employee_name": employee["name"],
        "emp_code": employee["emp_code"],
        "pay_type": pay_type,
        "pay_rate": pay_rate,
        "start_date": start_date,
        "end_date": end_date,
        "days_present": days_present,
        "total_days_in_range": total_days_in_range,
        "total_hours": round(total_hours, 2),
        "total_pay": total_pay,
        "daily_breakdown": daily_breakdown,
    }


def compute_salary_all(conn, start_date, end_date):
    employees = conn.execute("SELECT id FROM employees WHERE active = 1").fetchall()
    results = []
    for e in employees:
        r = compute_salary(conn, e["id"], start_date, end_date)
        if r:
            results.append(r)
    return results
