import os
import csv
import io
import re
import random
import secrets
from datetime import datetime, time as dtime, timedelta

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, flash, Response, send_from_directory
)
from werkzeug.security import check_password_hash, generate_password_hash

from database import get_db, init_db, DB_PATH, get_setting, set_setting
from face_engine import (
    decode_base64_image, save_training_face, train_model, recognize_face,
    detect_face_quality, _anti_spoofing_available, check_prompted_liveness,
)
from geo_utils import is_within_geofence
from timeutils import now_ist, now_ist_str, today_ist_str
from sms_utils import send_otp_sms
import salary as salary_mod

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# Accepts an optional leading + and 7-15 digits — permissive on purpose since
# this only gates "looks like a phone number", not a specific country format.
PHONE_RE = re.compile(r"^\+?[0-9]{7,15}$")
USERNAME_RE = re.compile(r"^[a-zA-Z0-9._]{3,20}$")


def _normalize_username(raw):
    return (raw or "").strip().lower()


def _normalize_phone(raw):
    """Strips spaces/dashes so the same number typed different ways
    (e.g. '98765 43210' vs '9876543210') matches on login."""
    return re.sub(r"[\s\-()]", "", (raw or "").strip())

BASE_DIR = os.path.dirname(__file__)
ATTENDANCE_PHOTO_DIR = os.path.join(BASE_DIR, "attendance_photos")
os.makedirs(ATTENDANCE_PHOTO_DIR, exist_ok=True)

# In-memory holding area for signups that haven't finished phone
# verification + PIN creation yet. Keyed by "role:phone". Nothing here is
# written to the database until create_pin() succeeds, so an abandoned
# signup never leaves a half-created account behind.
otp_store = {}


def _pending_key(role, phone):
    return f"{role}:{_normalize_phone(phone)}"


def _gen_otp():
    return f"{random.randint(100000, 999999)}"


app = Flask(__name__)
# Reads SECRET_KEY from the environment in production; falls back to a
# random key generated at process start (fine for local/dev use — sessions
# just get invalidated on restart) so there's never a hardcoded secret in code.
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
DEBUG_MODE = os.environ.get("FLASK_DEBUG", "0") == "1"

init_db()  # safe: uses CREATE TABLE IF NOT EXISTS
train_model()  # warm the in-memory face-embedding cache on startup


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def login_required(view):
    from functools import wraps

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("admin_id"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def employee_login_required(view):
    from functools import wraps

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("employee_id"):
            return redirect(url_for("employee_login"))
        return view(*args, **kwargs)
    return wrapped


@app.route("/register", methods=["GET", "POST"])
def register():
    """Admin sign-up, step 1: name + mobile number. Sends an OTP by SMS and
    moves on to /verify-otp. Nothing is written to the DB until the PIN
    step succeeds."""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        phone = _normalize_phone(request.form.get("phone", ""))

        if not name or not phone or not PHONE_RE.match(phone):
            flash("Enter your name and a valid mobile number.", "error")
            return render_template("register.html")

        conn = get_db()
        existing = conn.execute("SELECT id FROM admin_users WHERE phone = ?", (phone,)).fetchone()
        conn.close()
        if existing:
            flash("An admin account already exists for this mobile number. Please log in.", "error")
            return redirect(url_for("login"))

        otp = _gen_otp()
        otp_store[_pending_key("admin", phone)] = {
            "otp": otp, "role": "admin", "phone": phone, "name": name,
            "expires": now_ist() + timedelta(minutes=10), "verified": False,
        }
        send_otp_sms(phone, otp, "verify your RegistraX Solar admin account")
        session["pending_role"] = "admin"
        session["pending_phone"] = phone
        flash("We've sent a 6-digit verification code by SMS.", "info")
        return redirect(url_for("verify_otp"))

    return render_template("register.html")


@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():
    """Step 2, shared by admin and employee signup: confirm the SMS OTP
    really belongs to this phone before letting anyone create a PIN for it —
    this is the actual identity check; the PIN alone would let anyone type
    in someone else's number and claim their account."""
    role = session.get("pending_role")
    phone = session.get("pending_phone")
    key = _pending_key(role, phone) if role and phone else None

    if not key or key not in otp_store:
        flash("Your signup session expired. Please start again.", "error")
        return redirect(url_for("register") if role == "admin" else url_for("employee_register"))

    if request.method == "POST":
        entered = request.form.get("otp", "").strip()
        pending = otp_store[key]

        if now_ist() > pending["expires"]:
            otp_store.pop(key, None)
            session.pop("pending_role", None)
            session.pop("pending_phone", None)
            flash("Code expired. Please sign up again.", "error")
            return redirect(url_for("register") if role == "admin" else url_for("employee_register"))

        if entered == pending["otp"]:
            pending["verified"] = True
            flash("Mobile number verified! Now set a 6-digit PIN — you'll use this to log in.", "success")
            return redirect(url_for("create_pin"))
        flash("Incorrect code. Please try again.", "error")

    return render_template("verify_otp.html", phone=phone)


@app.route("/create-pin", methods=["GET", "POST"])
def create_pin():
    """Step 3: the verified phone gets a username + PIN. This is the ONLY
    place an admin_users/employees row is actually created — an
    unverified or abandoned signup never reaches the database.

    The mobile number is only ever used here, once, to prove the signup
    is really theirs. From this point on they log in with the username +
    PIN chosen on this screen — phone number is never asked for again.

    For employees this only creates a bare, 'incomplete' row (name + phone +
    PIN) — no site/role/personal details yet, and no admin request sent
    yet. They're logged straight in and walked through completing their
    profile and enrolling their face next; only once both of those are
    done does their status flip to 'requested' and they show up for admin
    approval (see api_employee_enroll)."""
    role = session.get("pending_role")
    phone = session.get("pending_phone")
    key = _pending_key(role, phone) if role and phone else None
    pending = otp_store.get(key) if key else None

    if not pending or not pending.get("verified"):
        flash("Please verify your mobile number first.", "error")
        return redirect(url_for("register") if role == "admin" else url_for("employee_register"))

    if request.method == "POST":
        username = _normalize_username(request.form.get("username", ""))
        pin = request.form.get("pin", "").strip()
        pin_confirm = request.form.get("pin_confirm", "").strip()

        if not USERNAME_RE.match(username):
            flash("Username must be 3-20 characters: letters, numbers, dots, or underscores.", "error")
            return render_template("create_pin.html", phone=phone, username=username)
        if not (pin.isdigit() and len(pin) == 6):
            flash("PIN must be exactly 6 digits.", "error")
            return render_template("create_pin.html", phone=phone, username=username)
        if pin != pin_confirm:
            flash("The two PINs don't match.", "error")
            return render_template("create_pin.html", phone=phone, username=username)

        table = "admin_users" if role == "admin" else "employees"
        conn = get_db()
        taken = conn.execute(f"SELECT id FROM {table} WHERE username = ?", (username,)).fetchone()
        if taken:
            conn.close()
            flash("That username is already taken. Please choose another.", "error")
            return render_template("create_pin.html", phone=phone, username="")

        pin_hash = generate_password_hash(pin)

        if role == "admin":
            conn.execute(
                "INSERT INTO admin_users (username, phone, phone_verified, password_hash) VALUES (?, ?, 1, ?)",
                (username, phone, pin_hash),
            )
            conn.commit()
            conn.close()
            otp_store.pop(key, None)
            session.pop("pending_role", None)
            session.pop("pending_phone", None)
            flash("Admin account created! Log in with your username and PIN.", "success")
            return redirect(url_for("login"))

        else:
            temp_code = f"TMP{secrets.token_hex(4)}"
            cur = conn.execute(
                """INSERT INTO employees
                   (emp_code, name, phone, phone_verified, username, password_hash,
                    must_change_password, status, profile_completed, active)
                   VALUES (?, ?, ?, 1, ?, ?, 0, 'incomplete', 0, 0)""",
                (temp_code, pending["name"], phone, username, pin_hash),
            )
            new_id = cur.lastrowid
            conn.execute("UPDATE employees SET emp_code = ? WHERE id = ?", (f"EMP{new_id:04d}", new_id))
            conn.commit()
            conn.close()
            otp_store.pop(key, None)
            session.pop("pending_role", None)
            session.pop("pending_phone", None)
            session["employee_id"] = new_id
            flash("Account created! Now tell us a bit about yourself.", "success")
            return redirect(url_for("employee_complete_profile"))

    return render_template("create_pin.html", phone=phone, username="")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = _normalize_username(request.form.get("username", ""))
        pin = request.form.get("pin", "").strip()
        conn = get_db()
        user = conn.execute("SELECT * FROM admin_users WHERE username = ?", (username,)).fetchone()
        conn.close()
        if user and user["password_hash"] and check_password_hash(user["password_hash"], pin):
            session["admin_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        flash("Invalid username or PIN.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Employee self-service auth
# ---------------------------------------------------------------------------
@app.route("/employee/register", methods=["GET", "POST"])
def employee_register():
    """Employee self-signup, step 1: just name + mobile number. Site, job
    role, and personal details are collected next, on the profile page,
    once they're logged in (see employee_complete_profile) — this first
    step's only job is proving they own this phone number."""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        phone = _normalize_phone(request.form.get("phone", ""))

        if not (name and phone and PHONE_RE.match(phone)):
            flash("Name and a valid mobile number are required.", "error")
            return render_template("employee_register.html")

        conn = get_db()
        existing = conn.execute("SELECT id FROM employees WHERE phone = ?", (phone,)).fetchone()
        conn.close()
        if existing:
            flash("An account already exists for this mobile number. Please log in.", "error")
            return redirect(url_for("employee_login"))

        otp = _gen_otp()
        otp_store[_pending_key("employee", phone)] = {
            "otp": otp, "role": "employee", "phone": phone, "name": name,
            "expires": now_ist() + timedelta(minutes=10), "verified": False,
        }
        send_otp_sms(phone, otp, "verify your RegistraX Solar employee account")
        session["pending_role"] = "employee"
        session["pending_phone"] = phone
        flash("We've sent a 6-digit verification code by SMS.", "info")
        return redirect(url_for("verify_otp"))

    return render_template("employee_register.html")


@app.route("/employee/login", methods=["GET", "POST"])
def employee_login():
    if request.method == "POST":
        username = _normalize_username(request.form.get("username", ""))
        pin = request.form.get("pin", "").strip()
        conn = get_db()
        emp = conn.execute("SELECT * FROM employees WHERE username = ?", (username,)).fetchone()
        conn.close()

        if not emp or not emp["password_hash"] or not check_password_hash(emp["password_hash"], pin):
            flash("Invalid username or PIN.", "error")
            return render_template("employee_login.html")

        if emp["status"] == "terminated":
            flash("Your account has been terminated. Contact your admin.", "error")
            return render_template("employee_login.html")

        # incomplete / requested / active all get in — employee_portal()
        # routes them to whichever step (profile, face enrollment, waiting
        # screen, or the real portal) matches their status.
        session["employee_id"] = emp["id"]
        return redirect(url_for("employee_portal"))

    return render_template("employee_login.html")


@app.route("/employee/logout")
def employee_logout():
    session.pop("employee_id", None)
    return redirect(url_for("employee_login"))


@app.route("/employee/portal")
@employee_login_required
def employee_portal():
    conn = get_db()
    employee = conn.execute(
        """SELECT e.*, s.name as site_name FROM employees e
           LEFT JOIN sites s ON e.site_id = s.id WHERE e.id = ?""",
        (session["employee_id"],)
    ).fetchone()
    if employee is None:
        conn.close()
        session.pop("employee_id", None)
        flash("Account not found. Please log in again.", "error")
        return redirect(url_for("employee_login"))

    if employee["status"] == "terminated":
        conn.close()
        session.pop("employee_id", None)
        flash("Your account has been terminated. Contact your admin.", "error")
        return redirect(url_for("employee_login"))

    # Onboarding isn't finished yet — send them to whichever step is next
    # rather than showing an (empty) attendance portal.
    if not employee["profile_completed"]:
        conn.close()
        return redirect(url_for("employee_complete_profile"))
    if not employee["face_trained"]:
        conn.close()
        return redirect(url_for("employee_enroll_self"))

    records = conn.execute(
        """SELECT * FROM attendance WHERE employee_id = ?
           ORDER BY timestamp DESC LIMIT 30""",
        (employee["id"],)
    ).fetchall()
    conn.close()

    # Salary is admin-only now (see /salary and /employees/<id>/salary) —
    # employees see their own attendance history here, not pay figures.
    return render_template("employee_portal.html", employee=employee, records=records)


@app.route("/employee/complete-profile", methods=["GET", "POST"])
@employee_login_required
def employee_complete_profile():
    """Step 2 of employee onboarding: personal + professional details.
    Runs right after signup (or on re-login if it was never finished).
    Nothing here sends the admin a request yet — that happens once face
    enrollment (step 3) also succeeds, in api_employee_enroll."""
    conn = get_db()
    employee = conn.execute("SELECT * FROM employees WHERE id = ?", (session["employee_id"],)).fetchone()
    if employee is None:
        conn.close()
        session.pop("employee_id", None)
        return redirect(url_for("employee_login"))
    if employee["status"] == "terminated":
        conn.close()
        session.pop("employee_id", None)
        flash("Your account has been terminated. Contact your admin.", "error")
        return redirect(url_for("employee_login"))

    all_sites = conn.execute("SELECT * FROM sites ORDER BY name").fetchall()

    if request.method == "POST":
        job_role = request.form.get("role", "").strip()
        site_id = request.form.get("site_id")
        address = request.form.get("address", "").strip()
        dob = request.form.get("date_of_birth", "").strip()
        email = request.form.get("email", "").strip().lower() or None

        if not (job_role and site_id):
            flash("Job role and site are required.", "error")
            conn.close()
            return render_template("employee_complete_profile.html", employee=employee, sites=all_sites)
        if email and not EMAIL_RE.match(email):
            flash("That email address doesn't look valid.", "error")
            conn.close()
            return render_template("employee_complete_profile.html", employee=employee, sites=all_sites)

        try:
            conn.execute(
                """UPDATE employees
                   SET role = ?, site_id = ?, address = ?, date_of_birth = ?,
                       email = ?, profile_completed = 1
                   WHERE id = ?""",
                (job_role, int(site_id), address, dob, email, employee["id"]),
            )
            conn.commit()
        except Exception as e:
            flash(f"Could not save details: {e}", "error")
            conn.close()
            return render_template("employee_complete_profile.html", employee=employee, sites=all_sites)
        conn.close()
        flash("Details saved. Now enroll your face so admin can review your request.", "success")
        return redirect(url_for("employee_enroll_self"))

    conn.close()
    return render_template("employee_complete_profile.html", employee=employee, sites=all_sites)


@app.route("/employee/enroll")
@employee_login_required
def employee_enroll_self():
    """Step 3 of employee onboarding: enroll their own face. Guarded so it
    can't be reached before the profile step is done."""
    conn = get_db()
    employee = conn.execute("SELECT * FROM employees WHERE id = ?", (session["employee_id"],)).fetchone()
    conn.close()
    if employee is None:
        session.pop("employee_id", None)
        return redirect(url_for("employee_login"))
    if not employee["profile_completed"]:
        return redirect(url_for("employee_complete_profile"))
    return render_template("employee_face_enroll.html", employee=employee)


@app.route("/api/employee/enroll", methods=["POST"])
@employee_login_required
def api_employee_enroll():
    """Employee's own version of api_enroll — same underlying face-capture
    logic, but scoped to the logged-in employee's own id, and on success it
    also flips status 'incomplete' -> 'requested', which is what actually
    sends the request to the admin (they show up on the Employees page's
    Requests list from this point on)."""
    employee_id = session["employee_id"]
    data = request.get_json()
    frames = data.get("frames", [])
    if not frames:
        return jsonify({"success": False, "message": "No frames received."}), 400

    saved = 0
    for i, frame_data in enumerate(frames):
        img = decode_base64_image(frame_data)
        if img is not None and save_training_face(employee_id, img, i):
            saved += 1

    if saved == 0:
        return jsonify({"success": False, "message": "No face detected in any frame. Try better lighting."}), 400

    train_model()

    conn = get_db()
    emp = conn.execute("SELECT status FROM employees WHERE id = ?", (employee_id,)).fetchone()
    conn.execute("UPDATE employees SET face_trained = 1 WHERE id = ?", (employee_id,))
    if emp and emp["status"] == "incomplete":
        conn.execute("UPDATE employees SET status = 'requested' WHERE id = ?", (employee_id,))
    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "message": f"Enrolled {saved} face samples. Your request has been sent to your admin for approval.",
    })


@app.route("/employee/change_password", methods=["POST"])
@employee_login_required
def employee_change_password():
    new_pin = request.form.get("new_password", "").strip()
    if not (new_pin.isdigit() and len(new_pin) == 6):
        flash("New PIN must be exactly 6 digits.", "error")
        return redirect(url_for("employee_portal"))
    conn = get_db()
    conn.execute(
        "UPDATE employees SET password_hash = ?, must_change_password = 0 WHERE id = ?",
        (generate_password_hash(new_pin), session["employee_id"])
    )
    conn.commit()
    conn.close()
    flash("PIN updated.", "success")
    return redirect(url_for("employee_portal"))


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@app.route("/")
@login_required
def dashboard():
    conn = get_db()
    total_sites = conn.execute("SELECT COUNT(*) c FROM sites").fetchone()["c"]
    total_employees = conn.execute("SELECT COUNT(*) c FROM employees WHERE status='active'").fetchone()["c"]
    pending_requests = conn.execute("SELECT COUNT(*) c FROM employees WHERE status='requested'").fetchone()["c"]
    today = today_ist_str()
    today_checkins = conn.execute(
        "SELECT COUNT(DISTINCT employee_id) c FROM attendance WHERE date(timestamp) = ? AND check_type='in'",
        (today,)
    ).fetchone()["c"]
    flagged_today = conn.execute(
        "SELECT COUNT(*) c FROM attendance WHERE date(timestamp) = ? AND status != 'verified'",
        (today,)
    ).fetchone()["c"]

    recent = conn.execute("""
        SELECT a.*, e.name as emp_name, e.emp_code, s.name as site_name
        FROM attendance a
        JOIN employees e ON a.employee_id = e.id
        JOIN sites s ON a.site_id = s.id
        ORDER BY a.timestamp DESC LIMIT 15
    """).fetchall()

    per_site = conn.execute("""
        SELECT s.id, s.name,
               COUNT(DISTINCT e.id) as employee_count,
               COUNT(DISTINCT CASE WHEN date(a.timestamp) = ? AND a.check_type='in' THEN a.employee_id END) as present_today
        FROM sites s
        LEFT JOIN employees e ON e.site_id = s.id AND e.active = 1
        LEFT JOIN attendance a ON a.site_id = s.id
        GROUP BY s.id
    """, (today,)).fetchall()

    conn.close()
    return render_template(
        "dashboard.html",
        total_sites=total_sites,
        total_employees=total_employees,
        pending_requests=pending_requests,
        today_checkins=today_checkins,
        flagged_today=flagged_today,
        recent=recent,
        per_site=per_site,
    )


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------
@app.route("/sites")
@login_required
def sites():
    conn = get_db()
    all_sites = conn.execute("""
        SELECT s.*, COUNT(e.id) as employee_count
        FROM sites s LEFT JOIN employees e ON e.site_id = s.id AND e.active = 1
        GROUP BY s.id ORDER BY s.created_at DESC
    """).fetchall()
    conn.close()
    return render_template("sites.html", sites=all_sites)


@app.route("/sites/add", methods=["POST"])
@login_required
def add_site():
    name = request.form.get("name", "").strip()
    address = request.form.get("address", "").strip()
    lat = request.form.get("latitude")
    lon = request.form.get("longitude")
    radius = request.form.get("radius_meters", 200)

    if not (name and lat and lon):
        flash("Site name, latitude and longitude are required.", "error")
        return redirect(url_for("sites"))

    conn = get_db()
    conn.execute(
        "INSERT INTO sites (name, address, latitude, longitude, radius_meters) VALUES (?, ?, ?, ?, ?)",
        (name, address, float(lat), float(lon), int(radius)),
    )
    conn.commit()
    conn.close()
    flash(f"Site '{name}' added.", "success")
    return redirect(url_for("sites"))


@app.route("/sites/<int:site_id>/delete", methods=["POST"])
@login_required
def delete_site(site_id):
    conn = get_db()
    conn.execute("DELETE FROM sites WHERE id = ?", (site_id,))
    conn.commit()
    conn.close()
    flash("Site removed.", "success")
    return redirect(url_for("sites"))


# ---------------------------------------------------------------------------
# Employees
# ---------------------------------------------------------------------------
@app.route("/employees")
@login_required
def employees():
    conn = get_db()
    # Self-signed-up requests (profile + face enrollment already done by the
    # employee) waiting on admin approval — surfaced first so they can't
    # get missed among the regular staff list.
    requested_employees = conn.execute("""
        SELECT e.*, s.name as site_name
        FROM employees e LEFT JOIN sites s ON e.site_id = s.id
        WHERE e.status = 'requested'
        ORDER BY e.created_at DESC
    """).fetchall()
    active_employees = conn.execute("""
        SELECT e.*, s.name as site_name
        FROM employees e LEFT JOIN sites s ON e.site_id = s.id
        WHERE e.status = 'active'
        ORDER BY e.created_at DESC
    """).fetchall()
    terminated_employees = conn.execute("""
        SELECT e.*, s.name as site_name
        FROM employees e LEFT JOIN sites s ON e.site_id = s.id
        WHERE e.status = 'terminated'
        ORDER BY e.created_at DESC
    """).fetchall()
    all_sites = conn.execute("SELECT * FROM sites ORDER BY name").fetchall()
    conn.close()
    return render_template(
        "employees.html",
        employees=active_employees,
        pending_employees=requested_employees,
        terminated_employees=terminated_employees,
        sites=all_sites,
    )


@app.route("/employees/<int:employee_id>/approve", methods=["POST"])
@login_required
def approve_employee(employee_id):
    """Approves a self-signed-up request. By this point the employee has
    already completed their profile and enrolled their face themselves
    (that's what got them onto the Requests list), so approval is just
    flipping them to active — no separate enrollment step needed here."""
    conn = get_db()
    emp = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
    if emp is None:
        conn.close()
        flash("Request not found.", "error")
        return redirect(url_for("employees"))
    conn.execute("UPDATE employees SET status = 'active', active = 1 WHERE id = ?", (employee_id,))
    conn.commit()
    conn.close()
    if emp["face_trained"]:
        flash(f"{emp['name']} approved and can now check in.", "success")
        return redirect(url_for("employees"))
    flash(f"{emp['name']} approved. Their face isn't enrolled yet — enroll it so they can check in.", "success")
    return redirect(url_for("enroll_face", employee_id=employee_id))


@app.route("/employees/<int:employee_id>/terminate", methods=["POST"])
@login_required
def terminate_employee(employee_id):
    """Terminates an employee — from a pending request (declining it) or
    from the active list (ending someone's employment). Either way they're
    permanently blocked from checking in or logging in until an admin
    explicitly reactivates them (see reactivate_employee)."""
    conn = get_db()
    emp = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
    if emp is None:
        conn.close()
        flash("Employee not found.", "error")
        return redirect(url_for("employees"))
    conn.execute("UPDATE employees SET status = 'terminated', active = 0 WHERE id = ?", (employee_id,))
    conn.commit()
    conn.close()
    flash(f"{emp['name']} has been terminated and can no longer check in.", "success")
    return redirect(url_for("employees"))


@app.route("/employees/<int:employee_id>/reactivate", methods=["POST"])
@login_required
def reactivate_employee(employee_id):
    conn = get_db()
    emp = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
    if emp is None:
        conn.close()
        flash("Employee not found.", "error")
        return redirect(url_for("employees"))
    conn.execute("UPDATE employees SET status = 'active', active = 1 WHERE id = ?", (employee_id,))
    conn.commit()
    conn.close()
    flash(f"{emp['name']} reactivated.", "success")
    return redirect(url_for("employees"))


@app.route("/employees/add", methods=["POST"])
@login_required
def add_employee():
    """Admin adding an employee directly (as opposed to the employee
    self-signing-up). Since the admin is entering everything themselves,
    this employee is created already 'active' with profile_completed=1 —
    no approval request needed. Face still needs enrolling (by the admin,
    same as before) before they can check in."""
    emp_code = request.form.get("emp_code", "").strip()
    name = request.form.get("name", "").strip()
    phone = _normalize_phone(request.form.get("phone", ""))
    email = request.form.get("email", "").strip().lower() or None
    role = request.form.get("role", "").strip()
    site_id = request.form.get("site_id")
    pay_type = request.form.get("pay_type", "daily")
    pay_rate = request.form.get("pay_rate") or 0
    expected_hours = request.form.get("expected_hours_per_day") or 8

    if not (emp_code and name and site_id and phone and PHONE_RE.match(phone)):
        flash("Employee code, name, a valid mobile number, and site are required.", "error")
        return redirect(url_for("employees"))
    if email and not EMAIL_RE.match(email):
        flash("That email address doesn't look valid.", "error")
        return redirect(url_for("employees"))

    conn = get_db()
    try:
        cur = conn.execute(
            """INSERT INTO employees
               (emp_code, name, phone, phone_verified, role, site_id, email,
                pay_type, pay_rate, expected_hours_per_day, status, profile_completed, active)
               VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, 'active', 1, 1)""",
            (emp_code, name, phone, role, int(site_id), email, pay_type, float(pay_rate), float(expected_hours)),
        )
        conn.commit()
        new_id = cur.lastrowid
    except Exception as e:
        conn.close()
        flash(f"Could not add employee (duplicate code or mobile number?): {e}", "error")
        return redirect(url_for("employees"))
    conn.close()
    flash(f"Employee '{name}' added. Now enroll their face.", "success")
    return redirect(url_for("enroll_face", employee_id=new_id))


@app.route("/employees/<int:employee_id>/portal_login", methods=["POST"])
@login_required
def create_portal_login(employee_id):
    """Issues/resets a username + temporary PIN for an admin-added
    employee's own portal login (self-signed-up employees already have
    one from signup). Mobile number is never the login identifier —
    only used once, if at all, for their own OTP if they'd self-signed-up."""
    username = _normalize_username(request.form.get("username", ""))
    conn = get_db()
    emp = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
    if emp is None:
        conn.close()
        flash("Employee not found.", "error")
        return redirect(url_for("employees"))

    if not emp["username"]:
        if not USERNAME_RE.match(username):
            conn.close()
            flash("Enter a username: 3-20 characters, letters/numbers/dots/underscores.", "error")
            return redirect(url_for("employees"))
        taken = conn.execute(
            "SELECT id FROM employees WHERE username = ? AND id != ?", (username, employee_id)
        ).fetchone()
        if taken:
            conn.close()
            flash("That username is already taken. Please choose another.", "error")
            return redirect(url_for("employees"))
    else:
        username = emp["username"]  # already has one — this call is just a PIN reset

    temp_pin = f"{random.randint(0, 999999):06d}"
    conn.execute(
        "UPDATE employees SET username = ?, password_hash = ?, must_change_password = 1 WHERE id = ?",
        (username, generate_password_hash(temp_pin), employee_id),
    )
    conn.commit()
    conn.close()
    flash(
        f"Portal login ready for {emp['name']} — they log in at Employee Login with "
        f"username \"{username}\" and this temporary PIN: {temp_pin}. "
        f"Share it with them directly; it won't be shown again.",
        "success",
    )
    return redirect(url_for("employees"))


# ---------------------------------------------------------------------------
# Face enrollment
# ---------------------------------------------------------------------------
@app.route("/employees/<int:employee_id>/enroll")
@login_required
def enroll_face(employee_id):
    conn = get_db()
    emp = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
    conn.close()
    if emp is None:
        flash("Employee not found.", "error")
        return redirect(url_for("employees"))
    return render_template("enroll.html", employee=emp)


@app.route("/api/enroll/<int:employee_id>", methods=["POST"])
@login_required
def api_enroll(employee_id):
    """Receives a batch of base64 webcam frames, saves detected faces, retrains model."""
    data = request.get_json()
    frames = data.get("frames", [])
    if not frames:
        return jsonify({"success": False, "message": "No frames received."}), 400

    saved = 0
    for i, frame_data in enumerate(frames):
        img = decode_base64_image(frame_data)
        if img is not None and save_training_face(employee_id, img, i):
            saved += 1

    if saved == 0:
        return jsonify({"success": False, "message": "No face detected in any frame. Try better lighting."}), 400

    train_model()

    conn = get_db()
    conn.execute("UPDATE employees SET face_trained = 1 WHERE id = ?", (employee_id,))
    conn.commit()
    conn.close()

    return jsonify({"success": True, "message": f"Enrolled {saved} face samples and retrained model."})


# ---------------------------------------------------------------------------
# Attendance window + cooldown helpers
# ---------------------------------------------------------------------------
def _parse_hhmm(s, fallback):
    try:
        h, m = s.split(":")
        return dtime(int(h), int(m))
    except Exception:
        h, m = fallback.split(":")
        return dtime(int(h), int(m))


def is_within_attendance_window(now=None):
    """Admin-configured daily window (e.g. 08:00–21:00) during which
    check-in/out is allowed at all. Outside this window the kiosk stays in
    scanning mode but silently refuses to register anything."""
    now = now or now_ist()
    start = _parse_hhmm(get_setting("checkin_window_start", "08:00"), "08:00")
    end = _parse_hhmm(get_setting("checkin_window_end", "21:00"), "21:00")
    return start <= now.time() <= end, start, end


def recent_duplicate(conn, employee_id, check_type, cooldown_minutes):
    """Guards against the kiosk's auto-scan loop (every ~2.5s) writing a
    fresh attendance row every time it re-recognizes the same person.

    This used to only check for a recent row of the *same* check_type,
    which meant that the moment a check-in was registered, the very next
    scan tick (2-3 seconds later, since the employee is still standing in
    frame) would see determine_check_type() correctly return 'out' — and
    register it, having no idea 'out' happens to be a different type from
    the row it just wrote a moment ago. In practice that produced
    back-to-back check-in/check-out pairs seconds apart, each independently
    evaluated against the geofence, which is why the kiosk could flash
    "flagged" repeatedly for what was really a single visit.

    Now it blocks ANY new attendance row for this employee within the
    cooldown window, regardless of type, so a check-in and its check-out
    can never land within cooldown_minutes of each other by accident."""
    cutoff = (now_ist() - timedelta(minutes=cooldown_minutes)).strftime("%Y-%m-%d %H:%M:%S")
    row = conn.execute(
        """SELECT id FROM attendance
           WHERE employee_id = ? AND timestamp >= ?
           ORDER BY timestamp DESC, id DESC LIMIT 1""",
        (employee_id, cutoff)
    ).fetchone()
    return row is not None


def _parse_ts(ts):
    return datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")


def determine_check_type(conn, employee_id):
    """
    Smart toggle, not a one-shot-per-day gate: the first scan of the day is
    IN, and every scan after that alternates OUT/IN — so a normal day looks
    like IN (arrival) -> OUT (lunch) -> IN (back) -> OUT (tea break) ->
    IN (back) -> OUT (leaving), all logged as individual rows in
    `attendance` for payroll/reporting to pair up later (see salary.py).
    There's no cap on how many pairs happen in a day; what stops the
    every-few-seconds auto-scan loop from spamming rows is recent_duplicate()
    below (the anti-passback cooldown), not a type restriction here.

    Returns (check_type, None) if a scan should be registered, or
    (None, reason) if it should be silently/politely refused.
    """
    today = today_ist_str()
    last_row = conn.execute(
        """SELECT check_type FROM attendance
           WHERE employee_id = ? AND date(timestamp) = ?
           ORDER BY timestamp DESC, id DESC LIMIT 1""",
        (employee_id, today)
    ).fetchone()

    if last_row is None:
        return "in", None
    return ("out" if last_row["check_type"] == "in" else "in"), None


def check_gps_accuracy(accuracy):
    """Rejects check-ins where the browser couldn't get a precise GPS fix
    (permission denied, Wi-Fi/IP-only location, indoors with no signal,
    GPS turned off, etc.) rather than trusting a low-confidence position
    against the geofence — this is what stops a coarse IP/network-based
    location (which can be tens of kilometers off, e.g. resolving to the
    ISP's city instead of the actual site) from ever being treated as
    equivalent to a real GPS fix. Threshold is admin-tunable via Settings
    (default 100m, matching what the frontend also enforces client-side
    before it will even let a scan through — this is the server-side
    backstop for that, since client-side checks alone can be bypassed).
    """
    max_accuracy = float(get_setting("gps_max_accuracy_meters", "100"))
    if accuracy is None:
        return False, max_accuracy
    try:
        return float(accuracy) <= max_accuracy, max_accuracy
    except (TypeError, ValueError):
        return False, max_accuracy


def _register_attendance(conn, employee, site, img, check_type, lat, lon, confidence):
    """Shared insert logic used by both the manual and auto-scan check-in paths."""
    within, distance = is_within_geofence(
        float(lat), float(lon), site["latitude"], site["longitude"], site["radius_meters"]
    )
    status = "verified" if within else "flagged"

    ts = now_ist_str()
    fname = f"{employee['id']}_{int(now_ist().timestamp())}.jpg"
    fpath = os.path.join(ATTENDANCE_PHOTO_DIR, fname)
    import cv2
    cv2.imwrite(fpath, img)

    conn.execute("""
        INSERT INTO attendance
        (employee_id, site_id, check_type, timestamp, latitude, longitude, distance_from_site,
         within_geofence, match_confidence, photo_path, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        employee["id"], site["id"], check_type, ts, float(lat), float(lon), distance,
        1 if within else 0, confidence, fname, status
    ))
    conn.commit()

    message = f"Welcome {employee['name']}! Attendance marked ({check_type})."
    if not within:
        message += f" You're {int(distance)}m from site (outside the {site['radius_meters']}m zone) — flagged for review."

    return {
        "success": True,
        "message": message,
        "employee_name": employee["name"],
        "emp_code": employee["emp_code"],
        "within_geofence": within,
        "distance": round(distance, 1),
        "status": status,
    }


# ---------------------------------------------------------------------------
# Check-in (this is the page employees use, no login required so it works
# directly from a shared site tablet / employee's own phone browser)
# ---------------------------------------------------------------------------
@app.route("/checkin")
def checkin_page():
    conn = get_db()
    all_sites = conn.execute("SELECT * FROM sites ORDER BY name").fetchall()
    conn.close()
    win_start = get_setting("checkin_window_start", "08:00")
    win_end = get_setting("checkin_window_end", "21:00")
    cooldown = get_setting("cooldown_minutes", "2")
    return render_template("checkin.html", sites=all_sites, win_start=win_start, win_end=win_end, cooldown=cooldown)


@app.route("/api/checkin", methods=["POST"])
def api_checkin():
    """Manual 'Scan Now' fallback endpoint (kept for cases where auto-scan
    can't run, e.g. very old browsers, or when someone wants to trigger a
    verification deliberately rather than waiting for the passive loop).

    Because this is a one-off action rather than the kiosk's continuous
    scan loop (which naturally accumulates several seconds of motion
    history to check against), the frontend prompts the person to blink or
    turn their head slightly and sends TWO frames — liveness_check_frame
    (captured first) and image (captured ~1s later, after the prompt).
    check_prompted_liveness() confirms something physically changed
    between them before recognition even runs, which a printed photo or a
    phone screen held up to the camera can't fake on cue."""
    data = request.get_json()
    image_data = data.get("image")
    liveness_frame_data = data.get("liveness_check_frame")
    site_id = data.get("site_id")
    lat = data.get("latitude")
    lon = data.get("longitude")
    accuracy = data.get("accuracy")

    if not (image_data and liveness_frame_data and site_id and lat is not None and lon is not None):
        return jsonify({"success": False, "message": "Missing photo, liveness frame, site, or GPS location."}), 400

    within_window, start, end = is_within_attendance_window()
    if not within_window:
        return jsonify({
            "success": False,
            "message": f"Attendance can only be marked between {start.strftime('%I:%M %p')} and {end.strftime('%I:%M %p')}."
        }), 200

    conn = get_db()
    site = conn.execute("SELECT * FROM sites WHERE id = ?", (site_id,)).fetchone()
    if site is None:
        conn.close()
        return jsonify({"success": False, "message": "Invalid site."}), 400

    accurate_enough, max_accuracy = check_gps_accuracy(accuracy)
    if not accurate_enough:
        conn.close()
        return jsonify({
            "success": False,
            "message": f"Precise location is required (need better than \u00b1{int(max_accuracy)}m accuracy). "
                       f"Enable High Accuracy / GPS mode on your device and try again."
        }), 200

    img = decode_base64_image(image_data)
    liveness_img = decode_base64_image(liveness_frame_data)

    live_ok, live_reason = check_prompted_liveness(liveness_img, img)
    if not live_ok:
        conn.close()
        msg = "Please blink or turn your head slightly when prompted, then try again."
        if live_reason == "no_movement_detected":
            msg = "No movement detected between frames \u2014 a live person is required. Please blink or move slightly and try again."
        return jsonify({"success": False, "reason": "liveness_failed", "message": msg}), 200

    employee_id, confidence = recognize_face(img, tracking_key=f"manual_{site_id}")

    if employee_id is None:
        conn.close()
        return jsonify({
            "success": False,
            "message": "Face not recognized. Please make sure you are enrolled and try again in good lighting."
        }), 200

    employee = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
    if employee is None or employee["status"] != "active":
        conn.close()
        msg = "Employee record not found or inactive."
        if employee is not None and employee["status"] == "requested":
            msg = f"{employee['name']}, your request is still awaiting admin approval."
        elif employee is not None and employee["status"] == "terminated":
            msg = f"{employee['name']}, your account has been terminated. Contact your admin."
        return jsonify({"success": False, "message": msg}), 200

    check_type, _ = determine_check_type(conn, employee_id)

    cooldown = float(get_setting("cooldown_minutes", "2"))
    if recent_duplicate(conn, employee_id, check_type, cooldown):
        conn.close()
        return jsonify({
            "success": False,
            "message": f"{employee['name']}, that was just registered. Please wait a moment.",
            "cooldown": True,
        }), 200

    result = _register_attendance(conn, employee, site, img, check_type, lat, lon, confidence)
    conn.close()
    return jsonify(result)


@app.route("/api/checkin_pin", methods=["POST"])
def api_checkin_pin():
    """Multi-factor fallback: verify identity with Employee Code/username +
    PIN instead of (or in addition to — the photo is still captured and
    stored for the audit trail) face matching. This exists for cases where
    face recognition genuinely can't confirm someone (poor lighting, a
    face covering, camera trouble) but the business still needs someone
    checked in — PIN + physical presence (GPS) + a stored photo gives
    100% deterministic identity confirmation, at the cost of relying on
    the PIN being kept private (make sure that trade-off is communicated
    to the client — PIN-only verification is inherently more vulnerable to
    buddy-punching than face match, which is why this is a fallback, not
    the default).

    Still requires the prompted liveness check and GPS accuracy gate —
    those protect against a very different failure mode (someone spoofing
    presence entirely) than what the PIN protects against (confirming
    *who*), so both apply regardless of which identity method is used.
    """
    data = request.get_json()
    identifier = _normalize_username(data.get("emp_code_or_username", ""))
    pin = (data.get("pin") or "").strip()
    image_data = data.get("image")
    liveness_frame_data = data.get("liveness_check_frame")
    site_id = data.get("site_id")
    lat = data.get("latitude")
    lon = data.get("longitude")
    accuracy = data.get("accuracy")

    if not (identifier and pin and image_data and liveness_frame_data and site_id and lat is not None and lon is not None):
        return jsonify({"success": False, "message": "Employee code/username, PIN, photo, and GPS location are all required."}), 400

    within_window, start, end = is_within_attendance_window()
    if not within_window:
        return jsonify({
            "success": False,
            "message": f"Attendance can only be marked between {start.strftime('%I:%M %p')} and {end.strftime('%I:%M %p')}."
        }), 200

    conn = get_db()
    site = conn.execute("SELECT * FROM sites WHERE id = ?", (site_id,)).fetchone()
    if site is None:
        conn.close()
        return jsonify({"success": False, "message": "Invalid site."}), 400

    accurate_enough, max_accuracy = check_gps_accuracy(accuracy)
    if not accurate_enough:
        conn.close()
        return jsonify({
            "success": False,
            "message": f"Precise location is required (need better than \u00b1{int(max_accuracy)}m accuracy). "
                       f"Enable High Accuracy / GPS mode on your device and try again."
        }), 200

    # Accept either their emp_code (e.g. "EMP0007") or their login username.
    employee = conn.execute(
        "SELECT * FROM employees WHERE lower(emp_code) = ? OR username = ?",
        (identifier, identifier),
    ).fetchone()
    if employee is None or not employee["password_hash"] or not check_password_hash(employee["password_hash"], pin):
        conn.close()
        return jsonify({"success": False, "message": "Employee code/username or PIN is incorrect."}), 200
    if employee["status"] != "active":
        conn.close()
        msg = "Employee record not found or inactive."
        if employee["status"] == "requested":
            msg = f"{employee['name']}, your request is still awaiting admin approval."
        elif employee["status"] == "terminated":
            msg = f"{employee['name']}, your account has been terminated. Contact your admin."
        return jsonify({"success": False, "message": msg}), 200

    img = decode_base64_image(image_data)
    liveness_img = decode_base64_image(liveness_frame_data)
    live_ok, live_reason = check_prompted_liveness(liveness_img, img)
    if not live_ok:
        conn.close()
        msg = "Please blink or turn your head slightly when prompted, then try again."
        if live_reason == "no_movement_detected":
            msg = "No movement detected between frames \u2014 a live person is required. Please blink or move slightly and try again."
        return jsonify({"success": False, "reason": "liveness_failed", "message": msg}), 200

    check_type, _ = determine_check_type(conn, employee["id"])
    cooldown = float(get_setting("cooldown_minutes", "2"))
    if recent_duplicate(conn, employee["id"], check_type, cooldown):
        conn.close()
        return jsonify({
            "success": False,
            "message": f"{employee['name']}, that was just registered. Please wait a moment.",
            "cooldown": True,
        }), 200

    # confidence=None marks this row as PIN-verified rather than face-matched,
    # for anyone auditing the attendance log later.
    result = _register_attendance(conn, employee, site, img, check_type, lat, lon, None)
    conn.close()
    return jsonify(result)


@app.route("/api/auto_checkin", methods=["POST"])
def api_auto_checkin():
    """
    Called every couple of seconds by the kiosk page's live scan loop.
    Unlike /api/checkin, this is designed to fail *quietly* most of the
    time (no face in frame yet, face not lined up, nobody currently
    standing there) — the UI only reacts loudly when a real match happens.
    """
    data = request.get_json()
    image_data = data.get("image")
    site_id = data.get("site_id")
    lat = data.get("latitude")
    lon = data.get("longitude")
    accuracy = data.get("accuracy")

    if not (image_data and site_id and lat is not None and lon is not None):
        return jsonify({"success": False, "reason": "missing_input"}), 200

    within_window, start, end = is_within_attendance_window()
    if not within_window:
        return jsonify({
            "success": False,
            "reason": "outside_window",
            "message": f"Attendance window is closed. Allowed {start.strftime('%I:%M %p')}–{end.strftime('%I:%M %p')}."
        }), 200

    conn = get_db()
    site = conn.execute("SELECT * FROM sites WHERE id = ?", (site_id,)).fetchone()
    if site is None:
        conn.close()
        return jsonify({"success": False, "reason": "invalid_site"}), 200

    accurate_enough, max_accuracy = check_gps_accuracy(accuracy)
    if not accurate_enough:
        conn.close()
        return jsonify({
            "success": False,
            "reason": "gps_inaccurate",
            "message": f"GPS signal too weak (need better than \u00b1{int(max_accuracy)}m). Enable precise/high-accuracy location.",
        }), 200

    img = decode_base64_image(image_data)
    quality = detect_face_quality(img)
    if not quality["ok"]:
        conn.close()
        # no_face / eyes_not_visible — normal while someone is still walking
        # into frame, not an error worth showing the employee
        return jsonify({"success": False, "reason": quality["reason"]}), 200

    employee_id, confidence = recognize_face(img, tracking_key=f"kiosk_{site_id}")
    if employee_id is None:
        conn.close()
        return jsonify({"success": False, "reason": "unrecognized",
                         "message": "Face detected but not recognized."}), 200

    employee = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
    if employee is None or employee["status"] != "active":
        conn.close()
        if employee is not None and employee["status"] == "requested":
            return jsonify({
                "success": False, "reason": "pending_approval",
                "message": f"{employee['name']}, your request is still awaiting admin approval.",
            }), 200
        return jsonify({"success": False, "reason": "inactive_employee"}), 200

    check_type, _ = determine_check_type(conn, employee_id)

    cooldown = float(get_setting("cooldown_minutes", "2"))
    if recent_duplicate(conn, employee_id, check_type, cooldown):
        conn.close()
        return jsonify({
            "success": False,
            "reason": "cooldown",
            "employee_name": employee["name"],
            "message": f"{employee['name']}, already marked recently.",
        }), 200

    result = _register_attendance(conn, employee, site, img, check_type, lat, lon, confidence)
    conn.close()
    return jsonify(result)


# ---------------------------------------------------------------------------
# Attendance reports
# ---------------------------------------------------------------------------
@app.route("/attendance")
@login_required
def attendance_report():
    conn = get_db()
    site_filter = request.args.get("site_id", "")
    date_filter = request.args.get("date", "")
    status_filter = request.args.get("status", "")

    query = """
        SELECT a.*, e.name as emp_name, e.emp_code, s.name as site_name
        FROM attendance a
        JOIN employees e ON a.employee_id = e.id
        JOIN sites s ON a.site_id = s.id
        WHERE 1=1
    """
    params = []
    if site_filter:
        query += " AND a.site_id = ?"
        params.append(site_filter)
    if date_filter:
        query += " AND date(a.timestamp) = ?"
        params.append(date_filter)
    if status_filter:
        query += " AND a.status = ?"
        params.append(status_filter)
    query += " ORDER BY a.timestamp DESC LIMIT 500"

    records = conn.execute(query, params).fetchall()
    all_sites = conn.execute("SELECT * FROM sites ORDER BY name").fetchall()
    conn.close()

    return render_template(
        "attendance.html",
        records=records,
        sites=all_sites,
        site_filter=site_filter,
        date_filter=date_filter,
        status_filter=status_filter,
    )


@app.route("/attendance/export")
@login_required
def export_attendance():
    conn = get_db()
    records = conn.execute("""
        SELECT a.timestamp, e.emp_code, e.name as emp_name, s.name as site_name,
               a.check_type, a.latitude, a.longitude, a.distance_from_site,
               a.within_geofence, a.status
        FROM attendance a
        JOIN employees e ON a.employee_id = e.id
        JOIN sites s ON a.site_id = s.id
        ORDER BY a.timestamp DESC
    """).fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Timestamp", "Employee Code", "Employee Name", "Site", "Type",
                      "Latitude", "Longitude", "Distance from Site (m)", "Within Geofence", "Status"])
    for r in records:
        writer.writerow([r["timestamp"], r["emp_code"], r["emp_name"], r["site_name"],
                          r["check_type"], r["latitude"], r["longitude"],
                          round(r["distance_from_site"], 1) if r["distance_from_site"] is not None else "",
                          "Yes" if r["within_geofence"] else "No", r["status"]])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=attendance_export.csv"}
    )


@app.route("/attendance_photos/<path:filename>")
@login_required
def attendance_photo(filename):
    return send_from_directory(ATTENDANCE_PHOTO_DIR, filename)


# ---------------------------------------------------------------------------
# Salary
# ---------------------------------------------------------------------------
@app.route("/salary")
@login_required
def salary_page():
    today = now_ist().date()
    default_start = today.replace(day=1).isoformat()
    default_end = today.isoformat()
    start_date = request.args.get("start", default_start)
    end_date = request.args.get("end", default_end)

    conn = get_db()
    results = salary_mod.compute_salary_all(conn, start_date, end_date)
    conn.close()

    return render_template("salary.html", results=results, start_date=start_date, end_date=end_date)


@app.route("/employees/<int:employee_id>/salary")
@login_required
def employee_salary(employee_id):
    today = now_ist().date()
    default_start = today.replace(day=1).isoformat()
    default_end = today.isoformat()
    start_date = request.args.get("start", default_start)
    end_date = request.args.get("end", default_end)

    conn = get_db()
    result = salary_mod.compute_salary(conn, employee_id, start_date, end_date)
    conn.close()

    if result is None:
        flash("Employee not found.", "error")
        return redirect(url_for("salary_page"))

    return render_template("employee_salary.html", result=result)


# ---------------------------------------------------------------------------
# Settings — attendance window, auto-checkin cooldown, admin password
# ---------------------------------------------------------------------------
@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings_page():
    if request.method == "POST":
        form_type = request.form.get("form_type")

        if form_type == "window":
            start = request.form.get("checkin_window_start", "08:00")
            end = request.form.get("checkin_window_end", "21:00")
            cooldown = request.form.get("cooldown_minutes", "2")
            gps_accuracy = request.form.get("gps_max_accuracy_meters", "100")
            set_setting("checkin_window_start", start)
            set_setting("checkin_window_end", end)
            set_setting("cooldown_minutes", str(float(cooldown)))
            set_setting("gps_max_accuracy_meters", str(float(gps_accuracy)))
            flash("Attendance window, cooldown & GPS accuracy updated.", "success")

        elif form_type == "password":
            current_pin = request.form.get("current_password", "")
            new_pin = request.form.get("new_password", "").strip()
            conn = get_db()
            user = conn.execute("SELECT * FROM admin_users WHERE id = ?", (session["admin_id"],)).fetchone()
            if not user or not check_password_hash(user["password_hash"], current_pin):
                flash("Current PIN is incorrect.", "error")
            elif not (new_pin.isdigit() and len(new_pin) == 6):
                flash("New PIN must be exactly 6 digits.", "error")
            else:
                conn.execute(
                    "UPDATE admin_users SET password_hash = ? WHERE id = ?",
                    (generate_password_hash(new_pin), session["admin_id"])
                )
                conn.commit()
                flash("PIN updated.", "success")
            conn.close()

        return redirect(url_for("settings_page"))

    win_start = get_setting("checkin_window_start", "08:00")
    win_end = get_setting("checkin_window_end", "21:00")
    cooldown = get_setting("cooldown_minutes", "2")
    gps_accuracy = get_setting("gps_max_accuracy_meters", "100")
    return render_template(
        "settings.html", win_start=win_start, win_end=win_end, cooldown=cooldown,
        gps_accuracy=gps_accuracy,
        antispoof_available=_anti_spoofing_available(),
    )


# ---------------------------------------------------------------------------
# PWA support files
# ---------------------------------------------------------------------------
@app.route("/sw.js")
def service_worker():
    """Served from the root path (not /static/) so its scope covers the
    whole app, not just /static/ — required for install-to-home-screen."""
    response = send_from_directory(os.path.join(BASE_DIR, "static"), "sw.js")
    response.headers["Service-Worker-Allowed"] = "/"
    response.headers["Cache-Control"] = "no-cache"
    return response


if __name__ == "__main__":
    app.run(debug=DEBUG_MODE, host="0.0.0.0", port=5000)
