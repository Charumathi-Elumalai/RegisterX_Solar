"""
RegistraX Solar — database layer.
Plain SQLite. No ORM, kept intentionally simple so it's easy to inspect,
back up (it's a single file), and migrate later to Postgres/MySQL if the
deployment ever needs it.
"""
import os
import sqlite3

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "registrax_solar.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _ensure_column(conn, table, col_name, col_def_sql):
    """Adds a column to an existing table if it isn't there yet. SQLite's
    ALTER TABLE ADD COLUMN doesn't support UNIQUE constraints, so uniqueness
    for email columns is enforced separately via a partial unique index."""
    existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if col_name not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def_sql}")


def _relax_employees_site_id(conn):
    """One-time rebuild for pre-existing databases created before employee
    signup was split into (1) phone+OTP+PIN then (2) choose site as part of
    completing their profile. Those older DBs still have site_id declared
    NOT NULL, which would reject step (1). SQLite has no ALTER COLUMN, so
    the table is rebuilt with the same data, just without that constraint."""
    row = conn.execute("PRAGMA table_info(employees)").fetchall()
    if not row:
        return  # table doesn't exist yet — CREATE TABLE below handles it
    site_col = next((r for r in row if r["name"] == "site_id"), None)
    if not site_col or not site_col["notnull"]:
        return  # already nullable (fresh installs), nothing to do

    col_names = [r["name"] for r in row]
    conn.execute("ALTER TABLE employees RENAME TO employees_old_migrating")
    conn.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emp_code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            phone TEXT,
            role TEXT,
            site_id INTEGER,
            active INTEGER NOT NULL DEFAULT 1,
            face_trained INTEGER NOT NULL DEFAULT 0,
            address TEXT,
            date_of_birth TEXT,
            username TEXT UNIQUE,
            password_hash TEXT,
            must_change_password INTEGER NOT NULL DEFAULT 1,
            pay_type TEXT NOT NULL DEFAULT 'daily',
            pay_rate REAL NOT NULL DEFAULT 0,
            expected_hours_per_day REAL NOT NULL DEFAULT 8,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (site_id) REFERENCES sites(id)
        )
    """)
    cols_sql = ", ".join(col_names)
    conn.execute(f"INSERT INTO employees ({cols_sql}) SELECT {cols_sql} FROM employees_old_migrating")
    conn.execute("DROP TABLE employees_old_migrating")
    conn.commit()


def init_db():
    conn = get_db()
    c = conn.cursor()
    _relax_employees_site_id(conn)

    c.execute("""
        CREATE TABLE IF NOT EXISTS admin_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS sites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            address TEXT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            radius_meters INTEGER NOT NULL DEFAULT 200,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emp_code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            phone TEXT,
            role TEXT,                      -- job title/role at the site (e.g. "Mason", "Supervisor")
            site_id INTEGER,                -- nullable: not chosen yet until profile-completion step
            active INTEGER NOT NULL DEFAULT 1,
            face_trained INTEGER NOT NULL DEFAULT 0,
            address TEXT,
            date_of_birth TEXT,

            -- employee self-service portal login (separate from admin_users)
            username TEXT UNIQUE,
            password_hash TEXT,
            must_change_password INTEGER NOT NULL DEFAULT 1,

            -- salary configuration
            pay_type TEXT NOT NULL DEFAULT 'daily',   -- 'daily' | 'monthly' | 'hourly'
            pay_rate REAL NOT NULL DEFAULT 0,
            expected_hours_per_day REAL NOT NULL DEFAULT 8,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (site_id) REFERENCES sites(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            sample_index INTEGER NOT NULL DEFAULT 0,
            encoding TEXT NOT NULL,          -- JSON list of 512 floats (ArcFace embedding)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            site_id INTEGER NOT NULL,
            check_type TEXT NOT NULL,        -- 'in' | 'out'
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            latitude REAL,
            longitude REAL,
            distance_from_site REAL,
            within_geofence INTEGER,
            match_confidence REAL,
            photo_path TEXT,
            status TEXT NOT NULL DEFAULT 'verified',   -- 'verified' | 'flagged'
            FOREIGN KEY (employee_id) REFERENCES employees(id),
            FOREIGN KEY (site_id) REFERENCES sites(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    conn.commit()

    # ------------------------------------------------------------------
    # Migration: mobile number + PIN based signup/login for both admins
    # and employees (replaces the old username/password-only accounts,
    # and later replaced email OTP verification with SMS OTP — see
    # sms_utils.py). `password_hash` is reused to store the hashed 6-digit
    # PIN — same column, same check_password_hash() call, just different
    # contents — so no data migration is needed for that part.
    # `email` is kept as an optional profile field (collected as part of
    # an employee's personal details) but is no longer used for OTP
    # verification or login for anyone.
    # ------------------------------------------------------------------
    _ensure_column(conn, "admin_users", "email", "email TEXT")
    _ensure_column(conn, "admin_users", "email_verified", "email_verified INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "admin_users", "phone", "phone TEXT")
    _ensure_column(conn, "admin_users", "phone_verified", "phone_verified INTEGER NOT NULL DEFAULT 0")

    _ensure_column(conn, "employees", "email", "email TEXT")
    _ensure_column(conn, "employees", "email_verified", "email_verified INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "employees", "phone_verified", "phone_verified INTEGER NOT NULL DEFAULT 0")
    # Employee lifecycle, self-signup path:
    #   'incomplete' -> phone verified + PIN set, but personal/professional
    #                   details and face enrollment aren't done yet. Not
    #                   shown to admin at all (there's no "request" yet).
    #   'requested'  -> details + face enrollment done; waiting on admin
    #                   approval. This is what shows up on the admin's
    #                   Employees page as a "Request".
    #   'active'     -> approved by admin, can check in.
    #   'terminated' -> declined or terminated by admin; permanently
    #                   blocked from checking in / logging in until an
    #                   admin explicitly re-activates them.
    # Employees added directly by an admin (no self-signup) start 'active'.
    _ensure_column(conn, "employees", "status", "status TEXT NOT NULL DEFAULT 'active'")
    _ensure_column(conn, "employees", "profile_completed", "profile_completed INTEGER NOT NULL DEFAULT 0")

    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_admin_email ON admin_users(email) WHERE email IS NOT NULL")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_employee_email ON employees(email) WHERE email IS NOT NULL")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_admin_phone ON admin_users(phone) WHERE phone IS NOT NULL")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_employee_phone ON employees(phone) WHERE phone IS NOT NULL")
    conn.commit()

    # Data migration for accounts created under the old email-OTP /
    # two-status ('pending'/'rejected') scheme, so existing deployments
    # don't lose their history when this update is installed:
    #   - old 'pending' (self-signed-up, awaiting approval) -> 'requested'
    #     (they'd already finished signup under the old flow, so treat
    #     their details as complete rather than dropping them back into
    #     the profile/face-enrollment steps)
    #   - old 'rejected' -> 'terminated'
    #   - anyone previously deactivated (active=0) while still marked
    #     'active' -> 'terminated', so they now show correctly under
    #     Terminated instead of silently vanishing from every list
    conn.execute("UPDATE employees SET status = 'requested', profile_completed = 1 WHERE status = 'pending'")
    conn.execute("UPDATE employees SET status = 'terminated' WHERE status = 'rejected'")
    conn.execute("UPDATE employees SET status = 'terminated' WHERE active = 0 AND status = 'active'")
    conn.execute("UPDATE employees SET profile_completed = 1 WHERE status IN ('requested', 'active', 'terminated')")
    conn.commit()

    # No hardcoded/seeded admin account any more — the very first admin
    # account is created by signing up at /register (name + mobile OTP
    # verification + 6-digit PIN), same flow every admin after them uses.

    conn.close()


def get_setting(key, default=None):
    conn = get_db()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def set_setting(key, value):
    conn = get_db()
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()
    conn.close()
