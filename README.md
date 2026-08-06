# RegistraX Solar — Face + GPS Attendance & Salary System

கையொப்பம் (RegistraX Solar) — Tamil for "signature / mark of presence."

> **This build**: signup/login now verifies a **mobile number** instead of
> email (SMS OTP); the admin Employees page has a full **Requests / Active
> / Terminated** workflow; employees now complete their own profile and
> face enrollment as part of signup before their request reaches the
> admin; a cooldown bug that could register a spurious check-out seconds
> after check-in (and show repeated "flagged" toasts) is fixed; and the
> check-in page now reverse-geocodes GPS into a readable address. See the
> notes below for details on each.
>
> **Latest round**: phone numbers are now *only* used for the one-time
> signup OTP — every login (admin and employee) uses a chosen **username +
> PIN** instead. Attendance is no longer capped at one check-in/check-out
> per day: each scan now alternates IN/OUT so lunch and tea breaks are
> logged correctly, gated by a 2-minute anti-passback cooldown. GPS now
> strictly enforces `enableHighAccuracy` + no cached fixes, and blocks the
> scan client-side with an alert if accuracy is worse than 100m (the
> "shows Chennai instead of Vellore" IP-fallback problem). Face matching
> is stricter (tightened similarity threshold and twin-separation margin
> in `face_engine.py`), a manual "Verify now" button and an "Employee Code
> + PIN" fallback now both require a prompted blink/head-turn between two
> captured frames before accepting the scan, and the camera preview is
> now mirrored (cosmetic only — the captured/stored photo is unaffected).

A Flask web app, installable as a PWA (Progressive Web App) on Android and
iPhone, for site-based teams: employees check in/out with live face
recognition + GPS geofencing, admins manage sites/staff/attendance, and
salary is calculated automatically from attendance records.

---

## 1. What this actually does (read this before demoing to your client)

- **Face recognition**: MTCNN face detection + ArcFace 512-dimension face
  embeddings (both real, published, widely-used models — via the `deepface`
  library). Not a toy photo-comparison.
- **Look-alike protection**: when a face is scanned, the system requires the
  best match to clearly beat the second-best match by a margin
  (`MATCH_MARGIN` in `face_engine.py`). If two enrolled people score too
  close together — which is what happens with siblings/close relatives —
  it **refuses to guess** and asks the person to try again, rather than
  silently marking the wrong person present. This is the honest, real-world
  answer to "family members look alike": no face system on the market can
  promise 100% separation between very close look-alikes in all conditions,
  but this refuses-to-guess behavior is the correct mitigation.
- **Liveness / anti-spoofing**: a duplicate-frame + zero-motion heuristic
  runs on every deployment. If `torch` is installed (see requirements.txt),
  it also runs a real trained anti-spoofing model (Silent-Face-Anti-Spoofing,
  MiniFASNet) that looks at a single frame for signs it's a printed photo or
  a phone/screen replay. If `torch` isn't installed, the app logs a warning
  and runs on the lighter heuristic only — it never silently pretends to
  have full liveness checking.
- **GPS**: real haversine great-circle distance between the employee's phone
  GPS and the site's registered coordinates, checked against the site's
  geofence radius. Outside the radius = attendance is still recorded but
  flagged for admin review.
- **Two roles**: Admin (full dashboard, manage sites/staff, reports, salary,
  settings) and Employee (self-service portal: own attendance history).
  Salary is admin-only.
- **Sign up / login**: both roles sign up with their real mobile number —
  get a 6-digit OTP by SMS, verify it, then set a 6-digit PIN. That
  mobile number + PIN is the login from then on (no usernames/passwords,
  no email required). By default OTP codes are printed to the server
  terminal (fine for local testing); set `SMS_GATEWAY_URL` / `SMS_API_KEY`
  / `SMS_SENDER_ID` env vars in production to send real SMS — see
  `sms_utils.py`. Email is now an optional profile field only (not used
  for login or verification).
- **Employee self-signup flow**: phone + OTP + PIN → **complete profile**
  (personal details + job role + site) → **enroll their own face** → their
  request is automatically sent to the admin. On the admin's Employees
  page this shows up under **Requests**, with the site they asked for and
  whether their face is enrolled, ready to Approve or Decline. Employees
  who signed themselves up can't check in until an admin approves them —
  that also means once an admin **Terminates** someone, they're
  permanently blocked from checking in *and logging in* until an admin
  explicitly **Reactivates** them; they can't just sign up again. Admins
  can also add employees directly (no approval step needed in that case) —
  see the "Add an employee" form on the Employees page.
- **Attendance rules**: the kiosk has no manual Check-In/Check-Out toggle —
  an employee's first scan of the day is automatically the check-in, and
  their next scan is the check-out (only one of each per day). Checkout is
  only accepted within 12 hours of that day's check-in; a scan outside
  that needs an admin to sort out rather than silently registering an
  open-ended shift. The auto-scan cooldown (Settings → "Cooldown minutes")
  also gates *any* two scans for the same employee, not just repeats of
  the same type — this stops the kiosk from registering a check-out just
  a couple of seconds after a check-in while someone is still standing in
  frame, which used to produce spurious back-to-back "flagged" entries.
  All timestamps are recorded and displayed in Indian Standard Time
  regardless of the server's own clock/timezone.
- **GPS**: real haversine great-circle distance between the employee's
  phone GPS and the site's registered coordinates, checked against the
  site's geofence radius. Outside the radius = attendance is still
  recorded but flagged for admin review. Location permission is requested
  automatically as soon as the check-in page loads; once GPS locks, the
  raw coordinates are also reverse-geocoded (via OpenStreetMap's free
  Nominatim API) into a readable address, shown under the GPS status pill
  — similar to a dropped pin on Google Maps — so the employee can see at
  a glance that the location actually looks right. GPS failures now show
  a specific reason (permission denied / signal too weak / device GPS
  off) instead of a generic error.
- **Salary**: pairs each day's check-in/check-out into worked hours, then
  applies daily / hourly / monthly pay type. This is a clear, inspectable
  starting formula — real payroll rules (paid leave, holidays, overtime,
  PF/ESI deductions) vary business to business. Treat `salary.py` as the
  place to customize once you know the client's exact rules.

**What this is not**: a native Play Store / App Store app. It's a PWA — see
section 4. Also, no camera-based system, from any vendor, can promise
perfect accuracy against identical twins or in very poor lighting; set that
expectation with your client up front.

---

## 2. Local setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

First run creates `registrax_solar.db` (SQLite — single file, easy to back up).
There's no seeded admin account any more — open the app and go to **Sign
up** to create the first admin (name + mobile OTP + PIN), same flow every
admin after them uses.

Open `http://localhost:5000` for the admin panel, or
`http://localhost:5000/checkin` for the kiosk check-in screen.

> First install: `pip install -r requirements.txt` will download the
> ArcFace model weights (~140MB) and the anti-spoofing weights (~2MB)
> automatically from their public GitHub releases the first time face
> recognition runs — this needs a normal internet connection once, then
> works offline.

---

## 3. Deploying for real (cloud server)

Any small Linux VPS works (DigitalOcean, Hetzner, AWS Lightsail, Oracle
Cloud Free Tier). Rough spec: 2 vCPU / 2GB+ RAM (face matching is CPU-bound
for 50 employees, this is comfortable).

```bash
# on the server
git clone <your repo>   # or upload the zip and unzip
cd registrax_solar
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# quick test
python app.py

# production (keeps running, restarts on crash — use with a process
# manager like systemd or pm2, and a reverse proxy like nginx for HTTPS)
gunicorn -w 2 -b 0.0.0.0:5000 --timeout 120 app:app
```

Put nginx (or Caddy, simpler for free auto-HTTPS) in front for a real
domain + SSL certificate — camera and GPS access in the browser **require
HTTPS** on real phones (localhost is exempt, production is not).

Set a real `SECRET_KEY` environment variable in production instead of the
auto-generated one (which resets sessions on every restart):
```bash
export SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
```

---

## 4. Installing it as an app on employees' phones (PWA)

No Play Store / App Store listing needed:

1. Employee opens the site's URL (e.g. `https://yourdomain.com/checkin`) in
   Chrome (Android) or Safari (iPhone).
2. **Android/Chrome**: menu (⋮) → "Add to Home screen" / "Install app".
   **iPhone/Safari**: Share icon → "Add to Home Screen".
3. It now behaves like a normal app: own icon, opens full-screen, no
   browser address bar, camera and GPS work exactly as they would in a
   native app.

This is faster, cheaper, and works on both platforms from one codebase.
A true native app (listed on Play Store/App Store) is a separate, larger
project — different tooling (React Native/Flutter), developer accounts,
and app review — worth doing later once the business is validated on this.

---

## 5. Project structure

```
app.py              — routes/controllers
database.py          — SQLite schema + connection helpers
face_engine.py        — face detection, matching, liveness
geo_utils.py          — GPS geofence math
sms_utils.py           — outgoing SMS for signup OTP (pluggable gateway)
salary.py             — attendance → pay calculation
templates/             — all HTML pages (Jinja2)
static/css/style.css   — design system
static/js/kiosk.js      — check-in camera + GPS + live scan loop
static/js/enroll.js     — guided multi-angle face enrollment
static/manifest.json    — PWA manifest
static/sw.js            — service worker (installability + basic offline shell)
attendance_photos/       — saved photo evidence per check-in/out
registrax_solar.db              — SQLite database (created on first run)
```

---

## 6. Sensible next steps (not built yet, flagging honestly)

- **Payroll rules refinement**: paid leave, holidays, overtime multipliers —
  once you know the client's exact policy.
- **Multi-admin roles** (e.g. site supervisor who can only see their own
  site) — currently all admins see everything.
- **SMS/WhatsApp notification** on flagged (outside-geofence) attendance.
- **Native app** — if the client later wants Play Store/App Store presence.
