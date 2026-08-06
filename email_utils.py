"""
RegistraX Solar — outgoing email for signup verification codes.

Reads SMTP settings from the environment so the real client deployment can
send real emails:
    SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASS, SMTP_FROM

If SMTP_HOST isn't set (local/dev), falls back to printing the email to the
terminal so signup still works without any mail account configured — same
spirit as the old "OTP printed to terminal" demo behaviour, just wired
through one place so switching to real email in production is a one-line
env var change, not a code change.
"""
import os
import smtplib
import ssl
import logging
from email.mime.text import MIMEText

logger = logging.getLogger("registrax.email")


def send_email(to_email, subject, body):
    host = os.environ.get("SMTP_HOST")

    if not host:
        print("\n" + "=" * 60)
        print(f"[DEV EMAIL — SMTP_HOST not set, printing instead] To: {to_email}")
        print(f"Subject: {subject}")
        print(body)
        print("=" * 60 + "\n")
        return True

    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    pwd = os.environ.get("SMTP_PASS")
    from_addr = os.environ.get("SMTP_FROM", user or "no-reply@registraxsolar.local")

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_email

    try:
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls(context=ssl.create_default_context())
            if user and pwd:
                server.login(user, pwd)
            server.sendmail(from_addr, [to_email], msg.as_string())
        return True
    except Exception:
        logger.exception("Failed to send email to %s", to_email)
        # Fall back to console so signup doesn't hard-fail if SMTP is briefly down
        print(f"\n[EMAIL SEND FAILED — fallback print] To: {to_email}\nSubject: {subject}\n{body}\n")
        return False


def send_otp_email(to_email, otp, purpose="verify your RegistraX Solar account"):
    subject = "Your RegistraX Solar verification code"
    body = (
        f"Your one-time verification code is: {otp}\n\n"
        f"Use this to {purpose}. This code expires in 10 minutes.\n\n"
        "If you didn't request this, you can safely ignore this email."
    )
    return send_email(to_email, subject, body)
