"""
RegistraX Solar — outgoing SMS for signup verification codes.

Signup/login now verifies a mobile number instead of an email address (see
app.py). This module is intentionally built the same way email_utils.py
was: if no real SMS gateway is configured, it prints the OTP to the server
terminal so signup still works out of the box for local testing — then
switching to a real SMS gateway in production is a config change here, not
a change anywhere else in the app.

To wire up a real gateway later, set these environment variables and fill
in send_sms() below with that provider's API call:
    SMS_GATEWAY_URL   — the provider's send-SMS API endpoint
    SMS_API_KEY       — the provider's API key / auth token
    SMS_SENDER_ID     — the "from" name/number shown to the recipient

Common providers (pick one when you're ready, e.g. Twilio, MSG91,
TextLocal, Fast2SMS) all follow the same shape: a POST request with the
phone number, message text, and an API key. Once you have an account,
this is a small, contained edit — nothing in app.py needs to change.
"""
import os
import logging

logger = logging.getLogger("registrax.sms")


def send_sms(phone, message):
    gateway_url = os.environ.get("SMS_GATEWAY_URL")

    if not gateway_url:
        print("\n" + "=" * 60)
        print(f"[DEV SMS \u2014 SMS_GATEWAY_URL not set, printing instead] To: {phone}")
        print(message)
        print("=" * 60 + "\n")
        return True

    try:
        import requests
        api_key = os.environ.get("SMS_API_KEY", "")
        sender_id = os.environ.get("SMS_SENDER_ID", "REGXSL")
        resp = requests.post(
            gateway_url,
            json={"to": phone, "message": message, "sender_id": sender_id},
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except Exception:
        logger.exception("Failed to send SMS to %s", phone)
        # Fall back to console so signup doesn't hard-fail if the gateway is briefly down
        print(f"\n[SMS SEND FAILED \u2014 fallback print] To: {phone}\n{message}\n")
        return False


def send_otp_sms(phone, otp, purpose="verify your RegistraX Solar account"):
    message = (
        f"RegistraX Solar: your OTP is {otp}. "
        f"Use this to {purpose}. Valid for 10 minutes. Do not share this code with anyone."
    )
    return send_sms(phone, message)
