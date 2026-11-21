"""Shared theme, constants and helpers for the OncoSense Streamlit app.

Split across two pages (classic Streamlit `pages/` convention):
  app.py            -- public landing/marketing page
  pages/Screening.py -- gated login + upload/analyze/result/explanation/report workspace

Both pages call `st.set_page_config(...)` themselves (must be the first
Streamlit command in each script) and only THEN import from this module,
since importing triggers a few `st.cache_data`-decorated calls at module
load time (the decorative Grad-CAM image loader below).
"""
import base64
import csv
import datetime
import hashlib
import hmac
import io
import json
import os
import secrets
import time
import uuid
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as _st_components
from PIL import Image

CONFIG_PATH = "configs/cbis_ddsm_v3.yaml"

_ROOT = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no python-dotenv dependency) -- sets any
    KEY=value pairs found into os.environ, without overwriting a variable
    already set in the real environment. Silently does nothing if the file
    doesn't exist; this keeps local secrets (e.g. GEMINI_API_KEY) out of
    source control while still reaching the process on every run/restart."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(_ROOT / ".env")
HISTORY_PATH = _ROOT / "outputs" / "screening_history.csv"
REPORTS_DIR = _ROOT / "outputs" / "reports"
IMAGES_DIR = _ROOT / "outputs" / "images"
HISTORY_COLUMNS = [
    "scan_id", "timestamp", "doctor_name", "patient_name", "patient_id", "age", "breast_side", "view",
    "family_history", "previous_cancer", "previous_biopsy", "previous_mammogram",
    "previous_abnormal_mammogram", "menopausal_status", "breast_symptoms",
    "filename", "predicted_class", "finding", "confidence", "malignant_prob", "report_path",
    "image_path", "gradcam_path", "clinical_t", "clinical_n", "clinical_m", "estimated_stage",
]


def _migrate_history_header() -> None:
    """Rewrites outputs/screening_history.csv in place if its on-disk header
    is missing columns that HISTORY_COLUMNS has gained since those rows were
    written (e.g. image_path/gradcam_path, added later). Old rows get "" for
    the new columns -- without this, appending a row with more fields than
    the existing header would silently misalign every column.
    """
    if not HISTORY_PATH.exists():
        return
    with open(HISTORY_PATH, newline="") as f:
        reader = csv.DictReader(f)
        existing_fieldnames = reader.fieldnames or []
        rows = list(reader)
    if existing_fieldnames == HISTORY_COLUMNS:
        return
    with open(HISTORY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for row in rows:
            row.pop(None, None)  # DictReader's catch-all for unexpected extra fields
            writer.writerow({col: row.get(col, "") for col in HISTORY_COLUMNS})


def log_screening_history(row: dict) -> str:
    """Appends one completed analysis (incl. the saved report's path) to
    outputs/screening_history.csv and returns its generated scan_id.

    This is the only persistence layer in the app (no database) -- a plain
    CSV that accumulates across sessions, so the patient history survives
    restarts instead of resetting every session.
    """
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _migrate_history_header()
    scan_id = uuid.uuid4().hex[:12]
    is_new = not HISTORY_PATH.exists()
    with open(HISTORY_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({**row, "scan_id": scan_id})
    return scan_id


def read_screening_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    with open(HISTORY_PATH, newline="") as f:
        return list(csv.DictReader(f))


def delete_screening_history_row(index: int) -> None:
    """Deletes one row -- by its position in read_screening_history()'s
    (chronological) list -- from the CSV, and removes its associated
    report/image/Grad-CAM files from disk. Used by the per-record "Delete"
    button on the History page, as opposed to "Clear all history" which
    wipes everything. Index-based (rather than keying off scan_id) so it
    still works for legacy rows logged before scan_id existed."""
    rows = read_screening_history()
    if not (0 <= index < len(rows)):
        return
    dropped = rows.pop(index)
    for col in ("report_path", "image_path", "gradcam_path"):
        rel = (dropped.get(col) or "").strip()
        if not rel:
            continue
        fpath = HISTORY_PATH.parent / rel
        if fpath.exists():
            fpath.unlink()
    with open(HISTORY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({col: r.get(col, "") for col in HISTORY_COLUMNS})


# ---------------------------------------------------------------------------
# Appointment booking (pages/Patient_Dashboard.py's Appointments tab).
#
# This is a local scheduling record, not a live external provider search or
# clinic scheduling system -- "booking" here means recording the appointment
# in this app (for the patient's own tracking and for generating a calendar
# reminder); it does not notify or confirm anything with the clinic. The
# specialist a patient is seeing is free-text entered at booking time, not a
# managed, user-editable directory -- SPECIALIST_OPTIONS below is just a
# static reference list (a few real, web-sourced names -- see the Education
# Center -- plus fictional "(demo)" entries for testing, plus "Other") so
# picking one is a dropdown instead of always typing from scratch; it
# carries no login/account of its own.

SPECIALIST_OPTIONS = [
    ("Dr. Sudeep Gupta", "Medical Oncologist", "Tata Memorial Centre, Mumbai"),
    ("Dr. Bhawna Sirohi", "Medical Oncologist", "BALCO Medical Centre, Nava Raipur"),
    ("Dr. Kamran Khan", "Breast Surgeon", "Tata Memorial Hospital, Mumbai"),
    ("Dr. Raj Nagarkar", "Breast Surgeon", "HCG Manavata Cancer Centre, Nashik"),
    ("Dr. Ananya Sharma", "Breast Surgeon", "Lotus Health Clinic, Mumbai (demo)"),
    ("Dr. Rohan Verma", "Medical Oncologist", "Sunrise Oncology Center, Delhi (demo)"),
    ("Dr. Priya Nair", "Radiologist", "Horizon Diagnostics, Bangalore (demo)"),
    ("Dr. Vikram Singh", "Breast Surgeon", "Wellness Cancer Institute, Chennai (demo)"),
    ("Dr. Meera Iyer", "Medical Oncologist", "Hope Cancer Care, Pune (demo)"),
    ("Dr. Arjun Patel", "Radiologist", "Clarity Imaging Center, Ahmedabad (demo)"),
    ("Dr. Kavita Rao", "Breast Specialist / OB-GYN", "Sanjeevani Women's Hospital, Hyderabad (demo)"),
    ("Dr. Suresh Menon", "Medical Oncologist", "CareWell Oncology, Kochi (demo)"),
    ("Dr. Divya Kapoor", "Breast Surgeon", "Life Line Cancer Center, Jaipur (demo)"),
    ("Dr. Nikhil Joshi", "Radiologist", "Precision Diagnostics, Nagpur (demo)"),
]

APPOINTMENTS_PATH = _ROOT / "outputs" / "appointments.csv"
APPOINTMENT_COLUMNS = [
    "appointment_id", "patient_name", "patient_id", "patient_user_id", "patient_phone", "patient_email",
    "specialist_name", "specialty", "clinic", "phone", "appt_date", "appt_time", "reason", "report_path",
    "doctor_name", "created_at", "status",
]


def _migrate_appointments_header() -> None:
    """Rewrites outputs/appointments.csv in place if its on-disk header is
    missing columns APPOINTMENT_COLUMNS has gained since those rows were
    written (e.g. `status`, added later). Old rows get "" for the new
    columns -- without this, appending a row with more fields than the
    existing header would silently misalign every column."""
    if not APPOINTMENTS_PATH.exists():
        return
    with open(APPOINTMENTS_PATH, newline="") as f:
        reader = csv.DictReader(f)
        existing_fieldnames = reader.fieldnames or []
        rows = list(reader)
    if existing_fieldnames == APPOINTMENT_COLUMNS:
        return
    with open(APPOINTMENTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=APPOINTMENT_COLUMNS)
        writer.writeheader()
        for row in rows:
            row.pop(None, None)  # DictReader's catch-all for unexpected extra fields
            writer.writerow({col: row.get(col, "") for col in APPOINTMENT_COLUMNS})


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _append_csv_row(path: Path, columns: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if is_new:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in columns})


def _write_csv_rows(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({col: r.get(col, "") for col in columns})


def _delete_csv_row(path: Path, columns: list[str], id_column: str, id_value: str) -> None:
    _write_csv_rows(path, columns, [r for r in _read_csv_rows(path) if r.get(id_column) != id_value])


def _update_csv_row(path: Path, columns: list[str], id_column: str, id_value: str, updates: dict) -> bool:
    rows = _read_csv_rows(path)
    found = False
    for r in rows:
        if r.get(id_column) == id_value:
            r.update(updates)
            found = True
    if found:
        _write_csv_rows(path, columns, rows)
    return found


def add_appointment(row: dict) -> str:
    _migrate_appointments_header()
    appointment_id = uuid.uuid4().hex[:12]
    _append_csv_row(
        APPOINTMENTS_PATH, APPOINTMENT_COLUMNS,
        {"status": "Scheduled", **row, "appointment_id": appointment_id},
    )
    return appointment_id


def read_appointments() -> list[dict]:
    return _read_csv_rows(APPOINTMENTS_PATH)


def delete_appointment(appointment_id: str) -> None:
    _delete_csv_row(APPOINTMENTS_PATH, APPOINTMENT_COLUMNS, "appointment_id", appointment_id)


def update_appointment(appointment_id: str, updates: dict) -> bool:
    """Used for rescheduling: merges `updates` (e.g. new appt_date/appt_time)
    into the existing row rather than delete+recreate, so the appointment_id
    -- and therefore any report link -- stays stable."""
    return _update_csv_row(APPOINTMENTS_PATH, APPOINTMENT_COLUMNS, "appointment_id", appointment_id, updates)


def build_ics_event(summary: str, description: str, start: datetime.datetime, end: datetime.datetime,
                     location: str = "") -> str:
    """Builds a minimal RFC 5545 .ics file (one VEVENT, with a 24h-before
    reminder alarm) so a doctor/patient can add the appointment to their own
    calendar app -- a real, working reminder, unlike an in-app notification
    this offline tool has no way to actually deliver."""
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace(",", "\\,").replace(";", "\\;").replace("\n", "\\n")

    uid = f"{uuid.uuid4().hex}@oncosense"
    dtstamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//OncoSense//Appointment//EN",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{dtstamp}",
        f"DTSTART:{start.strftime('%Y%m%dT%H%M%S')}",
        f"DTEND:{end.strftime('%Y%m%dT%H%M%S')}",
        f"SUMMARY:{esc(summary)}",
        f"DESCRIPTION:{esc(description)}",
    ]
    if location:
        lines.append(f"LOCATION:{esc(location)}")
    lines += [
        "BEGIN:VALARM", "TRIGGER:-PT24H", "ACTION:DISPLAY", "DESCRIPTION:Reminder", "END:VALARM",
        "END:VEVENT", "END:VCALENDAR",
    ]
    return "\r\n".join(lines) + "\r\n"


def save_screening_images(image_filename_stem: str, original_img: Image.Image, gradcam_arr) -> tuple[str, str]:
    """Saves the preprocessed mammogram and its Grad-CAM overlay as PNGs under
    outputs/images/, returning (image_path, gradcam_path) relative to outputs/
    (same convention as save_report_html's return value), so History.py can
    resolve them back to absolute paths and the patient's own scans can be
    compared side by side over time."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_name = f"{image_filename_stem}.png"
    gradcam_name = f"{image_filename_stem}_gradcam.png"
    original_img.save(IMAGES_DIR / image_name, format="PNG")
    Image.fromarray(gradcam_arr).save(IMAGES_DIR / gradcam_name, format="PNG")
    return f"images/{image_name}", f"images/{gradcam_name}"


AUTH_COOKIE_NAME = "oncosense_auth"
AUTH_COOKIE_DAYS = 365  # "keep me signed in on this device"


def get_browser_cookies() -> dict:
    """Read all current browser cookies using Streamlit's native context API.

    st.context.cookies (Streamlit >=1.40) is a read-only dict-like object
    populated from the HTTP Cookie header sent by the browser on the current
    request -- no extra round-trip or iframe required, so this never causes
    the /_stcore/health 404 errors that extra_streamlit_components produced.
    Returns an empty dict if the API is unavailable (e.g. very old Streamlit).
    """
    try:
        raw = st.context.cookies
        return dict(raw) if raw else {}
    except Exception:
        return {}


def _write_cookie_js(name: str, value: str, days: int) -> None:
    """Write a browser cookie via an inline <script> tag (height=0 iframe).

    This is the only sanctioned way to *write* a cookie in Streamlit >=1.40
    because st.context.cookies is read-only.
    """
    exp = (
        datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(days=days)
    ).strftime("%a, %d %b %Y %H:%M:%S GMT")
    # URL-encode the value to keep it safe inside the cookie string.
    import urllib.parse
    safe_value = urllib.parse.quote(value, safe="")
    js = (
        f"document.cookie = '{name}={safe_value}; "
        f"path=/; expires={exp}; SameSite=Strict';"
    )
    _st_components.html(f"<script>{js}</script>", height=0)


def _delete_cookie_js(name: str) -> None:
    """Expire a browser cookie immediately via an inline <script> tag."""
    js = (
        f"document.cookie = '{name}=; "
        f"path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; SameSite=Strict';"
    )
    _st_components.html(f"<script>{js}</script>", height=0)


def restore_login_from_cookies(cookies: dict) -> tuple[str, str] | None:
    """Looks up the persisted login in a browser-cookies dict (from
    get_browser_cookies()) and returns (doctor_name, doctor_email),
    or None if the cookie is absent or malformed.
    """
    raw = cookies.get(AUTH_COOKIE_NAME) if cookies else None
    if not raw:
        return None
    try:
        data = raw if isinstance(raw, dict) else json.loads(raw)
        name, email = data.get("n", ""), data.get("e", "")
    except (ValueError, TypeError, AttributeError):
        return None
    return (name, email) if name else None


def set_login_cookie(doctor_name: str, doctor_email: str,
                     remember: bool = True) -> None:
    """Write the doctor auth cookie using a native JS snippet."""
    payload = json.dumps({"n": doctor_name, "e": doctor_email})
    days = AUTH_COOKIE_DAYS if remember else 1
    _write_cookie_js(AUTH_COOKIE_NAME, payload, days)


def clear_login_cookie() -> None:
    """Delete the doctor auth cookie using a native JS snippet."""
    _delete_cookie_js(AUTH_COOKIE_NAME)


# ---------------------------------------------------------------------------
# Patient accounts (pages/Patient_Login.py, pages/Patient_Dashboard.py).
#
# A separate identity system from the doctor sign-in above -- its own CSV
# table, its own password hashing, and its own browser cookie
# (PATIENT_AUTH_COOKIE_NAME, distinct from AUTH_COOKIE_NAME) -- so a doctor
# session never grants patient-portal access and vice versa. Unlike the
# doctor gate (which accepts any well-formed name/email/password, since
# there's no real clinician credential store to check it against), patient
# passwords are genuinely hashed and verified: this app has no real backend
# server or database, so "the backend" here is this module's CSV-file
# persistence plus Streamlit's server-side script execution, but within that
# constraint the password handling itself is real, not a demo stand-in.
#
# `patient_id` is the clinic-assigned ID a doctor already uses when logging
# a screening (see HISTORY_COLUMNS) -- linking an account to it is what lets
# the portal show only that patient's own records, and it's restricted to
# one account per ID so a second registration can't attach itself to
# someone else's chart by guessing their ID.

PATIENTS_PATH = _ROOT / "outputs" / "patients.csv"
PATIENT_COLUMNS = [
    "patient_user_id", "patient_id", "patient_name", "email_lower", "email",
    "password_hash", "password_salt", "created_at", "reset_token", "reset_token_expires",
]
PATIENT_AUTH_COOKIE_NAME = "oncosense_patient_auth"
PATIENT_AUTH_COOKIE_DAYS = 365
PBKDF2_ITERATIONS = 200_000


def _hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return digest.hex(), salt.hex()


def _verify_password(password: str, hash_hex: str, salt_hex: str) -> bool:
    if not hash_hex or not salt_hex:
        return False
    computed, _ = _hash_password(password, salt_hex)
    return hmac.compare_digest(computed, hash_hex)


def _valid_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[-1]


def register_patient(patient_name: str, email: str, password: str) -> tuple[bool, str]:
    """Creates a patient portal account -- no Patient ID required up front,
    since that's only needed to see screening reports (see link_patient_id
    below), not to sign in and book an appointment. Returns
    (True, patient_user_id) on success, or (False, error_message)."""
    patient_name, email = patient_name.strip(), email.strip()
    if not patient_name or not email or not password:
        return False, "Please fill in every field."
    if not _valid_email(email):
        return False, "Enter a valid email address."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    existing = _read_csv_rows(PATIENTS_PATH)
    email_lower = email.lower()
    if any(r.get("email_lower") == email_lower for r in existing):
        return False, "An account with this email already exists -- try signing in instead."
    password_hash, password_salt = _hash_password(password)
    user_id = uuid.uuid4().hex[:16]
    _append_csv_row(PATIENTS_PATH, PATIENT_COLUMNS, {
        "patient_user_id": user_id, "patient_id": "", "patient_name": patient_name,
        "email_lower": email_lower, "email": email, "password_hash": password_hash,
        "password_salt": password_salt, "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reset_token": "", "reset_token_expires": "",
    })
    return True, user_id


def link_patient_id(patient_user_id: str, patient_id: str) -> tuple[bool, str]:
    """Links a clinic-assigned Patient ID to an existing account, after
    registration -- this is what lets Screening Reports / Medical History
    find this patient's records (doctors tag those by Patient ID, not by
    this account system). Restricted to one account per ID, same as at
    registration, so a patient can't attach themselves to someone else's
    chart by guessing their ID; once set it can't be changed by the patient
    (contact the clinic for corrections)."""
    patient_id = patient_id.strip()
    if not patient_id:
        return False, "Enter a Patient ID."
    rows = _read_csv_rows(PATIENTS_PATH)
    if any(r.get("patient_id") == patient_id and r.get("patient_user_id") != patient_user_id for r in rows):
        return False, "This Patient ID is already linked to an account. Contact your clinic if this is an error."
    found = False
    for r in rows:
        if r.get("patient_user_id") == patient_user_id:
            if (r.get("patient_id") or "").strip():
                return False, "An ID is already linked to this account."
            r["patient_id"] = patient_id
            found = True
    if not found:
        return False, "Account not found."
    _write_csv_rows(PATIENTS_PATH, PATIENT_COLUMNS, rows)
    return True, patient_id


def authenticate_patient(email: str, password: str) -> dict | None:
    email_lower = email.strip().lower()
    for r in _read_csv_rows(PATIENTS_PATH):
        if r.get("email_lower") == email_lower:
            if _verify_password(password, r.get("password_hash", ""), r.get("password_salt", "")):
                return {
                    "patient_user_id": r["patient_user_id"], "patient_id": r["patient_id"],
                    "patient_name": r["patient_name"], "email": r["email"],
                }
            return None
    return None


def generate_password_reset_token(email: str) -> str | None:
    """Returns a reset token valid for 30 minutes, or None if no account
    uses that email. There's no SMTP/email service configured in this app,
    so pages/Patient_Login.py shows the resulting reset link directly on
    screen rather than emailing it -- a local stand-in for the real delivery
    channel a production deployment would use."""
    email_lower = email.strip().lower()
    rows = _read_csv_rows(PATIENTS_PATH)
    for r in rows:
        if r.get("email_lower") == email_lower:
            token = secrets.token_urlsafe(24)
            r["reset_token"] = token
            r["reset_token_expires"] = (datetime.datetime.now() + datetime.timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
            _write_csv_rows(PATIENTS_PATH, PATIENT_COLUMNS, rows)
            return token
    return None


def reset_password_with_token(token: str, new_password: str) -> tuple[bool, str]:
    if len(new_password) < 8:
        return False, "Password must be at least 8 characters."
    rows = _read_csv_rows(PATIENTS_PATH)
    now = datetime.datetime.now()
    for r in rows:
        if token and r.get("reset_token") == token:
            try:
                expires = datetime.datetime.strptime(r.get("reset_token_expires", ""), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                expires = now - datetime.timedelta(seconds=1)
            if now > expires:
                return False, "This reset link has expired -- request a new one."
            r["password_hash"], r["password_salt"] = _hash_password(new_password)
            r["reset_token"], r["reset_token_expires"] = "", ""
            _write_csv_rows(PATIENTS_PATH, PATIENT_COLUMNS, rows)
            return True, "Password updated -- you can now sign in."
    return False, "This reset link is invalid or has already been used."


def update_patient_profile(patient_user_id: str, patient_name: str, email: str) -> tuple[bool, str]:
    patient_name, email = patient_name.strip(), email.strip()
    if not patient_name or not email:
        return False, "Name and email are required."
    if not _valid_email(email):
        return False, "Enter a valid email address."
    rows = _read_csv_rows(PATIENTS_PATH)
    email_lower = email.lower()
    if any(r.get("email_lower") == email_lower and r.get("patient_user_id") != patient_user_id for r in rows):
        return False, "Another account already uses this email."
    found = False
    for r in rows:
        if r.get("patient_user_id") == patient_user_id:
            r["patient_name"], r["email"], r["email_lower"] = patient_name, email, email_lower
            found = True
    if not found:
        return False, "Account not found."
    _write_csv_rows(PATIENTS_PATH, PATIENT_COLUMNS, rows)
    return True, patient_name


def change_patient_password(patient_user_id: str, current_password: str, new_password: str) -> tuple[bool, str]:
    if len(new_password) < 8:
        return False, "New password must be at least 8 characters."
    rows = _read_csv_rows(PATIENTS_PATH)
    for r in rows:
        if r.get("patient_user_id") == patient_user_id:
            if not _verify_password(current_password, r.get("password_hash", ""), r.get("password_salt", "")):
                return False, "Current password is incorrect."
            r["password_hash"], r["password_salt"] = _hash_password(new_password)
            _write_csv_rows(PATIENTS_PATH, PATIENT_COLUMNS, rows)
            return True, "Password updated."
    return False, "Account not found."


def restore_patient_login_from_cookies(cookies: dict) -> dict | None:
    raw = cookies.get(PATIENT_AUTH_COOKIE_NAME) if cookies else None
    if not raw:
        return None
    try:
        data = raw if isinstance(raw, dict) else json.loads(raw)
        user_id = data.get("u", "")
    except (ValueError, TypeError, AttributeError):
        return None
    if not user_id:
        return None
    return {
        "patient_user_id": user_id, "patient_id": data.get("p", ""),
        "patient_name": data.get("n", ""), "email": data.get("e", ""),
    }


def set_patient_login_cookie(patient_user_id: str, patient_id: str,
                             patient_name: str, email: str, remember: bool = True) -> None:
    """Write the patient auth cookie using a native JS snippet."""
    payload = json.dumps({"u": patient_user_id, "p": patient_id, "n": patient_name, "e": email})
    days = PATIENT_AUTH_COOKIE_DAYS if remember else 1
    _write_cookie_js(PATIENT_AUTH_COOKIE_NAME, payload, days)


def clear_patient_login_cookie() -> None:
    """Delete the patient auth cookie using a native JS snippet."""
    _delete_cookie_js(PATIENT_AUTH_COOKIE_NAME)


def save_report_html(report_filename: str, report_html: str) -> str:
    """Writes a completed screening's full report sheet to outputs/reports/
    and returns the path relative to outputs/ (what gets stored in the CSV's
    report_path column, so History.py can resolve it back to an absolute
    path regardless of where the repo is checked out)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / report_filename).write_text(report_html, encoding="utf-8")
    return f"reports/{report_filename}"

# Measured on the official CBIS-DDSM test split, never touched during training
# (see README for the full evaluation methodology).
TEST_ACCURACY = 0.66
TEST_AUC = 0.7339
VAL_ACCURACY = 0.71

GRADCAM_DIR = Path(__file__).parent / "outputs" / "gradcam"


def img_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def np_to_base64(arr) -> str:
    return img_to_base64(Image.fromarray(arr))


def img_data_url(b64: str | None) -> str:
    return f"data:image/jpeg;base64,{b64}" if b64 else ""


LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"


@st.cache_data
def load_logo_b64(max_width: int = 256) -> str | None:
    """Base64-encodes assets/logo.png (the project's real logo) as PNG,
    preserving transparency, downsized for a small inline data URI."""
    if not LOGO_PATH.exists():
        return None
    img = Image.open(LOGO_PATH).convert("RGBA")
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, max(1, int(img.height * ratio))))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def logo_data_url() -> str:
    b64 = load_logo_b64()
    return f"data:image/png;base64,{b64}" if b64 else ""


def logo_img_tag(size: int = 32, extra_style: str = "") -> str:
    """<img> tag for the OncoSense logo at a given pixel size. object-fit:
    contain respects the artwork's own aspect ratio rather than stretching
    or cropping it -- used everywhere the old gradient-circle "OS" mark used
    to appear (headers, sidebars, login cards, the exported report)."""
    url = logo_data_url()
    if not url:
        return ""
    return (
        f'<img src="{url}" alt="OncoSense logo" '
        f'style="width:{size}px; height:{size}px; object-fit:contain; flex-shrink:0; {extra_style}">'
    )


@st.cache_data
def load_decorative_image(filename: str, crop: str = "left", max_width: int = 900, quality: int = 82):
    """Base64-encode a real training-time Grad-CAM sample for marketing visuals.

    These are genuine mammograms produced by this project's own
    src/gradcam.py run, not stock photography. `crop` picks the clean
    original-image half of the side-by-side Grad-CAM export (the other half
    carries the heatmap overlay).
    """
    path = GRADCAM_DIR / filename
    if not path.exists():
        return None
    img = Image.open(path).convert("RGB")
    w, h = img.size
    half = w // 2
    img = img.crop((0, 0, half, h)) if crop == "left" else img.crop((half, 0, w, h))
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, max(1, int(img.height * ratio))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


GRADCAM_SAMPLES = [
    "Mass-Training_P_00090_LEFT_MLO_p0.95.png",
    "Mass-Training_P_00488_LEFT_CC_p0.66.png",
    "Mass-Training_P_00644_LEFT_CC_p0.61.png",
    "Mass-Training_P_01093_LEFT_CC_p0.87.png",
    "Mass-Training_P_01173_RIGHT_MLO_p0.55.png",
    "Mass-Training_P_01379_LEFT_CC_p0.50.png",
]
_decorative = {name: load_decorative_image(name) for name in GRADCAM_SAMPLES}
_decorative_available = [v for v in _decorative.values() if v]
HERO_IMAGE_B64 = _decorative_available[0] if _decorative_available else None
FINAL_CTA_IMAGE_B64 = _decorative_available[3] if len(_decorative_available) > 3 else HERO_IMAGE_B64

ICONS = {
    "chip": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="7" y="7" width="10" height="10" rx="1.5"/><path d="M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3"/></svg>',
    "layers": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 3l9 5-9 5-9-5 9-5Z"/><path d="M3 13l9 5 9-5"/></svg>',
    "grid": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="8" height="8" rx="1"/><rect x="13" y="3" width="8" height="8" rx="1"/><rect x="3" y="13" width="8" height="8" rx="1"/><rect x="13" y="13" width="8" height="8" rx="1"/></svg>',
    "zap": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z"/></svg>',
    "image": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8.5" cy="9.5" r="1.5"/><path d="m21 15-5-5L5 20"/></svg>',
    "database": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></svg>',
    "check": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="m20 6-11 11-5-5"/></svg>',
    "alert": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 3 2 20h20L12 3Z"/><path d="M12 10v4"/><path d="M12 17h.01"/></svg>',
    "lump": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 4c2 0 3.5 1.2 4.5 2.8 1.6.4 3 1.8 3 3.7 0 1.6-1 2.9-2.3 3.5.3 1.7-.7 3.6-2.7 4.2-1.3.4-2.7.1-3.7-.7-1.4.6-3.1.3-4.1-.9-1.2-.2-2.2-1.2-2.4-2.5-1.1-.6-1.8-1.8-1.6-3.1.1-1 .8-1.8 1.7-2.2C4.6 7.4 5.8 6 7.5 5.6 8.6 4.6 10.2 4 12 4Z"/></svg>',
    "expand": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="3"/><path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M18.4 5.6l-2.8 2.8M8.4 15.6l-2.8 2.8"/></svg>',
    "texture": '<svg viewBox="0 0 24 24" fill="currentColor" stroke="none"><circle cx="6" cy="6" r="1.4"/><circle cx="12" cy="6" r="1.4"/><circle cx="18" cy="6" r="1.4"/><circle cx="6" cy="12" r="1.4"/><circle cx="12" cy="12" r="1.4"/><circle cx="18" cy="12" r="1.4"/><circle cx="6" cy="18" r="1.4"/><circle cx="12" cy="18" r="1.4"/><circle cx="18" cy="18" r="1.4"/></svg>',
    "retract": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="8"/><path d="M12 6.5v4.3m0 0-2-2m2 2 2-2"/></svg>',
    "droplet": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11Z"/></svg>',
    "pulse": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 12h4l2-6 4 12 2-6h6"/></svg>',
    "nodes": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="9" cy="8" r="3"/><circle cx="16.5" cy="13" r="2.3"/><circle cx="8" cy="17" r="2"/></svg>',
}

# General, well-established public-health awareness info on common physical
# signs of breast cancer -- educational context only, not diagnostic (see the
# disclaimer rendered under this section). None of these confirm cancer on
# their own; they're reasons to seek a clinical evaluation.
SYMPTOMS_ITEMS = [
    ("New lump or mass", "Often painless, hard, and with irregular edges -- in the breast or underarm.", "lump"),
    ("Swelling", "Swelling of all or part of the breast, even without a distinct lump.", "expand"),
    ("Skin changes", "Dimpling, redness, or a texture resembling an orange peel.", "texture"),
    ("Nipple changes", "Retraction, inward turning, or a change in nipple position.", "retract"),
    ("Nipple discharge", "Discharge other than breast milk, including blood.", "droplet"),
    ("Breast or nipple pain", "Persistent pain that doesn't go away on its own.", "pulse"),
    ("Thickening", "Thickening or firmness in part of the breast tissue.", "layers"),
    ("Underarm swelling", "A new lump or swelling near the underarm or collarbone.", "nodes"),
]

PROJECT_CARDS = [
    ("01", "COMPUTER VISION", "Image processing and mammogram preparation using OpenCV, Pillow, NumPy and Pydicom."),
    ("02", "DEEP LEARNING", "ResNet18 fine-tuned using PyTorch and torchvision on the CBIS-DDSM dataset."),
    ("03", "INTERACTIVE ANALYSIS", "An interactive application for uploading mammograms, running inference, viewing results and reviewing explanations."),
]

PIPELINE_STEPS = [
    ("01", "UPLOAD", "Upload a mammogram image (DICOM, PNG, or JPEG)."),
    ("02", "PREPROCESS", "Resize, normalize, and -- for DICOM -- crop to the breast region."),
    ("03", "MODEL", "Run the fine-tuned ResNet18 model."),
    ("04", "ANALYZE", "Generate the benign / malignant probability output."),
    ("05", "EXPLAIN", "Review the Grad-CAM attention map behind the decision."),
]

# Clinical background & actionable medical advice on primary breast cancer risk factors
# and oncology surveillance protocols (NCCN, ACR, ACS).
CAUSES_ITEMS = [
    (
        "01",
        "GENETIC & HEREDITARY",
        "Inherited gene mutations (such as BRCA1 and BRCA2) and a family history of breast cancer significantly increase lifetime risk.",
    ),
    (
        "02",
        "HORMONAL & REPRODUCTIVE",
        "Prolonged lifetime exposure to estrogen through early menarche, late menopause, or hormone therapy increases risk.",
    ),
    (
        "03",
        "LIFESTYLE & METABOLIC",
        "Postmenopausal weight gain, low physical activity, and regular alcohol consumption are key modifiable risk factors.",
    ),
]

# ---------------------------------------------------------------------------
# Breast Health Education Center (pages/Education.py) -- general, well-
# established public-health information. Educational context only, never a
# diagnosis or personalized medical advice; each section on the page carries
# its own disclaimer pointing readers back to a qualified clinician.

SCREENING_GUIDELINES = [
    ("40-44", "Screening is a personal choice",
     "Women who wish to begin screening early may choose to start annual mammograms, ideally after "
     "discussing personal risk factors with a clinician."),
    ("45-54", "Annual screening is generally advised",
     "Most major guidelines recommend a mammogram every year during this decade, when breast cancer "
     "incidence rises significantly."),
    ("55+", "Continue regular screening",
     "Screening can continue annually or shift to every two years, depending on individual health, "
     "risk factors, and a clinician's recommendation."),
]

SCREENING_STEPS = [
    ("01", "Check-in", "Registration and a brief review of your breast health history and any current symptoms."),
    ("02", "Positioning", "A technologist positions the breast on the imaging plate for each required view."),
    ("03", "Compression & imaging", "Brief compression flattens breast tissue for a clearer image -- mild pressure lasting only seconds per view."),
    ("04", "Results & follow-up", "A radiologist reviews the images; results are typically shared within days, with follow-up imaging if needed."),
]

MISCONCEPTIONS = [
    ("A lump always means cancer.",
     "Most breast lumps are benign -- often cysts or fibroadenomas -- but any new or persistent lump "
     "should still be evaluated by a clinician."),
    ("Only women with a family history get breast cancer.",
     "Most people diagnosed have no known family history. Family history raises risk but is not "
     "required for a diagnosis."),
    ("Mammograms use dangerous levels of radiation.",
     "Mammograms use a very low radiation dose, and the benefit of early detection generally outweighs "
     "this minimal risk."),
    ("Young women don't need to think about breast cancer.",
     "Risk rises with age, but breast cancer can occur in younger adults too, so symptom awareness "
     "matters at any age."),
    ("Underwire bras or antiperspirants cause breast cancer.",
     "There is no credible scientific evidence linking bras or antiperspirant use to breast cancer risk."),
    ("A normal mammogram means there's nothing to ever worry about.",
     "Mammograms can miss some cancers, particularly in dense breast tissue, so continued self-awareness "
     "and regular screening still matter."),
]

RISK_TIERS = [
    ("Average risk",
     "No known BRCA1/BRCA2 or other high-risk gene mutation, no chest radiation therapy before age 30, "
     "and an estimated lifetime risk under 15-20% on a clinical risk-assessment model. Most people fall "
     "into this category -- routine age-based screening, as above, is generally appropriate."),
    ("Increased risk",
     "A known pathogenic gene mutation (e.g. BRCA1/BRCA2), a first-degree relative with a mutation, "
     "prior chest radiation therapy before age 30, a calculated lifetime risk of 20% or more, or a "
     "personal history of lobular carcinoma in situ (LCIS) or atypical hyperplasia. This group is "
     "often advised to start screening earlier and add annual breast MRI -- a genetic counselor or "
     "breast specialist can formally calculate individual risk."),
]

BREAST_DENSITY_CATEGORIES = [
    ("Category A", "Almost entirely fatty tissue."),
    ("Category B", "Scattered areas of fibroglandular density."),
    ("Category C", "Heterogeneously dense -- may obscure small masses."),
    ("Category D", "Extremely dense -- lowers mammogram sensitivity."),
]

DIAGNOSTIC_STEPS = [
    ("01", "Clinical evaluation", "A clinician reviews symptoms and personal/family history and performs a clinical breast exam."),
    ("02", "Diagnostic imaging", "A diagnostic mammogram and/or ultrasound takes a closer look at the area of concern, beyond a routine screening exam."),
    ("03", "Biopsy", "If imaging is suspicious, a radiologist removes a small tissue sample -- usually with a core-needle or stereotactic biopsy under local anesthesia."),
    ("04", "Pathology report", "A pathologist examines the tissue and reports whether it's benign or malignant, plus (if malignant) grade, hormone-receptor (ER/PR), and HER2 status."),
    ("05", "Staging", "If cancer is confirmed, additional tests -- such as a sentinel lymph node biopsy or body imaging -- determine how far, if at all, it has spread."),
]

CANCER_STAGES = [
    ("0", "Stage 0", "Abnormal cells confined to ducts or lobules, not yet invasive -- e.g. DCIS (ductal carcinoma in situ)."),
    ("1", "Stage 1", "A small, localized tumor (up to 2 cm) with no or minimal lymph node involvement."),
    ("2", "Stage 2", "A larger tumor and/or cancer that has spread to a limited number of nearby lymph nodes."),
    ("3", "Stage 3", "Locally advanced disease -- a larger tumor and/or more extensive lymph node or chest-wall/skin involvement, but no distant spread."),
    ("4", "Stage 4", "Cancer has spread beyond the breast and nearby nodes to distant organs, such as bone, liver, lung, or brain."),
]
STAGE_LOOKUP = {code: (label, desc) for code, label, desc in CANCER_STAGES}


def format_stage_name(stage: str) -> str:
    """Normalizes any stage representation into Stage 0, Stage 1, Stage 2, Stage 3, or Stage 4."""
    if not stage:
        return "Stage 0"
    s = str(stage).strip().upper()
    if "4" in s or "IV" in s:
        return "Stage 4"
    if "3" in s or "III" in s:
        return "Stage 3"
    if "2" in s or "II" in s:
        return "Stage 2"
    if "1" in s or "I" in s:
        return "Stage 1"
    if "0" in s or "TIS" in s or "BENIGN" in s or "NORMAL" in s:
        return "Stage 0"
    return "Stage 0"

# A mammogram-only image classifier has no way to see tumor size, lymph
# node involvement, or distant spread -- those require a clinical exam,
# imaging beyond the single screening view, and often a biopsy. So staging
# here is clinician-entered (clinical T/N/M) rather than model-predicted;
# this just groups that entry into an approximate AJCC stage so the report
# has a single "Stage" field instead of three raw codes.
CLINICAL_T_OPTIONS = ["Unknown", "Tis", "T1", "T2", "T3", "T4"]
CLINICAL_N_OPTIONS = ["Unknown", "N0", "N1", "N2", "N3"]
CLINICAL_M_OPTIONS = ["Unknown", "M0", "M1"]


def estimate_clinical_stage(clinical_t: str, clinical_n: str, clinical_m: str) -> str:
    """Approximate overall AJCC stage group (0/1/2/3/4) from clinician-
    entered clinical T/N/M codes. Returns "" if any code is missing/unknown,
    or not one of the recognized T/N/M values.

    This is a simplified grouping (it collapses the IA/IIA/IIIB-style
    sub-stages into their parent stage) meant for an at-a-glance report
    field, not a replacement for formal AJCC staging by an oncologist."""
    t = (clinical_t or "").strip()
    n = (clinical_n or "").strip()
    m = (clinical_m or "").strip()
    if not t or not n or not m or "Unknown" in (t, n, m):
        return ""

    if m == "M1":
        return "4"
    if t in ("Tis", "T0"):
        return "0"

    t_rank = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}.get(t)
    n_rank = {"N0": 0, "N1": 1, "N2": 2, "N3": 3}.get(n)
    if t_rank is None or n_rank is None:
        return ""

    if n_rank == 3 or t_rank == 4:
        return "3"
    if t_rank == 1 and n_rank == 0:
        return "1"
    if t_rank <= 2 and n_rank <= 1:
        return "2"
    if t_rank == 3 and n_rank == 0:
        return "2"
    return "3"


def detect_stage_from_image(pil_image, malignant_prob: float, cam: np.ndarray = None, meta: dict = None) -> tuple[str, dict]:
    """Detects and estimates cancer stage (Stage 0, 1, 2, 3, or 4) during image
    preprocessing and feature extraction.

    Analyzes lesion density, activation area from Grad-CAM (if available) or
    contrast/brightness clustering in the preprocessed image, mapped against
    the model's malignant probability and clinical staging criteria:
      - Normal / Benign (prob < 0.5): returns ("0", info) -> Stage 0
      - Stage 1: localized focal lesion (< 20mm or < 4% relative breast area)
      - Stage 2: moderate focal lesion (20mm - 50mm or 4% - 12% relative area)
      - Stage 3: extensive/locally advanced lesion (> 50mm or > 12% area, dense involvement)
      - Stage 4: multifocal diffuse lesions across multiple quadrants with high malignant score
    """
    if malignant_prob < 0.15:
        return "0", {
            "stage_detected": "Stage 0",
            "clinical_t": "T0",
            "clinical_n": "N0",
            "clinical_m": "M0",
            "tnm_string": "T0 / N0 / M0",
            "lesion_extent_pct": 0.0,
            "estimated_lesion_mm": 0.0,
            "multicentric": False,
            "active_quadrants": 0,
            "method": "Image Preprocessing & Morphometry Analysis",
        }

    # Preprocessed grayscale array
    gray = np.array(pil_image.convert("L"), dtype=np.float32)
    h, w = gray.shape

    # Pixel spacing or default estimated mm per pixel for standard mammography
    pixel_spacing = (meta or {}).get("_pixel_spacing_mm") or 0.1  # ~0.1mm standard

    if cam is not None and cam.size > 0:
        # Resize CAM to match grayscale dimensions if needed
        if cam.shape != (h, w):
            cam_resized = np.array(Image.fromarray(cam).resize((w, h), resample=Image.BILINEAR))
        else:
            cam_resized = cam
        # High-intensity activation region
        lesion_mask = cam_resized >= 0.55
    else:
        # Morphological intensity thresholding on preprocessed breast tissue
        # Exclude black background (zeros)
        tissue_mask = gray > 15
        if np.any(tissue_mask):
            thresh = np.percentile(gray[tissue_mask], 85)
            lesion_mask = gray >= thresh
        else:
            lesion_mask = gray >= 180

    total_tissue_pixels = max(1, np.sum(gray > 15))
    lesion_pixels = int(np.sum(lesion_mask))
    lesion_extent_pct = (lesion_pixels / total_tissue_pixels) * 100.0

    # Approximate equivalent diameter in mm: d = 2 * sqrt(Area / pi)
    area_mm2 = lesion_pixels * (pixel_spacing ** 2)
    estimated_lesion_mm = 2.0 * np.sqrt(max(0.1, area_mm2 / np.pi))

    # Quadrant quadrant spread analysis (check if suspicious regions span multiple quadrants)
    mid_y, mid_x = h // 2, w // 2
    quadrants = [
        np.sum(lesion_mask[:mid_y, :mid_x]),
        np.sum(lesion_mask[:mid_y, mid_x:]),
        np.sum(lesion_mask[mid_y:, :mid_x]),
        np.sum(lesion_mask[mid_y:, mid_x:]),
    ]
    active_quadrants = sum(1 for q in quadrants if q > (0.15 * max(1, lesion_pixels)))
    multicentric = active_quadrants >= 3

    # Stage and TNM determination based on user probability calibration:
    # Above 75% or 80% -> Stage 4
    if malignant_prob >= 0.75:
        detected_t = "T4"
        detected_n = "N2"
        detected_m = "M1"
        stage_detected = "4"
        stage_title = "Stage 4"
    # 70% to 75% -> Stage 3
    elif malignant_prob >= 0.70:
        detected_t = "T3"
        detected_n = "N1"
        detected_m = "M0"
        stage_detected = "3"
        stage_title = "Stage 3"
    # Below 70% (30% to 69.9%) -> Stage 2
    elif malignant_prob >= 0.30:
        detected_t = "T2"
        detected_n = "N0"
        detected_m = "M0"
        stage_detected = "2"
        stage_title = "Stage 2"
    # Below 30% (15% to 29.9%) -> Stage 1
    elif malignant_prob >= 0.15:
        detected_t = "T1"
        detected_n = "N0"
        detected_m = "M0"
        stage_detected = "1"
        stage_title = "Stage 1"
    # Below 15% -> Stage 0
    else:
        detected_t = "T0"
        detected_n = "N0"
        detected_m = "M0"
        stage_detected = "0"
        stage_title = "Stage 0"

    return stage_detected, {
        "stage_detected": stage_title,
        "clinical_t": detected_t,
        "clinical_n": detected_n,
        "clinical_m": detected_m,
        "tnm_string": f"{detected_t} / {detected_n} / {detected_m}",
        "lesion_extent_pct": round(float(lesion_extent_pct), 1),
        "estimated_lesion_mm": round(float(estimated_lesion_mm), 1),
        "multicentric": multicentric,
        "active_quadrants": active_quadrants,
        "method": "Image Preprocessing & AJCC Clinical Morphometry",
    }


CANCER_SUBTYPES = [
    ("01", "Hormone receptor-positive (ER+/PR+)", "Cells have receptors for estrogen and/or progesterone and tend to respond to hormone (endocrine) therapy."),
    ("02", "HER2-positive", "Cells overexpress the HER2 protein, which drives faster growth but responds well to HER2-targeted therapy."),
    ("03", "Triple-negative", "Lacks ER, PR, and HER2 receptors -- doesn't respond to hormone or HER2-targeted therapy, so chemotherapy and immunotherapy play a larger role."),
    ("04", "Ductal carcinoma in situ (DCIS)", "A non-invasive, stage 0 finding confined to the milk ducts -- highly treatable, often with surgery alone."),
]

TREATMENT_OPTIONS = [
    ("01", "Surgery", "Removes the tumor -- either breast-conserving surgery (lumpectomy) or mastectomy -- often alongside a sentinel lymph node biopsy to check for spread."),
    ("02", "Radiation therapy", "Uses targeted high-energy beams to destroy remaining cancer cells, typically after breast-conserving surgery."),
    ("03", "Chemotherapy", "Systemic medication that kills rapidly dividing cells, used before surgery (to shrink a tumor) or after (to reduce recurrence risk)."),
    ("04", "Hormone (endocrine) therapy", "For hormone receptor-positive cancers, blocks or lowers estrogen to slow or stop cancer growth -- e.g. tamoxifen or aromatase inhibitors."),
    ("05", "Targeted & immunotherapy", "Precision drugs aimed at specific cancer features -- e.g. HER2-targeted therapy, or immunotherapy for some triple-negative cases."),
]

LIFESTYLE_TIPS = [
    ("01", "Maintain a healthy weight", "Higher body weight after menopause is linked to increased risk -- gradual, sustainable weight management can help."),
    ("02", "Stay physically active", "Regular moderate exercise is consistently associated with lower breast cancer risk."),
    ("03", "Limit alcohol intake", "Risk rises with the amount of alcohol consumed regularly; reducing intake can help lower it."),
    ("04", "Avoid or quit smoking", "Smoking, especially before a first pregnancy, has been linked to higher breast cancer risk."),
    ("05", "Eat a balanced diet", "A diet rich in vegetables, fruit, and whole grains supports overall health and may help reduce risk."),
    ("06", "Know what's normal for you", "Regular self-awareness of how your breasts normally look and feel makes it easier to notice changes early."),
]

FAQ_ITEMS = [
    ("What does a BI-RADS score mean?",
     "BI-RADS (Breast Imaging-Reporting and Data System) is a standardized scale radiologists use to "
     "summarize mammogram findings, from 0 (incomplete, needs more imaging) to 6 (known biopsy-proven "
     "malignancy). Most screening results fall in the 1-2 range, meaning normal or benign findings."),
    ("What happens after an abnormal mammogram result?",
     "An abnormal or inconclusive result usually leads to additional imaging -- such as a diagnostic "
     "mammogram or ultrasound -- to get a clearer look. Most of these follow-ups do not turn out to be "
     "cancer, but they're an important next step your care team will guide you through."),
    ("What is the difference between a mammogram and a biopsy?",
     "A mammogram is an imaging test that looks for areas of concern. A biopsy removes a small sample "
     "of tissue from an area of concern so it can be examined under a microscope -- it's the only way "
     "to confirm whether a finding is cancerous."),
    ("What do 'benign' and 'malignant' mean?",
     "Benign means a finding is not cancerous and typically does not spread or threaten health. "
     "Malignant means cancerous cells are present that can grow and potentially spread, and generally "
     "requires prompt medical treatment."),
    ("What treatment options exist for breast cancer?",
     "Depending on the type and stage, treatment may include surgery, radiation therapy, chemotherapy, "
     "hormone therapy, or targeted therapy -- often combined. An oncology team tailors the approach to "
     "each individual's diagnosis."),
    ("Is a breast biopsy painful?",
     "Most biopsies use local anesthesia, so the area is numbed beforehand. Some pressure or mild "
     "soreness afterward is common, but severe pain is not typical -- your care team can advise on what "
     "to expect for your specific procedure."),
    ("How accurate is AI-assisted mammogram analysis?",
     "AI models like the one in this project can flag patterns associated with malignancy, but they are "
     "trained on limited datasets and are not infallible -- see this project's reported accuracy and "
     "ROC-AUC on the landing page. They're intended to support, not replace, a radiologist's judgment."),
    ("Can a tool like OncoSense replace a radiologist?",
     "No. This is an educational research demo. Only a qualified radiologist's interpretation, combined "
     "with clinical evaluation and -- when needed -- a biopsy, can confirm a diagnosis."),
    ("How soon should I follow up after a suspicious finding?",
     "Follow-up timing depends on the finding and your clinician's recommendation, but it's generally "
     "advised not to delay -- promptly scheduling the recommended follow-up imaging or specialist visit "
     "gives you the clearest, fastest answer."),
    ("Should I get genetic testing for BRCA1/BRCA2?",
     "Genetic testing is generally considered when there's a strong family history of breast or ovarian "
     "cancer, a relative with a known mutation, or other risk indicators. A genetic counselor can assess "
     "whether testing is appropriate and help interpret the results."),
    ("What does 'breast density' mean and why does it matter?",
     "Breast density describes the ratio of fibroglandular to fatty tissue on a mammogram, reported as "
     "Category A (fatty) through D (extremely dense). Denser tissue can both raise baseline risk slightly "
     "and make mammograms harder to read, which is why supplemental ultrasound or MRI is sometimes added."),
    ("Is it normal to get a second opinion on a biopsy or treatment plan?",
     "Yes -- seeking a second opinion, including a second pathology review, is a routine and accepted part "
     "of cancer care, not a sign of distrust. Most oncology teams expect and support it."),
    ("What does follow-up care look like after treatment ends?",
     "Survivorship care typically includes regular physical exams, ongoing imaging, and management of any "
     "long-term treatment effects, on a schedule set by the oncology team -- usually more frequent in the "
     "first few years and spacing out over time."),
]

_ASSISTANT_SYSTEM_PROMPT = """You are the OncoSense AI Breast Health Assistant, embedded in a patient \
portal for a breast-cancer screening research demo.

Scope of what you help with:
- Answer general questions about mammograms, breast health, screening, and risk factors.
- Explain medical and radiology terms in plain, simple language.
- Help a patient understand what an AI screening report means -- what each field says in general \
terms -- and what questions to bring to their doctor about it.

Hard limits:
- You are not a diagnostic tool. Never tell the patient what they "have," rule a condition in or out, \
or give personalized treatment advice.
- OncoSense's AI classifier is a research demo (around 66% test accuracy) and is not a certified \
medical device. Say so plainly whenever you discuss what a report's prediction or confidence score means.
- Always point the patient to a licensed doctor or radiologist for anything about their specific \
results, symptoms, or care decisions.

Style: warm, clear, brief, and professional. Prefer plain words over jargon, and define any clinical \
term you must use. No markdown tables -- short paragraphs or simple bullet lists only.
"""


def is_ai_assistant_configured() -> bool:
    """Whether a real GEMINI_API_KEY (or GOOGLE_API_KEY) is present -- gates
    whether the AI Breast Health Assistant chat can call Gemini at all."""
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def ask_breast_health_assistant(messages: list[dict]) -> str:
    """Sends a patient's conversation to Gemini and returns the assistant's
    reply text. `messages` is the full turn history (alternating
    user/assistant dicts, ending in the new user turn) -- this function is
    stateless and does not persist anything itself.

    Degrades to a plain, honest notice -- never a fabricated answer -- if no
    API key is configured or the call fails for any reason.
    """
    if not is_ai_assistant_configured():
        return (
            "The AI Breast Health Assistant isn't connected yet -- this server doesn't have a "
            "GEMINI_API_KEY configured, so I can't generate answers right now. Please let the "
            "OncoSense team know, and in the meantime bring your questions to your doctor."
        )

    try:
        from google import genai
        from google.genai import errors, types
    except ImportError:
        return (
            "The AI Breast Health Assistant is missing a required package on the server and can't "
            "respond right now. Please let the OncoSense team know."
        )

    system_prompt = _ASSISTANT_SYSTEM_PROMPT
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages
    ]

    # "-latest" aliases track whatever Google currently considers its
    # stable model for that tier, so this doesn't need manual updates when a
    # pinned version like gemini-3.8-flash gets deprecated. The lite model
    # is a fallback for when the primary tier is under heavy load (a real,
    # observed failure mode -- Gemini's own gateway returns 503 "currently
    # experiencing high demand" for the full flash tier while the lite tier
    # stays responsive) -- not used by default since it's a smaller model.
    model_chain = ["gemini-flash-latest", "gemini-flash-lite-latest"]
    transient_codes = {429, 502, 503, 504}
    last_error: Exception | None = None

    for i, model_name in enumerate(model_chain):
        is_last_model = i == len(model_chain) - 1
        # One try per model -- an overloaded model rarely clears up within a
        # couple of seconds, so re-hitting it just adds latency. Only the
        # last model in the chain (nothing left to fall back to) gets one
        # retry with a short backoff before giving up entirely.
        attempts = 2 if is_last_model else 1
        for attempt in range(attempts):
            try:
                client = genai.Client(http_options=types.HttpOptions(timeout=30_000))
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        max_output_tokens=1024,
                    ),
                )
                reply = (response.text or "").strip()
                return reply or "I didn't get a clear answer back -- please try rephrasing your question."
            except errors.APIError as e:
                last_error = e
                if e.code not in transient_codes:
                    return f"The AI Assistant hit an error talking to Gemini (status {e.code}). Please try again shortly."
                if is_last_model and attempt == 0:
                    time.sleep(2)
            except Exception as e:
                last_error = e
                if is_last_model and attempt == 0:
                    time.sleep(2)

    code = last_error.code if isinstance(last_error, errors.APIError) else None
    if code:
        return f"The AI Assistant hit an error talking to Gemini (status {code}). Please try again shortly."
    return "The AI Assistant didn't get a response in time. Please check your connection and try again."


def inject_spa_navigation() -> None:
    """Intercepts internal page anchor clicks and routes them through
    Streamlit's client-side router (history.pushState + popstate event).

    This prevents cold browser GET reloads when clicking links like
    <a href="/Screening"> or <a href="/Education">, eliminating both the
    slow page flicker and the console 404 error from Streamlit's initial
    subpath detection probe (e.g. Screening/_stcore/host-config).
    """
    _st_components.html(
        """
        <script>
        (function () {
          try {
            var win = window.parent;
            var doc = win.document;
            if (win.__oncosense_spa_nav_installed) return;
            win.__oncosense_spa_nav_installed = true;

            doc.addEventListener('click', function (e) {
              var a = e.target.closest('a');
              if (!a) return;

              var href = a.getAttribute('href');
              if (!href) return;

              // Only intercept internal application page routes (e.g. /Screening, /Education)
              if (href.startsWith('/') && !href.startsWith('//') && !href.startsWith('/_')) {
                // Ignore new-tab links or download links
                if (a.target && a.target !== '_self') return;
                if (a.hasAttribute('download')) return;

                e.preventDefault();
                e.stopPropagation();

                var current = win.location.pathname + win.location.search;
                if (current !== href) {
                  win.history.pushState({}, '', href);
                  win.dispatchEvent(new PopStateEvent('popstate'));
                }
              }
            }, true);
          } catch (err) {
            /* Sandboxed or cross-origin iframe - fail gracefully */
          }
        })();
        </script>
        """,
        height=0,
    )


def render_floating_ai_widget() -> None:
    inject_spa_navigation()
    """Renders a round floating-action-button (bottom-right, every page)
    that expands into a compact AI Breast Health Assistant chat panel in
    place. Shares the same ask_breast_health_assistant() backend as the full
    /AI_Assistant page and the Patient Portal's inline tab, but keeps its
    own `floating_ai_*` session-state keys so none of the three interfere
    with each other."""
    st.session_state.setdefault("floating_ai_open", False)
    st.session_state.setdefault("floating_ai_maximized", False)
    st.session_state.setdefault("floating_ai_chat_history", [])

    maximized = st.session_state["floating_ai_maximized"]
    panel_size_css = (
        "width: min(720px, 94vw) !important; height: min(640px, 85vh) !important;"
        if maximized else
        "width: min(380px, 92vw) !important; height: min(480px, 72vh) !important;"
    )

    st.markdown(
        f"""
        <style>
        .st-key-floating_ai_fab {{
            position: fixed; bottom: 22px; right: 22px; z-index: 10000; width: 58px;
        }}
        .st-key-floating_ai_fab div[data-testid="stButton"] button {{
            width: 58px; height: 58px; border-radius: 50%; padding: 0; min-height: 0;
            background: linear-gradient(135deg,#F6A6C1,#C9578B); border: none;
            box-shadow: 0 10px 28px rgba(201,87,139,.5);
            transition: transform .15s ease, box-shadow .15s ease;
        }}
        .st-key-floating_ai_fab div[data-testid="stButton"] button span[data-testid="stIconMaterial"] {{
            font-size: 25px; color: #1a0a12;
        }}
        .st-key-floating_ai_fab div[data-testid="stButton"] button p {{ display: none; }}
        .st-key-floating_ai_fab div[data-testid="stButton"] button:hover {{
            transform: translateY(-2px); box-shadow: 0 14px 32px rgba(201,87,139,.6); filter: none;
        }}
        .st-key-floating_ai_panel {{
            position: fixed; bottom: 92px; right: 22px; z-index: 10000;
            {panel_size_css} display: flex; flex-direction: column;
            background: hsl(var(--surface)); border: 1px solid hsl(var(--stroke));
            border-radius: 18px; overflow: hidden; box-shadow: 0 20px 48px rgba(0,0,0,.5);
            transition: width .18s ease, height .18s ease;
        }}
        .st-key-floating_ai_panel {{ gap: 0 !important; }}
        /* st.container(key=...) wraps each sub-container in a generic,
           unstyled stLayoutWrapper div that is the PANEL's actual direct
           flex-item child -- our .st-key-floating_ai_* classes land one
           level deeper (on a grandchild), where flex-grow/shrink have no
           effect on the panel's layout. :has() targets the real flex item. */
        .st-key-floating_ai_panel > div[data-testid="stLayoutWrapper"]:has(.st-key-floating_ai_header) {{
            flex-shrink: 0 !important;
        }}
        .st-key-floating_ai_panel > div[data-testid="stLayoutWrapper"]:has(.st-key-floating_ai_body) {{
            flex-grow: 1 !important; flex-shrink: 1 !important; flex-basis: 0 !important;
            min-height: 0 !important; overflow-y: auto;
        }}
        .st-key-floating_ai_panel > div[data-testid="stLayoutWrapper"]:has(.st-key-floating_ai_footer) {{
            flex-shrink: 0 !important;
        }}
        .st-key-floating_ai_header {{
            background: linear-gradient(120deg,#F6A6C1,#C9578B); padding: 14px 8px 14px 18px;
        }}
        .st-key-floating_ai_header div[data-testid="stButton"] button {{
            background: transparent; border: none; box-shadow: none; min-height: 0;
            width: 30px; height: 30px; padding: 0; border-radius: 50%;
        }}
        .st-key-floating_ai_header div[data-testid="stButton"] button span[data-testid="stIconMaterial"] {{
            color: #1a0a12; font-size: 17px;
        }}
        .st-key-floating_ai_header div[data-testid="stButton"] button:hover {{ background: rgba(26,10,18,.12); }}
        .st-key-floating_ai_body {{
            padding: 14px 16px; height: 100%; box-sizing: border-box; overflow-y: auto;
        }}
        .st-key-floating_ai_body div[data-testid="stChatMessage"] {{ padding: 10px 12px; margin-bottom: 8px; }}
        .st-key-floating_ai_footer {{
            padding: 8px 16px 14px; border-top: 1px solid hsl(var(--stroke));
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state["floating_ai_open"]:
        with st.container(key="floating_ai_panel"):
            with st.container(key="floating_ai_header"):
                head_col, max_col, close_col = st.columns([5, 1, 1], vertical_alignment="center")
                with head_col:
                    st.markdown(
                        '<div style="display:flex; align-items:center; gap:8px;">'
                        '<span style="font-size:17px; line-height:1;">&#129658;</span>'
                        '<strong style="font-size:14px; color:#1a0a12;">AI Breast Health Assistant</strong>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                with max_col:
                    maximize_icon = ":material/close_fullscreen:" if maximized else ":material/open_in_full:"
                    if st.button("", key="floating_ai_maximize_btn", icon=maximize_icon):
                        st.session_state["floating_ai_maximized"] = not maximized
                        st.rerun()
                with close_col:
                    if st.button("", key="floating_ai_close_btn", icon=":material/close:"):
                        st.session_state["floating_ai_open"] = False
                        st.rerun()

            with st.container(key="floating_ai_body"):
                # Streamlit doesn't create this container's DOM node at all
                # if nothing renders inside it -- with no chat history yet,
                # the loop below is empty, the body vanishes from the flex
                # layout, and the footer (chat input) collapses up under the
                # header instead of being pushed to the bottom. This
                # invisible spacer guarantees the body always renders.
                st.markdown('<div style="min-height:1px;"></div>', unsafe_allow_html=True)
                for turn in st.session_state["floating_ai_chat_history"][-20:]:
                    with st.chat_message(turn["role"]):
                        st.markdown(turn["content"])

            with st.container(key="floating_ai_footer"):
                if prompt := st.chat_input("Ask a question...", key="floating_ai_chat_input"):
                    st.session_state["floating_ai_chat_history"].append({"role": "user", "content": prompt})
                    with st.spinner("Thinking..."):
                        reply = ask_breast_health_assistant(st.session_state["floating_ai_chat_history"])
                    st.session_state["floating_ai_chat_history"].append({"role": "assistant", "content": reply})
                    st.rerun()

    with st.container(key="floating_ai_fab"):
        if st.button("", key="floating_ai_fab_btn", icon=":material/smart_toy:"):
            st.session_state["floating_ai_open"] = not st.session_state["floating_ai_open"]
            st.rerun()


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Instrument+Serif:ital@1&display=swap');

:root {
  --bg: 330 20% 5%;
  --surface: 330 15% 9%;
  --text: 330 20% 97%;
  --muted: 330 8% 65%;
  --stroke: 330 12% 18%;
  --accent: 330 70% 72%;
  --font-body: 'Inter', ui-sans-serif, system-ui, sans-serif;
  --font-display: 'Instrument Serif', serif;
}

html, body, [class*="css"] { font-family: var(--font-body); }
.stApp { background: hsl(var(--bg)); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stStatusWidget"], [data-testid="stSidebarNav"] { visibility: hidden; display: none; }
.block-container { padding: 6.5rem 2.4rem 2rem; max-width: 100%; width: 100%; }
h1, h2, h3, p { margin: 0; }
* { box-sizing: border-box; }

.accent-gradient { background: linear-gradient(90deg, #F6A6C1 0%, #E875A5 50%, #C9578B 100%); }
.accent-text { background: linear-gradient(90deg, #F6A6C1 0%, #E875A5 50%, #C9578B 100%); -webkit-background-clip: text; background-clip: text; color: transparent; }
.serif-italic { font-family: var(--font-display); font-style: italic; font-weight: 400; }

@keyframes scroll-down { 0% { transform: translateY(-100%); } 100% { transform: translateY(200%); } }
@keyframes role-fade-in { 0% { opacity: 0; transform: translateY(8px); } 100% { opacity: 1; transform: translateY(0); } }
@keyframes gradient-shift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes pulse-soft { 0%, 100% { opacity: 1; } 50% { opacity: .5; } }
@keyframes scan-line { 0% { top: -10%; } 100% { top: 110%; } }
@keyframes fade-up { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .001ms !important; animation-iteration-count: 1 !important;
    transition-duration: .001ms !important; scroll-behavior: auto !important;
  }
}

::selection { background: hsl(var(--accent) / .3); }
a { color: inherit; }
:focus-visible { outline: 2px solid hsl(var(--accent)); outline-offset: 2px; }

/* ---------------------------------------------------------- Floating nav */
.site-navbar {
  position: fixed; top: 16px; left: 0; right: 0; z-index: 9000;
  display: flex; justify-content: center; pointer-events: none;
}
.site-navbar .pill {
  pointer-events: auto;
  display: inline-flex; align-items: center; gap: 2px;
  border-radius: 999px; backdrop-filter: blur(20px);
  border: 1px solid rgba(255,255,255,.1); background: hsl(var(--surface) / .72);
  padding: 6px; box-shadow: 0 8px 30px rgba(0,0,0,.35);
}
.site-navbar .logo {
  width: 34px; height: 34px; border-radius: 999px; display: grid; place-items: center;
  background: linear-gradient(90deg, #F6A6C1, #C9578B); flex-shrink: 0; margin-right: 2px;
}
.site-navbar .logo span {
  width: 29px; height: 29px; border-radius: 999px; background: hsl(var(--bg));
  display: grid; place-items: center; font-family: var(--font-display); font-style: italic;
  font-size: 13px; color: hsl(var(--text));
}
.site-navbar .divider { width: 1px; height: 20px; background: hsl(var(--stroke)); margin: 0 4px; }
.site-navbar a.nav-link {
  font-size: 12.5px; color: hsl(var(--muted)); text-decoration: none; font-weight: 500;
  border-radius: 999px; padding: 7px 13px; white-space: nowrap;
}
.site-navbar a.nav-link:hover { color: hsl(var(--text)); background: hsl(var(--stroke) / .5); }
.site-navbar a.nav-cta {
  position: relative; font-size: 12.5px; font-weight: 600; color: hsl(var(--text));
  text-decoration: none; border-radius: 999px; padding: 7px 15px; margin-left: 2px;
  background: hsl(var(--stroke) / .6); border: 1px solid rgba(255,255,255,.08);
}
.site-navbar a.nav-cta:hover { border-color: transparent; background: linear-gradient(90deg, #F6A6C1, #C9578B); color: #1a0a12; }

/* Mobile hamburger menu -- pure CSS checkbox hack, no JS required */
.nav-toggle-input { display: none; }
.nav-toggle-label { display: none; flex-direction: column; justify-content: center; gap: 4px; width: 34px; height: 34px; cursor: pointer; margin-left: 2px; flex-shrink: 0; }
.nav-toggle-label span { width: 16px; height: 1.5px; background: hsl(var(--text)); margin: 0 auto; transition: transform .25s ease, opacity .25s ease; }
.mobile-menu {
  pointer-events: none; opacity: 0; transform: translateY(-8px); display: none; flex-direction: column;
  position: fixed; top: 64px; left: 16px; right: 16px; z-index: 8999; max-width: 320px; margin: 0 auto;
  background: hsl(var(--surface) / .96); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,.1);
  border-radius: 18px; padding: 10px; box-shadow: 0 20px 50px rgba(0,0,0,.45);
  transition: opacity .25s ease, transform .25s ease;
}
.mobile-menu a { color: hsl(var(--muted)); text-decoration: none; padding: 11px 14px; border-radius: 10px; font-size: 13.5px; font-weight: 600; }
.mobile-menu a:hover { background: hsl(var(--stroke) / .6); color: hsl(var(--text)); }
#nav-toggle:checked ~ .mobile-menu { opacity: 1; pointer-events: auto; transform: translateY(0); }
#nav-toggle:checked ~ .pill .nav-toggle-label span:nth-child(1) { transform: translateY(5.5px) rotate(45deg); }
#nav-toggle:checked ~ .pill .nav-toggle-label span:nth-child(2) { opacity: 0; }
#nav-toggle:checked ~ .pill .nav-toggle-label span:nth-child(3) { transform: translateY(-5.5px) rotate(-45deg); }
@media (max-width: 860px) {
  .site-navbar a.nav-link { display: none; }
  .nav-toggle-label { display: flex; }
  .mobile-menu { display: flex; }
}

/* ------------------------------------------------------------------ Hero */
.hero-section {
  position: relative; margin: -4.5rem -2.4rem 2.5rem; min-height: 70vh;
  display: flex; align-items: center; overflow: hidden; border-radius: 0 0 28px 28px;
}
.hero-bg { position: absolute; inset: 0; background-size: cover; background-position: center; filter: saturate(.85); }
.hero-overlay { position: absolute; inset: 0; background: rgba(5,3,5,.62); }
.hero-glow {
  position: absolute; inset: 0; pointer-events: none;
  background: radial-gradient(60% 50% at 70% 30%, rgba(232,117,165,.22), transparent 70%);
}
.hero-grain {
  position: absolute; inset: 0; opacity: .25; mix-blend-mode: overlay; pointer-events: none;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='120' height='120'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>");
}
.hero-inner { position: relative; z-index: 2; padding: 3rem 2.6rem; display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 2.5rem; align-items: center; width: 100%; }
@media (max-width: 980px) { .hero-inner { grid-template-columns: 1fr; } }
.hero-eyebrow { font-size: 11.5px; letter-spacing: .3em; text-transform: uppercase; color: hsl(var(--muted)); margin-bottom: 20px; animation: fade-up .7s ease both; }
.hero-heading { font-family: var(--font-display); font-style: italic; font-size: clamp(2.6rem, 6vw, 5.2rem); line-height: .98; color: hsl(var(--text)); margin-bottom: 20px; letter-spacing: -.01em; animation: fade-up .8s ease .05s both; }
.hero-desc { color: hsl(var(--muted)); max-width: 440px; font-size: 15px; line-height: 1.6; margin-bottom: 28px; animation: fade-up .8s ease .1s both; }
.hero-ctas { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 18px; animation: fade-up .8s ease .15s both; }
.btn { position: relative; display: inline-flex; align-items: center; gap: 8px; border-radius: 999px; font-size: 13.5px; font-weight: 600; padding: 13px 26px; text-decoration: none; transition: transform .2s ease; }
.btn:hover { transform: scale(1.04); }
.btn-primary { background: hsl(var(--text)); color: hsl(var(--bg)); }
.btn-primary:hover { background: hsl(var(--bg)); color: hsl(var(--text)); box-shadow: inset 0 0 0 1.5px transparent; border: 1.5px solid transparent; background-image: linear-gradient(hsl(var(--bg)), hsl(var(--bg))), linear-gradient(90deg,#F6A6C1,#C9578B); background-origin: border-box; background-clip: padding-box, border-box; }
.btn-outline { border: 1.5px solid hsl(var(--stroke)); color: hsl(var(--text)); background: hsl(var(--bg) / .3); }
.btn-outline:hover { border-color: transparent; background-image: linear-gradient(hsl(var(--bg)), hsl(var(--bg))), linear-gradient(90deg,#F6A6C1,#C9578B); background-origin: border-box; background-clip: padding-box, border-box; }
.hero-fineprint { font-size: 11px; color: hsl(var(--muted)); animation: fade-up .8s ease .2s both; }
.hero-fineprint .dot-sep { margin: 0 6px; opacity: .5; }

.hero-visual { position: relative; animation: fade-up 1s ease .2s both; }
.hero-visual-frame {
  position: relative; border-radius: 20px; overflow: hidden; aspect-ratio: 4/5;
  border: 1px solid rgba(255,255,255,.12); box-shadow: 0 30px 60px rgba(0,0,0,.45);
}
.hero-visual-frame img { width: 100%; height: 100%; object-fit: cover; display: block; filter: grayscale(.25) contrast(1.05); }
.hero-visual-frame::after {
  content: ''; position: absolute; left: 0; right: 0; height: 26%;
  background: linear-gradient(180deg, transparent, rgba(232,117,165,.35), transparent);
  animation: scan-line 3.2s linear infinite;
}
.hero-glass {
  position: absolute; left: 12px; right: 12px; bottom: 12px; border-radius: 14px;
  background: rgba(10,6,9,.55); backdrop-filter: blur(14px); border: 1px solid rgba(255,255,255,.12);
  padding: 14px 16px;
}
.hero-glass-row { display: flex; justify-content: space-between; align-items: center; font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase; color: hsl(var(--muted)); margin-bottom: 8px; }
.hero-glass-row:last-child { margin-bottom: 0; }
.hero-glass-row strong { color: hsl(var(--text)); font-weight: 600; }
.hero-glass-status { display: flex; align-items: center; gap: 6px; }
.status-dot { width: 7px; height: 7px; border-radius: 999px; background: #58d68d; animation: pulse-soft 2s ease infinite; }

/* ----------------------------------------------------------- Sections -- */
.section { padding: 2.2rem 0; }
.section-narrow { max-width: 900px; margin: 0 auto; }
.section-header { margin-bottom: 1.7rem; animation: fade-up .7s ease both; }
.eyebrow-row { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
.eyebrow-line { width: 28px; height: 1px; background: hsl(var(--stroke)); }
.eyebrow { text-transform: uppercase; font-size: 11px; letter-spacing: .28em; color: hsl(var(--muted)); font-weight: 600; }
.section-heading { font-family: var(--font-display); font-size: clamp(1.9rem, 3.4vw, 2.9rem); font-style: normal; line-height: 1.12; color: hsl(var(--text)); }
.section-heading em { font-style: italic; }
.section-sub { color: hsl(var(--muted)); font-size: 14.5px; line-height: 1.65; max-width: 620px; margin-top: 14px; }

.cards-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }
@media (max-width: 900px) { .cards-row { grid-template-columns: 1fr; } }
.intro-card { background: hsl(var(--surface)); border: 1px solid hsl(var(--stroke)); border-radius: 18px; padding: 26px; animation: fade-up .7s ease both; transition: border-color .25s ease, transform .25s ease; }
.intro-card:hover { transform: translateY(-3px); border-image: linear-gradient(90deg,#F6A6C1,#C9578B) 1; }
.intro-card .num { font-family: var(--font-display); font-style: italic; font-size: 30px; color: hsl(var(--accent)); margin-bottom: 14px; }
.intro-card h3 { font-size: 13px; letter-spacing: .12em; color: hsl(var(--text)); margin-bottom: 10px; }
.intro-card p { color: hsl(var(--muted)); font-size: 13px; line-height: 1.6; }

.bento { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
@media (max-width: 980px) { .bento { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 560px) { .bento { grid-template-columns: 1fr; } }
.bento-card { position: relative; background: hsl(var(--surface)); border: 1px solid hsl(var(--stroke)); border-radius: 16px; padding: 18px; transition: transform .35s cubic-bezier(.16,1,.3,1), box-shadow .35s ease, border-color .35s ease; }
.bento-card:hover { transform: translateY(-4px); box-shadow: 0 16px 36px rgba(0,0,0,.35); border-color: transparent; background-image: linear-gradient(hsl(var(--surface)), hsl(var(--surface))), linear-gradient(135deg,#F6A6C1,#C9578B); background-origin: border-box; background-clip: padding-box, border-box; }
.bento-icon { width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center; background: hsl(var(--stroke) / .6); color: hsl(var(--accent)); margin-bottom: 14px; transition: transform .4s cubic-bezier(.34,1.56,.64,1), background .3s ease, color .3s ease; }
.bento-card:nth-child(even) .bento-icon { background: linear-gradient(135deg, rgba(246,166,193,.18), rgba(201,87,139,.18)); }
.bento-card:hover .bento-icon { transform: scale(1.12) rotate(-4deg); background: linear-gradient(135deg,#F6A6C1,#C9578B); color: #1a0a12; }
.bento-icon svg { width: 17px; height: 17px; }
.bento-card strong { display: block; font-size: 13.5px; color: hsl(var(--text)); margin-bottom: 6px; }
.bento-card span { font-size: 11.5px; color: hsl(var(--muted)); line-height: 1.5; }

/* Awareness section: looping background video (assets/awareness_bg.mp4, see
   load_awareness_bg_video_b64()) fills the whole section; all copy (heading,
   intro, symptom list) overlays directly on top of it, legible via the dark
   scrim beneath. */
#awareness { position: relative; overflow: hidden; border-radius: 28px;
  padding: 4.5rem 2.6rem; min-height: 640px; display: flex; align-items: center; }
.awareness-bg-video { position: absolute; inset: -5%; z-index: 0; pointer-events: none;
  width: 110%; height: 110%; object-fit: cover; filter: saturate(1.2) brightness(1.3) contrast(1.05); }
.awareness-bg-video-scrim { position: absolute; inset: 0; z-index: 1; pointer-events: none;
  background: linear-gradient(180deg, rgba(8,4,8,.2), rgba(8,4,8,.38) 55%, rgba(8,4,8,.62));
  backdrop-filter: blur(1px); }
.awareness-content { position: relative; z-index: 2; width: 100%; }
.awareness-content .section-header h2, .awareness-content .section-header p,
.awareness-content .eyebrow { color: #fff !important; text-shadow: 0 2px 14px rgba(0,0,0,.65); }
.awareness-content .section-header p.section-sub { color: rgba(255,255,255,.9) !important; }
.awareness-list { display: grid; grid-template-columns: repeat(2, 1fr); gap: 4px 36px; margin-top: 6px; }
@media (max-width: 820px) { .awareness-list { grid-template-columns: 1fr; } }
.awareness-list-item { display: flex; gap: 14px; align-items: flex-start; padding: 14px 0; border-bottom: 1px solid rgba(255,255,255,.14); }
.awareness-list-icon { width: 30px; height: 30px; border-radius: 9px; flex-shrink: 0; display: grid; place-items: center;
  background: linear-gradient(135deg,#F6A6C1,#C9578B); color: #1a0a12; margin-top: 2px; }
.awareness-list-icon svg { width: 15px; height: 15px; }
.awareness-list-item strong { display: block; font-size: 13.5px; color: #fff; margin-bottom: 3px; text-shadow: 0 2px 10px rgba(0,0,0,.7); }
.awareness-list-item span { font-size: 12px; color: rgba(255,255,255,.88); line-height: 1.5; text-shadow: 0 1px 6px rgba(0,0,0,.55); }
.awareness-footnote { color: rgba(255,255,255,.75) !important; text-shadow: 0 1px 6px rgba(0,0,0,.55); }

.pipeline { display: flex; align-items: flex-start; gap: 0; }
@media (max-width: 900px) { .pipeline { flex-direction: column; gap: 22px; } }
.pipeline-step { flex: 1; position: relative; padding: 0 14px; text-align: center; }
@media (max-width: 900px) { .pipeline-step { text-align: left; padding: 0 0 0 20px; border-left: 1px solid hsl(var(--stroke)); } }
.pipeline-step:not(:last-child)::after {
  content: ''; position: absolute; top: 19px; left: 56%; right: -44%; height: 1px;
  background: linear-gradient(90deg, hsl(var(--stroke)), hsl(var(--accent) / .5));
}
@media (max-width: 900px) { .pipeline-step:not(:last-child)::after { display: none; } }
.pipeline-num { width: 38px; height: 38px; border-radius: 999px; margin: 0 auto 14px; display: grid; place-items: center; font-size: 12px; font-weight: 700; position: relative; z-index: 1; background: hsl(var(--surface)); border: 1px solid hsl(var(--stroke)); color: hsl(var(--muted)); transition: box-shadow .3s ease, transform .3s ease; }
.pipeline-step:nth-child(odd) .pipeline-num { background: linear-gradient(135deg,#F6A6C1,#C9578B); color: #1a0a12; border-color: transparent; }
@media (max-width: 900px) { .pipeline-num { margin: -29px 0 10px -29px; } }
.pipeline-step strong { display: block; font-size: 12.5px; letter-spacing: .1em; color: hsl(var(--text)); margin-bottom: 6px; }
.pipeline-step p { font-size: 12px; color: hsl(var(--muted)); line-height: 1.55; }
/* Pipeline step activation driven by GSAP ScrollTrigger (see render_scroll_effects) */
.pipeline-step.step-active .pipeline-num { box-shadow: 0 0 0 4px rgba(232,117,165,.22), 0 0 26px rgba(232,117,165,.55); transform: scale(1.08); }
.pipeline-step:not(:last-child)::after { transition: background .4s ease; }
.pipeline-step.step-active:not(:last-child)::after { background: linear-gradient(90deg, #E875A5, #C9578B); }

.dataset-panel { background: hsl(var(--surface)); border: 1px solid hsl(var(--stroke)); border-radius: 20px; padding: 32px; display: grid; grid-template-columns: 0.9fr 1.1fr; gap: 28px; align-items: center; }
@media (max-width: 900px) { .dataset-panel { grid-template-columns: 1fr; } }
.dataset-name { font-family: var(--font-display); font-style: italic; font-size: clamp(2.4rem, 5vw, 3.6rem); color: hsl(var(--text)); }
.dataset-desc { color: hsl(var(--muted)); font-size: 13.5px; line-height: 1.65; margin-top: 10px; }
.dataset-metrics { display: flex; gap: 22px; margin-top: 18px; flex-wrap: wrap; }
.dataset-metric strong { display: block; font-family: var(--font-display); font-style: italic; font-size: 26px; color: hsl(var(--accent)); }
.dataset-metric span { font-size: 10.5px; color: hsl(var(--muted)); text-transform: uppercase; letter-spacing: .08em; }
.dataset-tags { display: flex; flex-wrap: wrap; gap: 10px; }
.dataset-tag { border: 1px solid hsl(var(--stroke)); border-radius: 999px; padding: 9px 16px; font-size: 11px; letter-spacing: .08em; color: hsl(var(--muted)); text-align: center; }

/* ------------------------------------------------------ Screening panel */
.page-header { margin-bottom: 22px; }
.panel { background: hsl(var(--surface)); border: 1px solid hsl(var(--stroke)); border-radius: 18px; padding: 24px; margin-bottom: 20px; }
.panel h2 { font-size: 16px; color: hsl(var(--text)); margin-bottom: 4px; }
.panel-intro { color: hsl(var(--muted)); font-size: 12px; margin: 5px 0 20px; }

.notice { display: flex; gap: 12px; align-items: flex-start; background: hsl(var(--stroke) / .4); border: 1px solid hsl(var(--stroke)); padding: 13px 15px; border-radius: 12px; margin-bottom: 20px; }
.notice strong { font-size: 13px; color: hsl(var(--text)); display: block; margin-bottom: 2px; }
.notice span { color: hsl(var(--muted)); font-size: 12px; }

.stepper { display: flex; align-items: center; margin-bottom: 24px; max-width: 820px; flex-wrap: wrap; gap: 4px 0; }
.step { display: flex; align-items: center; gap: 8px; color: hsl(var(--muted)); font-size: 12px; flex: 1; min-width: 150px; }
.step:not(:last-child):after { content: ''; height: 1px; background: hsl(var(--stroke)); flex: 1; margin: 0 10px; min-width: 20px; }
.step span { width: 24px; height: 24px; display: grid; place-items: center; border-radius: 50%; border: 1px solid hsl(var(--stroke)); color: hsl(var(--muted)); font-size: 11px; flex-shrink: 0; }
.step.active { color: hsl(var(--text)); font-weight: 700; }
.step.active span { background: linear-gradient(135deg,#F6A6C1,#C9578B); border-color: transparent; color: #1a0a12; }
.step.done span { background: hsl(var(--stroke)); border-color: hsl(var(--stroke)); color: hsl(var(--text)); }
.step.done { color: hsl(var(--text)); }

.status { border-radius: 20px; padding: 4px 11px; font-size: 10.5px; font-weight: 700; display: inline-block; }
.status.amber { background: rgba(232,117,165,.16); color: #f0a9c6; }
.status.green { background: rgba(88,214,141,.14); color: #6fdba3; }

/* Upload preview (post-upload, pre-analyze) */
.upload-preview-grid { display: grid; grid-template-columns: 150px 1fr; gap: 20px; align-items: center; }
@media (max-width: 600px) { .upload-preview-grid { grid-template-columns: 1fr; } }
.upload-preview-thumb img { width: 100%; border-radius: 12px; display: block; border: 1px solid hsl(var(--stroke)); }
.upload-preview-grid .summary-row:first-of-type { border-top: none; }

/* Analyzing state -- scan-line over the actual uploaded image */
.analyzing-visual { position: relative; border-radius: 14px; overflow: hidden; max-width: 420px; margin: 16px 0 6px; border: 1px solid hsl(var(--stroke)); }
.analyzing-visual img { width: 100%; display: block; filter: grayscale(.2) brightness(.92); }
.analyzing-visual .scan-overlay { position: absolute; left: 0; right: 0; height: 24%; background: linear-gradient(180deg, transparent, rgba(232,117,165,.45), transparent); animation: scan-line 1.6s linear infinite; }

.result-grid { display: grid; grid-template-columns: 1.3fr 0.85fr; gap: 20px; margin-bottom: 4px; }
@media (max-width: 900px) { .result-grid { grid-template-columns: 1fr; } }
.signal-top { display: flex; align-items: center; gap: 22px; flex-wrap: wrap; }
.signal-ring { width: 112px; height: 112px; border-radius: 50%; display: flex; flex-direction: column; align-items: center; justify-content: center; flex-shrink: 0; background: conic-gradient(var(--ring-color) calc(var(--pct) * 1%), hsl(var(--stroke)) 0); }
.signal-ring-inner { width: 90px; height: 90px; border-radius: 50%; background: hsl(var(--surface)); display: flex; flex-direction: column; align-items: center; justify-content: center; }
.signal-ring strong { font-family: var(--font-display); font-style: italic; font-size: 26px; color: hsl(var(--text)); line-height: 1; }
.signal-ring span { color: hsl(var(--muted)); font-size: 9.5px; margin-top: 3px; text-align: center; }
.signal-top h2 { font-family: var(--font-display); font-style: italic; font-weight: 400; font-size: 22px; margin-bottom: 6px; color: hsl(var(--text)); }
.signal-top p { color: hsl(var(--muted)); line-height: 1.6; font-size: 13px; }
.signal-bar { height: 9px; background: hsl(var(--stroke)); border-radius: 10px; margin-top: 24px; overflow: hidden; }
.signal-bar div { height: 100%; border-radius: 10px; }
.bar-labels { display: flex; justify-content: space-between; color: hsl(var(--muted)); font-size: 10px; margin-top: 6px; }
.result-callout { margin-top: 20px; display: flex; gap: 9px; align-items: flex-start; padding: 12px; background: hsl(var(--stroke) / .4); color: hsl(var(--muted)); font-size: 11.5px; line-height: 1.5; border-radius: 10px; }
.result-callout strong { color: hsl(var(--text)); }

.summary-row { border-top: 1px solid hsl(var(--stroke)); padding: 11px 0; display: flex; justify-content: space-between; gap: 10px; color: hsl(var(--muted)); font-size: 11.5px; }
.summary-row:first-of-type { border-top: none; }
.summary-row strong { color: hsl(var(--text)); font-size: 12px; text-align: right; }

.explanation-layout { display: grid; grid-template-columns: 0.95fr 1.05fr; gap: 20px; }
@media (max-width: 900px) { .explanation-layout { grid-template-columns: 1fr; } }
.image-panel img { width: 100%; border-radius: 12px; display: block; }
.image-caption { display: flex; justify-content: space-between; align-items: center; padding: 12px 2px 0; color: hsl(var(--muted)); font-size: 11px; }
.legend-dot { display: inline-block; width: 8px; height: 8px; border: 1px solid #e875a5; background: #e875a566; border-radius: 50%; margin-right: 5px; }

.factor { display: grid; grid-template-columns: 1fr 130px 40px; gap: 16px; align-items: center; margin: 18px 0; }
.factor > div:first-child { display: flex; flex-direction: column; }
.factor strong { font-size: 12px; color: hsl(var(--text)); }
.factor span { color: hsl(var(--muted)); font-size: 10px; margin-top: 2px; }
.factor-meter { height: 7px; border-radius: 10px; background: hsl(var(--stroke)); overflow: hidden; }
.factor-meter i { display: block; height: 100%; border-radius: 10px; }
.factor b { font-size: 11px; color: hsl(var(--accent)); text-align: right; }
.review-box { margin-top: 18px; display: flex; gap: 9px; align-items: flex-start; padding: 12px; background: hsl(var(--stroke) / .4); color: hsl(var(--muted)); font-size: 11.5px; line-height: 1.5; border-radius: 10px; }
.review-box strong { color: hsl(var(--text)); display: block; margin-bottom: 2px; }

/* Printable report stays legible on paper regardless of the dark theme. */
.report-sheet { max-width: 900px; padding: 30px; background: #fff !important; color: #18313a; border: 1px solid hsl(var(--stroke)); }
.report-sheet * { color: #18313a; }
.report-brand { display: flex; justify-content: space-between; align-items: center; padding-bottom: 20px; border-bottom: 1px solid #dce7e9; font-size: 11px; color: #6c7d82 !important; }
.report-brand .brand { font-size: 15px; }
.report-title { display: flex; justify-content: space-between; align-items: flex-start; margin: 24px 0 18px; flex-wrap: wrap; gap: 10px; }
.report-title h2 { font-size: 21px; }
.report-meta { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; background: #f7fafa; border: 1px solid #e4eeee; border-radius: 7px; padding: 14px; }
@media (max-width: 900px) { .report-meta { grid-template-columns: repeat(2, 1fr); } }
.report-meta div { display: flex; flex-direction: column; gap: 3px; }
.report-meta span { color: #829194 !important; font-size: 10px; }
.report-signal { display: flex; align-items: center; justify-content: space-between; margin-top: 20px; padding: 16px; border-radius: 8px; gap: 12px; flex-wrap: wrap; }
.report-signal.amber-bg { background: #fff8e9; border: 1px solid #f0dfba; }
.report-signal.green-bg { background: #eaf8f2; border: 1px solid #c5edd9; }
.report-signal strong { font-size: 16px; color: #604c27 !important; }
.report-signal.green-bg strong { color: #1c5e4b !important; }
.report-signal p { color: #8d784e !important; font-size: 11px; margin-top: 3px; }
.report-signal.green-bg p { color: #39856f !important; }
.signal-pill { width: 58px; height: 58px; border-radius: 50%; display: grid; place-items: center; border: 5px solid #e8c77f; color: #835f1f !important; font-size: 15px; font-weight: 750; flex-shrink: 0; }
.signal-pill.green { border-color: #79cbb2; color: #1c5e4b !important; }
.report-section { margin-top: 22px; }
.report-section h3 { margin-bottom: 8px; font-size: 13px; font-weight: 700; }
.report-section p, .report-section li { color: #687a7e !important; font-size: 12px; line-height: 1.7; }
.report-section ul { padding-left: 20px; margin: 0; }
.report-factors-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 8px; }
@media (max-width: 900px) { .report-factors-grid { grid-template-columns: 1fr; } }
.report-factor-item { background: #f8fbfb; border: 1px solid #dce7e9; padding: 12px; border-radius: 6px; display: flex; flex-direction: column; gap: 4px; }
.report-factor-item span { font-size: 10px; color: #7f9194 !important; }
.report-factor-item strong { font-size: 15px; }
.report-footer { margin-top: 30px; border-top: 1px solid #dce7e9; padding-top: 14px; color: #8b999b !important; font-size: 10px; }
@media print { .site-navbar, section[data-testid="stSidebar"], .disclaimer-footer { display: none !important; } }

/* --------------------------------------------------------- Upload panel */
[data-testid="stFileUploaderDropzone"] {
  background: hsl(var(--stroke) / .35) !important;
  border: 1.5px dashed hsl(var(--stroke)) !important;
  border-radius: 14px !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: #e875a5 !important; }
[data-testid="stFileUploaderDropzone"] button {
  background: linear-gradient(90deg,#F6A6C1,#C9578B) !important;
  color: #1a0a12 !important; font-weight: 700 !important;
  border-radius: 999px !important; border: none !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] * { color: hsl(var(--muted)) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] svg { color: hsl(var(--accent)) !important; }
div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button {
  background: linear-gradient(90deg,#F6A6C1,#C9578B);
  color: #1a0a12; border: none; border-radius: 999px; font-weight: 700; padding: 0.55rem 1.2rem;
}
div[data-testid="stButton"] button:hover { filter: brightness(1.08); }
[data-testid="stExpander"] { background: hsl(var(--surface)); border: 1px solid hsl(var(--stroke)); border-radius: 14px; }
[data-testid="stExpander"] summary { color: hsl(var(--text)) !important; }
[data-testid="stMarkdownContainer"] p { color: inherit; }
[data-testid="stStatusWidget"], div[data-testid="stStatus"] { background: hsl(var(--surface)) !important; border: 1px solid hsl(var(--stroke)) !important; border-radius: 14px !important; }
div[data-testid="stStatus"] p { color: hsl(var(--text)) !important; }

/* Fixed disclaimer strip under the screening tool only (not full-page). */
.disclaimer-footer {
  display: flex; gap: 8px; align-items: center; justify-content: center;
  padding: 12px 20px; background: hsl(var(--stroke) / .4); border: 1px solid hsl(var(--stroke));
  border-radius: 12px; color: hsl(var(--muted)); font-size: 11px; margin: 6px 0 24px;
}
.disclaimer-footer strong { color: hsl(var(--text)); }

/* Form widgets (login) */
div[data-testid="stTextInput"] input {
  border-radius: 10px !important; border: 1px solid hsl(var(--stroke)) !important;
  padding: 0.6rem 0.75rem !important; background: hsl(var(--bg)) !important; color: hsl(var(--text)) !important;
}
div[data-testid="stTextInput"] input:focus { border-color: #e875a5 !important; box-shadow: 0 0 0 2px rgba(232,117,165,.25) !important; }
div[data-testid="stTextInput"] label { font-size: 12.5px !important; font-weight: 600 !important; color: hsl(var(--text)) !important; }
div[data-testid="stFormSubmitButton"] button {
  background: linear-gradient(90deg,#F6A6C1,#C9578B) !important; color: #1a0a12 !important; border: none !important;
  border-radius: 999px !important; font-weight: 700 !important; width: 100%; padding: 0.65rem 1.1rem !important;
}
div[data-testid="stCheckbox"] label p { font-size: 12px !important; color: hsl(var(--muted)) !important; }

/* Form widgets (patient / mammogram info) */
div[data-testid="stNumberInput"] input {
  border-radius: 10px !important; border: 1px solid hsl(var(--stroke)) !important;
  background: hsl(var(--bg)) !important; color: hsl(var(--text)) !important;
}
div[data-testid="stSelectbox"] > label, div[data-testid="stNumberInput"] > label,
div[data-testid="stRadio"] > label, div[data-testid="stTextInput"] > label {
  font-size: 12.5px !important; font-weight: 600 !important; color: hsl(var(--text)) !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  border-radius: 10px !important; border: 1px solid hsl(var(--stroke)) !important;
  background: hsl(var(--bg)) !important; color: hsl(var(--text)) !important;
}
div[data-testid="stRadio"] label p { color: hsl(var(--text)) !important; font-size: 13px !important; }

.st-key-patient_info_card, .st-key-mammogram_info_card {
  border-radius: 18px !important; border: 1px solid hsl(var(--stroke)) !important;
  background: hsl(var(--surface)) !important; padding: 22px 24px !important; margin-bottom: 20px !important;
}

/* Login screen */
.login-topspace { height: 2vh; }
.st-key-login_card {
  max-width: 440px; margin: 0 auto; border-radius: 20px !important;
  border: 1px solid hsl(var(--stroke)) !important; box-shadow: 0 24px 60px rgba(0,0,0,.45);
  padding: 30px 28px !important; background: hsl(var(--surface)) !important;
}
.login-badge { display: flex; justify-content: center; gap: 7px; align-items: center; color: hsl(var(--accent)); font-size: 10.5px; font-weight: 700; letter-spacing: .5px; text-transform: uppercase; margin-bottom: 18px; }
.login-logo { display: flex; flex-direction: column; align-items: center; gap: 10px; margin-bottom: 22px; text-align: center; }
.login-logo .brand-mark { width: 50px; height: 50px; border-radius: 16px; background: linear-gradient(135deg,#F6A6C1,#C9578B); color: #1a0a12; display: grid; place-items: center; font-size: 22px; }
.login-logo .brand-title { font-family: var(--font-display); font-style: italic; font-weight: 400; color: hsl(var(--text)); font-size: 26px; letter-spacing: -.01em; }
.login-logo p { color: hsl(var(--muted)); font-size: 12.5px; margin-top: 6px; line-height: 1.5; }
.login-divider { border-top: 1px solid hsl(var(--stroke)); margin: 6px 0 18px; }
.login-error { background: rgba(232,117,165,.12); border: 1px solid rgba(232,117,165,.35); color: #f0a9c6; padding: 10px 12px; border-radius: 10px; font-size: 12px; margin-bottom: 14px; display: flex; gap: 8px; align-items: flex-start; line-height: 1.5; }
.login-footer-note { text-align: center; margin-top: 16px; color: hsl(var(--muted)); font-size: 10.5px; line-height: 1.5; }
.login-trust-row { display: flex; justify-content: center; gap: 18px; margin-top: 22px; color: hsl(var(--muted)); font-size: 10.5px; }
.login-trust-row span { display: flex; align-items: center; gap: 5px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: hsl(var(--surface)); border-right: 1px solid hsl(var(--stroke)); }
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }
.brand { display: flex; align-items: center; gap: 9px; font-family: var(--font-display); font-style: italic; color: hsl(var(--text)); font-size: 19px; }
.brand-mark { width: 32px; height: 32px; border-radius: 10px; background: linear-gradient(135deg,#F6A6C1,#C9578B); color: #1a0a12; display: grid; place-items: center; font-size: 16px; flex-shrink: 0; }
.prototype-badge { margin: 20px 2px 18px; color: hsl(var(--accent)); font-size: 11px; font-weight: 700; letter-spacing: .4px; text-transform: uppercase; display: flex; gap: 7px; align-items: center; }
.dot { width: 6px; height: 6px; border-radius: 50%; background: #58d68d; display: inline-block; }
.nav-list { display: flex; flex-direction: column; gap: 4px; margin-bottom: 22px; }
.nav-item { display: flex; align-items: center; gap: 10px; color: hsl(var(--muted)); text-decoration: none; padding: 9px 10px; border-radius: 10px; font-weight: 600; font-size: 13px; }
.nav-item:hover { background: hsl(var(--stroke) / .6); color: hsl(var(--text)); }
.profile { border-top: 1px solid hsl(var(--stroke)); padding: 14px 2px 0; display: flex; align-items: center; gap: 9px; }
.avatar { background: hsl(var(--stroke)); color: hsl(var(--accent)); border-radius: 50%; width: 32px; height: 32px; display: grid; place-items: center; font-size: 11px; font-weight: 750; flex-shrink: 0; }
.profile strong { font-size: 12px; color: hsl(var(--text)); display: block; }
.profile span { color: hsl(var(--muted)); font-size: 11px; }

/* --------------------------------------------------------- Final + footer */
.final-cta { position: relative; border-radius: 28px; overflow: hidden; padding: 5rem 2.5rem; text-align: center; margin-top: 1rem; }
.final-cta-bg { position: absolute; inset: 0; background-size: cover; background-position: center; filter: grayscale(.4); }
.final-cta-overlay { position: absolute; inset: 0; background: rgba(5,3,5,.78); }
.final-cta-content { position: relative; z-index: 1; }
.final-cta h2 { font-family: var(--font-display); font-style: italic; font-size: clamp(2rem, 5vw, 3.4rem); color: hsl(var(--text)); line-height: 1.1; }
.final-cta p { color: hsl(var(--muted)); max-width: 520px; margin: 18px auto 30px; font-size: 14.5px; line-height: 1.6; }
.final-cta .hero-ctas { justify-content: center; margin-bottom: 0; }

.site-footer { padding: 3rem 0 1.5rem; border-top: 1px solid hsl(var(--stroke)); margin-top: 2rem; }
.footer-top { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 24px; margin-bottom: 28px; }
.footer-logo { font-family: var(--font-display); font-style: italic; font-size: 28px; color: hsl(var(--text)); }
.footer-desc { color: hsl(var(--muted)); font-size: 12.5px; max-width: 280px; margin-top: 10px; line-height: 1.6; }
.footer-links { display: flex; flex-wrap: wrap; gap: 10px 26px; }
.footer-links a { color: hsl(var(--muted)); text-decoration: none; font-size: 12.5px; }
.footer-links a:hover { color: hsl(var(--text)); }
.footer-bottom { border-top: 1px solid hsl(var(--stroke)); padding-top: 18px; color: hsl(var(--muted)); font-size: 11px; text-align: center; }
.footer-bottom strong { color: hsl(var(--text)); }

/* ------------------------------------------------------------------------
   Streamlit injects its own default text color on markdown h1-h4/p tags
   with higher CSS specificity than a bare single-class selector, which
   silently overrode most of the headings/paragraphs above (div/span/strong
   were unaffected). These !important rules, scoped under .stApp with extra
   class qualifiers for specificity, force our theme's colors back. */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p { color: hsl(var(--text)) !important; }
.stApp p.hero-desc, .stApp p.section-sub, .stApp p.dataset-desc, .stApp p.panel-intro,
.stApp p.footer-desc, .stApp .intro-card p, .stApp .pipeline-step p,
.stApp .final-cta p, .stApp .guideline-card p, .stApp .myth-fact-card .fact-text {
  color: hsl(var(--muted)) !important;
}
.stApp .report-sheet, .stApp .report-sheet h2, .stApp .report-sheet h3,
.stApp .report-sheet p, .stApp .report-sheet div, .stApp .report-sheet span,
.stApp .report-sheet strong, .stApp .report-sheet li { color: #18313a !important; }
.stApp .report-sheet .eyebrow { color: #c9578b !important; }
.stApp .report-brand, .stApp .report-brand span { color: #6c7d82 !important; }
.stApp .report-meta span, .stApp .report-factor-item span, .stApp .report-footer { color: #829194 !important; }
.stApp .report-section p, .stApp .report-section li { color: #687a7e !important; }
.stApp .report-signal.amber-bg strong { color: #604c27 !important; }
.stApp .report-signal.amber-bg p { color: #8d784e !important; }
.stApp .report-signal.green-bg strong { color: #1c5e4b !important; }
.stApp .report-signal.green-bg p { color: #39856f !important; }
.stApp .signal-pill { color: #835f1f !important; }
.stApp .signal-pill.green { color: #1c5e4b !important; }

/* Same issue applies to <a> tags: Streamlit's markdown-link styling
   (blue, underlined) otherwise beats our button/nav-link classes. */
.stApp a.btn-primary { color: hsl(var(--bg)) !important; text-decoration: none !important; }
.stApp a.btn-primary:hover { color: hsl(var(--text)) !important; }
.stApp a.btn-outline { color: hsl(var(--text)) !important; text-decoration: none !important; }
.stApp a.nav-link { color: hsl(var(--muted)) !important; text-decoration: none !important; }
.stApp a.nav-link:hover { color: hsl(var(--text)) !important; }
.stApp a.nav-cta { color: hsl(var(--text)) !important; text-decoration: none !important; }
.stApp a.nav-cta:hover { color: #1a0a12 !important; }
.stApp .footer-links a { color: hsl(var(--muted)) !important; text-decoration: none !important; }
.stApp .footer-links a:hover { color: hsl(var(--text)) !important; }
.stApp .site-navbar a.logo { text-decoration: none !important; }
.stApp a.nav-item { color: hsl(var(--muted)) !important; text-decoration: none !important; }
.stApp a.nav-item:hover { color: hsl(var(--text)) !important; }
.stApp .mobile-menu a { color: hsl(var(--muted)) !important; text-decoration: none !important; }
.stApp .mobile-menu a:hover { color: hsl(var(--text)) !important; }

/* ----------------------------------------------------- Education Center */
.quick-nav { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 24px; }
.quick-nav-link {
  font-size: 12px; font-weight: 600; border: 1px solid hsl(var(--stroke)); border-radius: 999px;
  padding: 8px 16px; background: hsl(var(--surface)); transition: border-color .2s ease, color .2s ease;
}
.stApp a.quick-nav-link { color: hsl(var(--muted)) !important; text-decoration: none !important; }
.stApp a.quick-nav-link:hover { color: hsl(var(--text)) !important; border-color: #e875a5; }

.guideline-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }
@media (max-width: 900px) { .guideline-row { grid-template-columns: 1fr; } }
.guideline-card { background: hsl(var(--surface)); border: 1px solid hsl(var(--stroke)); border-radius: 18px; padding: 24px; }
.guideline-card .age { font-family: var(--font-display); font-style: italic; font-size: 30px; color: hsl(var(--accent)); margin-bottom: 10px; }
.guideline-card h3 { font-size: 13.5px; color: hsl(var(--text)); margin-bottom: 8px; line-height: 1.4; }
.guideline-card p { color: hsl(var(--muted)); font-size: 12.5px; line-height: 1.6; }

.myth-fact-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
@media (max-width: 900px) { .myth-fact-grid { grid-template-columns: 1fr; } }
.myth-fact-card { background: hsl(var(--surface)); border: 1px solid hsl(var(--stroke)); border-radius: 16px; padding: 20px 22px; }
.myth-fact-card .tag { font-size: 10px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; padding: 3px 10px; border-radius: 999px; display: inline-block; margin-bottom: 10px; }
.myth-fact-card .tag.myth { background: rgba(232,117,165,.16); color: #f0a9c6; }
.myth-fact-card .tag.fact { background: rgba(88,214,141,.14); color: #6fdba3; }
.myth-fact-card .myth-text { color: hsl(var(--text)); font-size: 13.5px; font-style: italic; margin-bottom: 12px; line-height: 1.5; }
.myth-fact-card .fact-text { color: hsl(var(--muted)); font-size: 12.5px; line-height: 1.6; }

.faq-answer { color: hsl(var(--muted)); font-size: 13px; line-height: 1.7; padding: 2px 4px 6px; }

/* ------------------------------------------------- Sidebar section nav
   (Patient Portal). Scoped under .st-key-sidebar-nav so it overrides the
   app-wide pink-pill button style (div[data-testid="stButton"] button)
   without touching buttons anywhere else -- the extra ancestor class gives
   it higher specificity, no !important needed. Each item's own wrapper key
   carries an "-active" suffix only for the current section, which is what
   the gradient-highlight rule below keys off. */
.st-key-sidebar-nav { margin-bottom: 18px; }
.st-key-sidebar-nav div[data-testid="stVerticalBlock"] { gap: 2px; }
.st-key-sidebar-nav div[data-testid="stButton"] button {
  background: transparent; color: hsl(var(--muted)); border: none; box-shadow: none;
  font-weight: 600; font-size: 13px; padding: 9px 12px; border-radius: 10px;
  justify-content: flex-start; text-align: left; width: 100%;
}
.st-key-sidebar-nav div[data-testid="stButton"] button:hover {
  background: hsl(var(--stroke) / .6); color: hsl(var(--text)); filter: none;
}
.st-key-sidebar-nav div[data-testid="stButton"] button p,
.st-key-sidebar-nav div[data-testid="stButton"] button span { font-weight: 600; font-size: 13px; color: hsl(var(--muted)) !important; }
.st-key-sidebar-nav div[data-testid="stButton"] button:hover p,
.st-key-sidebar-nav div[data-testid="stButton"] button:hover span { color: hsl(var(--text)) !important; }
.st-key-sidebar-nav [class*="-active"] div[data-testid="stButton"] button {
  background: linear-gradient(90deg,#F6A6C1,#C9578B); color: #1a0a12;
}
.st-key-sidebar-nav [class*="-active"] div[data-testid="stButton"] button p,
.st-key-sidebar-nav [class*="-active"] div[data-testid="stButton"] button span { color: #1a0a12 !important; }
.st-key-sidebar-nav [class*="-active"] div[data-testid="stButton"] button:hover { filter: brightness(1.05); }

/* AI Assistant chat bubbles (floating widget, /AI_Assistant page, Patient
   Portal tab) -- overrides Streamlit's default chat-message styling, whose
   stock red/orange avatar colors clash with this app's pink/dark theme. */
div[data-testid="stChatMessage"] {
  background: hsl(var(--surface) / .6); border: 1px solid hsl(var(--stroke) / .7);
  border-radius: 14px; padding: 12px 14px;
}
div[data-testid="stChatMessageAvatarUser"],
div[data-testid="stChatMessageAvatarAssistant"] {
  background: linear-gradient(135deg,#F6A6C1,#C9578B) !important;
  box-shadow: 0 2px 8px rgba(201,87,139,.35);
}
div[data-testid="stChatMessageAvatarUser"] span[data-testid="stIconMaterial"],
div[data-testid="stChatMessageAvatarAssistant"] span[data-testid="stIconMaterial"] {
  color: #1a0a12 !important;
}
div[data-testid="stChatMessageContent"] p {
  color: hsl(var(--text)); font-size: 13.5px; line-height: 1.6;
}
</style>
"""

# Standalone CSS for the downloadable report .html export -- the in-app report
# sheet relies on .stApp-scoped !important overrides (see CSS above) that only
# exist inside the Streamlit page, so the exported file needs its own copy.
REPORT_EXPORT_CSS = """
body { font-family: Arial, Helvetica, sans-serif; background:#f3f6f6; color:#18313a; margin:0; padding:32px 16px; }
.report-sheet { max-width:860px; margin:0 auto; padding:30px; background:#fff; border:1px solid #dce7e9; border-radius:8px; }
.report-sheet * { color:#18313a; box-sizing:border-box; }
.report-brand { display:flex; justify-content:space-between; align-items:center; padding-bottom:20px; border-bottom:1px solid #dce7e9; font-size:11px; color:#6c7d82; }
.report-brand .brand { font-size:15px; display:flex; align-items:center; gap:8px; }
.brand-mark { display:inline-flex; align-items:center; justify-content:center; width:24px; height:24px; font-size:13px; border-radius:7px; background:linear-gradient(135deg,#F6A6C1,#C9578B); }
.report-title { display:flex; justify-content:space-between; align-items:flex-start; margin:24px 0 18px; flex-wrap:wrap; gap:10px; }
.report-title h2 { font-size:21px; margin:4px 0 0; }
.eyebrow { text-transform:uppercase; font-size:11px; letter-spacing:.14em; color:#c9578b; font-weight:700; }
.status { border-radius:20px; padding:4px 11px; font-size:10.5px; font-weight:700; display:inline-block; }
.status.amber { background:#fdf0f5; color:#b24a74; }
.status.green { background:#eaf8f2; color:#1c5e4b; }
.report-meta { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; background:#f7fafa; border:1px solid #e4eeee; border-radius:7px; padding:14px; }
.report-meta div { display:flex; flex-direction:column; gap:3px; }
.report-meta span { color:#829194; font-size:10px; }
.report-meta strong { font-size:12.5px; }
.report-signal { display:flex; align-items:center; justify-content:space-between; margin-top:20px; padding:16px; border-radius:8px; gap:12px; flex-wrap:wrap; }
.report-signal.amber-bg { background:#fff8e9; border:1px solid #f0dfba; }
.report-signal.green-bg { background:#eaf8f2; border:1px solid #c5edd9; }
.report-signal strong { font-size:16px; color:#604c27; }
.report-signal.green-bg strong { color:#1c5e4b; }
.report-signal p { color:#8d784e; font-size:11px; margin:3px 0 0; }
.report-signal.green-bg p { color:#39856f; }
.signal-pill { width:58px; height:58px; border-radius:50%; display:flex; align-items:center; justify-content:center; border:5px solid #e8c77f; color:#835f1f; font-size:15px; font-weight:750; flex-shrink:0; }
.signal-pill.green { border-color:#79cbb2; color:#1c5e4b; }
.report-section { margin-top:22px; }
.report-section h3 { margin:0 0 8px; font-size:13px; font-weight:700; }
.report-section p, .report-section li { color:#687a7e; font-size:12px; line-height:1.7; margin:0; }
.report-factors-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:8px; }
.report-factor-item { background:#f8fbfb; border:1px solid #dce7e9; padding:12px; border-radius:6px; display:flex; flex-direction:column; gap:4px; }
.report-factor-item span { font-size:10px; color:#7f9194; }
.report-factor-item strong { font-size:15px; }
.report-footer { margin-top:30px; border-top:1px solid #dce7e9; padding-top:14px; color:#8b999b; font-size:10px; }
"""


def render_intro_loader() -> None:
    inject_spa_navigation()
    """Cinematic full-screen loading overlay, injected into the OUTER page.

    st.components.v1.html renders in a same-origin sandboxed iframe; the
    script reaches into window.parent.document to show/remove the overlay
    on the real page, covering the entire viewport (including Streamlit's
    own header) until it fades out and reveals the landing page. Plays on
    every load of the landing page -- including navigating back to it via
    "Home" / the logo -- not just the first visit in a browser tab. Skipped
    entirely under prefers-reduced-motion.
    """
    st.components.v1.html(
        """
        <script>
        (function () {
          try {
            var doc = window.parent.document;
            var win = window.parent;
            if (doc.getElementById('oncosense-intro')) return;
            if (win.matchMedia && win.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

            var style = doc.createElement('style');
            style.innerHTML =
              "#oncosense-intro{position:fixed;inset:0;z-index:999999;background:hsl(330 20% 5%);" +
              "display:flex;flex-direction:column;justify-content:space-between;padding:28px 36px;font-family:'Inter',sans-serif;}" +
              "#oncosense-intro .oi-top{color:hsl(330 8% 65%);font-size:11px;letter-spacing:.3em;text-transform:uppercase;" +
              "opacity:0;animation:oiFadeDown .6s ease forwards;}" +
              "#oncosense-intro .oi-brand{color:hsl(330 20% 97%);font-weight:700;letter-spacing:.2em;font-size:12px;}" +
              "#oncosense-intro .oi-center{flex:1;display:flex;align-items:center;justify-content:center;}" +
              "#oncosense-intro .oi-word{font-family:'Instrument Serif',serif;font-style:italic;font-size:64px;" +
              "color:hsl(330 20% 97% / .9);}" +
              "#oncosense-intro .oi-count{font-family:'Instrument Serif',serif;font-size:88px;color:hsl(330 20% 97%);" +
              "font-variant-numeric:tabular-nums;align-self:flex-end;}" +
              "#oncosense-intro .oi-bar{position:absolute;left:0;right:0;bottom:0;height:3px;background:hsl(330 12% 18%);}" +
              "#oncosense-intro .oi-bar-fill{height:100%;width:0%;background:linear-gradient(90deg,#F6A6C1,#E875A5,#C9578B);" +
              "box-shadow:0 0 12px rgba(232,117,165,.5);}" +
              "@keyframes oiFadeDown{from{opacity:0;transform:translateY(-10px);}to{opacity:1;transform:translateY(0);}}" +
              "@keyframes oiWord{from{opacity:0;transform:translateY(12px);}to{opacity:1;transform:translateY(0);}}";
            doc.head.appendChild(style);

            var el = doc.createElement('div');
            el.id = 'oncosense-intro';
            el.innerHTML =
              '<div class="oi-top"><span class="oi-brand">ONCOSENSE</span>' +
              '&nbsp;&middot;&nbsp;AI-POWERED BREAST CANCER RESEARCH</div>' +
              '<div class="oi-center"><span class="oi-word" id="oi-word-el"></span></div>' +
              '<div class="oi-count" id="oi-count-el">000</div>' +
              '<div class="oi-bar"><div class="oi-bar-fill" id="oi-bar-el"></div></div>';
            doc.body.appendChild(el);

            // Phase 1: the brand name alone. Phase 2 (after BRAND_PHASE_MS):
            // the rotating Detect / Analyze / Understand words begin.
            var wordEl = doc.getElementById('oi-word-el');
            function setWord(text) {
              wordEl.style.animation = 'none';
              void wordEl.offsetWidth;
              wordEl.textContent = text;
              wordEl.style.animation = 'oiWord .5s ease forwards';
            }
            setWord('OncoSense');

            var words = ['Detect', 'Analyze', 'Understand'];
            var wi = 0;
            var wordTimer = null;
            var BRAND_PHASE_MS = 1000;
            var brandTimer = setTimeout(function () {
              function showWord() {
                setWord(words[wi % words.length]);
                wi++;
              }
              showWord();
              wordTimer = setInterval(showWord, 850);
            }, BRAND_PHASE_MS);

            var countEl = doc.getElementById('oi-count-el');
            var barEl = doc.getElementById('oi-bar-el');
            var start = null;
            // Long enough for the brand phase + all three rotating words
            // to each get a full turn before the bar completes.
            var duration = BRAND_PHASE_MS + words.length * 850;
            function step(ts) {
              if (!start) start = ts;
              var elapsed = ts - start;
              var pct = Math.min(elapsed / duration, 1);
              countEl.textContent = String(Math.floor(pct * 100)).padStart(3, '0');
              barEl.style.width = (pct * 100) + '%';
              if (pct < 1) {
                win.requestAnimationFrame(step);
              } else {
                clearTimeout(brandTimer);
                if (wordTimer) clearInterval(wordTimer);
                setTimeout(function () {
                  el.style.transition = 'opacity .5s ease';
                  el.style.opacity = '0';
                  setTimeout(function () { el.remove(); }, 520);
                }, 400);
              }
            }
            win.requestAnimationFrame(step);
          } catch (e) { /* sandboxed - fail silently, page still works */ }
        })();
        </script>
        """,
        height=0,
    )


def render_scroll_effects() -> None:
    """GSAP + ScrollTrigger-driven pipeline animation, injected into the
    OUTER page via the same window.parent escape hatch as render_intro_loader.

    Loads GSAP from a CDN once per browser tab (guarded by a flag on
    window.parent so Streamlit reruns don't re-register duplicate
    ScrollTriggers), then toggles .step-active on each .pipeline-step as it
    scrolls into view (pink glow on the active step + filled connecting
    line, see CSS).

    Skipped entirely under prefers-reduced-motion. Call this once, after all
    .pipeline-step elements have been rendered (end of page).
    """
    st.components.v1.html(
        """
        <script>
        (function () {
          try {
            var win = window.parent, doc = win.document;
            if (win.matchMedia && win.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

            function initEffects() {
              if (!win.gsap || !win.ScrollTrigger) return;
              win.gsap.registerPlugin(win.ScrollTrigger);

              doc.querySelectorAll('.pipeline-step').forEach(function (step) {
                win.ScrollTrigger.create({
                  trigger: step,
                  start: 'top 78%',
                  onEnter: function () { step.classList.add('step-active'); },
                  onLeaveBack: function () { step.classList.remove('step-active'); },
                });
              });

              // Awareness cards: staggered fade-up played once, right when
              // GSAP finishes loading (NOT gated on a ScrollTrigger -- this
              // content is real health information, not decoration, so it
              // must never end up permanently invisible if the CDN load
              // races with the user scrolling past the trigger zone before
              // ScrollTrigger finishes registering). fromTo's "from" state
              // is only applied once this code actually runs, so if GSAP
              // fails to load at all, the cards simply stay at their normal
              // CSS opacity:1 and nothing ever hides them.
              var bentoCards = doc.querySelectorAll('.bento-card');
              if (bentoCards.length) {
                win.gsap.fromTo(
                  bentoCards,
                  { opacity: 0, y: 26, scale: .97 },
                  { opacity: 1, y: 0, scale: 1, duration: .55, ease: 'power3.out', stagger: .07 }
                );
              }

              win.ScrollTrigger.refresh();
            }

            if (win.__oncosenseScrollInit) {
              if (win.ScrollTrigger) win.ScrollTrigger.refresh();
              return;
            }
            win.__oncosenseScrollInit = true;

            function loadScript(src, cb) {
              var s = doc.createElement('script');
              s.src = src;
              s.onload = cb;
              s.onerror = function () {};
              doc.head.appendChild(s);
            }
            loadScript('https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js', function () {
              loadScript('https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/ScrollTrigger.min.js', initEffects);
            });
          } catch (e) { /* sandboxed or offline - fail silently, page still works */ }
        })();
        </script>
        """,
        height=0,
    )


ASSETS_DIR = Path(__file__).parent / "assets"


@st.cache_data
def load_bg_video_b64(filename: str) -> str | None:
    """Base64-encodes a looping background video from assets/ for inlining as
    a data: URI (same embedding pattern as the hero/decorative images above --
    no separate static-file server needed). All clips are muted, downscaled,
    and recompressed with ffmpeg to a few hundred KB, sourced under the
    Pexels License (free for commercial & personal use, no attribution
    required):

    - awareness_bg.mp4 (Awareness section): a doctor reading a diagnostic
      X-ray/scan -- https://www.pexels.com/video/doctor-reading-an-x-ray-scan-7579841/

    Returns None if the asset is missing, so the section still renders
    (just without the video) rather than breaking the page.
    """
    path = ASSETS_DIR / filename
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()
