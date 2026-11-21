"""OncoSense -- doctor view of patient-booked appointments.

A clinician-facing read view, reachable at /Appointments: shows the details
of appointments patients have booked with specialists (via the Patient
Portal), with a "days away" reminder and a calendar/report download.

Booking a new appointment is a patient self-service action -- see
pages/Patient_Dashboard.py -- so it doesn't live here; this page is read-only
for the booking details themselves (no creating, rescheduling, or
cancelling). The one management action it does offer is marking a booked
appointment "Completed" once the visit has happened.
"""
import datetime
import sys
import time
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
_logo_favicon = _ROOT / "assets" / "logo.png"
st.set_page_config(
    page_title="OncoSense -- Appointments",
    page_icon=str(_logo_favicon) if _logo_favicon.exists() else "\U0001FA7B",
    layout="wide",
)

sys.path.insert(0, str(_ROOT))

from oncosense_common import (  # noqa: E402
    CSS,
    REPORTS_DIR,
    build_ics_event,
    clear_login_cookie,
    get_browser_cookies,
    logo_img_tag,
    read_appointments,
    render_floating_ai_widget,
    restore_login_from_cookies,
    update_appointment,
)

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<style>.block-container { padding-top: 2rem !important; }</style>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div style="position:fixed; inset:0; z-index:-1; overflow:hidden; pointer-events:none;">
      <div class="hero-glow"></div>
      <div class="hero-grain"></div>
    </div>
    """,
    unsafe_allow_html=True,
)
render_floating_ai_widget()

browser_cookies = get_browser_cookies()

st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("doctor_name", "")
st.session_state.setdefault("doctor_email", "")

if not st.session_state["authenticated"]:
    if not browser_cookies:
        st.stop()
    restored = restore_login_from_cookies(browser_cookies)
    if restored:
        st.session_state["authenticated"] = True
        st.session_state["doctor_name"], st.session_state["doctor_email"] = restored

if not st.session_state["authenticated"]:
    st.markdown('<div class="login-topspace"></div>', unsafe_allow_html=True)
    _, col_center, _ = st.columns([1, 1.15, 1])
    with col_center:
        with st.container(key="login_card", border=True):
            st.markdown(
                f"""
                <div class="login-badge"><span class="dot"></span> Clinician access only</div>
                <div class="login-logo">
                  {logo_img_tag(50)}
                  <div class="brand-title">OncoSense</div>
                  <p>Sign in on the screening workspace to view<br>appointments.</p>
                </div>
                <div class="login-divider"></div>
                <a class="btn btn-primary" href="/Screening" target="_self"
                   style="width:100%; justify-content:center; box-sizing:border-box;">Go to sign in</a>
                """,
                unsafe_allow_html=True,
            )
    st.stop()

doctor_name = st.session_state.get("doctor_name") or "Doctor"
doctor_email = st.session_state.get("doctor_email", "")
doctor_initials = "".join(part[0].upper() for part in doctor_name.replace("Dr.", "").split()[:2]) or "DR"

with st.sidebar:
    st.markdown(
        f"""
        <div class="brand">
          {logo_img_tag(28)}
          <span>OncoSense</span>
        </div>
        <div class="profile" style="border-top:none; padding-top:0; margin:18px 0 16px;">
          <div class="avatar">{doctor_initials}</div>
          <div>
            <strong>{doctor_name}</strong>
            <span>{doctor_email or "Signed in &middot; Attending"}</span>
          </div>
        </div>
        <a class="btn btn-outline" href="/Screening" target="_self"
           style="width:100%; justify-content:center; box-sizing:border-box; margin-bottom:10px;">
          &larr; Back to Screening
        </a>
        <a class="btn btn-outline" href="/History" target="_self"
           style="width:100%; justify-content:center; box-sizing:border-box; margin-bottom:10px;">
          Patient Screening History &rarr;
        </a>
        <a class="btn btn-outline" href="/Education" target="_self"
           style="width:100%; justify-content:center; box-sizing:border-box; margin-bottom:10px;">
          Breast Health Education &rarr;
        </a>
        <a class="btn btn-outline" href="/AI_Assistant" target="_self"
           style="width:100%; justify-content:center; box-sizing:border-box; margin-bottom:10px;">
          AI Breast Health Assistant &rarr;
        </a>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Sign out", key="logout_btn", use_container_width=True):
        for k in ("authenticated", "doctor_name", "doctor_email"):
            st.session_state.pop(k, None)
        clear_login_cookie()
        time.sleep(0.6)
        st.rerun()

st.markdown(
    """
    <section class="section section-narrow">
      <div class="section-header">
        <h2 class="section-heading">Patient appointment<br><em>bookings.</em></h2>
        <p class="section-sub">Appointments patients have booked with specialists through their portal --
        for your visibility, not for managing here. Patients book, reschedule, or cancel their own
        appointments from the Patient Portal.</p>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

appointments = read_appointments()
if not appointments:
    st.markdown(
        """
        <div class="notice">
          <strong>No appointments booked yet</strong>
          <span>Once a patient books an appointment from their portal, it will show up here with a
          reminder countdown and a calendar download.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    today = datetime.date.today()

    def _sort_key(a: dict):
        try:
            return datetime.datetime.strptime(f"{a['appt_date']} {a['appt_time']}", "%Y-%m-%d %H:%M")
        except (ValueError, KeyError):
            return datetime.datetime.max

    appointments.sort(key=_sort_key)

    for appt in appointments:
        try:
            a_date = datetime.datetime.strptime(appt["appt_date"], "%Y-%m-%d").date()
        except (ValueError, KeyError):
            a_date = None
        is_completed = (appt.get("status") or "").strip().lower() == "completed"

        days_away = (a_date - today).days if a_date else None
        if is_completed:
            when_label, tone = "Completed", "green"
        elif days_away is None:
            when_label, tone = "Date unknown", "amber"
        elif days_away == 0:
            when_label, tone = "Today", "amber"
        elif days_away == 1:
            when_label, tone = "Tomorrow", "amber"
        elif days_away > 1:
            when_label, tone = f"In {days_away} days", "green"
        else:
            when_label, tone = f"{abs(days_away)} days ago", "green"

        with st.container(key=f"appt_card_{appt['appointment_id']}", border=True):
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:14px; flex-wrap:wrap;">
                  <div>
                    <div class="eyebrow" style="margin-bottom:6px;">{appt.get('appt_date', 'n/a')} at {appt.get('appt_time', 'n/a')}</div>
                    <h3 style="font-size:16px; color:hsl(var(--text)); margin-bottom:2px;">{appt.get('patient_name', 'n/a')}</h3>
                    <span style="color:hsl(var(--muted)); font-size:12px;">
                      with {appt.get('specialist_name', 'n/a')} &middot; {appt.get('specialty', 'n/a')} &middot;
                      {appt.get('clinic') or 'n/a'}
                    </span>
                  </div>
                  <span class="status {tone}">{when_label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if appt.get("created_at"):
                st.markdown(
                    f'<div class="summary-row" style="border-top:none;"><span>Booked on</span>'
                    f'<strong>{appt["created_at"]}</strong></div>',
                    unsafe_allow_html=True,
                )
            if appt.get("patient_phone"):
                st.markdown(
                    f'<div class="summary-row"><span>Patient phone</span><strong>{appt["patient_phone"]}</strong></div>',
                    unsafe_allow_html=True,
                )
            if appt.get("patient_email"):
                st.markdown(
                    f'<div class="summary-row"><span>Patient email</span><strong>{appt["patient_email"]}</strong></div>',
                    unsafe_allow_html=True,
                )
            if appt.get("reason"):
                st.markdown(
                    f'<div class="summary-row"><span>Reason</span><strong>{appt["reason"]}</strong></div>',
                    unsafe_allow_html=True,
                )
            if appt.get("phone"):
                st.markdown(
                    f'<div class="summary-row"><span>Clinic phone</span><strong>{appt["phone"]}</strong></div>',
                    unsafe_allow_html=True,
                )

            col_ics, col_report, col_complete = st.columns(3)
            with col_ics:
                if a_date:
                    try:
                        a_time = datetime.datetime.strptime(appt.get("appt_time", "09:00"), "%H:%M").time()
                    except ValueError:
                        a_time = datetime.time(9, 0)
                    start_dt = datetime.datetime.combine(a_date, a_time)
                    end_dt = start_dt + datetime.timedelta(minutes=30)
                    ics_text = build_ics_event(
                        summary=f"Appointment with {appt.get('specialist_name', 'n/a')} ({appt.get('specialty', 'n/a')})",
                        description=(
                            f"Patient: {appt.get('patient_name', 'n/a')}\n"
                            f"Reason: {appt.get('reason') or 'n/a'}\n"
                            f"Booked via OncoSense"
                        ),
                        start=start_dt, end=end_dt, location=appt.get("clinic") or "",
                    )
                    st.download_button(
                        "Add to calendar", data=ics_text.encode("utf-8"),
                        file_name=f"appointment_{appt['appointment_id']}.ics", mime="text/calendar",
                        use_container_width=True, key=f"ics_{appt['appointment_id']}",
                    )
            with col_report:
                report_path = (appt.get("report_path") or "").strip()
                report_file = (REPORTS_DIR.parent / report_path) if report_path else None
                if report_file and report_file.exists():
                    st.download_button(
                        "Download report", data=report_file.read_bytes(),
                        file_name=f"oncosense_report_{appt.get('patient_id') or 'patient'}.html", mime="text/html",
                        use_container_width=True, key=f"dl_report_{appt['appointment_id']}",
                    )
                else:
                    st.caption("No report attached")
            with col_complete:
                if is_completed:
                    st.button(
                        "Completed", key=f"completed_{appt['appointment_id']}",
                        icon=":material/check_circle:", use_container_width=True, disabled=True,
                    )
                else:
                    if st.button(
                        "Mark as completed", key=f"complete_{appt['appointment_id']}",
                        icon=":material/check:", use_container_width=True,
                    ):
                        update_appointment(appt["appointment_id"], {"status": "Completed"})
                        st.success("Marked as completed.")
                        time.sleep(0.4)
                        st.rerun()

st.markdown(
    """
    <footer class="site-footer">
      <div class="footer-links"><a href="/" target="_self">&larr; Back to Home</a></div>
      <div class="footer-bottom">
        <strong>Educational research demo</strong> &bull; Not for clinical use &bull; Appointments recorded
        here are not confirmed with any clinic -- always book or verify directly.
      </div>
    </footer>
    """,
    unsafe_allow_html=True,
)
