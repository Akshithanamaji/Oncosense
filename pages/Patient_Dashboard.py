"""OncoSense -- Patient Portal.

Gated by its own patient session/cookie (see pages/Patient_Login.py) --
entirely separate from the clinician sign-in on pages/Screening.py, so a
doctor session never grants access here and a patient session never grants
access there.

Registration doesn't collect a clinic Patient ID (see pages/Patient_Login.py
-- it's not needed just to sign in and book an appointment), so appointments
are scoped to `patient_user_id` (this account's own id) instead. Screening
reports and medical history, though, are tagged by doctors using the clinic
Patient ID, not this account system -- those only appear once the patient
links that ID below, which is also when appointments start additionally
picking up any the clinic booked on their chart. Either way, every query is
scoped to this one patient; nothing here ever shows another patient's data.
"""
import datetime
import sys
import time
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
_logo_favicon = _ROOT / "assets" / "logo.png"
st.set_page_config(
    page_title="OncoSense -- Patient Portal",
    page_icon=str(_logo_favicon) if _logo_favicon.exists() else "\U0001FA7B",
    layout="wide",
)

sys.path.insert(0, str(_ROOT))

from oncosense_common import (  # noqa: E402
    CSS,
    REPORTS_DIR,
    SPECIALIST_OPTIONS,
    SYMPTOMS_ITEMS,
    add_appointment,
    ask_breast_health_assistant,
    build_ics_event,
    change_patient_password,
    clear_patient_login_cookie,
    delete_appointment,
    format_stage_name,
    get_browser_cookies,
    is_ai_assistant_configured,
    link_patient_id,
    logo_img_tag,
    read_appointments,
    read_screening_history,
    render_floating_ai_widget,
    restore_patient_login_from_cookies,
    update_appointment,
    update_patient_profile,
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

# ------------------------------------------------------------------- Auth gate
browser_cookies = get_browser_cookies()

st.session_state.setdefault("patient_authenticated", False)
st.session_state.setdefault("patient_user_id", "")
st.session_state.setdefault("patient_id", "")
st.session_state.setdefault("patient_name", "")
st.session_state.setdefault("patient_email", "")

if not st.session_state["patient_authenticated"]:
    if not browser_cookies:
        st.stop()
    restored = restore_patient_login_from_cookies(browser_cookies)
    if restored:
        st.session_state["patient_authenticated"] = True
        st.session_state["patient_user_id"] = restored["patient_user_id"]
        st.session_state["patient_id"] = restored["patient_id"]
        st.session_state["patient_name"] = restored["patient_name"]
        st.session_state["patient_email"] = restored["email"]

if not st.session_state["patient_authenticated"]:
    st.markdown('<div class="login-topspace"></div>', unsafe_allow_html=True)
    _, col_center, _ = st.columns([1, 1.15, 1])
    with col_center:
        with st.container(key="login_card", border=True):
            st.markdown(
                f"""
                <div class="login-badge"><span class="dot"></span> Patient access only</div>
                <div class="login-logo">
                  {logo_img_tag(50)}
                  <div class="brand-title">OncoSense</div>
                  <p>Sign in to view your portal.</p>
                </div>
                <div class="login-divider"></div>
                <a class="btn btn-primary" href="/Patient_Login" target="_self"
                   style="width:100%; justify-content:center; box-sizing:border-box;">Go to sign in</a>
                """,
                unsafe_allow_html=True,
            )
    st.stop()

patient_name = st.session_state.get("patient_name") or "Patient"
patient_id = st.session_state.get("patient_id", "")
patient_user_id = st.session_state.get("patient_user_id", "")
patient_email = st.session_state.get("patient_email", "")
patient_initials = "".join(part[0].upper() for part in patient_name.split()[:2]) or "PT"

# ----------------------------------------------- Data scoped to this patient
# patient_id (the clinic chart ID) is only set once the patient links it --
# guard against matching on it being blank, since legacy/incomplete rows
# could also have an empty patient_id and would otherwise look like a match.
raw_history = read_screening_history()
my_records = [r for r in raw_history if patient_id and (r.get("patient_id") or "").strip() == patient_id]
my_records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

# Appointments this patient booked themselves are tagged with their own
# account id (patient_user_id) so booking works before any ID is linked;
# once linked, appointments the clinic booked under that chart ID (which
# carry a patient_id but no patient_user_id) show up too.
raw_appointments = read_appointments()
my_appointments = [
    a for a in raw_appointments
    if (a.get("patient_user_id") or "").strip() == patient_user_id
    or (patient_id and (a.get("patient_id") or "").strip() == patient_id)
]

with st.sidebar:
    st.markdown(
        f"""
        <div class="brand">
          {logo_img_tag(28)}
          <span>OncoSense</span>
        </div>
        <div class="profile" style="border-top:none; padding-top:0; margin:18px 0 16px;">
          <div class="avatar">{patient_initials}</div>
          <div>
            <strong>{patient_name}</strong>
            <span>{f'Patient ID {patient_id}' if patient_id else 'Patient ID not linked'}</span>
          </div>
        </div>
        <a class="btn btn-outline" href="/" target="_self"
           style="width:100%; justify-content:center; box-sizing:border-box; margin-bottom:10px;">
          &larr; Back to Home
        </a>
        """,
        unsafe_allow_html=True,
    )
    NAV_ITEMS = [
        ("Screening Reports", "description"),
        ("Appointments", "event"),
        ("Medical History", "folder_shared"),
        ("AI Assistant", "smart_toy"),
        ("Notifications", "notifications"),
        ("Education", "school"),
        ("Profile Settings", "settings"),
    ]
    st.session_state.setdefault("patient_dashboard_section", "Screening Reports")
    with st.container(key="sidebar-nav"):
        for label, icon in NAV_ITEMS:
            is_active = st.session_state["patient_dashboard_section"] == label
            slug = label.replace(" ", "_")
            with st.container(key=f"navitem-{slug}-active" if is_active else f"navitem-{slug}"):
                if st.button(label, key=f"navbtn_{slug}", icon=f":material/{icon}:", use_container_width=True):
                    st.session_state["patient_dashboard_section"] = label
                    st.rerun()
    section = st.session_state["patient_dashboard_section"]
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    if st.button("Sign out", key="patient_logout_btn", use_container_width=True):
        for k in ("patient_authenticated", "patient_user_id", "patient_id", "patient_name", "patient_email"):
            st.session_state.pop(k, None)
        clear_patient_login_cookie()
        time.sleep(0.6)
        st.switch_page("pages/Patient_Login.py")

st.markdown(
    f"""
    <section class="section section-narrow">
      <div class="section-header">
        <h2 class="section-heading">Welcome back,<br><em>{patient_name.split()[0] if patient_name else 'there'}.</em></h2>
        <p class="section-sub">Your screening reports, appointments, and account -- all in one place.</p>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

today = datetime.date.today()


def _appt_sort_key(a: dict):
    try:
        return datetime.datetime.strptime(f"{a['appt_date']} {a['appt_time']}", "%Y-%m-%d %H:%M")
    except (ValueError, KeyError):
        return datetime.datetime.max


my_appointments.sort(key=_appt_sort_key)


def _render_link_id_prompt(key_suffix: str) -> None:
    """Shown wherever reports/history need a linked Patient ID but the
    account doesn't have one yet -- a one-time step, not required at signup."""
    st.markdown(
        """
        <div class="notice">
          <strong>Link your Patient ID to see your reports</strong>
          <span>Your clinic gave you a Patient ID when you were screened -- enter it below to connect
          this account to your records. You only need to do this once.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    link_col1, link_col2 = st.columns([3, 1])
    with link_col1:
        entered_id = st.text_input("Patient ID", key=f"link_id_input_{key_suffix}", placeholder="P-00001", label_visibility="collapsed")
    with link_col2:
        if st.button("Link ID", key=f"link_id_btn_{key_suffix}", use_container_width=True):
            ok, msg = link_patient_id(patient_user_id, entered_id)
            if ok:
                st.session_state["patient_id"] = msg
                st.success("Patient ID linked.")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(msg)


# ======================================================== Screening Reports
if section == "Screening Reports":
    if not patient_id:
        _render_link_id_prompt("reports")
    elif not my_records:
        st.markdown(
            """
            <div class="notice">
              <strong>No reports on file yet</strong>
              <span>Once your clinician logs a mammogram analysis under your Patient ID, it will appear
              here.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    for i, r in enumerate(my_records):
        pred = (r.get("predicted_class") or "").upper()
        tone = "amber" if pred == "MALIGNANT" else "green"
        with st.container(key=f"pt_report_{i}", border=True):
            header_col, image_col = st.columns([3, 1])
            with header_col:
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:14px; flex-wrap:wrap;">
                      <div>
                        <div class="eyebrow" style="margin-bottom:6px;">{r.get('timestamp', '')}</div>
                        <h3 style="font-size:15px; color:hsl(var(--text)); margin-bottom:2px;">{r.get('finding', 'n/a')}</h3>
                        <span style="color:hsl(var(--muted)); font-size:12px;">Screened by {r.get('doctor_name') or 'n/a'}</span>
                      </div>
                      <span class="status {tone}">{pred or 'N/A'}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="summary-row" style="border-top:none;"><span>Image file</span><strong>{r.get('filename', 'n/a')}</strong></div>
                    <div class="summary-row"><span>Confidence</span><strong>{(lambda c: f'{float(c)*100:.1f}%' if c else 'n/a')(r.get('confidence'))}</strong></div>
                    """,
                    unsafe_allow_html=True,
                )
            image_path = (r.get("image_path") or "").strip()
            image_file = (REPORTS_DIR.parent / image_path) if image_path else None
            with image_col:
                if image_file and image_file.exists():
                    st.image(str(image_file), caption="Mammogram", use_container_width=True)
                else:
                    st.caption("No archived image (legacy entry).")

            gradcam_path = (r.get("gradcam_path") or "").strip()
            gradcam_file = (REPORTS_DIR.parent / gradcam_path) if gradcam_path else None
            if gradcam_file and gradcam_file.exists():
                with st.expander("Grad-CAM heatmap (model explanation)"):
                    st.image(str(gradcam_file), use_container_width=True)

            report_path = (r.get("report_path") or "").strip()
            report_file = (REPORTS_DIR.parent / report_path) if report_path else None
            if report_file and report_file.exists():
                with st.expander("View report"):
                    st.components.v1.html(report_file.read_text(encoding="utf-8"), height=700, scrolling=True)
            else:
                st.caption("Report file not found for this entry.")

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                if report_file and report_file.exists():
                    st.download_button(
                        "Download report (share with your doctor)", data=report_file.read_bytes(),
                        file_name=f"oncosense_report_{r.get('timestamp', '').replace(':', '').replace(' ', '_')}.html",
                        mime="text/html", use_container_width=True, key=f"pt_dl_report_{i}",
                    )
            with col_dl2:
                if image_file and image_file.exists():
                    st.download_button(
                        "Download mammogram image", data=image_file.read_bytes(),
                        file_name=f"oncosense_image_{r.get('timestamp', '').replace(':', '').replace(' ', '_')}.png",
                        mime="image/png", use_container_width=True, key=f"pt_dl_image_{i}",
                    )

# ============================================================= Appointments
elif section == "Appointments":
    with st.container(key="pt_book_card", border=True):
        st.markdown('<h2 style="margin-bottom:4px;">Book an appointment</h2>', unsafe_allow_html=True)
        st.markdown(
            '<p class="panel-intro">This records the appointment in OncoSense for your own tracking -- '
            'it does not notify or confirm anything with the clinic, so please call or book with them '
            'directly as well.</p>',
            unsafe_allow_html=True,
        )
        if patient_id:
            st.markdown(
                f'<div class="summary-row" style="border-top:none;"><span>Patient ID</span><strong>{patient_id}</strong></div>',
                unsafe_allow_html=True,
            )
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            contact_name = st.text_input("Patient name", value=patient_name, key="pt_appt_contact_name")
        with pcol2:
            contact_phone = st.text_input("Phone number", key="pt_appt_contact_phone", placeholder="e.g. 98765 43210")
        with pcol3:
            contact_email = st.text_input("Email", value=patient_email, key="pt_appt_contact_email")

        specialist_option_labels = [f"{name} -- {specialty}, {clinic}" for name, specialty, clinic in SPECIALIST_OPTIONS]
        specialist_choice = st.selectbox(
            "Doctor / specialist", specialist_option_labels + ["Other (enter manually)"], key="pt_appt_specialist_choice",
        )
        st.caption(
            "A few real, publicly-sourced specialists to pick from -- verify current availability before "
            "relying on this for care. Not listed? Choose \"Other\" and enter their details."
        )

        if specialist_choice == "Other (enter manually)":
            scol1, scol2 = st.columns(2)
            with scol1:
                spec_name = st.text_input("Specialist name", key="pt_appt_specialist_name")
                spec_clinic = st.text_input("Clinic / hospital", key="pt_appt_clinic")
            with scol2:
                spec_specialty = st.selectbox(
                    "Specialty", ["Radiologist", "Breast Surgeon", "Medical Oncologist", "Breast Specialist / OB-GYN", "Other"],
                    key="pt_appt_specialty",
                )
                spec_phone = st.text_input("Phone", key="pt_appt_phone")
        else:
            chosen = SPECIALIST_OPTIONS[specialist_option_labels.index(specialist_choice)]
            spec_name, spec_specialty, spec_clinic = chosen[0], chosen[1], chosen[2]
            spec_phone = ""

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            appt_date = st.date_input("Appointment date", key="pt_appt_date", min_value=today)
        with dcol2:
            appt_time = st.time_input("Appointment time", key="pt_appt_time", value=datetime.time(9, 0))

        reason = st.text_area(
            "Reason / notes", key="pt_appt_reason",
            placeholder="e.g. Follow-up on my latest screening result",
        )

        chosen_report_path = ""
        if my_records:
            report_labels = [f"{r.get('timestamp', 'n/a')} -- {(r.get('predicted_class') or 'n/a').upper()}" for r in my_records]
            report_choice = st.selectbox(
                "Share a report with this appointment (optional)", ["None"] + report_labels, key="pt_appt_report_choice",
            )
            if report_choice != "None":
                chosen_report_path = my_records[report_labels.index(report_choice)].get("report_path", "")

        if st.button("Book appointment", key="pt_book_appt_btn", use_container_width=True):
            spec_name_val = spec_name.strip() if isinstance(spec_name, str) else spec_name
            contact_name_val = contact_name.strip() if isinstance(contact_name, str) else contact_name
            if not spec_name_val:
                st.warning("Enter the specialist's name before booking.")
            elif not contact_name_val:
                st.warning("Enter the patient's name before booking.")
            else:
                add_appointment({
                    "patient_name": contact_name_val,
                    "patient_id": patient_id,
                    "patient_user_id": patient_user_id,
                    "patient_phone": contact_phone.strip() if isinstance(contact_phone, str) else contact_phone,
                    "patient_email": contact_email.strip() if isinstance(contact_email, str) else contact_email,
                    "specialist_name": spec_name_val,
                    "specialty": spec_specialty,
                    "clinic": spec_clinic.strip() if isinstance(spec_clinic, str) else spec_clinic,
                    "phone": spec_phone.strip() if isinstance(spec_phone, str) else spec_phone,
                    "appt_date": appt_date.strftime("%Y-%m-%d"),
                    "appt_time": appt_time.strftime("%H:%M"),
                    "reason": reason.strip(),
                    "report_path": chosen_report_path,
                    "doctor_name": "",
                    "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                st.success(f"Appointment recorded for {appt_date.strftime('%b %d, %Y')}.")
                time.sleep(0.6)
                st.rerun()

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    st.markdown('<h2 style="margin-bottom:10px;">My appointments</h2>', unsafe_allow_html=True)

    if not my_appointments:
        st.caption("No appointments booked yet.")
    for appt in my_appointments:
        try:
            a_date = datetime.datetime.strptime(appt["appt_date"], "%Y-%m-%d").date()
        except (ValueError, KeyError):
            a_date = None
        days_away = (a_date - today).days if a_date else None
        if days_away is None:
            when_label, tone = "Date unknown", "amber"
        elif days_away == 0:
            when_label, tone = "Today", "amber"
        elif days_away == 1:
            when_label, tone = "Tomorrow", "amber"
        elif days_away > 1:
            when_label, tone = f"In {days_away} days", "green"
        else:
            when_label, tone = f"{abs(days_away)} days ago", "green"

        with st.container(key=f"pt_appt_card_{appt['appointment_id']}", border=True):
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:14px; flex-wrap:wrap;">
                  <div>
                    <div class="eyebrow" style="margin-bottom:6px;">{appt.get('appt_date', 'n/a')} at {appt.get('appt_time', 'n/a')}</div>
                    <h3 style="font-size:15px; color:hsl(var(--text));">{appt.get('specialist_name', 'n/a')}</h3>
                    <span style="color:hsl(var(--muted)); font-size:12px;">{appt.get('specialty', 'n/a')} &middot; {appt.get('clinic') or 'n/a'}</span>
                  </div>
                  <span class="status {tone}">{when_label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if appt.get("reason"):
                st.markdown(
                    f'<div class="summary-row"><span>Reason</span><strong>{appt["reason"]}</strong></div>',
                    unsafe_allow_html=True,
                )

            with st.expander("Reschedule"):
                try:
                    cur_date = datetime.datetime.strptime(appt["appt_date"], "%Y-%m-%d").date()
                except (ValueError, KeyError):
                    cur_date = today
                try:
                    cur_time = datetime.datetime.strptime(appt.get("appt_time", "09:00"), "%H:%M").time()
                except ValueError:
                    cur_time = datetime.time(9, 0)
                rcol1, rcol2 = st.columns(2)
                with rcol1:
                    new_date = st.date_input(
                        "New date", value=max(cur_date, today), min_value=today,
                        key=f"resched_date_{appt['appointment_id']}",
                    )
                with rcol2:
                    new_time = st.time_input("New time", value=cur_time, key=f"resched_time_{appt['appointment_id']}")
                if st.button("Save new date/time", key=f"resched_save_{appt['appointment_id']}", use_container_width=True):
                    update_appointment(appt["appointment_id"], {
                        "appt_date": new_date.strftime("%Y-%m-%d"), "appt_time": new_time.strftime("%H:%M"),
                    })
                    st.success("Appointment rescheduled.")
                    time.sleep(0.5)
                    st.rerun()

            col_ics, col_report, col_cancel = st.columns(3)
            with col_ics:
                if a_date:
                    try:
                        a_time = datetime.datetime.strptime(appt.get("appt_time", "09:00"), "%H:%M").time()
                    except ValueError:
                        a_time = datetime.time(9, 0)
                    start_dt = datetime.datetime.combine(a_date, a_time)
                    ics_text = build_ics_event(
                        summary=f"Appointment with {appt.get('specialist_name', 'n/a')} ({appt.get('specialty', 'n/a')})",
                        description=f"Reason: {appt.get('reason') or 'n/a'}\nBooked via OncoSense patient portal.",
                        start=start_dt, end=start_dt + datetime.timedelta(minutes=30), location=appt.get("clinic") or "",
                    )
                    st.download_button(
                        "Add to calendar", data=ics_text.encode("utf-8"),
                        file_name=f"appointment_{appt['appointment_id']}.ics", mime="text/calendar",
                        use_container_width=True, key=f"pt_ics_{appt['appointment_id']}",
                    )
            with col_report:
                report_path = (appt.get("report_path") or "").strip()
                report_file = (REPORTS_DIR.parent / report_path) if report_path else None
                if report_file and report_file.exists():
                    st.download_button(
                        "Download shared report", data=report_file.read_bytes(),
                        file_name=f"oncosense_report_{patient_id}.html", mime="text/html",
                        use_container_width=True, key=f"pt_dl_appt_report_{appt['appointment_id']}",
                    )
            with col_cancel:
                if st.button("Cancel appointment", key=f"pt_cancel_{appt['appointment_id']}", use_container_width=True):
                    delete_appointment(appt["appointment_id"])
                    st.rerun()

# =========================================================== Medical History
elif section == "Medical History":
    if not patient_id:
        _render_link_id_prompt("history")
    elif not my_records:
        st.caption("No medical history on file yet.")
    for i, r in enumerate(my_records):
        with st.container(key=f"pt_history_{i}", border=True):
            st.markdown(
                f"""
                <div class="eyebrow" style="margin-bottom:8px;">{r.get("timestamp", "n/a")}</div>
                <span style="color:hsl(var(--muted)); font-size:12px;">Screened by {r.get('doctor_name') or 'n/a'}</span>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="summary-row" style="border-top:none;"><span>Age at visit</span><strong>{r.get('age', 'n/a')}</strong></div>
                <div class="summary-row"><span>Breast side / view</span><strong>{r.get('breast_side', 'n/a')} / {r.get('view', 'n/a')}</strong></div>
                <div class="summary-row"><span>Family history of breast cancer</span><strong>{r.get('family_history', 'n/a')}</strong></div>
                <div class="summary-row"><span>Previous breast cancer</span><strong>{r.get('previous_cancer', 'n/a')}</strong></div>
                <div class="summary-row"><span>Previous breast biopsy</span><strong>{r.get('previous_biopsy', 'n/a')}</strong></div>
                <div class="summary-row"><span>Previous mammogram</span><strong>{r.get('previous_mammogram', 'n/a')}</strong></div>
                <div class="summary-row"><span>Previous abnormal mammogram</span><strong>{r.get('previous_abnormal_mammogram', 'n/a')}</strong></div>
                <div class="summary-row"><span>Menopausal status</span><strong>{r.get('menopausal_status', 'n/a')}</strong></div>
                <div class="summary-row"><span>Breast symptoms</span><strong>{r.get('breast_symptoms', 'n/a')}</strong></div>
                {f'<div class="summary-row"><span>Clinical stage estimate</span><strong>{format_stage_name(r.get("estimated_stage"))}</strong></div>' if (r.get('estimated_stage') or '').strip() else ''}
                """,
                unsafe_allow_html=True,
            )

# ============================================================= AI Assistant
elif section == "AI Assistant":
    st.session_state.setdefault("ai_chat_history", [])

    st.markdown(
        """
        <style>
        .block-container { padding-bottom: 96px !important; }
        div[data-testid="stChatInput"] {
            position: fixed !important; bottom: 0 !important; left: 300px !important; right: 0 !important;
            top: auto !important; max-width: 760px; margin: 0 auto; z-index: 9500;
            background: hsl(var(--bg)); padding: 14px 24px 20px; border-top: 1px solid hsl(var(--stroke));
        }
        </style>
        <div class="notice">
          <strong>Educational only -- not a diagnosis.</strong>
          <span>This assistant can explain mammograms, breast health, and what your report means in
          plain language. It cannot diagnose you or recommend treatment -- always discuss your specific
          results and care decisions with your doctor.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not is_ai_assistant_configured():
        st.caption("The AI service isn't connected on this server yet -- the assistant will say so below.")

    if st.session_state["ai_chat_history"]:
        if st.button("Clear chat", key="ai_chat_clear_btn"):
            st.session_state["ai_chat_history"] = []
            st.rerun()

    if my_records:
        if st.button("Explain my most recent report", key="ai_chat_explain_report_btn"):
            latest = my_records[0]
            conf = latest.get("confidence")
            conf_pct = f"{float(conf) * 100:.1f}%" if conf else "n/a"
            report_summary = (
                f"Date: {latest.get('timestamp', 'n/a')}\n"
                f"Finding: {latest.get('finding', 'n/a')}\n"
                f"AI prediction: {(latest.get('predicted_class') or 'n/a').upper()}\n"
                f"Confidence: {conf_pct}\n"
                f"Estimated stage (clinician-entered, if any): {format_stage_name(latest.get('estimated_stage')) if latest.get('estimated_stage') else 'not recorded'}"
            )
            question = (
                f"Here is my latest OncoSense screening report:\n{report_summary}\n\n"
                "Can you explain in simple terms what this means?"
            )
            st.session_state["ai_chat_history"].append({"role": "user", "content": question})
            with st.spinner("Thinking..."):
                reply = ask_breast_health_assistant(st.session_state["ai_chat_history"])
            st.session_state["ai_chat_history"].append({"role": "assistant", "content": reply})
            st.rerun()

    for turn in st.session_state["ai_chat_history"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    if prompt := st.chat_input("Ask about mammograms, breast health, or your report..."):
        st.session_state["ai_chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = ask_breast_health_assistant(st.session_state["ai_chat_history"])
            st.markdown(reply)
        st.session_state["ai_chat_history"].append({"role": "assistant", "content": reply})

# ============================================================= Notifications
elif section == "Notifications":
    st.caption("In-app notifications only -- this demo has no email/SMS service connected.")
    notifications = []
    for appt in my_appointments:
        try:
            a_date = datetime.datetime.strptime(appt["appt_date"], "%Y-%m-%d").date()
        except (ValueError, KeyError):
            continue
        days_away = (a_date - today).days
        if 0 <= days_away <= 7:
            when = "today" if days_away == 0 else ("tomorrow" if days_away == 1 else f"in {days_away} days")
            notifications.append((
                appt.get("created_at", ""),
                f"Appointment reminder",
                f"Your appointment with {appt.get('specialist_name', 'n/a')} is {when} ({appt.get('appt_date')} at {appt.get('appt_time')}).",
            ))
        try:
            created = datetime.datetime.strptime(appt.get("created_at", ""), "%Y-%m-%d %H:%M:%S")
            if (datetime.datetime.now() - created).days < 2:
                notifications.append((
                    appt.get("created_at", ""), "Appointment confirmed",
                    f"Your appointment with {appt.get('specialist_name', 'n/a')} on {appt.get('appt_date')} has been recorded.",
                ))
        except ValueError:
            pass
    for r in my_records[:5]:
        notifications.append((
            r.get("timestamp", ""), "Report available",
            f"Your {r.get('timestamp', 'n/a')} screening report is ready to view -- finding: {r.get('finding', 'n/a')}.",
        ))

    notifications.sort(key=lambda n: n[0], reverse=True)
    if not notifications:
        st.markdown(
            """
            <div class="notice">
              <strong>No notifications</strong>
              <span>Reminders for upcoming appointments and new reports will show up here.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    for ts, title, body in notifications:
        st.markdown(
            f"""
            <div class="notice">
              <strong>{title} &middot; <span style="font-weight:400; color:hsl(var(--muted));">{ts}</span></strong>
              <span>{body}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ================================================================ Education
elif section == "Education":
    symptoms_preview_html = "".join(
        f"<li>{name} -- {desc}</li>" for name, desc, _ in SYMPTOMS_ITEMS[:4]
    )
    st.markdown(
        f"""
        <div class="panel">
          <h2>Breast Health Education</h2>
          <p class="panel-intro">A few warning signs to know -- the full Education Center covers
          screening guidance, risk factors, staging, treatment, and FAQs.</p>
          <ul style="color:hsl(var(--muted)); font-size:13px; line-height:1.9; margin:0 0 18px 18px; padding:0;">
            {symptoms_preview_html}
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<a class="btn btn-primary" href="/Education" target="_self">Visit the full Education Center &rarr;</a>',
        unsafe_allow_html=True,
    )

# ============================================================ Profile Settings
elif section == "Profile Settings":
    with st.container(key="pt_profile_card", border=True):
        st.markdown('<h2 style="margin-bottom:16px;">Profile</h2>', unsafe_allow_html=True)
        if patient_id:
            st.markdown(
                f'<div class="summary-row" style="border-top:none;"><span>Patient ID</span><strong>{patient_id}</strong></div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No Patient ID linked yet -- link it to see your screening reports.")
            _render_link_id_prompt("profile")
        pf_name = st.text_input("Full name", value=patient_name, key="pf_name")
        pf_email = st.text_input("Email", value=patient_email, key="pf_email")
        if st.button("Save profile", key="pt_save_profile_btn"):
            ok, msg = update_patient_profile(patient_user_id, pf_name, pf_email)
            if ok:
                st.session_state["patient_name"] = pf_name.strip()
                st.session_state["patient_email"] = pf_email.strip()
                st.success("Profile updated.")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(msg)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    with st.container(key="pt_password_card", border=True):
        st.markdown('<h2 style="margin-bottom:16px;">Change password</h2>', unsafe_allow_html=True)
        cur_pw = st.text_input("Current password", type="password", key="pf_current_pw")
        new_pw = st.text_input("New password (min. 8 characters)", type="password", key="pf_new_pw")
        confirm_pw = st.text_input("Confirm new password", type="password", key="pf_confirm_pw")
        if st.button("Update password", key="pt_change_pw_btn"):
            if new_pw != confirm_pw:
                st.error("New passwords do not match.")
            else:
                ok, msg = change_patient_password(patient_user_id, cur_pw, new_pw)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    if st.button("Sign out", key="pt_profile_signout_btn", use_container_width=True):
        for k in ("patient_authenticated", "patient_user_id", "patient_id", "patient_name", "patient_email"):
            st.session_state.pop(k, None)
        clear_patient_login_cookie()
        time.sleep(0.6)
        st.switch_page("pages/Patient_Login.py")

st.markdown(
    """
    <footer class="site-footer">
      <div class="footer-links"><a href="/" target="_self">&larr; Back to Home</a></div>
      <div class="footer-bottom">
        <strong>Educational research demo</strong> &bull; Not for clinical use
      </div>
    </footer>
    """,
    unsafe_allow_html=True,
)
