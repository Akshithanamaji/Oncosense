"""OncoSense -- patient screening history.

Every completed analysis on the Screening page (pages/Screening.py) is
logged here: who was screened, their details, and the full report that was
generated for them. This page is the clinician's record of past screenings,
reachable at /History.
"""
import sys
import time
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
_logo_favicon = _ROOT / "assets" / "logo.png"
st.set_page_config(
    page_title="OncoSense -- Patient History",
    page_icon=str(_logo_favicon) if _logo_favicon.exists() else "\U0001FA7B",
    layout="wide",
)

sys.path.insert(0, str(_ROOT))

from oncosense_common import (  # noqa: E402
    CSS,
    HISTORY_PATH,
    IMAGES_DIR,
    REPORTS_DIR,
    clear_login_cookie,
    delete_screening_history_row,
    format_stage_name,
    get_browser_cookies,
    logo_img_tag,
    read_screening_history,
    render_floating_ai_widget,
    restore_login_from_cookies,
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
                  <p>Sign in on the screening workspace to view<br>patient history.</p>
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
        <a class="btn btn-outline" href="/Appointments" target="_self"
           style="width:100%; justify-content:center; box-sizing:border-box; margin-bottom:10px;">
          Appointments &rarr;
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
        <h2 class="section-heading">Patient screening<br><em>history.</em></h2>
        <p class="section-sub">Every mammogram analyzed on this workstation, with the patient's
        details and the full report generated at the time.</p>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

raw_rows = read_screening_history()
indexed_rows = list(reversed(list(enumerate(raw_rows))))  # (original_index, row), newest first
patient_count = len({r.get("patient_id", "").strip() for r in raw_rows if r.get("patient_id", "").strip()})

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(
        f"""<div class="panel" style="margin-bottom:0;"><div class="summary-row" style="border-top:none;">
        <span>Patients screened</span><strong>{patient_count}</strong></div></div>""",
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(
        f"""<div class="panel" style="margin-bottom:0;"><div class="summary-row" style="border-top:none;">
        <span>Images analyzed</span><strong>{len(raw_rows)}</strong></div></div>""",
        unsafe_allow_html=True,
    )
with col_c:
    if st.button("Clear screening history", key="clear_history_btn", use_container_width=True):
        if HISTORY_PATH.exists():
            HISTORY_PATH.unlink()
        if REPORTS_DIR.exists():
            for f in REPORTS_DIR.glob("*.html"):
                f.unlink()
        if IMAGES_DIR.exists():
            for f in IMAGES_DIR.glob("*.png"):
                f.unlink()
        st.rerun()

st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

if not raw_rows:
    st.markdown(
        """
        <div class="notice">
          <strong>No screenings logged yet</strong>
          <span>Analyze a mammogram on the Screening page and it will show up here, with the
          patient's details and a link back to the full report.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # ---------------------------------------------- Compare scans over time
    # Group by patient_id so a patient with multiple visits can have their
    # mammograms viewed side by side, newest alongside oldest.
    by_patient: dict[str, list[tuple[int, dict]]] = {}
    for idx, r in enumerate(raw_rows):
        pid = (r.get("patient_id") or "").strip()
        if pid:
            by_patient.setdefault(pid, []).append((idx, r))
    comparable_patients = {
        pid: entries for pid, entries in by_patient.items()
        if len([e for e in entries if (e[1].get("image_path") or "").strip()]) >= 2
    }

    if comparable_patients:
        with st.expander(f"Compare scans over time ({len(comparable_patients)} patient(s) with multiple visits)"):
            pid_options = sorted(
                comparable_patients.keys(),
                key=lambda pid: comparable_patients[pid][0][1].get("patient_name") or pid,
            )
            label_for = {
                pid: f"{comparable_patients[pid][0][1].get('patient_name') or 'n/a'} (ID {pid})"
                for pid in pid_options
            }
            chosen_pid = st.selectbox(
                "Patient", pid_options, key="compare_patient",
                format_func=lambda pid: label_for[pid],
            )
            entries = [e for e in comparable_patients[chosen_pid] if (e[1].get("image_path") or "").strip()]
            entry_labels = [f"{r.get('timestamp', 'n/a')} · {(r.get('predicted_class') or 'n/a').upper()}" for _, r in entries]
            chosen = st.multiselect(
                "Scans to compare (defaults to the earliest and latest)",
                options=list(range(len(entries))),
                default=[0, len(entries) - 1] if len(entries) > 1 else [0],
                format_func=lambda i: entry_labels[i],
                key="compare_scans",
            )
            if chosen:
                compare_cols = st.columns(len(chosen))
                for col, i in zip(compare_cols, sorted(chosen)):
                    _, r = entries[i]
                    with col:
                        img_file = REPORTS_DIR.parent / r["image_path"]
                        if img_file.exists():
                            st.image(str(img_file), use_container_width=True)
                        pred = (r.get("predicted_class") or "n/a").upper()
                        tone = "amber" if pred == "MALIGNANT" else "green"
                        st.markdown(
                            f"""
                            <div style="text-align:center; margin-top:6px;">
                              <div class="eyebrow" style="margin-bottom:4px;">{r.get('timestamp', 'n/a')}</div>
                              <span class="status {tone}">{pred} &middot; {r.get('malignant_prob', 'n/a')}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                st.caption("Select at least one scan above to view it.")
        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------- Record list
    search = st.text_input(
        "Search by patient name or ID", key="history_search", placeholder="e.g. Jane Doe or P-00001",
    )
    rows = indexed_rows
    if search.strip():
        q = search.strip().lower()
        rows = [
            (idx, r) for idx, r in rows
            if q in (r.get("patient_name") or "").lower() or q in (r.get("patient_id") or "").lower()
        ]

    for orig_idx, r in rows:
        pred = (r.get("predicted_class") or "").upper()
        is_malignant = pred == "MALIGNANT"
        tone = "amber" if is_malignant else "green"
        patient_name = r.get("patient_name") or "n/a"
        patient_id = r.get("patient_id") or "n/a"
        malignant_prob = r.get("malignant_prob") or "n/a"
        confidence = r.get("confidence") or ""
        try:
            confidence_display = f"{float(confidence) * 100:.1f}%"
        except ValueError:
            confidence_display = "n/a"

        with st.container(key=f"history_row_{orig_idx}", border=True):
            header_col, image_col = st.columns([3, 1])
            with header_col:
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:14px; flex-wrap:wrap;">
                      <div>
                        <div class="eyebrow" style="margin-bottom:6px;">{r.get('timestamp', '')}</div>
                        <h3 style="font-size:16px; color:hsl(var(--text)); margin-bottom:2px;">{patient_name}</h3>
                        <span style="color:hsl(var(--muted)); font-size:12px;">
                          Patient ID {patient_id} &middot; Age {r.get('age', 'n/a')} &middot;
                          {r.get('breast_side', 'n/a')} {r.get('view', 'n/a')} &middot; Screened by {r.get('doctor_name', 'n/a')}
                        </span>
                      </div>
                      <span class="status {tone}">{pred or 'N/A'} &middot; {malignant_prob}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="summary-row"><span>Image file</span><strong>{r.get('filename', 'n/a')}</strong></div>
                    <div class="summary-row"><span>Confidence</span><strong>{confidence_display}</strong></div>
                    <div class="summary-row"><span>AI finding</span><strong>{r.get('finding', 'n/a')}</strong></div>
                    {f'<div class="summary-row"><span>Clinical stage estimate</span><strong>{format_stage_name(r.get("estimated_stage"))}</strong></div>' if (r.get('estimated_stage') or '').strip() else ''}
                    """,
                    unsafe_allow_html=True,
                )
            with image_col:
                image_path = (r.get("image_path") or "").strip()
                image_file = (REPORTS_DIR.parent / image_path) if image_path else None
                if image_file and image_file.exists():
                    st.image(str(image_file), caption="Mammogram", use_container_width=True)
                else:
                    st.caption("No archived image for this record (legacy entry).")

            with st.expander("Patient details (intake form)"):
                st.markdown(
                    f"""
                    <div class="summary-row" style="border-top:none;"><span>Family history of breast cancer</span><strong>{r.get('family_history', 'n/a')}</strong></div>
                    <div class="summary-row"><span>Previous breast cancer</span><strong>{r.get('previous_cancer', 'n/a')}</strong></div>
                    <div class="summary-row"><span>Previous breast biopsy</span><strong>{r.get('previous_biopsy', 'n/a')}</strong></div>
                    <div class="summary-row"><span>Previous mammogram</span><strong>{r.get('previous_mammogram', 'n/a')}</strong></div>
                    <div class="summary-row"><span>Previous abnormal mammogram</span><strong>{r.get('previous_abnormal_mammogram', 'n/a')}</strong></div>
                    <div class="summary-row"><span>Menopausal status</span><strong>{r.get('menopausal_status', 'n/a')}</strong></div>
                    <div class="summary-row"><span>Breast symptoms</span><strong>{r.get('breast_symptoms', 'n/a')}</strong></div>
                    <div class="summary-row"><span>Clinical T / N / M</span><strong>{r.get('clinical_t') or 'n/a'} / {r.get('clinical_n') or 'n/a'} / {r.get('clinical_m') or 'n/a'}</strong></div>
                    """,
                    unsafe_allow_html=True,
                )

            if image_file and image_file.exists():
                gradcam_path = (r.get("gradcam_path") or "").strip()
                gradcam_file = (REPORTS_DIR.parent / gradcam_path) if gradcam_path else None
                if gradcam_file and gradcam_file.exists():
                    with st.expander("Grad-CAM heatmap (model explanation)"):
                        st.image(str(gradcam_file), use_container_width=True)

            report_path = r.get("report_path") or ""
            report_file = (REPORTS_DIR.parent / report_path) if report_path else None
            if report_file and report_file.exists():
                report_html = report_file.read_text(encoding="utf-8")
                with st.expander("View report"):
                    st.components.v1.html(report_html, height=900, scrolling=True)
            else:
                st.caption("Report file not found for this entry.")

            ts_safe = r.get("timestamp", "").replace(":", "").replace(" ", "_")
            col_dl_report, col_dl_image, col_delete = st.columns(3)
            with col_dl_report:
                if report_file and report_file.exists():
                    st.download_button(
                        "Download report",
                        data=report_file.read_bytes(),
                        file_name=f"oncosense_report_{patient_id}_{ts_safe}.html",
                        mime="text/html",
                        use_container_width=True,
                        key=f"dl_report_{orig_idx}",
                    )
            with col_dl_image:
                if image_file and image_file.exists():
                    st.download_button(
                        "Download image",
                        data=image_file.read_bytes(),
                        file_name=f"oncosense_image_{patient_id}_{ts_safe}.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"dl_image_{orig_idx}",
                    )
            with col_delete:
                if st.button("Delete this record", key=f"delete_{orig_idx}", use_container_width=True):
                    delete_screening_history_row(orig_idx)
                    st.rerun()

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
