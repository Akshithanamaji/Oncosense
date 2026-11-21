"""OncoSense -- gated screening workspace.

Upload a mammogram (DICOM .dcm, or an already-processed PNG/JPG) and get a
benign/malignant prediction with a Grad-CAM heatmap showing which region of
the image drove the decision.

Separate Streamlit page (classic `pages/` convention) from the public
landing page at app.py, reachable at /Screening. Clinician sign-in happens
here; once authenticated, the full upload -> analyze -> result ->
explanation -> report flow renders on this same page (no further page
navigation needed for the actual screening work).
"""
import datetime
import re
import sys
import time
from pathlib import Path

import numpy as np
import pydicom
import streamlit as st
import torch
import yaml
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
_logo_favicon = _ROOT / "assets" / "logo.png"
st.set_page_config(
    page_title="OncoSense -- Screening",
    page_icon=str(_logo_favicon) if _logo_favicon.exists() else "\U0001FA7B",
    layout="wide",
)

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.datasets import build_transforms  # noqa: E402
from src.gradcam import (  # noqa: E402
    GradCAM,
    get_target_layer,
    overlay_heatmap,
    compute_saliency_metrics,
    evaluate_lesion_correspondence,
    extract_breast_tissue_mask,
    COLORMAP_DICT,
)
from src.model import get_model  # noqa: E402
from src.prepare_cbis_ddsm import dicom_to_pil  # noqa: E402
from src.quality import check_mammogram_quality  # noqa: E402
from src.utils import get_device, load_checkpoint  # noqa: E402

from oncosense_common import (  # noqa: E402
    CONFIG_PATH,
    CSS,
    REPORT_EXPORT_CSS,
    STAGE_LOOKUP,
    clear_login_cookie,
    detect_stage_from_image,
    estimate_clinical_stage,
    get_browser_cookies,
    img_to_base64,
    log_screening_history,
    logo_img_tag,
    np_to_base64,
    render_floating_ai_widget,
    restore_login_from_cookies,
    save_report_html,
    save_screening_images,
    set_login_cookie,
)

st.markdown(CSS, unsafe_allow_html=True)
# No navbar on this page anymore, so reclaim the top space that was
# reserved to clear it.
st.markdown(
    '<style>.block-container { padding-top: 2rem !important; }</style>',
    unsafe_allow_html=True,
)
# Rendered before the auth gate below so the floating AI assistant shows
# on this page regardless of sign-in state (including the login screen).
render_floating_ai_widget()

# Ambient background (reuses the hero's glow + grain treatment) so the plain
# dark page -- especially the login screen, which has no hero image of its
# own -- doesn't look flat. Fixed + z-index:-1 so it sits behind everything
# and never intercepts clicks.
st.markdown(
    """
    <div style="position:fixed; inset:0; z-index:-1; overflow:hidden; pointer-events:none;">
      <div class="hero-glow"></div>
      <div class="hero-grain"></div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    # ResNet-18 is a lightweight ~44MB model that executes in <50ms on CPU.
    # Using CPU for Streamlit web serving ensures thread-safe inference across
    # concurrent sessions and prevents CUDA multi-threading context errors (CUDA 201/999).
    device = torch.device("cpu")
    model = get_model(cfg["model"]["arch"], cfg["num_classes"], cfg["model"]["pretrained"]).to(device)
    ckpt_path = Path(cfg["output"]["checkpoint_dir"]) / f"{cfg['output']['run_name']}_best.pth"
    model = load_checkpoint(model, ckpt_path, device)
    model.eval()
    target_layer = get_target_layer(model, cfg["model"]["arch"])
    cam_engine = GradCAM(model, target_layer)
    return cfg, device, model, cam_engine


def load_uploaded_image(uploaded_file, image_size: int):
    """Returns (preprocessed PIL image, dict of source metadata for the details panel)."""
    name = uploaded_file.name.lower()
    if name.endswith(".dcm"):
        ds = pydicom.dcmread(uploaded_file)
        spacing = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
        meta = {
            "Source format": "DICOM",
            "Original size": f"{ds.Rows} x {ds.Columns} px",
            "Photometric interpretation": getattr(ds, "PhotometricInterpretation", "n/a"),
            "Laterality / view": f"{getattr(ds, 'ImageLaterality', 'n/a')} {getattr(ds, 'ViewPosition', '')}".strip(),
            "_original_width_px": int(ds.Columns),
            "_pixel_spacing_mm": float(spacing[0]) if spacing else None,
        }
        return dicom_to_pil(ds, image_size), meta

    original = Image.open(uploaded_file).convert("RGB")
    meta = {
        "Source format": (name.rsplit(".", 1)[-1]).upper(),
        "Original size": f"{original.size[1]} x {original.size[0]} px",
        "_original_width_px": original.size[0],
        "_pixel_spacing_mm": None,
    }
    return original.resize((image_size, image_size)), meta


def risk_tone(malignant_prob: float) -> str:
    return "amber" if malignant_prob >= 0.5 else "green"


def build_clinical_narrative(is_elevated: bool, age, family_history, previous_cancer,
                              previous_biopsy, previous_abnormal_mammogram,
                              menopausal_status, breast_symptoms) -> dict:
    """Rule-based (not model-generated) clinical narrative built from the patient
    intake form + the model's benign/malignant call. This is static domain
    knowledge, not a second AI prediction -- the model only produces the
    malignant probability; everything else here is templated text so the
    report reads as a complete clinical summary instead of a bare number."""
    risk_factors = []
    try:
        if age is not None and int(age) >= 50:
            risk_factors.append("Age 50 or older (breast cancer risk increases with age)")
    except (TypeError, ValueError):
        pass
    if family_history == "Yes":
        risk_factors.append("Family history of breast cancer")
    if previous_cancer == "Yes":
        risk_factors.append("Personal history of breast cancer")
    if previous_biopsy == "Yes":
        risk_factors.append("Previous breast biopsy")
    if previous_abnormal_mammogram == "Yes":
        risk_factors.append("Previous abnormal mammogram finding")
    if menopausal_status == "Post":
        risk_factors.append("Postmenopausal status (prolonged hormonal exposure)")
    if breast_symptoms and breast_symptoms != "None":
        risk_factors.append(f"Reported symptom: {breast_symptoms}")
    if not risk_factors:
        risk_factors.append("No major risk factors identified from the information provided")

    if is_elevated:
        next_steps = (
            "Consult a breast specialist or radiologist promptly to review this result. "
            "Additional diagnostic imaging (e.g. a diagnostic mammogram or ultrasound) or a "
            "biopsy may be recommended to confirm or rule out malignancy. Correlate the "
            "Grad-CAM highlighted region with the patient's clinical history, and do not "
            "delay follow-up care."
        )
    else:
        next_steps = (
            "Continue routine screening mammograms on the schedule recommended by a healthcare "
            "provider (typically annually or every two years, depending on age and personal risk). "
            "Report any new breast symptoms -- a lump, pain, skin changes, or nipple discharge -- "
            "to a doctor promptly, even between scheduled screenings."
        )

    return {
        "risk_factors": risk_factors,
        "why_it_may_occur": (
            "Breast cancer does not have a single cause. Research points to a combination of "
            "factors: increasing age, inherited genetic mutations (such as BRCA1/BRCA2), a family "
            "history of the disease, hormonal factors (prolonged estrogen exposure, late menopause, "
            "or hormone therapy), and lifestyle factors (obesity, alcohol use, physical inactivity). "
            "Having one or more of these factors does not mean cancer will develop, and their absence "
            "does not rule it out -- they describe population-level risk, not individual certainty."
        ),
        "next_steps": next_steps,
        "risk_reduction": [
            "Maintain a healthy body weight",
            "Engage in regular physical activity",
            "Avoid smoking and tobacco use",
            "Limit alcohol consumption",
            "Follow recommended breast cancer screening schedules",
            "Discuss personal and family risk factors with a healthcare provider, including "
            "genetic counseling if indicated",
        ],
    }


def render_login() -> None:
    """Gated clinician sign-in screen shown before the screening workspace."""
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
                  <p>Sign in with your clinical credentials to access the<br>
                  mammography screening workspace.</p>
                </div>
                <div class="login-divider"></div>
                """,
                unsafe_allow_html=True,
            )

            error = st.session_state.pop("login_error", None)
            if error:
                st.markdown(f'<div class="login-error">&#9888;&nbsp; {error}</div>', unsafe_allow_html=True)

            with st.form("doctor_login_form", clear_on_submit=False):
                full_name = st.text_input("Full name", placeholder="Dr. Jane Smith")
                email = st.text_input("Work email", placeholder="jane.smith@hospital.org")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                st.checkbox("Keep me signed in on this device", key="login_remember")
                submitted = st.form_submit_button("Sign in to OncoSense")

            if submitted:
                if not full_name.strip() or not email.strip() or not password:
                    st.session_state["login_error"] = "Please fill in your name, email, and password to continue."
                elif "@" not in email or "." not in email.split("@")[-1]:
                    st.session_state["login_error"] = "Enter a valid work email address."
                elif len(password) < 4:
                    st.session_state["login_error"] = "Password must be at least 4 characters."
                else:
                    st.session_state["authenticated"] = True
                    st.session_state["doctor_name"] = full_name.strip()
                    st.session_state["doctor_email"] = email.strip()
                    set_login_cookie(
                        full_name.strip(), email.strip(),
                        remember=st.session_state.get("login_remember", True),
                    )
                st.rerun()



browser_cookies = get_browser_cookies()

st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("doctor_name", "")
st.session_state.setdefault("doctor_email", "")

if not st.session_state["authenticated"]:
    restored = restore_login_from_cookies(browser_cookies)
    if restored:
        st.session_state["authenticated"] = True
        st.session_state["doctor_name"], st.session_state["doctor_email"] = restored

if not st.session_state["authenticated"]:
    render_login()
    st.stop()

# ---------------------------------------------------------------- Sidebar --
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
        <a class="btn btn-outline" href="/History" target="_self"
           style="width:100%; justify-content:center; box-sizing:border-box; margin-bottom:10px;">
          Patient Screening History &rarr;
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
        # Same race as the login cookie write: give the delete component a
        # moment to actually run in the browser before rerunning, otherwise
        # the next run can still see the old cookie and silently log back in.
        time.sleep(0.6)
        st.rerun()

# ------------------------------------------------------------- Main flow --
cfg, device, model, cam_engine = load_model_and_config()

# ----------------------------------------------------------------- Screening
st.markdown('<div id="upload-section"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <section class="section section-narrow">
      <div class="section-header">
        <h2 class="section-heading">Analyze a<br><em>mammogram.</em></h2>
        <p class="section-sub">Upload a mammography image (DICOM, PNG, or JPEG) and run the research
        model. The model looks for benign vs. malignant patterns and highlights the region that
        drove its decision.</p>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.container(key="patient_info_card", border=True):
    st.markdown('<h2 style="margin-bottom:16px;">Patient Information</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text_input("Patient Name", key="patient_name", placeholder="Jane Doe")
        st.text_input("Patient ID", key="patient_id", placeholder="P-00001")
        st.selectbox("Family History of Breast Cancer", ["No", "Yes"], key="family_history")
        st.selectbox("Breast Symptoms", ["None", "Lump", "Pain", "Nipple Discharge"], key="breast_symptoms")
    with col2:
        st.number_input("Age", min_value=0, max_value=120, value=50, step=1, key="patient_age")
        st.selectbox("Previous Breast Cancer", ["No", "Yes"], key="previous_cancer")
        st.selectbox("Menopausal Status", ["Pre", "Post", "Unknown"], key="menopausal_status")
    with col3:
        st.selectbox("Previous Breast Biopsy", ["No", "Yes"], key="previous_biopsy")
        st.selectbox("Previous Mammogram", ["No", "Yes"], key="previous_mammogram")
        st.selectbox("Previous Abnormal Mammogram", ["No", "Yes"], key="previous_abnormal_mammogram")



with st.container(key="mammogram_info_card", border=True):
    st.markdown('<h2 style="margin-bottom:16px;">Mammogram Information</h2>', unsafe_allow_html=True)
    st.radio("Breast Side", ["Left", "Right"], key="breast_side", horizontal=True)
    uploaded_file = st.file_uploader(
        "Upload a mammogram", type=["dcm", "png", "jpg", "jpeg"], label_visibility="collapsed",
    )

if uploaded_file is None:
    st.session_state["analyze_clicked"] = False
    st.markdown(
        """
        <div class="stepper">
          <div class="step active"><span>1</span><strong>Upload</strong></div>
          <div class="step"><span>2</span><strong>Analyze</strong></div>
          <div class="step"><span>3</span><strong>Result</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # A fresh file resets any previous "analyze" confirmation for the old one.
    if st.session_state.get("last_uploaded_name") != uploaded_file.name:
        st.session_state["analyze_clicked"] = False
        st.session_state["last_uploaded_name"] = uploaded_file.name

    quality_report = check_mammogram_quality(uploaded_file, target_size=cfg["image_size"])
    file_size_kb = uploaded_file.size / 1024

    if quality_report.status == "REJECTED":
        st.session_state["analyze_clicked"] = False
        st.markdown(
            """
            <div class="stepper">
              <div class="step done"><span>&#10003;</span><strong>Upload</strong></div>
              <div class="step active" style="border-color:#e74c3c;"><span style="color:#e74c3c;">&#10007;</span><strong>Quality Rejected</strong></div>
              <div class="step"><span>3</span><strong>Result</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="panel" style="border-left:4px solid #e74c3c;">
              <span class="status red" style="margin-bottom:12px; display:inline-block;">
                &#10007; Quality Screening Rejected &mdash; Image Unsuitable for Inference (Score: {quality_report.overall_score}/100)
              </span>
              <h3 style="margin:4px 0 10px; font-size:16px; color:#e74c3c;">Mammogram Quality Screening Rejection</h3>
              <p style="color:hsl(var(--text)); font-size:13px; line-height:1.6;">
                The uploaded file <strong>{uploaded_file.name}</strong> failed automated diagnostic quality assurance.
                Running ResNet18 on corrupted, unreadable, or unsuitable images causes unreliable predictions.
              </p>
              <div style="background:rgba(231,76,60,0.08); border-radius:8px; padding:12px 14px; margin:14px 0;">
                <strong style="color:#e74c3c; font-size:12.5px;">Detected Failure Reasons:</strong>
                <ul style="margin:6px 0 0 18px; padding:0; font-size:12.5px; color:hsl(var(--text)); line-height:1.7;">
                  {''.join(f'<li>{err}</li>' for err in quality_report.errors)}
                </ul>
              </div>
              <p style="font-size:12px; color:hsl(var(--muted)); line-height:1.6; margin:0;">
                <em>Research Value: Pre-inference quality screening intercepts unreadable files, blur, and severe noise, preventing false positives and hallucinations on out-of-distribution inputs.</em>
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Inspection: Quality Screening Telemetry & Checklist"):
            for chk in quality_report.checks:
                badge_color = "#58d68d" if chk.status == "PASS" else "#f39c12" if chk.status == "WARN" else "#e74c3c"
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid hsl(var(--border)); font-size:12.5px;">
                      <div><strong>{chk.name}</strong>: <span style="color:hsl(var(--muted));">{chk.message}</span></div>
                      <span style="padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; color:{badge_color}; border:1px solid {badge_color};">{chk.status}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.stop()

    # Image passed or acceptable with warning
    if quality_report.preprocessed_image is not None:
        pil_image = quality_report.preprocessed_image
        source_meta = {
            "Source format": quality_report.meta.get("format", "UNKNOWN"),
            "Original size": f"{quality_report.meta.get('original_height', 0)} x {quality_report.meta.get('original_width', 0)} px",
            "_original_width_px": quality_report.meta.get("original_width", cfg["image_size"]),
            "_pixel_spacing_mm": quality_report.meta.get("pixel_spacing_mm"),
        }
    else:
        pil_image, source_meta = load_uploaded_image(uploaded_file, cfg["image_size"])

    preview_b64 = img_to_base64(pil_image)

    # Automatically analyze lesion morphometry during preprocessing to detect Clinical TNM
    initial_stage_code, initial_stage_metrics = detect_stage_from_image(
        pil_image, malignant_prob=0.85, cam=None, meta=source_meta
    )
    detected_preview_t = initial_stage_metrics.get("clinical_t", "T1")
    detected_preview_n = initial_stage_metrics.get("clinical_n", "N0")
    detected_preview_m = initial_stage_metrics.get("clinical_m", "M0")

    if not st.session_state.get("analyze_clicked"):
        # ------------------------------------------------- Upload preview
        st.markdown(
            """
            <div class="stepper">
              <div class="step done"><span>&#10003;</span><strong>Upload</strong></div>
              <div class="step active"><span>2</span><strong>Analyze</strong></div>
              <div class="step"><span>3</span><strong>Result</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        is_warn = quality_report.status == "WARNING"
        status_badge_html = (
            f'<span class="status amber" style="margin-bottom:12px; display:inline-block;">&#9888; Quality Warning (Score: {quality_report.overall_score}/100) &middot; Suboptimal Quality</span>'
            if is_warn else
            f'<span class="status green" style="margin-bottom:12px; display:inline-block;">&#10003; Quality Verified (Score: {quality_report.overall_score}/100) &middot; Ready for Inference</span>'
        )

        st.markdown(
            f"""
            <div class="panel">
              <div class="upload-preview-grid">
                <div class="upload-preview-thumb">
                  <img src="data:image/png;base64,{preview_b64}" alt="Preview of the uploaded mammogram" />
                </div>
                <div>
                  {status_badge_html}
                  <div class="summary-row"><span>Filename</span><strong>{uploaded_file.name}</strong></div>
                  <div class="summary-row"><span>File size</span><strong>{file_size_kb:.1f} KB</strong></div>
                  <div class="summary-row"><span>Source format</span><strong>{source_meta.get('Source format', 'n/a')}</strong></div>
                  <div class="summary-row"><span>Original resolution</span><strong>{source_meta.get('Original size', 'n/a')}</strong></div>
                  <div class="summary-row"><span>Quality Score</span><strong style="color:{'#f39c12' if is_warn else '#58d68d'};">{quality_report.overall_score}/100 ({quality_report.status})</strong></div>
                  <div class="summary-row"><span>Preprocessed Clinical T / N / M</span><strong style="color:#F6A6C1;">{detected_preview_t} / {detected_preview_n} / {detected_preview_m}</strong></div>
                  <div class="summary-row"><span>Estimated Lesion Dimension</span><strong>~{initial_stage_metrics.get('estimated_lesion_mm', 0)} mm</strong></div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if quality_report.warnings:
            st.warning("⚠️ **Quality Advisory:** " + " | ".join(quality_report.warnings))

        with st.expander("Inspection: Quality Assurance Telemetry & Metrics"):
            for chk in quality_report.checks:
                badge_color = "#58d68d" if chk.status == "PASS" else "#f39c12" if chk.status == "WARN" else "#e74c3c"
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid hsl(var(--border)); font-size:12px;">
                      <div><strong>{chk.name}</strong> ({chk.value_display}): <span style="color:hsl(var(--muted));">{chk.message}</span></div>
                      <span style="padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; color:{badge_color}; border:1px solid {badge_color};">{chk.status}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        btn_label = "Analyze Mammogram (Proceed Despite Warning) →" if is_warn else "Analyze Mammogram →"
        if st.button(btn_label, key="analyze_btn"):
            st.session_state["analyze_clicked"] = True
            st.rerun()
        st.markdown(
            """
            <div class="disclaimer-footer">
              <span>&#8505;&#65039;</span>
              <span><strong>Educational coursework project.</strong> Results are illustrative only, not a diagnosis or medical advice.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # --------------------------------------------------- Analyzing state
        st.markdown(
            """
            <div class="stepper">
              <div class="step done"><span>&#10003;</span><strong>Upload</strong></div>
              <div class="step active"><span>2</span><strong>Analyze</strong></div>
              <div class="step"><span>3</span><strong>Result</strong></div>
            </div>
            <div class="eyebrow">Analyzing Mammogram</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="analyzing-visual">
              <img src="data:image/png;base64,{preview_b64}" alt="Mammogram currently being analyzed" />
              <div class="scan-overlay"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------- Inference
        with st.status("Analyzing mammogram...", expanded=True) as status:
            st.write("Loading image ✓")
            time.sleep(0.2)

            st.write("Preprocessing image & analyzing tissue morphology ✓")
            transform = build_transforms(cfg["image_size"], train=False)
            input_tensor = transform(pil_image).unsqueeze(0).to(device)
            time.sleep(0.2)

            st.write(f"Running neural network & computing Grad-CAM ({cfg['model']['arch']}) ✓")
            cams, probs, pred_idx_auto = cam_engine.generate_all_classes(input_tensor)
            time.sleep(0.2)

            threshold = cfg.get("decision_threshold", 0.5)
            malignant_prob = float(probs[1])
            benign_prob = float(probs[0])
            pred_idx = 1 if malignant_prob >= threshold else 0
            pred_label = cfg["class_names"][pred_idx]
            confidence = float(probs[pred_idx])
            tone = risk_tone(malignant_prob)
            is_elevated = tone == "amber"
            finding = "Elevated review signal" if is_elevated else "Low suspicion signal"
            score_percent = round(malignant_prob * 100)

            # Stage detection during image analysis (uses malignant CAM for lesion morphometry)
            st.write("Detecting cancer stage (Stage 0 to 4 analysis) ✓")
            detected_stage_code, stage_metrics = detect_stage_from_image(
                pil_image, malignant_prob=malignant_prob, cam=cams[1], meta=source_meta
            )
            time.sleep(0.2)

            # Staging is fully automatic — no manual override inputs
            clinical_t_entered = "Unknown"
            clinical_n_entered = "Unknown"
            clinical_m_entered = "Unknown"

            # If user left TNM as Unknown, detect and populate TNM using AJCC medical staging rules from image morphometry
            if detected_stage_code and stage_metrics:
                clinical_t = stage_metrics.get("clinical_t") if clinical_t_entered == "Unknown" else clinical_t_entered
                clinical_n = stage_metrics.get("clinical_n") if clinical_n_entered == "Unknown" else clinical_n_entered
                clinical_m = stage_metrics.get("clinical_m") if clinical_m_entered == "Unknown" else clinical_m_entered
                tnm_source_note = "Estimated via AJCC criteria & image morphometry" if clinical_t_entered == "Unknown" else "Clinician entered"
            elif not is_elevated:
                clinical_t = "T0" if clinical_t_entered == "Unknown" else clinical_t_entered
                clinical_n = "N0" if clinical_n_entered == "Unknown" else clinical_n_entered
                clinical_m = "M0" if clinical_m_entered == "Unknown" else clinical_m_entered
                tnm_source_note = "Benign / No lesion detected"
            else:
                clinical_t = clinical_t_entered
                clinical_n = clinical_n_entered
                clinical_m = clinical_m_entered
                tnm_source_note = "Pending clinical staging"

            clinical_stage_code = estimate_clinical_stage(clinical_t, clinical_n, clinical_m)

            # Prioritize detected stage from image analysis if malignant, with clinical TNM fallback/reconciliation
            estimated_stage = detected_stage_code if detected_stage_code else clinical_stage_code
            if not estimated_stage:
                estimated_stage = "0"
            stage_label, stage_desc = STAGE_LOOKUP.get(str(estimated_stage), ("Stage 0", ""))
            if detected_stage_code and stage_metrics and is_elevated:
                stage_desc = (
                    f"{stage_desc} (AJCC clinical classification: {clinical_t} [tumor size ~{stage_metrics['estimated_lesion_mm']}mm], "
                    f"{clinical_n} [regional nodes], {clinical_m} [metastasis indicator])."
                )

            resized = pil_image.resize((cfg["image_size"], cfg["image_size"]))
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            status.update(label="Analysis complete", state="complete", expanded=False)

        if st.button("← Analyze a different mammogram", key="reanalyze_btn"):
            st.session_state["analyze_clicked"] = False
            st.rerun()

        # -------------------------------------------------------- Stepper
        st.markdown(
            """
            <div class="stepper">
              <div class="step done"><span>&#10003;</span><strong>Upload</strong></div>
              <div class="step done"><span>&#10003;</span><strong>Analyze</strong></div>
              <div class="step active"><span>3</span><strong>Result</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------ Result grid
        st.markdown('<div id="result-section"></div>', unsafe_allow_html=True)
        ring_color = "#e8779c" if is_elevated else "#58d68d"
        bar_gradient = (
            "linear-gradient(90deg, #F6A6C1, #C9578B)" if is_elevated else "linear-gradient(90deg, #7fc9c5, #58d68d)"
        )

        st.markdown(
            f"""
            <div class="result-grid">
              <div class="panel signal-card">
                <div class="signal-top">
                  <div class="signal-ring" style="--pct:{score_percent}; --ring-color:{ring_color};">
                    <div class="signal-ring-inner">
                      <strong>{malignant_prob:.2f}</strong>
                      <span>malignant<br>probability</span>
                    </div>
                  </div>
                  <div>
                    <div class="eyebrow">Model output &middot; {cfg['model']['arch']}</div>
                    <h2>{finding}</h2>
                    {f'<p>The model flagged a region with patterns commonly associated with malignant findings in training data. Review recommended.</p><div style="margin-top:10px; display:inline-block; padding:5px 12px; border-radius:16px; background:rgba(232,119,156,0.18); border:1px solid rgba(232,119,156,0.4); color:#F6A6C1; font-weight:600; font-size:12.5px;">Detected Stage: {stage_label}</div>' if is_elevated else f'<p>Mammogram screening shows benign / normal tissue features.</p><div style="margin-top:10px; display:inline-block; padding:5px 12px; border-radius:16px; background:rgba(88,214,141,0.18); border:1px solid rgba(88,214,141,0.4); color:#58d68d; font-weight:600; font-size:12.5px;">Detected Stage: {stage_label}</div>'}
                  </div>
                </div>
                <div class="signal-bar"><div style="width:{score_percent}%; background:{bar_gradient};"></div></div>
                <div class="bar-labels"><span>0.00 (Benign)</span><span>{threshold:.2f} (Threshold)</span><span>1.00 (Malignant)</span></div>
              </div>
              <div class="panel summary-card">
                <h2>Screening Summary</h2>
                <div class="summary-row"><span>Patient Name</span><strong>{st.session_state.get('patient_name') or 'n/a'}</strong></div>
                <div class="summary-row"><span>Patient ID</span><strong>{st.session_state.get('patient_id') or 'n/a'}</strong></div>
                <div class="summary-row"><span>Age</span><strong>{st.session_state.get('patient_age', 'n/a')}</strong></div>
                <div class="summary-row"><span>Breast side / view</span><strong>{st.session_state.get('breast_side', 'n/a')} &middot; {st.session_state.get('mammogram_view', 'n/a')}</strong></div>
                <div class="summary-row"><span>Predicted class</span><strong>{pred_label.upper()}</strong></div>
                <div class="summary-row"><span>Confidence</span><strong>{confidence * 100:.1f}%</strong></div>
                <div class="summary-row"><span>Decision threshold</span><strong>{threshold:.3f}</strong></div>
                <div class="summary-row"><span>Cancer Staging</span><strong style="color:{'#F6A6C1' if is_elevated else '#58d68d'};">{stage_label}</strong></div>
                <div class="summary-row"><span>Clinical T / N / M</span><strong>{clinical_t} / {clinical_n} / {clinical_m} <span style="font-size:11px; font-weight:normal; opacity:0.8;">({tnm_source_note})</span></strong></div>
                <div class="summary-row"><span>Image Quality Check</span><strong style="color:{'#58d68d' if quality_report.status == 'PASSED' else '#f39c12'};">{quality_report.status} ({quality_report.overall_score}/100)</strong></div>

              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------- Clinical narrative
        narrative = build_clinical_narrative(
            is_elevated,
            st.session_state.get("patient_age"),
            st.session_state.get("family_history"),
            st.session_state.get("previous_cancer"),
            st.session_state.get("previous_biopsy"),
            st.session_state.get("previous_abnormal_mammogram"),
            st.session_state.get("menopausal_status"),
            st.session_state.get("breast_symptoms"),
        )
        risk_factors_html = "".join(f"<li>{rf}</li>" for rf in narrative["risk_factors"])
        risk_reduction_html = "".join(f"<li>{rr}</li>" for rr in narrative["risk_reduction"])

        st.markdown(
            f"""
            <div class="panel">
              <h2>AI Analysis & Recommendations</h2>
              <p class="panel-intro">A rule-based clinical summary combining the model's output with the
              patient information entered before upload. This is templated guidance, not a second AI
              prediction.</p>

              <h3 style="font-size:14px; margin:4px 0 8px; color:hsl(var(--text));">Cancer Risk & Stage Detection</h3>
              <p style="color:hsl(var(--muted)); font-size:13px; line-height:1.6;">
                Estimated probability of a suspicious finding: <strong style="color:hsl(var(--text));">{score_percent}%</strong><br>
                Detected Cancer Stage: <strong style="color:{'#F6A6C1' if is_elevated else '#58d68d'};">{stage_label}</strong>
              </p>
              {f'<p style="color:hsl(var(--muted)); font-size:12.5px; line-height:1.5; margin-top:4px;"><em>Staging details: {stage_desc}</em></p>' if is_elevated and stage_desc else ''}

              <h3 style="font-size:14px; margin:16px 0 8px; color:hsl(var(--text));">AI Analysis</h3>
              <p style="color:hsl(var(--muted)); font-size:13px; line-height:1.6;">
                {f"The mammogram appears <strong style='color:hsl(var(--text));'>Suspicious</strong>. The model detected a region with patterns commonly associated with malignant findings -- see the Grad-CAM heatmap below. During image preprocessing, lesion morphometry classified the suspicious lesion as <strong>{stage_label}</strong>." if is_elevated else "The mammogram appears <strong style='color:hsl(var(--text));'>Normal / Benign</strong>. Tissue architecture is within typical limits."}
              </p>

              <h3 style="font-size:14px; margin:16px 0 8px; color:hsl(var(--text));">Possible Risk Factors</h3>
              <ul style="color:hsl(var(--muted)); font-size:13px; line-height:1.8; margin:0 0 0 18px; padding:0;">
                {risk_factors_html}
              </ul>

              <h3 style="font-size:14px; margin:16px 0 8px; color:hsl(var(--text));">Why It May Occur</h3>
              <p style="color:hsl(var(--muted)); font-size:13px; line-height:1.6;">{narrative['why_it_may_occur']}</p>

              <h3 style="font-size:14px; margin:16px 0 8px; color:hsl(var(--text));">Risk Reduction</h3>
              <ul style="color:hsl(var(--muted)); font-size:13px; line-height:1.8; margin:0 0 0 18px; padding:0;">
                {risk_reduction_html}
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------------------------------------------------- Explanation
        st.markdown('<div id="explanation-section"></div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="page-header" style="margin-top:22px;">
              <div class="eyebrow">Explainable AI &mdash; Grad-CAM</div>
              <h2 class="section-heading" style="font-size:22px;">ResNet18 Feature<br>Attribution &amp; <em>Interpretability.</em></h2>
              <p class="section-sub">Visualizing gradient-weighted activation maps to evaluate which mammographic tissue regions influenced ResNet18's prediction.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="background:hsl(var(--surface)); border:1px solid hsl(var(--border)); border-radius:12px; padding:14px 18px 8px; margin-bottom:16px;">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <strong style="color:hsl(var(--text)); font-size:13px; text-transform:uppercase; letter-spacing:0.04em;">Attribution &amp; Heatmap Controls</strong>
                <span style="font-size:11.5px; color:hsl(var(--muted));">Interactive colormaps, opacity &amp; class target</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.3, 1.0, 1.0, 1.0])
        with ctrl_col1:
            target_choice = st.selectbox(
                "Attribution Target Class",
                options=[
                    f"ResNet18 Prediction ({pred_label.capitalize()})",
                    "Malignant Suspicion Evidence (Class 1)",
                    "Benign / Normal Evidence (Class 0)",
                ],
                index=0,
                key="gradcam_target_choice",
                help="Select which class prediction to explain. 'ResNet18 Prediction' explains the actual decision reached by the model.",
            )
        with ctrl_col2:
            colormap_choice = st.selectbox(
                "Colormap",
                options=["Turbo (Perceptual)", "Jet (Classic)", "Viridis", "Inferno", "Magma"],
                index=0,
                key="gradcam_colormap_choice",
                help="Scientific colormap used to render activation intensity.",
            )
        with ctrl_col3:
            alpha_val = st.slider(
                "Heatmap Opacity (α)",
                min_value=0.10,
                max_value=0.90,
                value=0.45,
                step=0.05,
                key="gradcam_alpha_slider",
                help="Transparency blending factor between original mammogram and heatmap.",
            )
        with ctrl_col4:
            saliency_thresh = st.slider(
                "Noise Cutoff Threshold",
                min_value=0.00,
                max_value=0.70,
                value=0.15,
                step=0.05,
                key="gradcam_thresh_slider",
                help="Suppresses background gradient noise below this threshold.",
            )

        anno_col1, anno_col2 = st.columns(2)
        with anno_col1:
            show_bbox_opt = st.checkbox("Highlight Peak Saliency Bounding Box", value=True, key="gradcam_bbox_cb")
        with anno_col2:
            show_crosshair_opt = st.checkbox("Mark Peak Saliency Centroid Crosshair", value=True, key="gradcam_cross_cb")

        # Select the active CAM based on user selection
        if "Malignant" in target_choice:
            active_cam = cams[1]
            active_label = "Malignant (Class 1)"
            active_class_idx = 1
        elif "Benign" in target_choice:
            active_cam = cams[0]
            active_label = "Benign (Class 0)"
            active_class_idx = 0
        else:
            active_cam = cams[pred_idx]
            active_label = f"ResNet18 Prediction ({pred_label.capitalize()})"
            active_class_idx = pred_idx

        cmap_name = colormap_choice.split()[0].lower()

        # Generate the overlays
        active_overlay = overlay_heatmap(
            resized,
            active_cam,
            alpha=alpha_val,
            colormap=cmap_name,
            threshold=saliency_thresh,
            show_bbox=show_bbox_opt,
            show_centroid=show_crosshair_opt,
            mode="overlay_only",
        )
        orig_img_arr = np.array(resized.convert("RGB"))
        overlay_heatmap_img = active_overlay

        img_b64 = np_to_base64(active_overlay)

        st.markdown(
            f"""
            <div class="explanation-layout">
              <div class="panel image-panel">
                <img src="data:image/png;base64,{img_b64}" alt="Grad-CAM heatmap over mammogram indicating regions that influenced ResNet18's prediction" />
                <div class="image-caption">
                  <span><span class="legend-dot"></span>Model attention region (Grad-CAM)</span>
                </div>
              </div>
              <div class="panel factors-panel">
                <h2>Class Probabilities</h2>
                <p class="panel-intro">Softmax output of the model's final layer for this image.</p>
                <div class="factor">
                  <div><strong>Malignant</strong><span>{'Above' if malignant_prob >= threshold else 'Below'} decision threshold ({threshold:.2f})</span></div>
                  <div class="factor-meter"><i style="width:{malignant_prob*100:.1f}%; background:#e875a5;"></i></div>
                  <b>{malignant_prob*100:.1f}%</b>
                </div>
                <div class="factor">
                  <div><strong>Benign</strong><span>Remaining softmax mass</span></div>
                  <div class="factor-meter"><i style="width:{benign_prob*100:.1f}%; background:#78bfbb;"></i></div>
                  <b>{benign_prob*100:.1f}%</b>
                </div>
              </div>
            </div>
            <p style="color:hsl(var(--muted)); font-size:11.5px; line-height:1.6; margin-top:14px;">
              Model visualizations are intended to help understand model behavior and should not be
              interpreted as clinical evidence.
            </p>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Input image side-by-side (preprocessed vs. heatmap)"):
            col1, col2 = st.columns(2)
            with col1:
                st.image(orig_img_arr, caption="Input (preprocessed)", use_container_width=True)
            with col2:
                st.image(active_overlay, caption="Grad-CAM overlay", use_container_width=True)

        # -------------------------------------------------------------- Report
        st.markdown('<div id="report-section"></div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="page-header" style="margin-top:22px;">
              <div class="eyebrow">Report</div>
              <h2 class="section-heading" style="font-size:22px;">Your analysis,<br><em>documented.</em></h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        report_sheet_inner = f"""
          <div class="report-brand">
            <div class="brand" style="color:#18313a;">{logo_img_tag(24)}<span>OncoSense</span></div>
            <span>Educational coursework project</span>
          </div>
          <div class="report-section">
            <h3>Patient History</h3>
            <div class="report-factors-grid">
              <div class="report-factor-item"><span>Patient Name</span><strong>{st.session_state.get('patient_name') or 'n/a'}</strong></div>
              <div class="report-factor-item"><span>Patient ID</span><strong>{st.session_state.get('patient_id') or 'n/a'}</strong></div>
              <div class="report-factor-item"><span>Age</span><strong>{st.session_state.get('patient_age', 'n/a')}</strong></div>
              <div class="report-factor-item"><span>Breast side / view</span><strong>{st.session_state.get('breast_side', 'n/a')} / {st.session_state.get('mammogram_view', 'n/a')}</strong></div>
              <div class="report-factor-item"><span>Menopausal status</span><strong>{st.session_state.get('menopausal_status', 'n/a')}</strong></div>
              <div class="report-factor-item"><span>Family history of breast cancer</span><strong>{st.session_state.get('family_history', 'n/a')}</strong></div>
              <div class="report-factor-item"><span>Previous breast cancer</span><strong>{st.session_state.get('previous_cancer', 'n/a')}</strong></div>
              <div class="report-factor-item"><span>Previous breast biopsy</span><strong>{st.session_state.get('previous_biopsy', 'n/a')}</strong></div>
              <div class="report-factor-item"><span>Breast symptoms</span><strong>{st.session_state.get('breast_symptoms', 'n/a')}</strong></div>
              <div class="report-factor-item"><span>Previous mammogram</span><strong>{st.session_state.get('previous_mammogram', 'n/a')}</strong></div>
              <div class="report-factor-item"><span>Previous abnormal mammogram</span><strong>{st.session_state.get('previous_abnormal_mammogram', 'n/a')}</strong></div>
            </div>
          </div>
          <div class="report-signal {'amber-bg' if is_elevated else 'green-bg'}">
            <div>
              <div class="eyebrow" style="color:inherit;">Algorithmic signal</div>
              <strong>{finding}</strong>
              <p>Malignant score: {malignant_prob:.2f} / 1.00 &bull; threshold {threshold:.2f}</p>
            </div>
            <div class="signal-pill {tone}">{malignant_prob:.2f}</div>
          </div>
          <div class="report-section">
            <h3>Class Probability Breakdown</h3>
            <div class="report-factors-grid">
              <div class="report-factor-item"><span>Malignant probability</span><strong>{malignant_prob*100:.1f}%</strong></div>
              <div class="report-factor-item"><span>Benign probability</span><strong>{benign_prob*100:.1f}%</strong></div>
              <div class="report-factor-item"><span>Decision threshold</span><strong>{threshold:.3f}</strong></div>
            </div>
          </div>
          <div class="report-section">
            <h3>Clinical Stage</h3>
            <div class="report-factors-grid">
              <div class="report-factor-item"><span>Clinical T / N / M</span><strong>{clinical_t} / {clinical_n} / {clinical_m}</strong></div>
              <div class="report-factor-item"><span>Cancer Stage</span><strong>{stage_label}</strong></div>
            </div>
            <p style="color:hsl(var(--muted)); font-size:11.5px; line-height:1.6; margin-top:6px;">
              {stage_desc or "Screening indicates benign tissue features."}
            </p>
          </div>
          <div class="report-section">
            <h3>AI Analysis</h3>
            <p>Mammogram appears <strong>{'Suspicious' if is_elevated else 'Normal / Benign'}</strong>.
            {f'The model detected a region with patterns commonly associated with malignant findings -- see the attached Grad-CAM heatmap for the highlighted area. Morphometry analysis categorized the lesion as {stage_label}.' if is_elevated else 'No suspicious malignant mass or microcalcification cluster was identified in the analyzed views.'}</p>
          </div>
          <div class="report-section">
            <h3>Possible Risk Factors</h3>
            <ul style="margin:0 0 0 18px; padding:0; line-height:1.8;">{risk_factors_html}</ul>
          </div>
          <div class="report-section">
            <h3>Why It May Occur</h3>
            <p>{narrative['why_it_may_occur']}</p>
          </div>
          <div class="report-section">
            <h3>Risk Reduction</h3>
            <ul style="margin:0 0 0 18px; padding:0; line-height:1.8;">{risk_reduction_html}</ul>
          </div>
        """

        st.markdown(f'<div class="panel report-sheet">{report_sheet_inner}</div>', unsafe_allow_html=True)

        report_export_html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8" />
<title>OncoSense Report - {uploaded_file.name}</title>
<style>{REPORT_EXPORT_CSS}</style>
</head><body>
<div class="report-sheet">{report_sheet_inner}</div>
</body></html>"""

        # Persist this screening to the clinician-facing patient history: the
        # full report sheet saved to disk (so it can be reopened later from
        # the History page), plus a CSV row pointing at it. Gated on a
        # session flag so reruns of this page (opening an expander, etc.)
        # don't append duplicate rows for the same upload.
        if st.session_state.get("history_logged_for") != uploaded_file.name:
            safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(uploaded_file.name).stem)
            safe_patient = re.sub(r"[^A-Za-z0-9_-]+", "_", st.session_state.get("patient_id") or "unknown")
            file_stem = f"{now_str.replace(':', '').replace(' ', '_')}_{safe_patient}_{safe_stem}"
            report_filename = f"{file_stem}.html"
            report_path = save_report_html(report_filename, report_export_html)
            image_path, gradcam_path = save_screening_images(file_stem, pil_image, overlay_heatmap_img)
            log_screening_history({
                "timestamp": now_str,
                "doctor_name": doctor_name,
                "patient_name": st.session_state.get("patient_name") or "",
                "patient_id": st.session_state.get("patient_id") or "",
                "age": st.session_state.get("patient_age", ""),
                "breast_side": st.session_state.get("breast_side", ""),
                "view": st.session_state.get("mammogram_view", ""),
                "family_history": st.session_state.get("family_history", ""),
                "previous_cancer": st.session_state.get("previous_cancer", ""),
                "previous_biopsy": st.session_state.get("previous_biopsy", ""),
                "previous_mammogram": st.session_state.get("previous_mammogram", ""),
                "previous_abnormal_mammogram": st.session_state.get("previous_abnormal_mammogram", ""),
                "menopausal_status": st.session_state.get("menopausal_status", ""),
                "breast_symptoms": st.session_state.get("breast_symptoms", ""),
                "filename": uploaded_file.name,
                "predicted_class": pred_label,
                "finding": finding,
                "confidence": f"{confidence:.4f}",
                "malignant_prob": f"{malignant_prob:.4f}",
                "report_path": report_path,
                "image_path": image_path,
                "gradcam_path": gradcam_path,
                "clinical_t": clinical_t if clinical_t != "Unknown" else "",
                "clinical_n": clinical_n if clinical_n != "Unknown" else "",
                "clinical_m": clinical_m if clinical_m != "Unknown" else "",
                "estimated_stage": stage_label,
            })
            st.session_state["history_logged_for"] = uploaded_file.name

        col_dl, col_view = st.columns(2)
        with col_dl:
            st.download_button(
                "Download Report",
                data=report_export_html.encode("utf-8"),
                file_name=f"oncosense_report_{Path(uploaded_file.name).stem}.html",
                mime="text/html",
                use_container_width=True,
                key="download_report_btn",
            )
        with col_view:
            st.markdown(
                '<a class="btn btn-outline" href="#report-section" '
                'style="width:100%; justify-content:center; box-sizing:border-box; margin-top:2px;">View Report</a>',
                unsafe_allow_html=True,
            )


# -------------------------------------------------------------------- Footer
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
