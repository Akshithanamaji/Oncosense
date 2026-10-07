"""Streamlit demo app for the breast cancer detector.

Upload a mammogram (DICOM .dcm, or an already-processed PNG/JPG) and get a
benign/malignant prediction with a Grad-CAM heatmap showing which region of
the image drove the decision.

Visual design follows the "OncoScreen" clinical-triage UI reference
(teal/navy palette, signal-ring result gauge, factors panel, printable
report sheet) -- restyled here in pure Streamlit + injected CSS rather than
adopting its Next.js stack, since the ML backend is Python/PyTorch.

Run with:
    streamlit run app.py
"""
import base64
import datetime
import io
import sys
from pathlib import Path

import numpy as np
import pydicom
import streamlit as st
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

from datasets import build_transforms  # noqa: E402
from gradcam import GradCAM, get_target_layer, overlay_heatmap  # noqa: E402
from model import get_model  # noqa: E402
from prepare_cbis_ddsm import dicom_to_pil  # noqa: E402
from utils import get_device, load_checkpoint  # noqa: E402

st.set_page_config(page_title="OncoSense", page_icon="🩺", layout="wide")

CONFIG_PATH = "configs/cbis_ddsm_v3.yaml"

# Measured on the official CBIS-DDSM test split, never touched during training
# (see README for the full evaluation methodology).
TEST_ACCURACY = 0.66
TEST_AUC = 0.7339
VAL_ACCURACY = 0.71


def img_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def np_to_base64(arr: np.ndarray) -> str:
    return img_to_base64(Image.fromarray(arr))


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --background: #f5f8fa;
  --foreground: #18313a;
  --card: #ffffff;
  --border: #dce7e9;
  --primary: #147d83;
  --navy: #163940;
  --blue: #e4f1f3;
  --amber: #b67820;
  --amber-bg: #fff4dc;
  --green: #278469;
  --green-bg: #e4f5ef;
  --muted-foreground: #6c7d82;
}

html, body, [class*="css"] { font-family: 'Inter', ui-sans-serif, system-ui, sans-serif; }
.stApp { background: var(--background); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stStatusWidget"] { visibility: hidden; }
.block-container { padding: 1.6rem 2.4rem 6rem; max-width: 1180px; }
h1, h2, h3, p { margin: 0; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: #fff;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

.brand { display: flex; align-items: center; gap: 9px; font-weight: 800; color: var(--navy); font-size: 18px; letter-spacing: -0.5px; }
.brand-mark { width: 32px; height: 32px; border-radius: 9px; background: var(--primary); color: #fff; display: grid; place-items: center; font-size: 16px; flex-shrink: 0; }
.brand-accent { color: var(--primary); }
.prototype-badge { margin: 20px 2px 18px; color: var(--primary); font-size: 11px; font-weight: 700; letter-spacing: 0.4px; text-transform: uppercase; display: flex; gap: 7px; align-items: center; }
.dot { width: 6px; height: 6px; border-radius: 50%; background: #5bb4a3; display: inline-block; }
.nav-list { display: flex; flex-direction: column; gap: 4px; margin-bottom: 22px; }
.nav-item { display: flex; align-items: center; gap: 10px; color: #6d7e82; text-decoration: none; padding: 9px 10px; border-radius: 8px; font-weight: 600; font-size: 13px; }
.nav-item:hover { background: #f2f7f7; color: var(--navy); }
.help-card { background: #f3f8f8; border: 1px solid #e0ecec; border-radius: 10px; padding: 12px; display: flex; align-items: flex-start; gap: 10px; color: var(--primary); margin-bottom: 14px; }
.help-card div { display: flex; flex: 1; flex-direction: column; }
.help-card strong { font-size: 12px; color: var(--navy); }
.help-card span { color: var(--muted-foreground); font-size: 11px; line-height: 1.4; }
.profile { border-top: 1px solid var(--border); padding: 14px 2px 0; display: flex; align-items: center; gap: 9px; }
.avatar { background: #dcecef; color: var(--primary); border-radius: 50%; width: 32px; height: 32px; display: grid; place-items: center; font-size: 11px; font-weight: 750; flex-shrink: 0; }
.profile strong { font-size: 12px; color: var(--navy); display: block; }
.profile span { color: var(--muted-foreground); font-size: 11px; }

/* Topbar context */
.topbar-context { display: flex; align-items: center; gap: 8px; color: #718287; font-size: 12px; padding: 4px 0 18px; border-bottom: 1px solid var(--border); margin-bottom: 22px; }
.breadcrumb { display: flex; gap: 7px; align-items: center; color: #829296; font-size: 12px; margin-bottom: 18px; }
.breadcrumb strong { color: #36545b; font-weight: 600; }

.page-header { margin-bottom: 22px; }
.eyebrow { text-transform: uppercase; font-size: 10px; letter-spacing: 1.1px; color: var(--primary); font-weight: 800; margin-bottom: 7px; }
.page-header h1 { font-size: 27px; letter-spacing: -0.6px; color: var(--navy); }
.page-header p { color: var(--muted-foreground); margin-top: 7px; max-width: 650px; font-size: 13.5px; line-height: 1.55; }

.notice { display: flex; gap: 12px; align-items: flex-start; background: #eaf5f5; border: 1px solid #cfe5e5; padding: 13px 15px; border-radius: 9px; margin-bottom: 20px; }
.notice strong { font-size: 13px; color: var(--navy); display: block; margin-bottom: 2px; }
.notice span { color: #61777c; font-size: 12px; }

/* Panels */
.panel { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 20px; margin-bottom: 20px; }
.panel h2 { font-size: 16px; color: var(--navy); margin-bottom: 4px; }
.panel-intro { color: var(--muted-foreground); font-size: 12px; margin: 5px 0 20px; }

/* Stepper */
.stepper { display: flex; align-items: center; margin-bottom: 24px; max-width: 820px; flex-wrap: wrap; gap: 4px 0; }
.step { display: flex; align-items: center; gap: 8px; color: #8a9a9d; font-size: 12px; flex: 1; min-width: 150px; }
.step:not(:last-child):after { content: ''; height: 1px; background: var(--border); flex: 1; margin: 0 10px; min-width: 20px; }
.step span { width: 24px; height: 24px; display: grid; place-items: center; border-radius: 50%; border: 1px solid #cbd8da; color: #87979a; font-size: 11px; flex-shrink: 0; }
.step.active { color: var(--primary); font-weight: 700; }
.step.active span, .step.done span { background: var(--primary); border-color: var(--primary); color: #fff; }
.step.done { color: #536a70; }

/* Status pills */
.status { border-radius: 20px; padding: 4px 11px; font-size: 10.5px; font-weight: 700; display: inline-block; }
.status.amber { background: var(--amber-bg); color: var(--amber); }
.status.green { background: var(--green-bg); color: var(--green); }

/* Result grid */
.result-grid { display: grid; grid-template-columns: 1.3fr 0.85fr; gap: 20px; margin-bottom: 4px; }
@media (max-width: 900px) { .result-grid { grid-template-columns: 1fr; } }
.signal-top { display: flex; align-items: center; gap: 22px; flex-wrap: wrap; }
.signal-ring { width: 108px; height: 108px; border-radius: 50%; display: flex; flex-direction: column; align-items: center; justify-content: center; flex-shrink: 0; background: conic-gradient(var(--ring-color) calc(var(--pct) * 1%), #e7eeee 0); }
.signal-ring-inner { width: 86px; height: 86px; border-radius: 50%; background: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.signal-ring strong { font-size: 24px; color: var(--navy); line-height: 1; }
.signal-ring span { color: #809195; font-size: 9.5px; margin-top: 3px; text-align: center; }
.signal-top h2 { font-size: 19px; margin-bottom: 6px; color: var(--navy); }
.signal-top p { color: var(--muted-foreground); line-height: 1.6; font-size: 13px; }
.signal-bar { height: 9px; background: #e7eeee; border-radius: 10px; margin-top: 24px; overflow: hidden; }
.signal-bar div { height: 100%; border-radius: 10px; }
.bar-labels { display: flex; justify-content: space-between; color: #85969a; font-size: 10px; margin-top: 6px; }
.result-callout { margin-top: 20px; display: flex; gap: 9px; align-items: flex-start; padding: 12px; background: #fff9ed; color: #84672f; font-size: 11.5px; line-height: 1.5; border-radius: 6px; }
.result-callout strong { color: #6b5223; }

.summary-row { border-top: 1px solid #edf1f1; padding: 11px 0; display: flex; justify-content: space-between; gap: 10px; color: #849295; font-size: 11.5px; }
.summary-row:first-of-type { border-top: none; }
.summary-row strong { color: #3c5960; font-size: 12px; text-align: right; }

/* Explanation layout */
.explanation-layout { display: grid; grid-template-columns: 0.95fr 1.05fr; gap: 20px; }
@media (max-width: 900px) { .explanation-layout { grid-template-columns: 1fr; } }
.image-panel img { width: 100%; border-radius: 8px; display: block; }
.image-caption { display: flex; justify-content: space-between; align-items: center; padding: 12px 2px 0; color: #77898d; font-size: 11px; }
.legend-dot { display: inline-block; width: 8px; height: 8px; border: 1px solid #d8913d; background: #d8913d66; border-radius: 50%; margin-right: 5px; }

.factor { display: grid; grid-template-columns: 1fr 130px 40px; gap: 16px; align-items: center; margin: 18px 0; }
.factor > div:first-child { display: flex; flex-direction: column; }
.factor strong { font-size: 12px; color: var(--navy); }
.factor span { color: #8b999c; font-size: 10px; margin-top: 2px; }
.factor-meter { height: 7px; border-radius: 10px; background: #edf1f1; overflow: hidden; }
.factor-meter i { display: block; height: 100%; border-radius: 10px; }
.factor b { font-size: 11px; color: var(--primary); text-align: right; }
.review-box { margin-top: 18px; display: flex; gap: 9px; align-items: flex-start; padding: 12px; background: #eaf5f5; color: #527277; font-size: 11.5px; line-height: 1.5; border-radius: 6px; }
.review-box strong { color: #2e5a61; display: block; margin-bottom: 2px; }

/* Report sheet */
.report-sheet { max-width: 900px; padding: 30px; box-shadow: 0 4px 16px #18313a0a; }
.report-brand { display: flex; justify-content: space-between; align-items: center; padding-bottom: 20px; border-bottom: 1px solid var(--border); color: #829397; font-size: 11px; }
.report-brand .brand { font-size: 15px; }
.report-title { display: flex; justify-content: space-between; align-items: flex-start; margin: 24px 0 18px; flex-wrap: wrap; gap: 10px; }
.report-title h2 { font-size: 21px; color: var(--navy); }
.report-meta { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; background: #f7fafa; border: 1px solid #e4eeee; border-radius: 7px; padding: 14px; }
@media (max-width: 900px) { .report-meta { grid-template-columns: repeat(2, 1fr); } }
.report-meta div { display: flex; flex-direction: column; gap: 3px; }
.report-meta span { color: #829194; font-size: 10px; }
.report-meta strong { color: #39565d; font-size: 11.5px; }
.report-signal { display: flex; align-items: center; justify-content: space-between; margin-top: 20px; padding: 16px; border-radius: 8px; gap: 12px; flex-wrap: wrap; }
.report-signal.amber-bg { background: #fff8e9; border: 1px solid #f0dfba; }
.report-signal.green-bg { background: #eaf8f2; border: 1px solid #c5edd9; }
.report-signal strong { font-size: 16px; color: #604c27; }
.report-signal.green-bg strong { color: #1c5e4b; }
.report-signal p { color: #8d784e; font-size: 11px; margin-top: 3px; }
.report-signal.green-bg p { color: #39856f; }
.signal-pill { width: 58px; height: 58px; border-radius: 50%; display: grid; place-items: center; border: 5px solid #e8c77f; color: #835f1f; font-size: 15px; font-weight: 750; flex-shrink: 0; }
.signal-pill.green { border-color: #79cbb2; color: #1c5e4b; }
.report-section { margin-top: 22px; }
.report-section h3 { margin-bottom: 8px; color: #36545b; font-size: 13px; font-weight: 700; }
.report-section p, .report-section li { color: #687a7e; font-size: 12px; line-height: 1.7; }
.report-section ul { padding-left: 20px; margin: 0; }
.report-factors-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 8px; }
@media (max-width: 900px) { .report-factors-grid { grid-template-columns: 1fr; } }
.report-factor-item { background: #f8fbfb; border: 1px solid var(--border); padding: 12px; border-radius: 6px; display: flex; flex-direction: column; gap: 4px; }
.report-factor-item span { font-size: 10px; color: #7f9194; }
.report-factor-item strong { font-size: 15px; color: var(--navy); }
.report-footer { margin-top: 30px; border-top: 1px solid var(--border); padding-top: 14px; color: #8b999b; font-size: 10px; }

/* Streamlit widget restyling */
[data-testid="stFileUploaderDropzone"] {
  background: #fbfdfd !important;
  border: 1.5px dashed #a9c4c6 !important;
  border-radius: 9px !important;
}
[data-testid="stFileUploaderDropzone"] button {
  background: var(--primary) !important;
  color: #fff !important;
  border-radius: 6px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] * { color: #5e777c !important; }
[data-testid="stFileUploaderDropzoneInstructions"] svg { color: var(--primary) !important; }
div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button {
  background: var(--primary);
  color: #fff;
  border: none;
  border-radius: 7px;
  font-weight: 700;
  padding: 0.5rem 1.1rem;
}
div[data-testid="stButton"] button:hover { background: #0d6e74; }

/* Fixed disclaimer footer */
.disclaimer-footer {
  position: fixed; bottom: 0; left: 0; right: 0;
  padding: 10px 24px; background: #eff6f6; border-top: 1px solid #dceaea;
  display: flex; gap: 8px; align-items: center; justify-content: center;
  color: #718488; font-size: 11px; z-index: 999;
}
.disclaimer-footer strong { color: #536c71; }

/* Global form widget restyling (used on the login screen and elsewhere) */
div[data-testid="stTextInput"] input {
  border-radius: 7px !important;
  border: 1px solid var(--border) !important;
  padding: 0.6rem 0.75rem !important;
  background: #fff !important;
}
div[data-testid="stTextInput"] input:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 2px #147d8322 !important;
}
div[data-testid="stTextInput"] label {
  font-size: 12.5px !important;
  font-weight: 650 !important;
  color: var(--navy) !important;
}
div[data-testid="stFormSubmitButton"] button {
  background: var(--primary) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 7px !important;
  font-weight: 700 !important;
  width: 100%;
  padding: 0.6rem 1.1rem !important;
}
div[data-testid="stFormSubmitButton"] button:hover { background: #0d6e74 !important; }
div[data-testid="stForm"] {
  border: none !important;
  padding: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
div[data-testid="stCheckbox"] label p { font-size: 12px !important; color: var(--muted-foreground) !important; }

/* Login screen */
.login-topspace { height: 4vh; }
.st-key-login_card {
  max-width: 440px;
  margin: 0 auto;
  border-radius: 16px !important;
  border: 1px solid var(--border) !important;
  box-shadow: 0 18px 48px #18313a14;
  padding: 28px 26px !important;
  background: var(--card) !important;
}
.login-badge {
  display: flex; justify-content: center; gap: 7px; align-items: center;
  color: var(--primary); font-size: 10.5px; font-weight: 700; letter-spacing: 0.5px;
  text-transform: uppercase; margin-bottom: 18px;
}
.login-logo { display: flex; flex-direction: column; align-items: center; gap: 10px; margin-bottom: 22px; text-align: center; }
.login-logo .brand-mark { width: 48px; height: 48px; border-radius: 13px; background: var(--primary); color: #fff; display: grid; place-items: center; font-size: 22px; }
.login-logo .brand-title { font-weight: 800; color: var(--navy); font-size: 23px; letter-spacing: -0.5px; }
.login-logo .brand-title .brand-accent { color: var(--primary); }
.login-logo p { color: var(--muted-foreground); font-size: 12.5px; margin-top: 2px; line-height: 1.5; }
.login-divider { border-top: 1px solid var(--border); margin: 6px 0 18px; }
.login-error {
  background: #fdeeec; border: 1px solid #f3c6bd; color: #9a3b2a;
  padding: 10px 12px; border-radius: 8px; font-size: 12px; margin-bottom: 14px;
  display: flex; gap: 8px; align-items: flex-start; line-height: 1.5;
}
.login-footer-note { text-align: center; margin-top: 16px; color: #8b999c; font-size: 10.5px; line-height: 1.5; }
.login-trust-row { display: flex; justify-content: center; gap: 18px; margin-top: 22px; color: #8a9a9d; font-size: 10.5px; }
.login-trust-row span { display: flex; align-items: center; gap: 5px; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model_and_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    device = get_device()
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
        meta = {
            "Source format": "DICOM",
            "Original size": f"{ds.Rows} x {ds.Columns} px",
            "Photometric interpretation": getattr(ds, "PhotometricInterpretation", "n/a"),
            "Laterality / view": f"{getattr(ds, 'ImageLaterality', 'n/a')} {getattr(ds, 'ViewPosition', '')}".strip(),
        }
        return dicom_to_pil(ds, image_size), meta

    original = Image.open(uploaded_file).convert("RGB")
    meta = {
        "Source format": (name.rsplit(".", 1)[-1]).upper(),
        "Original size": f"{original.size[1]} x {original.size[0]} px",
    }
    return original.resize((image_size, image_size)), meta


def risk_tone(malignant_prob: float) -> str:
    return "amber" if malignant_prob >= 0.5 else "green"


def render_login() -> None:
    """Gated clinician sign-in screen shown before the screening workspace."""
    st.markdown('<div class="login-topspace"></div>', unsafe_allow_html=True)
    _, col_center, _ = st.columns([1, 1.15, 1])
    with col_center:
        with st.container(key="login_card", border=True):
            st.markdown(
                """
                <div class="login-badge"><span class="dot"></span> Clinician access only</div>
                <div class="login-logo">
                  <span class="brand-mark">🩺</span>
                  <div class="brand-title">Onco<span class="brand-accent">Sense</span></div>
                  <p>Sign in with your clinical credentials to access the<br>
                  mammography screening workspace.</p>
                </div>
                <div class="login-divider"></div>
                """,
                unsafe_allow_html=True,
            )

            error = st.session_state.pop("login_error", None)
            if error:
                st.markdown(f'<div class="login-error">⚠️&nbsp; {error}</div>', unsafe_allow_html=True)

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
                st.rerun()

            st.markdown(
                """
                <div class="login-footer-note">
                  Educational coursework demo &bull; credentials are not verified against a real
                  hospital directory and no patient data is stored.
                </div>
                <div class="login-trust-row">
                  <span>🔒&nbsp; Session-only access</span>
                  <span>🩻&nbsp; CBIS-DDSM model</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("doctor_name", "")
st.session_state.setdefault("doctor_email", "")

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
          <span class="brand-mark">🩺</span>
          <span>Onco<span class="brand-accent">Sense</span></span>
        </div>
        <div class="prototype-badge"><span class="dot"></span> Educational demo · v1</div>
        <div class="nav-list">
          <a class="nav-item" href="#upload-section">📤&nbsp;&nbsp;Upload</a>
          <a class="nav-item" href="#result-section">📊&nbsp;&nbsp;Result</a>
          <a class="nav-item" href="#explanation-section">🔍&nbsp;&nbsp;Explanation</a>
          <a class="nav-item" href="#report-section">📋&nbsp;&nbsp;Report</a>
        </div>
        <div class="help-card">
          <span>📎</span>
          <div>
            <strong>About this tool</strong>
            <span>ResNet18 fine-tuned on CBIS-DDSM mammograms. Coursework project, not a medical device.</span>
          </div>
        </div>
        <div class="profile">
          <div class="avatar">{doctor_initials}</div>
          <div>
            <strong>{doctor_name}</strong>
            <span>{doctor_email or "Signed in · Attending"}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Sign out", key="logout_btn", use_container_width=True):
        for k in ("authenticated", "doctor_name", "doctor_email"):
            st.session_state.pop(k, None)
        st.rerun()

# ------------------------------------------------------------- Main flow --
cfg, device, model, cam_engine = load_model_and_config()

st.markdown(
    """
    <div class="topbar-context">🛡️&nbsp; Educational research demo &nbsp;•&nbsp; Not for clinical use</div>
    <div class="breadcrumb"><span>Workspace</span>&nbsp;›&nbsp;<strong>Screening</strong></div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div id="upload-section"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="page-header">
      <div class="eyebrow">New screening</div>
      <h1>Upload Mammogram for Analysis</h1>
      <p>Provide a single mammography image (DICOM, PNG, or JPEG). The model looks for
      benign vs. malignant patterns and highlights the region that drove its decision.</p>
    </div>
    <div class="notice">
      <span>ℹ️</span>
      <div>
        <strong>Format support</strong>
        <span>DICOM (.dcm) gets the same breast-region cropping used in training. PNG/JPG are assumed to already be a mammogram crop.</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a mammogram", type=["dcm", "png", "jpg", "jpeg"], label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="panel" style="text-align:center; color:#6c7d82; padding:44px 20px;">
          <div style="font-size:32px; margin-bottom:10px;">📤</div>
          <strong style="color:var(--navy); font-size:14px;">Upload a mammogram to begin</strong>
          <p style="margin-top:6px; font-size:12.5px;">Supports .dcm, .png, .jpg &bull; Result, explanation, and report sections will appear here.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="disclaimer-footer">
          <span>ℹ️</span>
          <span><strong>Educational coursework project.</strong> Results are illustrative only, not a diagnosis or medical advice.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# --------------------------------------------------------------- Inference
pil_image, source_meta = load_uploaded_image(uploaded_file, cfg["image_size"])

transform = build_transforms(cfg["image_size"], train=False)
input_tensor = transform(pil_image).unsqueeze(0).to(device)

cam, probs = cam_engine(input_tensor, class_idx=1)
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

resized = pil_image.resize((cfg["image_size"], cfg["image_size"]))
overlay = overlay_heatmap(resized, cam)
overlay_original, overlay_heatmap_img = np.split(overlay, 2, axis=1)
now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ------------------------------------------------------------------ Stepper
st.markdown(
    """
    <div class="stepper">
      <div class="step done"><span>✓</span><strong>Upload</strong></div>
      <div class="step done"><span>✓</span><strong>Analyze</strong></div>
      <div class="step active"><span>3</span><strong>Result</strong></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------- Result grid
st.markdown('<div id="result-section"></div>', unsafe_allow_html=True)
ring_color = "#b67820" if is_elevated else "#278469"
bar_gradient = (
    "linear-gradient(90deg, #7fc9c5, #d28a43)" if is_elevated else "linear-gradient(90deg, #7fc9c5, #278469)"
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
            <div class="eyebrow">Model output · {cfg['model']['arch']}</div>
            <h2>{finding}</h2>
            <p>{'The model flagged a region with patterns commonly associated with malignant findings in training data. Review recommended.' if is_elevated else 'No high-confidence malignant pattern was identified by the model for this image.'}</p>
          </div>
        </div>
        <div class="signal-bar"><div style="width:{score_percent}%; background:{bar_gradient};"></div></div>
        <div class="bar-labels"><span>0.00 (Benign)</span><span>{threshold:.2f} (Threshold)</span><span>1.00 (Malignant)</span></div>
        <div class="result-callout">
          <span>ℹ️</span>
          <span><strong>Guardrail:</strong> This score is a model output, not a probability of cancer. It must be interpreted alongside clinical judgment and is not validated for diagnostic use.</span>
        </div>
      </div>
      <div class="panel summary-card">
        <h2>Screening Summary</h2>
        <div class="summary-row"><span>Predicted class</span><strong>{pred_label.upper()}</strong></div>
        <div class="summary-row"><span>Confidence</span><strong>{confidence * 100:.1f}%</strong></div>
        <div class="summary-row"><span>Decision threshold</span><strong>{threshold:.3f}</strong></div>
        <div class="summary-row"><span>Image source</span><strong>{uploaded_file.name}</strong></div>
        <div class="summary-row"><span>Model checkpoint</span><strong>{cfg['output']['run_name']}_best.pth</strong></div>
        <div class="summary-row"><span>Analyzed at</span><strong>{now_str}</strong></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------- Explanation
st.markdown('<div id="explanation-section"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="page-header" style="margin-top:6px;">
      <div class="eyebrow">Interpretability</div>
      <h1 style="font-size:20px;">Understand the Signal</h1>
      <p>Grad-CAM highlights the pixels that most influenced the malignant score, and the
      class-probability breakdown shows how confident the model is either way.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

img_b64 = np_to_base64(overlay_heatmap_img)
orig_b64 = np_to_base64(overlay_original)

st.markdown(
    f"""
    <div class="explanation-layout">
      <div class="panel image-panel">
        <img src="data:image/png;base64,{img_b64}" alt="Grad-CAM heatmap" />
        <div class="image-caption">
          <span><span class="legend-dot"></span>Model attention region (Grad-CAM)</span>
        </div>
      </div>
      <div class="panel factors-panel">
        <h2>Class Probabilities</h2>
        <p class="panel-intro">Softmax output of the model's final layer for this image.</p>
        <div class="factor">
          <div><strong>Malignant</strong><span>{'Above' if malignant_prob >= threshold else 'Below'} decision threshold ({threshold:.2f})</span></div>
          <div class="factor-meter"><i style="width:{malignant_prob*100:.1f}%; background:#d28a43;"></i></div>
          <b>{malignant_prob*100:.1f}%</b>
        </div>
        <div class="factor">
          <div><strong>Benign</strong><span>Remaining softmax mass</span></div>
          <div class="factor-meter"><i style="width:{benign_prob*100:.1f}%; background:#78bfbb;"></i></div>
          <b>{benign_prob*100:.1f}%</b>
        </div>
        <div class="review-box">
          <span>🛡️</span>
          <div><strong>Recommended next step</strong>Correlate the highlighted region with clinical history. This tool is for education/coursework, not clinical triage.</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Input image side-by-side (preprocessed vs. heatmap)"):
    col1, col2 = st.columns(2)
    with col1:
        st.image(overlay_original, caption="Input (preprocessed)", use_container_width=True)
    with col2:
        st.image(overlay_heatmap_img, caption="Grad-CAM overlay", use_container_width=True)

# ------------------------------------------------------------------ Report
st.markdown('<div id="report-section"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="page-header" style="margin-top:6px;">
      <div class="eyebrow">Clinical-style report</div>
      <h1 style="font-size:20px;">Screening Report</h1>
      <p>A standardized, printable summary sheet -- print this page (Ctrl/Cmd+P) to export as PDF.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

meta_rows = "".join(f"<div><span>{k}</span><strong>{v}</strong></div>" for k, v in source_meta.items())

st.markdown(
    f"""
    <div class="panel report-sheet">
      <div class="report-brand">
        <div class="brand"><span class="brand-mark" style="width:24px;height:24px;font-size:13px;">🩺</span><span>Onco<span class="brand-accent">Sense</span></span></div>
        <span>Educational coursework project</span>
      </div>
      <div class="report-title">
        <div>
          <div class="eyebrow">{uploaded_file.name}</div>
          <h2>Digital Mammography Screening Summary</h2>
        </div>
        <span class="status {tone}">{finding} ({pred_label.upper()})</span>
      </div>
      <div class="report-meta">
        {meta_rows}
        <div><span>Resized to (model input)</span><strong>{cfg['image_size']} x {cfg['image_size']} px</strong></div>
        <div><span>Architecture</span><strong>{cfg['model']['arch']} (ImageNet-pretrained)</strong></div>
        <div><span>Training data</span><strong>CBIS-DDSM mammograms (mass + calc)</strong></div>
        <div><span>Processed at</span><strong>{now_str}</strong></div>
      </div>
      <div class="report-signal {'amber-bg' if is_elevated else 'green-bg'}">
        <div>
          <div class="eyebrow">Algorithmic signal</div>
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
        <h3>Model Performance (held-out official test set)</h3>
        <p>Measured on CBIS-DDSM's official test split (never used in training or validation):
        accuracy <strong>{TEST_ACCURACY*100:.0f}%</strong>, ROC-AUC <strong>{TEST_AUC:.2f}</strong>
        (validation-set accuracy {VAL_ACCURACY*100:.0f}%). Reported honestly rather than only on
        validation data, since the two differ here -- a known train/test distribution shift in
        this benchmark. See project README for full methodology.</p>
      </div>
      <div class="report-section">
        <h3>Interpretation Notice</h3>
        <p>This is a coursework/demo model. Its output does not represent an automated diagnosis
        or cancer probability and must never be used in isolation, without clinical examination,
        prior imaging, and professional clinical judgment.</p>
      </div>
      <div class="report-footer">Generated by OncoSense educational demo &bull; Not for diagnostic use</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-footer">
      <span>ℹ️</span>
      <span><strong>Educational coursework project.</strong> Results are illustrative only, not a diagnosis or medical advice.</span>
    </div>
    """,
    unsafe_allow_html=True,
)
