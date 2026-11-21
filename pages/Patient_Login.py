"""OncoSense -- patient sign in, registration, and password reset.

Public page (reachable at /Patient_Login) separate from the clinician sign-in
on pages/Screening.py: its own account table (outputs/patients.csv), its own
password hashing, and its own browser cookie, so a doctor session never
grants patient-portal access and vice versa. On success, redirects to
pages/Patient_Dashboard.py.

There's no email service wired up in this app, so "forgot password" can't
actually send an email -- the reset link is shown directly on screen instead,
clearly labeled as a local stand-in for that delivery step.
"""
import sys
import time
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
_logo_favicon = _ROOT / "assets" / "logo.png"
st.set_page_config(
    page_title="OncoSense -- Patient Login",
    page_icon=str(_logo_favicon) if _logo_favicon.exists() else "\U0001FA7B",
    layout="wide",
    initial_sidebar_state="collapsed",
)

sys.path.insert(0, str(_ROOT))

from oncosense_common import (  # noqa: E402
    CSS,
    authenticate_patient,
    generate_password_reset_token,
    get_browser_cookies,
    logo_img_tag,
    register_patient,
    render_floating_ai_widget,
    reset_password_with_token,
    restore_patient_login_from_cookies,
    set_patient_login_cookie,
)

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { display: none !important; }
    div[data-testid="stSidebarCollapsedControl"] { display: none !important; }
    .block-container { padding-top: 2rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div style="position:fixed; inset:0; z-index:-1; overflow:hidden; pointer-events:none;">
      <div class="hero-glow"></div>
      <div class="hero-grain"></div>
    </div>
    <div style="position:fixed; top:18px; left:24px; z-index:9000; display:flex; align-items:center; gap:10px;">
      <a href="/" target="_self" style="text-decoration:none; display:flex; align-items:center; gap:10px;">
        {logo_img_tag(34)}
        <span style="font-family:var(--font-display); font-style:italic; font-size:21px;
                     color:hsl(var(--text)); letter-spacing:-.01em;">OncoSense</span>
      </a>
    </div>
    <div style="position:fixed; top:18px; right:24px; z-index:9000; display:flex; gap:10px;">
      <a class="btn btn-outline" href="/" target="_self" style="padding:9px 18px; font-size:12.5px;">Home</a>
      <a class="btn btn-outline" href="/Screening" target="_self" style="padding:9px 18px; font-size:12.5px;">Doctor Login</a>
    </div>
    """,
    unsafe_allow_html=True,
)
render_floating_ai_widget()

browser_cookies = get_browser_cookies()

st.session_state.setdefault("patient_authenticated", False)
st.session_state.setdefault("patient_user_id", "")
st.session_state.setdefault("patient_id", "")
st.session_state.setdefault("patient_name", "")
st.session_state.setdefault("patient_email", "")

if not st.session_state["patient_authenticated"] and browser_cookies:
    restored = restore_patient_login_from_cookies(browser_cookies)
    if restored:
        st.session_state["patient_authenticated"] = True
        st.session_state["patient_user_id"] = restored["patient_user_id"]
        st.session_state["patient_id"] = restored["patient_id"]
        st.session_state["patient_name"] = restored["patient_name"]
        st.session_state["patient_email"] = restored["email"]

if st.session_state["patient_authenticated"]:
    st.switch_page("pages/Patient_Dashboard.py")

st.markdown('<div class="login-topspace"></div>', unsafe_allow_html=True)
_, col_center, _ = st.columns([1, 1.25, 1])

reset_token = st.query_params.get("reset_token", "")

with col_center:
    with st.container(key="login_card", border=True):
        st.markdown(
            f"""
            <div class="login-badge"><span class="dot"></span> Patient access</div>
            <div class="login-logo">
              {logo_img_tag(50)}
              <div class="brand-title">OncoSense</div>
              <p>Sign in to view your screening reports, manage<br>appointments, and more.</p>
            </div>
            <div class="login-divider"></div>
            """,
            unsafe_allow_html=True,
        )

        if reset_token:
            # ------------------------------------------------- Reset password
            st.markdown("**Set a new password**")
            reset_error = st.session_state.pop("reset_pw_error", None)
            if reset_error:
                st.markdown(f'<div class="login-error">&#9888;&nbsp; {reset_error}</div>', unsafe_allow_html=True)
            with st.form("reset_password_form"):
                new_pw = st.text_input("New password", type="password", key="reset_new_pw")
                confirm_pw = st.text_input("Confirm new password", type="password", key="reset_confirm_pw")
                reset_submitted = st.form_submit_button("Update password")
            if reset_submitted:
                if new_pw != confirm_pw:
                    st.session_state["reset_pw_error"] = "Passwords do not match."
                else:
                    ok, msg = reset_password_with_token(reset_token, new_pw)
                    if ok:
                        st.success(f"{msg}")
                        st.markdown(
                            '<a class="btn btn-primary" href="/Patient_Login" target="_self" '
                            'style="width:100%; justify-content:center; box-sizing:border-box; margin-top:10px;">'
                            'Go to sign in</a>',
                            unsafe_allow_html=True,
                        )
                        st.stop()
                    else:
                        st.session_state["reset_pw_error"] = msg
                st.rerun()
        else:
            tab_signin, tab_register, tab_forgot = st.tabs(["Sign In", "Register", "Forgot Password"])

            # --------------------------------------------------------- Sign in
            with tab_signin:
                error = st.session_state.pop("patient_login_error", None)
                if error:
                    st.markdown(f'<div class="login-error">&#9888;&nbsp; {error}</div>', unsafe_allow_html=True)
                with st.form("patient_login_form"):
                    pl_email = st.text_input("Email", key="pl_email", placeholder="jane.doe@example.com")
                    pl_password = st.text_input("Password", type="password", key="pl_password")
                    pl_remember = st.checkbox("Keep me signed in on this device", value=True, key="pl_remember")
                    pl_submitted = st.form_submit_button("Sign in")
                if pl_submitted:
                    if not pl_email.strip() or not pl_password:
                        st.session_state["patient_login_error"] = "Enter your email and password."
                    else:
                        result = authenticate_patient(pl_email, pl_password)
                        if not result:
                            st.session_state["patient_login_error"] = "Incorrect email or password."
                        else:
                            st.session_state["patient_authenticated"] = True
                            st.session_state["patient_user_id"] = result["patient_user_id"]
                            st.session_state["patient_id"] = result["patient_id"]
                            st.session_state["patient_name"] = result["patient_name"]
                            st.session_state["patient_email"] = result["email"]
                            set_patient_login_cookie(remember=pl_remember, **result)
                    st.rerun()

            # ---------------------------------------------------------- Register
            with tab_register:
                reg_error = st.session_state.pop("patient_register_error", None)
                if reg_error:
                    st.markdown(f'<div class="login-error">&#9888;&nbsp; {reg_error}</div>', unsafe_allow_html=True)
                with st.form("patient_register_form"):
                    reg_name = st.text_input("Full name", key="pr_name", placeholder="Jane Doe")
                    reg_email = st.text_input("Email", key="pr_email", placeholder="jane.doe@example.com")
                    reg_password = st.text_input("Password (min. 8 characters)", type="password", key="pr_password")
                    reg_confirm = st.text_input("Confirm password", type="password", key="pr_confirm")
                    reg_submitted = st.form_submit_button("Create account")
                if reg_submitted:
                    if reg_password != reg_confirm:
                        st.session_state["patient_register_error"] = "Passwords do not match."
                    else:
                        ok, msg = register_patient(reg_name, reg_email, reg_password)
                        if not ok:
                            st.session_state["patient_register_error"] = msg
                        else:
                            result = authenticate_patient(reg_email, reg_password)
                            st.session_state["patient_authenticated"] = True
                            st.session_state["patient_user_id"] = result["patient_user_id"]
                            st.session_state["patient_id"] = result["patient_id"]
                            st.session_state["patient_name"] = result["patient_name"]
                            st.session_state["patient_email"] = result["email"]
                            set_patient_login_cookie(remember=True, **result)
                    st.rerun()

            # ----------------------------------------------------- Forgot password
            with tab_forgot:
                st.caption(
                    "This demo has no email service connected, so your reset link is shown directly "
                    "below instead of being emailed -- in a production deployment it would be sent to "
                    "your inbox instead."
                )
                with st.form("patient_forgot_form"):
                    fp_email = st.text_input("Email", key="fp_email", placeholder="jane.doe@example.com")
                    fp_submitted = st.form_submit_button("Get reset link")
                if fp_submitted:
                    token = generate_password_reset_token(fp_email)
                    if token:
                        st.success("Reset link generated (valid for 30 minutes):")
                        st.markdown(
                            f'<a class="btn btn-primary" href="/Patient_Login?reset_token={token}" target="_self" '
                            'style="width:100%; justify-content:center; box-sizing:border-box;">'
                            'Open reset link</a>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("If an account exists with that email, a reset link has been generated.")

st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
