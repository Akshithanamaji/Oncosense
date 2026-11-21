"""OncoSense -- AI Breast Health Assistant.

A public, non-gated page (reachable at /AI_Assistant) where anyone -- a
visitor from the landing page, a doctor, or a patient without an account --
can ask general questions about mammograms and breast health, and get
medical terms explained in plain language. It never diagnoses and always
points specific-care questions back to a real doctor.

Deliberately lightweight like app.py/Education.py (no torch/ML imports) so
it loads fast and needs no sign-in. The Patient Portal has its own copy of
this assistant (pages/Patient_Dashboard.py, section "AI Assistant") that
additionally has access to that patient's own report -- this page is the
general-purpose, logged-out version linked from the landing page and from
the doctor sidebar.
"""
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
_logo_favicon = _ROOT / "assets" / "logo.png"
st.set_page_config(
    page_title="OncoSense -- AI Breast Health Assistant",
    page_icon=str(_logo_favicon) if _logo_favicon.exists() else "\U0001FA7B",
    layout="wide",
    initial_sidebar_state="collapsed",
)

sys.path.insert(0, str(_ROOT))

from oncosense_common import (  # noqa: E402
    CSS,
    ask_breast_health_assistant,
    is_ai_assistant_configured,
    logo_img_tag,
)

st.markdown(CSS, unsafe_allow_html=True)
# Public page, no clinician sidebar -- same treatment as app.py / Education.py.
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { display: none !important; }
    div[data-testid="stSidebarCollapsedControl"] { display: none !important; }
    .block-container { padding-top: 2rem !important; padding-bottom: 96px !important; }
    div[data-testid="stChatInput"] {
        position: fixed !important; bottom: 0 !important; left: 0 !important; right: 0 !important;
        top: auto !important; max-width: 760px; margin: 0 auto; z-index: 9500;
        background: hsl(var(--bg)); padding: 14px 24px 20px; border-top: 1px solid hsl(var(--stroke));
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Minimal brand mark (top-left) + quick links (top-right), mirroring
# Education.py's header treatment since this page also has no sidebar.
st.markdown(
    f"""
    <div style="position:fixed; top:18px; left:24px; z-index:9000;
                display:flex; align-items:center; gap:10px;">
      <a href="/" target="_self" style="text-decoration:none; display:flex; align-items:center; gap:10px;">
        {logo_img_tag(34)}
        <span style="font-family:var(--font-display); font-style:italic; font-size:21px;
                     color:hsl(var(--text)); letter-spacing:-.01em;">OncoSense</span>
      </a>
    </div>
    <div style="position:fixed; top:18px; right:24px; z-index:9000; display:flex; gap:10px;">
      <a class="btn btn-outline" href="/" target="_self" style="padding:9px 18px; font-size:12.5px;">Home</a>
      <a class="btn btn-outline" href="/Education" target="_self" style="padding:9px 18px; font-size:12.5px;">Education Center</a>
      <a class="btn btn-primary" href="/Screening" target="_self" style="padding:9px 18px; font-size:12.5px;">Start Screening &rarr;</a>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="section section-narrow" style="padding-top:2.4rem;">
      <div class="eyebrow-row"><span class="eyebrow-line"></span><span class="eyebrow">AI Assistant</span></div>
      <h1 class="section-heading" style="font-size:clamp(2.1rem, 4.2vw, 3.2rem);">AI Breast Health<br><em>Assistant.</em></h1>
      <p class="section-sub" style="max-width:680px;">
        Ask about mammograms and breast health, or get a medical term explained in plain language.
        No sign-in needed.
      </p>
      <div class="notice" style="margin-top:4px;">
        <strong>Educational only -- not a diagnosis.</strong>
        <span>This assistant cannot diagnose you, read your personal results, or recommend treatment.
        For your own screening report, sign in to the Patient Portal, where this assistant can also
        explain that specific report. Always bring questions about your own health to a doctor.</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

if not is_ai_assistant_configured():
    st.caption("The AI service isn't connected on this server yet -- the assistant will say so below.")

_, chat_col, _ = st.columns([1, 10, 1])
with chat_col:
    st.session_state.setdefault("ai_assistant_chat_history", [])

    if st.session_state["ai_assistant_chat_history"]:
        if st.button("Clear chat", key="ai_assistant_chat_clear_btn"):
            st.session_state["ai_assistant_chat_history"] = []
            st.rerun()

    if not st.session_state["ai_assistant_chat_history"]:
        st.caption("Try asking:")
        starter_questions = [
            "What is a mammogram and how does it work?",
            "What does BI-RADS mean on a report?",
            "What does 'breast density' mean?",
            "What are common breast cancer warning signs?",
        ]
        starter_cols = st.columns(len(starter_questions))
        for col, question in zip(starter_cols, starter_questions):
            with col:
                if st.button(question, key=f"ai_assistant_starter_{question}", use_container_width=True):
                    st.session_state["ai_assistant_chat_history"].append({"role": "user", "content": question})
                    with st.spinner("Thinking..."):
                        reply = ask_breast_health_assistant(st.session_state["ai_assistant_chat_history"])
                    st.session_state["ai_assistant_chat_history"].append({"role": "assistant", "content": reply})
                    st.rerun()

    for turn in st.session_state["ai_assistant_chat_history"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    if prompt := st.chat_input("Ask about mammograms or breast health..."):
        st.session_state["ai_assistant_chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = ask_breast_health_assistant(st.session_state["ai_assistant_chat_history"])
            st.markdown(reply)
        st.session_state["ai_assistant_chat_history"].append({"role": "assistant", "content": reply})

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ Footer
st.markdown(
    """
    <footer class="site-footer">
      <div class="footer-top">
        <div>
          <div class="footer-logo">OncoSense</div>
          <p class="footer-desc">AI-powered mammogram analysis and breast health education for research
          and learning purposes.</p>
        </div>
        <div class="footer-links">
          <a href="/" target="_self">Home</a>
          <a href="/Education" target="_self">Education Center</a>
          <a href="/Screening" target="_self">Doctor Login</a>
          <a href="/Patient_Login" target="_self">Patient Login</a>
        </div>
      </div>
      <div class="footer-bottom">
        <strong>Educational research demo</strong> &bull; Not for clinical use &bull; Not a substitute
        for professional medical advice, diagnosis, or treatment.
      </div>
    </footer>
    """,
    unsafe_allow_html=True,
)
