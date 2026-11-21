"""OncoSense -- public landing / marketing page.

Visual design: a premium cinematic "AI research lab" aesthetic -- dark
charcoal theme, Inter + Instrument Serif italic typography, pink/magenta
accent gradient -- built as injected CSS (+ a small iframe->parent JS intro)
around the existing Streamlit widget tree.

The actual screening workspace (login, upload, ResNet18 inference, Grad-CAM
explanation, report) lives on a separate page: pages/Screening.py. This page
is purely informational/marketing and does not import torch/the ML stack,
so it loads fast. "Start Screening" buttons link to /Screening.

Run with:
    .venv\\Scripts\\python.exe -m streamlit run app.py
"""
from pathlib import Path

import streamlit as st

_logo_favicon = Path(__file__).resolve().parent / "assets" / "logo.png"
st.set_page_config(
    page_title="OncoSense",
    page_icon=str(_logo_favicon) if _logo_favicon.exists() else "\U0001FA7B",
    layout="wide", initial_sidebar_state="collapsed",
)

from oncosense_common import (  # noqa: E402
    CAUSES_ITEMS,
    CSS,
    FINAL_CTA_IMAGE_B64,
    HERO_IMAGE_B64,
    ICONS,
    PIPELINE_STEPS,
    PROJECT_CARDS,
    SYMPTOMS_ITEMS,
    TEST_ACCURACY,
    TEST_AUC,
    VAL_ACCURACY,
    img_data_url,
    load_bg_video_b64,
    logo_img_tag,
    inject_spa_navigation,
    render_floating_ai_widget,
    render_intro_loader,
    render_scroll_effects,
)

st.markdown(CSS, unsafe_allow_html=True)
inject_spa_navigation()
# This page never uses st.sidebar, but Streamlit's multipage framework still
# renders the sidebar chrome (with its page-switcher) on every page once a
# pages/ directory exists. Hide it completely here and let the main content
# reclaim the width -- the sidebar is only meaningful on the Screening page.
# Also: no floating navbar on this page (removed below), so reclaim the top
# space that was reserved to clear it.
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { display: none !important; }
    div[data-testid="stSidebarCollapsedControl"] { display: none !important; }
    .block-container { padding-top: 2rem !important; }
    .hero-section { margin-top: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
render_intro_loader()

# Minimal brand mark (logo + "OncoSense") in the header area -- intentionally
# NOT the old full navigation pill (links/CTA/hamburger), which was removed.
# Styles are inlined rather than reusing the .logo/.logo-span CSS classes,
# since those are scoped under .site-navbar (not present on this page).
st.markdown(
    f"""
    <div style="position:fixed; top:18px; left:24px; z-index:9000;
                display:flex; align-items:center; gap:10px;">
      <a href="#hero" style="text-decoration:none; display:flex; align-items:center;">
        {logo_img_tag(34)}
      </a>
      <a href="#hero" style="text-decoration:none; font-family:var(--font-display); font-style:italic;
                             font-size:21px; color:hsl(var(--text)); letter-spacing:-.01em;">OncoSense</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------- Hero
st.markdown(
    f"""
    <section id="hero" class="hero-section">
      <div class="hero-bg" style="background-image:url('{img_data_url(HERO_IMAGE_B64)}');"></div>
      <div class="hero-overlay"></div>
      <div class="hero-glow"></div>
      <div class="hero-grain"></div>
      <div class="hero-inner">
        <div>
          <div class="hero-eyebrow">AI-Powered Breast Cancer Research</div>
          <h1 class="hero-heading">Smarter imaging.<br>Earlier <em>insight.</em></h1>
          <p class="hero-desc">Explore AI-assisted mammogram analysis using deep learning and
          medical image processing -- upload a mammogram and see what a fine-tuned ResNet18
          model looks for.</p>
          <div class="hero-ctas">
            <a class="btn btn-primary" href="/Screening" target="_self">Doctor Login &rarr;</a>
            <a class="btn btn-primary" href="/Patient_Login" target="_self">Patient Login &rarr;</a>
            <a class="btn btn-outline" href="#awareness">Know the Signs</a>
          </div>
          <div class="hero-fineprint">Educational research demo<span class="dot-sep">&bull;</span>Not for clinical use</div>
        </div>
        <div class="hero-visual">
          <div class="hero-visual-frame">
            <img src="{img_data_url(HERO_IMAGE_B64)}" alt="Sample mammogram from the CBIS-DDSM training set used by this project" />
          </div>
          <div class="hero-glass">
            <div class="hero-glass-row"><span>Mammogram Analysis</span><strong>ResNet18</strong></div>
            <div class="hero-glass-row"><span class="hero-glass-status"><span class="status-dot"></span>Model Ready</span><strong>Research Mode</strong></div>
          </div>
        </div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------- Project intro
cards_html = "".join(
    f"""<div class="intro-card"><div class="num">{num}</div><h3>{title}</h3><p>{desc}</p></div>"""
    for num, title, desc in PROJECT_CARDS
)
st.markdown(
    f"""
    <section id="about" class="section">
      <div class="section-header">
        <div class="eyebrow-row"><span class="eyebrow-line"></span><span class="eyebrow">The Project</span></div>
        <h2 class="section-heading">Combining medical imaging<br>with <em>artificial intelligence.</em></h2>
        <p class="section-sub">This project explores how deep learning and computer vision can be
        applied to mammogram images for breast cancer research and image-based analysis.</p>
      </div>
      <div class="cards-row">{cards_html}</div>
    </section>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------- Awareness
symptoms_list_html = "".join(
    f"""<div class="awareness-list-item"><div class="awareness-list-icon">{ICONS[icon]}</div>
        <div><strong>{name}</strong><span>{desc}</span></div></div>"""
    for name, desc, icon in SYMPTOMS_ITEMS
)
_awareness_video_b64 = load_bg_video_b64("awareness_bg.mp4")
awareness_video_html = (
    f"""<video class="awareness-bg-video" autoplay muted loop playsinline preload="auto">
          <source src="data:video/mp4;base64,{_awareness_video_b64}" type="video/mp4" />
        </video>
        <div class="awareness-bg-video-scrim"></div>"""
    if _awareness_video_b64 else ""
)
st.markdown(
    f"""
    <section id="awareness" class="section">
      {awareness_video_html}
      <div class="awareness-content">
        <div class="section-header">
          <div class="eyebrow-row"><span class="eyebrow-line"></span><span class="eyebrow">Awareness</span></div>
          <h2 class="section-heading">Know the<br>warning <em>signs.</em></h2>
          <p class="section-sub">Breast cancer often has early physical signs. Recognizing them -- and
          knowing when to seek screening -- can support earlier diagnosis and better outcomes.</p>
        </div>
        <div class="awareness-list">{symptoms_list_html}</div>
        <p class="awareness-footnote" style="font-size:11.5px; line-height:1.6; margin-top:18px;">
          None of these signs confirm breast cancer on their own and many have other, non-cancerous
          causes. They are reasons to seek a clinical evaluation, not a diagnosis.
        </p>
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin-top:22px;">
          <a class="btn btn-outline" href="/Education" target="_self">
            Visit the Breast Health Education Center &rarr;
          </a>
          <a class="btn btn-outline" href="/AI_Assistant" target="_self">
            Ask the AI Breast Health Assistant &rarr;
          </a>
        </div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------- How it works
pipeline_html = "".join(
    f"""<div class="pipeline-step"><div class="pipeline-num">{num}</div><strong>{title}</strong><p>{desc}</p></div>"""
    for num, title, desc in PIPELINE_STEPS
)
st.markdown(
    f"""
    <section id="how-it-works" class="section">
      <div class="section-header">
        <div class="eyebrow-row"><span class="eyebrow-line"></span><span class="eyebrow">How It Works</span></div>
        <h2 class="section-heading">From image<br>to <em>AI insight.</em></h2>
      </div>
      <div class="pipeline">{pipeline_html}</div>
    </section>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- Dataset
tags_html = "".join(f'<div class="dataset-tag">{t}</div>' for t in ["DICOM", "MAMMOGRAPHY", "MEDICAL IMAGING", "RESEARCH DATA"])
st.markdown(
    f"""
    <section class="section">
      <div class="section-header">
        <div class="eyebrow-row"><span class="eyebrow-line"></span><span class="eyebrow">Dataset</span></div>
        <h2 class="section-heading">Built with<br>real medical <em>imaging data.</em></h2>
      </div>
      <div class="dataset-panel">
        <div>
          <div class="dataset-name">CBIS-DDSM</div>
          <p class="dataset-desc">The model was developed using mammography images from the
          CBIS-DDSM (Curated Breast Imaging Subset of DDSM) dataset, joining TCIA DICOM scans
          with case-description labels for benign vs. malignant findings.</p>
          <div class="dataset-metrics">
            <div class="dataset-metric"><strong>{TEST_ACCURACY*100:.0f}%</strong><span>Test accuracy</span></div>
            <div class="dataset-metric"><strong>{TEST_AUC:.2f}</strong><span>Test ROC-AUC</span></div>
            <div class="dataset-metric"><strong>{VAL_ACCURACY*100:.0f}%</strong><span>Val. accuracy</span></div>
          </div>
        </div>
        <div class="dataset-tags">{tags_html}</div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------- Why breast cancer occurs
causes_html = "".join(
    f"""<div class="intro-card"><div class="num">{num}</div><h3>{title}</h3><p>{desc}</p></div>"""
    for num, title, desc in CAUSES_ITEMS
)
st.markdown(
    f"""
    <section class="section">
      <div class="section-header">
        <div class="eyebrow-row"><span class="eyebrow-line"></span><span class="eyebrow">Background</span></div>
        <h2 class="section-heading">Why breast cancer<br><em>occurs.</em></h2>
        <p class="section-sub">Breast cancer develops when cells in breast tissue begin to grow and
        divide abnormally. There is no single cause -- researchers have instead identified several
        risk factors that raise the likelihood of it occurring.</p>
      </div>
      <div class="cards-row">{causes_html}</div>
    </section>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- Final CTA
st.markdown(
    f"""
    <section class="final-cta">
      <div class="final-cta-bg" style="background-image:url('{img_data_url(FINAL_CTA_IMAGE_B64)}');"></div>
      <div class="final-cta-overlay"></div>
      <div class="final-cta-content">
        <h2>Explore the future<br>of medical <em>AI.</em></h2>
        <p>Discover the technology and research behind OncoSense.</p>
        <div class="hero-ctas">
          <a class="btn btn-primary" href="/Screening" target="_self">Doctor Login &rarr;</a>
          <a class="btn btn-primary" href="/Patient_Login" target="_self">Patient Login &rarr;</a>
          <a class="btn btn-outline" href="/AI_Assistant" target="_self">Ask the AI Assistant &rarr;</a>
          <a class="btn btn-outline" href="#awareness">Know the Signs</a>
        </div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------- Footer
footer_links_html = (
    '<a href="#hero">Home</a>'
    '<a href="#awareness">Awareness</a>'
    '<a href="#how-it-works">How It Works</a>'
    '<a href="/Screening" target="_self">Doctor Login</a>'
    '<a href="/Patient_Login" target="_self">Patient Login</a>'
    '<a href="/Education" target="_self">Education Center</a>'
    '<a href="/AI_Assistant" target="_self">AI Assistant</a>'
    '<a href="#about">About</a>'
)
st.markdown(
    f"""
    <footer class="site-footer">
      <div class="footer-top">
        <div>
          <div class="footer-logo">OncoSense</div>
          <p class="footer-desc">AI-powered mammogram analysis for educational research.</p>
        </div>
        <div class="footer-links">{footer_links_html}</div>
      </div>
      <div class="footer-bottom">
        <strong>Educational research demo</strong> &bull; Not for clinical use
      </div>
    </footer>
    """,
    unsafe_allow_html=True,
)

# All .pipeline-step nodes above now exist in the DOM, so the scroll-driven
# GSAP effects can safely bind to them.
render_scroll_effects()

render_floating_ai_widget()
