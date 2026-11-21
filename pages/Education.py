"""OncoSense -- Breast Health Education Center.

A public, non-gated reference page (reachable at /Education) covering
general breast-health education: warning signs, screening/mammogram
guidance, risk factors and common misconceptions, healthy-lifestyle
guidance, and FAQs about diagnosis and treatment.

Deliberately lightweight like app.py (no torch/ML imports) so it loads
fast and needs no clinician sign-in -- this content is meant for anyone,
not just the gated screening workspace.
"""
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
_logo_favicon = _ROOT / "assets" / "logo.png"
st.set_page_config(
    page_title="OncoSense -- Breast Health Education Center",
    page_icon=str(_logo_favicon) if _logo_favicon.exists() else "\U0001FA7B",
    layout="wide",
    initial_sidebar_state="collapsed",
)

sys.path.insert(0, str(_ROOT))

from oncosense_common import (  # noqa: E402
    BREAST_DENSITY_CATEGORIES,
    CANCER_STAGES,
    CANCER_SUBTYPES,
    CAUSES_ITEMS,
    CSS,
    DIAGNOSTIC_STEPS,
    FAQ_ITEMS,
    ICONS,
    LIFESTYLE_TIPS,
    MISCONCEPTIONS,
    RISK_TIERS,
    SCREENING_GUIDELINES,
    SCREENING_STEPS,
    SYMPTOMS_ITEMS,
    TREATMENT_OPTIONS,
    logo_img_tag,
    render_floating_ai_widget,
)

st.markdown(CSS, unsafe_allow_html=True)
# Public page, no clinician sidebar -- same treatment as app.py.
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

# Minimal brand mark (top-left) + quick links (top-right), mirroring app.py's
# header treatment since this page also has no sidebar.
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
      <a class="btn btn-primary" href="/Screening" target="_self" style="padding:9px 18px; font-size:12.5px;">Start Screening &rarr;</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ Intro
quick_links = [
    ("symptoms", "Warning Signs"),
    ("screening", "Screening & Mammograms"),
    ("risk-factors", "Risk Factors & Myths"),
    ("diagnosis", "Diagnosis & Staging"),
    ("treatment", "Treatment Options"),
    ("lifestyle", "Healthy Lifestyle"),
    ("faqs", "FAQs"),
]
quick_nav_html = "".join(
    f'<a class="quick-nav-link" href="#{anchor}">{label}</a>' for anchor, label in quick_links
)
st.markdown(
    f"""
    <section class="section section-narrow" style="padding-top:2.4rem;">
      <div class="eyebrow-row"><span class="eyebrow-line"></span><span class="eyebrow">Patient Education</span></div>
      <h1 class="section-heading" style="font-size:clamp(2.1rem, 4.2vw, 3.2rem);">Breast Health<br><em>Education Center.</em></h1>
      <p class="section-sub" style="max-width:680px;">
        Clinical-grade reference information on breast health -- warning signs, screening, risk
        stratification, the diagnostic pathway, staging, and treatment options -- written in plain
        language for patients.
      </p>
      <div class="quick-nav">{quick_nav_html}</div>
    </section>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------- Warning signs
symptoms_list_html = "".join(
    f"""<div class="awareness-list-item"><div class="awareness-list-icon">{ICONS[icon]}</div>
        <div><strong>{name}</strong><span>{desc}</span></div></div>"""
    for name, desc, icon in SYMPTOMS_ITEMS
)
st.markdown(
    f"""
    <section id="symptoms" class="section">
      <div class="section-header">
        <div class="eyebrow-row"><span class="eyebrow-line"></span><span class="eyebrow">Symptoms</span></div>
        <h2 class="section-heading">Know the<br>warning <em>signs.</em></h2>
        <p class="section-sub">Breast cancer often has early physical signs. Recognizing them -- and
        knowing when to seek screening -- can support earlier diagnosis and better outcomes.</p>
      </div>
      <div class="awareness-list" style="grid-template-columns:repeat(2,1fr);">{symptoms_list_html}</div>
      <div class="notice" style="margin-top:22px;">
        <strong>When to see a doctor</strong>
        <span>Make an appointment if you notice any of the signs above, especially if they are new,
        persistent for more than a couple of weeks, or don't go away on their own. None of these signs
        confirm breast cancer by itself -- many have other, non-cancerous causes -- but only a clinical
        evaluation can tell the difference.</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------- Screening & mammograms
guideline_cards_html = "".join(
    f"""<div class="guideline-card"><div class="age">{age}</div><h3>{title}</h3><p>{desc}</p></div>"""
    for age, title, desc in SCREENING_GUIDELINES
)
screening_steps_html = "".join(
    f"""<div class="pipeline-step"><div class="pipeline-num">{num}</div><strong>{title}</strong><p>{desc}</p></div>"""
    for num, title, desc in SCREENING_STEPS
)
density_rows_html = "".join(
    f"""<div class="summary-row"><span>{cat}</span><strong>{desc}</strong></div>"""
    for cat, desc in BREAST_DENSITY_CATEGORIES
)
st.markdown(
    f"""
    <section id="screening" class="section">
      <div class="section-header">
        <div class="eyebrow-row"><span class="eyebrow-line"></span><span class="eyebrow">Screening</span></div>
        <h2 class="section-heading">Mammograms &amp;<br>when to <em>screen.</em></h2>
        <p class="section-sub">A mammogram is a low-dose X-ray of breast tissue used to detect changes
        before they can be felt. General age-based guidance is below -- your own screening schedule
        should be set with a clinician based on your personal and family history.</p>
      </div>
      <div class="guideline-row">{guideline_cards_html}</div>
      <h3 style="font-size:13px; letter-spacing:.12em; text-transform:uppercase; color:hsl(var(--muted));
                 margin:34px 0 18px;">What to expect during a mammogram</h3>
      <div class="pipeline">{screening_steps_html}</div>
      <p class="section-sub" style="margin-top:26px; max-width:720px;">
        Newer <strong style="color:hsl(var(--text));">3D mammography (tomosynthesis)</strong> captures
        multiple angles for a clearer view, which can help in denser breast tissue. If your result
        mentions a <strong style="color:hsl(var(--text));">BI-RADS</strong> category, that's a
        standardized scale radiologists use to summarize findings -- see the FAQ section below for what
        the numbers mean.
      </p>
      <div class="panel" style="margin-top:22px;">
        <h2>Breast density (BI-RADS composition categories)</h2>
        <p class="panel-intro">Your report may note a density category -- denser tissue can lower a
        mammogram's sensitivity, which is why your clinician may recommend supplemental imaging.</p>
        {density_rows_html}
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------- Risk factors & misconceptions
risk_tier_html = "".join(
    f"""<div class="guideline-card"><div class="age" style="font-size:20px;">{tier}</div><p>{desc}</p></div>"""
    for tier, desc in RISK_TIERS
)
causes_html = "".join(
    f"""<div class="intro-card"><div class="num">{num}</div><h3>{title}</h3><p>{desc}</p></div>"""
    for num, title, desc in CAUSES_ITEMS
)
myth_fact_html = "".join(
    f"""<div class="myth-fact-card">
          <span class="tag myth">Myth</span>
          <div class="myth-text">&ldquo;{myth}&rdquo;</div>
          <span class="tag fact">Fact</span>
          <p class="fact-text">{fact}</p>
        </div>"""
    for myth, fact in MISCONCEPTIONS
)
st.markdown(
    f"""
    <section id="risk-factors" class="section">
      <div class="section-header">
        <div class="eyebrow-row"><span class="eyebrow-line"></span><span class="eyebrow">Risk &amp; Myths</span></div>
        <h2 class="section-heading">Understand your<br><em>risk.</em></h2>
        <p class="section-sub">There is no single cause of breast cancer -- researchers have instead
        identified several risk factors that raise the likelihood of it occurring. A clinician or
        genetic counselor can formally classify your individual risk tier.</p>
      </div>
      <div class="guideline-row" style="grid-template-columns:repeat(2,1fr);">{risk_tier_html}</div>
      <h3 style="font-size:13px; letter-spacing:.12em; text-transform:uppercase; color:hsl(var(--muted));
                 margin:34px 0 18px;">Contributing risk factors</h3>
      <div class="cards-row">{causes_html}</div>
      <h3 style="font-size:13px; letter-spacing:.12em; text-transform:uppercase; color:hsl(var(--muted));
                 margin:34px 0 18px;">Common misconceptions</h3>
      <div class="myth-fact-grid">{myth_fact_html}</div>
      <p style="color:hsl(var(--muted)); font-size:11.5px; line-height:1.6; margin-top:18px;">
        General clinical background, not personalized medical advice. Consult a healthcare professional
        for your own risk assessment.
      </p>
    </section>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------- Diagnosis & staging
diagnostic_steps_html = "".join(
    f"""<div class="pipeline-step"><div class="pipeline-num">{num}</div><strong>{title}</strong><p>{desc}</p></div>"""
    for num, title, desc in DIAGNOSTIC_STEPS
)
stage_cards_html = "".join(
    f"""<div class="guideline-card"><div class="age">{num}</div><h3>{title}</h3><p>{desc}</p></div>"""
    for num, title, desc in CANCER_STAGES
)
subtype_cards_html = "".join(
    f"""<div class="intro-card"><div class="num">{num}</div><h3>{title}</h3><p>{desc}</p></div>"""
    for num, title, desc in CANCER_SUBTYPES
)
st.markdown(
    f"""
    <section id="diagnosis" class="section">
      <div class="section-header">
        <div class="eyebrow-row"><span class="eyebrow-line"></span><span class="eyebrow">Diagnosis</span></div>
        <h2 class="section-heading">From finding<br>to <em>diagnosis.</em></h2>
        <p class="section-sub">A suspicious screening result goes through a structured diagnostic
        pathway before any treatment decision is made.</p>
      </div>
      <div class="pipeline">{diagnostic_steps_html}</div>
      <h3 style="font-size:13px; letter-spacing:.12em; text-transform:uppercase; color:hsl(var(--muted));
                 margin:34px 0 18px;">Stages, at a glance</h3>
      <div class="guideline-row">{stage_cards_html}</div>
      <h3 style="font-size:13px; letter-spacing:.12em; text-transform:uppercase; color:hsl(var(--muted));
                 margin:34px 0 18px;">Common molecular subtypes</h3>
      <div class="cards-row" style="grid-template-columns:repeat(2,1fr);">{subtype_cards_html}</div>
      <p style="color:hsl(var(--muted)); font-size:11.5px; line-height:1.6; margin-top:18px;">
        Staging and subtype are determined by a pathologist and oncology team from biopsy and imaging
        results -- this is a general reference, not an interpretation of any individual's results.
      </p>
    </section>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------ Treatment
treatment_cards_html = "".join(
    f"""<div class="intro-card"><div class="num">{num}</div><h3>{title}</h3><p>{desc}</p></div>"""
    for num, title, desc in TREATMENT_OPTIONS
)
st.markdown(
    f"""
    <section id="treatment" class="section">
      <div class="section-header">
        <div class="eyebrow-row"><span class="eyebrow-line"></span><span class="eyebrow">Treatment</span></div>
        <h2 class="section-heading">Treatment<br><em>options.</em></h2>
        <p class="section-sub">Most breast cancer treatment combines two or more of the approaches below,
        tailored by a multidisciplinary oncology team to the cancer's stage and subtype.</p>
      </div>
      <div class="cards-row" style="grid-template-columns:repeat(3,1fr);">{treatment_cards_html}</div>
    </section>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------- Healthy lifestyle
lifestyle_html = "".join(
    f"""<div class="intro-card"><div class="num">{num}</div><h3>{title}</h3><p>{desc}</p></div>"""
    for num, title, desc in LIFESTYLE_TIPS
)
st.markdown(
    f"""
    <section id="lifestyle" class="section">
      <div class="section-header">
        <div class="eyebrow-row"><span class="eyebrow-line"></span><span class="eyebrow">Prevention</span></div>
        <h2 class="section-heading">Healthy habits,<br>lower <em>risk.</em></h2>
        <p class="section-sub">No habit can guarantee prevention, but these well-established, general
        lifestyle factors are consistently associated with lower breast cancer risk.</p>
      </div>
      <div class="cards-row" style="grid-template-columns:repeat(3,1fr);">{lifestyle_html}</div>
    </section>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------- FAQs
st.markdown(
    """
    <section id="faqs" class="section section-narrow">
      <div class="section-header">
        <div class="eyebrow-row"><span class="eyebrow-line"></span><span class="eyebrow">FAQs</span></div>
        <h2 class="section-heading">Diagnosis &amp;<br>treatment <em>questions.</em></h2>
        <p class="section-sub">Straightforward answers to common questions patients ask after a
        screening or an abnormal result.</p>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)
_, faq_col, _ = st.columns([1, 10, 1])
with faq_col:
    for question, answer in FAQ_ITEMS:
        with st.expander(question):
            st.markdown(f'<p class="faq-answer">{answer}</p>', unsafe_allow_html=True)

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

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
          <a href="#symptoms">Warning Signs</a>
          <a href="#screening">Screening</a>
          <a href="#diagnosis">Diagnosis</a>
          <a href="#treatment">Treatment</a>
          <a href="#faqs">FAQs</a>
          <a href="/Screening" target="_self">Start Screening</a>
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

render_floating_ai_widget()
