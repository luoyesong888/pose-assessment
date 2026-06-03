import os

import streamlit as st
from app_pipeline import commit_assessment, process_assessment
from report_export import build_report_html, build_report_pdf_bytes
from records_store import load_store, search_patients, total_assessments

if "lang" not in st.session_state:
    st.session_state.lang = "zh"

LANG = {
    "zh": {
        "lang_btn": "EN",
        "nav_home": "首页", "nav_flow": "流程", "nav_upload": "上传", "nav_about": "关于",
        "badge": "AI 驱动 · 专业康复评估",
        "title": "运动康复\n智能评估系统",
        "subtitle": "上传正面与侧面照片，AI 自动分析姿态，生成专业康复建议报告。",
        "chip1": "双视角分析", "chip2": "AI 报告", "chip3": "进度追踪",
        "stat1_label": "视角", "stat1_val": "正面 + 侧面",
        "stat2_label": "分析引擎", "stat2_val": "DeepSeek AI",
        "stat3_label": "输出", "stat3_val": "结构化报告",
        "flow_title": "使用流程",
        "s1_title": "上传图片", "s1_desc": "提交正面与侧面全身照片，确保光线均匀、站姿自然放松。",
        "s2_title": "AI 分析", "s2_desc": "系统自动识别关节关键点，检查肩线、骨盆与躯干平衡状态。",
        "s3_title": "查看建议", "s3_desc": "生成可读性强的结构化报告，包含康复建议与后续跟踪指引。",
        "upload_title": "开始评估",
        "profile_title": "患者信息",
        "patient_ph": "输入患者姓名或编号",
        "front_label": "正面全身照片",
        "side_label": "侧面全身照片",
        "upload_hint": "支持 JPG、PNG，单文件不超过 200MB",
        "front_tip": "正对镜头站立，完整露出头部到足部",
        "side_tip": "侧身站立，避免弯腰或转头",
        "analyze": "立即开始 AI 姿态分析",
        "waiting": "请上传正面与侧面全身照片后开始分析",
        "ready": "照片已就绪，点击按钮开始分析",
        "archive_title": "患者档案",
        "archive_search": "搜索患者姓名 / 编号",
        "archive_latest": "最新评估",
        "archive_load": "加载到主面板",
        "archive_empty": "暂无历史档案",
        "archive_stats_patients": "患者数",
        "archive_stats_assessments": "评估数",
        "api_title": "DeepSeek 设置",
        "api_hint": "支持通过环境变量 DEEPSEEK_API_KEY 或侧边栏输入密钥。",
        "patient_code_title": "患者编号",
        "patient_code_ph": "输入患者编号（可选）",
        "report_section": "治疗师报告",
        "report_empty": "完成一次分析后，这里会显示专业报告。",
        "report_download": "下载报告",
        "report_generated": "报告已生成",
        "footer": "KinetiQ · 基于 2D 姿态关键点的运动康复筛查 · 非医学诊断 · Powered by MediaPipe + DeepSeek AI",
    },
    "en": {
        "lang_btn": "中文",
        "nav_home": "Home", "nav_flow": "Flow", "nav_upload": "Upload", "nav_about": "About",
        "badge": "AI Powered · Clinical Grade Assessment",
        "title": "Sports Rehab\nAssessment",
        "subtitle": "Upload front and side photos. AI analyzes posture and generates a professional recovery report.",
        "chip1": "Dual View", "chip2": "AI Report", "chip3": "Progress Track",
        "stat1_label": "Views", "stat1_val": "Front + Side",
        "stat2_label": "Engine", "stat2_val": "DeepSeek AI",
        "stat3_label": "Output", "stat3_val": "Structured Report",
        "flow_title": "How It Works",
        "s1_title": "Upload", "s1_desc": "Submit full-body front and side photos with even lighting and a relaxed natural stance.",
        "s2_title": "AI Analysis", "s2_desc": "The system detects joint landmarks and checks shoulder line, pelvis and trunk balance.",
        "s3_title": "Review", "s3_desc": "Receive a clear structured report with recovery recommendations and follow-up guidance.",
        "upload_title": "Start Assessment",
        "profile_title": "Patient Info",
        "patient_ph": "Enter patient name or ID",
        "front_label": "Front View Photo",
        "side_label": "Side View Photo",
        "upload_hint": "JPG or PNG, max 200MB per file",
        "front_tip": "Stand facing the camera, full body in frame",
        "side_tip": "Stand sideways, avoid bending or turning head",
        "analyze": "Start AI Posture Analysis",
        "waiting": "Please upload front and side photos to begin",
        "ready": "Photos ready — click to start analysis",
        "archive_title": "Patient Archive",
        "archive_search": "Search patient name / ID",
        "archive_latest": "Latest Assessment",
        "archive_load": "Load to Main Panel",
        "archive_empty": "No history yet",
        "archive_stats_patients": "Patients",
        "archive_stats_assessments": "Assessments",
        "api_title": "DeepSeek Settings",
        "api_hint": "You can use DEEPSEEK_API_KEY from the environment or enter it here.",
        "patient_code_title": "Patient ID",
        "patient_code_ph": "Enter patient ID (optional)",
        "report_section": "Therapist Report",
        "report_empty": "Run one analysis to generate a professional report here.",
        "report_download": "Download Report",
        "report_generated": "Report generated",
        "footer": "KinetiQ · 2D posture landmark-based rehab screening · Not medical diagnosis · Powered by MediaPipe + DeepSeek AI",
    },
}

t = LANG[st.session_state.lang]

if "store" not in st.session_state:
    st.session_state.store = load_store()
if "selected_patient_key" not in st.session_state:
    st.session_state.selected_patient_key = ""
if "current_report" not in st.session_state:
    st.session_state.current_report = None
if "deepseek_key" not in st.session_state:
    st.session_state.deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")

st.set_page_config(page_title="KinetiQ", layout="wide", page_icon="⚡")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

:root {
    --bg:       #f5efe6;
    --surface:  #fdfaf6;
    --card:     #ffffff;
    --text:     #1c1510;
    --muted:    #7a6e65;
    --accent:   #b85c2a;
    --accent2:  #d4845a;
    --line:     rgba(28,21,16,0.09);
    --max:      1100px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
}

.stApp { background: var(--bg); }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

[data-testid="stAppViewContainer"] > .main > div {
    padding-top: 0 !important;
}

/* ── TOPBAR ── */
.topbar {
    position: fixed;
    inset: 0 0 auto 0;
    z-index: 999;
    height: 64px;
    background: rgba(245,239,230,0.92);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--line);
    display: flex;
    align-items: center;
    padding: 0 clamp(20px, 4vw, 64px);
    justify-content: space-between;
}

.topbar-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: var(--text);
    letter-spacing: -0.5px;
}

.topbar-logo em {
    font-style: normal;
    color: var(--accent);
}

.topbar-nav {
    display: flex;
    gap: 28px;
    align-items: center;
}

.topbar-nav a {
    font-size: 13px;
    font-weight: 500;
    color: var(--muted);
    text-decoration: none;
    transition: color 0.2s;
}
.topbar-nav a:hover { color: var(--accent); }

/* lang button override */
.lang-wrap {
    position: fixed;
    top: 14px;
    right: clamp(20px, 4vw, 64px);
    z-index: 1001;
}

.lang-wrap div[data-testid="stButton"] > button {
    background: transparent !important;
    border: 1px solid var(--line) !important;
    color: var(--muted) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 6px 14px !important;
    border-radius: 999px !important;
    width: auto !important;
    box-shadow: none !important;
    letter-spacing: 0.5px;
}
.lang-wrap div[data-testid="stButton"] > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    transform: none !important;
    opacity: 1 !important;
}

/* ── HERO ── */
.hero {
    margin-top: 64px;
    padding: 48px clamp(20px, 4vw, 64px) 40px;
    background: var(--bg);
}

.hero-inner {
    max-width: var(--max);
    margin: 0 auto;
    display: grid;
    grid-template-columns: 1.4fr 1fr;
    gap: 32px;
    align-items: start;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 48px 52px;
    box-shadow: 0 20px 60px rgba(28,21,16,0.07);
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 999px;
    background: rgba(184,92,42,0.09);
    border: 1px solid rgba(184,92,42,0.18);
    color: var(--accent);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 20px;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(36px, 4.5vw, 56px);
    line-height: 1.05;
    letter-spacing: -0.03em;
    color: var(--text);
    margin-bottom: 16px;
    white-space: pre-line;
}

.hero-sub {
    font-size: 15px;
    color: var(--muted);
    line-height: 1.75;
    max-width: 46ch;
    margin-bottom: 24px;
}

.hero-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.chip {
    padding: 7px 14px;
    border-radius: 999px;
    border: 1px solid var(--line);
    background: var(--card);
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
}

.stat-col {
    display: grid;
    gap: 12px;
}

.stat-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 4px 16px rgba(28,21,16,0.04);
}

.stat-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
}

.stat-val {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: var(--text);
    letter-spacing: -0.03em;
}

/* ── FLOW ── */
.flow {
    padding: 40px clamp(20px, 4vw, 64px);
}

.flow-inner {
    max-width: var(--max);
    margin: 0 auto;
}

.section-eyebrow {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    opacity: 0.6;
    margin-bottom: 20px;
    padding-left: 2px;
}

.steps {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
}

.step {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 24px 22px;
}

.step-num {
    width: 32px; height: 32px;
    border-radius: 10px;
    background: rgba(184,92,42,0.10);
    color: var(--accent);
    font-size: 13px;
    font-weight: 800;
    display: grid;
    place-items: center;
    margin-bottom: 14px;
}

.step-title {
    font-family: 'DM Serif Display', serif;
    font-size: 17px;
    color: var(--text);
    margin-bottom: 8px;
}

.step-desc {
    font-size: 13px;
    color: var(--muted);
    line-height: 1.7;
}

/* ── UPLOAD PANEL ── */
.upload-section {
    padding: 0 clamp(20px, 4vw, 64px) 60px;
}

.upload-panel {
    max-width: var(--max);
    margin: 0 auto;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 36px 40px;
    box-shadow: 0 16px 48px rgba(28,21,16,0.06);
}

.panel-title {
    font-family: 'DM Serif Display', serif;
    font-size: 26px;
    color: var(--text);
    letter-spacing: -0.03em;
    margin-bottom: 28px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--line);
}

.field-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 8px;
}

.field-tip {
    font-size: 12px;
    color: var(--muted);
    margin-top: 8px;
    line-height: 1.5;
}

.divider {
    height: 1px;
    background: var(--line);
    margin: 28px 0;
}

/* inputs */
div[data-testid="stTextInput"] input {
    background: var(--card) !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-size: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 12px 16px !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(184,92,42,0.10) !important;
}

[data-testid="stFileUploader"] section {
    background: var(--card) !important;
    border: 1.5px dashed rgba(28,21,16,0.15) !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--accent) !important;
}

/* CTA button */
.cta-wrap div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 14px 24px !important;
    width: 100% !important;
    box-shadow: 0 10px 32px rgba(184,92,42,0.22) !important;
    letter-spacing: 0.2px !important;
}
.cta-wrap div[data-testid="stButton"] > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
.cta-wrap div[data-testid="stButton"] > button:disabled {
    background: rgba(28,21,16,0.08) !important;
    color: var(--muted) !important;
    box-shadow: none !important;
}

.status-box {
    padding: 14px 18px;
    border-radius: 12px;
    background: rgba(184,92,42,0.07);
    border: 1px solid rgba(184,92,42,0.14);
    color: var(--muted);
    font-size: 13px;
    line-height: 1.6;
    margin-bottom: 16px;
}

.status-box.ready {
    background: rgba(24,169,87,0.07);
    border-color: rgba(24,169,87,0.18);
    color: #1a7a42;
}

.report-shell {
    max-width: var(--max);
    margin: 28px auto 0;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 32px 36px;
    box-shadow: 0 16px 48px rgba(28,21,16,0.06);
}

.report-head {
    display: flex;
    justify-content: space-between;
    gap: 18px;
    align-items: flex-start;
    margin-bottom: 22px;
    padding-bottom: 18px;
    border-bottom: 1px solid var(--line);
}

.report-title {
    font-family: 'DM Serif Display', serif;
    font-size: 26px;
    color: var(--text);
    letter-spacing: -0.03em;
}

.report-sub {
    margin-top: 8px;
    color: var(--muted);
    line-height: 1.6;
    font-size: 13px;
    max-width: 64ch;
}

.report-meta {
    display: grid;
    gap: 8px;
    min-width: 220px;
}

.meta-pill {
    background: rgba(184,92,42,0.08);
    border: 1px solid rgba(184,92,42,0.12);
    color: var(--accent);
    border-radius: 999px;
    padding: 8px 12px;
    font-size: 12px;
    font-weight: 700;
    text-align: center;
}

.report-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 18px;
}

.report-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 16px 18px;
}

.report-card-label {
    color: var(--muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.report-card-value {
    color: var(--text);
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    letter-spacing: -0.03em;
}

.report-markdown {
    background: rgba(255,255,255,0.75);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 20px 22px;
    color: var(--text);
    line-height: 1.8;
}

.report-markdown h1,
.report-markdown h2,
.report-markdown h3 {
    margin-top: 0.8em;
    margin-bottom: 0.4em;
}

.report-actions {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 18px;
}

.archive-summary {
    font-size: 12px;
    color: var(--muted);
    line-height: 1.55;
    padding: 6px 0 0;
}

/* ── FOOTER ── */
.footer {
    padding: 20px clamp(20px, 4vw, 64px) 32px;
    text-align: center;
    font-size: 12px;
    color: var(--muted);
    opacity: 0.6;
    border-top: 1px solid var(--line);
}

@media (max-width: 900px) {
    .hero-inner { grid-template-columns: 1fr; padding: 32px 24px; }
    .steps { grid-template-columns: 1fr; }
    .topbar-nav { display: none; }
}
</style>
""", unsafe_allow_html=True)

# ── TOPBAR ──
st.markdown(f"""
<div class="topbar">
    <div class="topbar-logo">Kineti<em>Q</em></div>
    <div class="topbar-nav">
        <a href="#">{t['nav_home']}</a>
        <a href="#">{t['nav_flow']}</a>
        <a href="#">{t['nav_upload']}</a>
        <a href="#">{t['nav_about']}</a>
    </div>
</div>
""", unsafe_allow_html=True)

# lang button (fixed top right)
st.markdown('<div class="lang-wrap">', unsafe_allow_html=True)
if st.button(t["lang_btn"], key="lang_switch"):
    st.session_state.lang = "en" if st.session_state.lang == "zh" else "zh"
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"### {t['archive_title']}")
    search_query = st.text_input(t["archive_search"], key="archive_search_query")
    store = st.session_state.store
    patients = search_patients(store, search_query)
    st.metric(t["archive_stats_patients"], len(store.get("patients", [])))
    st.metric(t["archive_stats_assessments"], total_assessments(store))
    st.markdown("---")

    if patients:
        patient_labels = []
        patient_lookup = {}
        for patient in patients:
            label = f"{patient.get('patient_name', 'Unnamed')} · {patient.get('patient_code', '')}".strip()
            latest = patient.get("assessments", [{}])[0]
            label = f"{label} | {t['archive_latest']}: {latest.get('created_at', '-')}"
            patient_labels.append(label)
            patient_lookup[label] = patient

        selected_label = st.selectbox(" ", patient_labels, label_visibility="collapsed")
        selected_patient = patient_lookup[selected_label]
        st.caption(
            f"{selected_patient.get('patient_name', '')} · "
            f"{len(selected_patient.get('assessments', []))} records"
        )
        if st.button(t["archive_load"], use_container_width=True):
            latest = selected_patient.get("assessments", [{}])[0]
            st.session_state.selected_patient_key = selected_patient.get("patient_key", "")
            st.session_state.current_report = latest
            st.rerun()
    else:
        st.caption(t["archive_empty"])

    st.markdown("---")
    st.markdown(f"### {t['api_title']}")
    st.session_state.deepseek_key = st.text_input(
        "DeepSeek API Key",
        type="password",
        value=st.session_state.deepseek_key,
        placeholder="sk-...",
        label_visibility="collapsed",
    )
    st.caption(t["api_hint"])

# ── HERO ──
st.markdown(f"""
<div class="hero">
    <div class="hero-inner">
        <div>
            <div class="hero-badge">⚡ {t['badge']}</div>
            <h1 class="hero-title">{t['title']}</h1>
            <p class="hero-sub">{t['subtitle']}</p>
            <div class="hero-chips">
                <span class="chip">{t['chip1']}</span>
                <span class="chip">{t['chip2']}</span>
                <span class="chip">{t['chip3']}</span>
            </div>
        </div>
        <div class="stat-col">
            <div class="stat-card">
                <div class="stat-label">{t['stat1_label']}</div>
                <div class="stat-val">{t['stat1_val']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">{t['stat2_label']}</div>
                <div class="stat-val">{t['stat2_val']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">{t['stat3_label']}</div>
                <div class="stat-val">{t['stat3_val']}</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── FLOW ──
st.markdown(f"""
<div class="flow">
    <div class="flow-inner">
        <div class="section-eyebrow">{t['flow_title']}</div>
        <div class="steps">
            <div class="step">
                <div class="step-num">01</div>
                <div class="step-title">{t['s1_title']}</div>
                <div class="step-desc">{t['s1_desc']}</div>
            </div>
            <div class="step">
                <div class="step-num">02</div>
                <div class="step-title">{t['s2_title']}</div>
                <div class="step-desc">{t['s2_desc']}</div>
            </div>
            <div class="step">
                <div class="step-num">03</div>
                <div class="step-title">{t['s3_title']}</div>
                <div class="step-desc">{t['s3_desc']}</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── UPLOAD PANEL ──
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown(f'<div class="upload-panel">', unsafe_allow_html=True)
st.markdown(f'<div class="panel-title">{t["upload_title"]}</div>', unsafe_allow_html=True)

# patient
st.markdown(f'<div class="field-label">{t["profile_title"]}</div>', unsafe_allow_html=True)
patient_name = st.text_input("p", placeholder=t["patient_ph"], label_visibility="collapsed")
patient_code = st.text_input(
    "patient_code",
    placeholder=t["patient_code_ph"],
    label_visibility="collapsed",
)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# uploads
col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown(f'<div class="field-label">{t["front_label"]}</div>', unsafe_allow_html=True)
    front_file = st.file_uploader(t["upload_hint"], type=["jpg","jpeg","png"], key="front")
    st.markdown(f'<div class="field-tip">{t["front_tip"]}</div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="field-label">{t["side_label"]}</div>', unsafe_allow_html=True)
    side_file = st.file_uploader(t["upload_hint"], type=["jpg","jpeg","png"], key="side")
    st.markdown(f'<div class="field-tip">{t["side_tip"]}</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# CTA
ready = front_file and side_file
status_class = "ready" if ready else ""
status_text = t["ready"] if ready else t["waiting"]
st.markdown(f'<div class="status-box {status_class}">{status_text}</div>', unsafe_allow_html=True)

st.markdown('<div class="cta-wrap">', unsafe_allow_html=True)
col_l, col_c, col_r = st.columns([1, 4, 1])
with col_c:
    if st.button(t["analyze"], key="cta", disabled=not ready):
        with st.spinner("AI 分析中..."):
            result = process_assessment(
                patient_name=patient_name,
                patient_code=patient_code,
                lang=st.session_state.lang,
                front_file=front_file,
                side_file=side_file,
                deepseek_key=st.session_state.deepseek_key,
            )
            assessment = result["assessment"]
            st.session_state.store = commit_assessment(result["profile"], assessment)
            st.session_state.current_report = assessment
            st.session_state.selected_patient_key = assessment["patient_key"]
            st.success(t["report_generated"])
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div></div>', unsafe_allow_html=True)

report = st.session_state.current_report
st.markdown('<div class="report-shell">', unsafe_allow_html=True)
st.markdown(f'<div class="report-head"><div><div class="report-title">{t["report_section"]}</div><div class="report-sub">{t["subtitle"]}</div></div></div>', unsafe_allow_html=True)

if report:
    summary = report.get("summary", {})
    acl = summary.get("acl_risk", {})
    kinetic_chain = summary.get("kinetic_chain", [])
    recommendations = summary.get("recommendations", [])
    muscle_map = report.get("muscle_map", {})
    dominant_targets = muscle_map.get("dominant_targets", [])
    primary_muscles = muscle_map.get("primary_muscles", [])

    st.markdown(
        f"""
<div class="report-grid">
    <div class="report-card">
        <div class="report-card-label">ACL 风险 / ACL Risk</div>
        <div class="report-card-value">{acl.get('label_zh', 'Low')} · {acl.get('score', 0):.2f}</div>
    </div>
    <div class="report-card">
        <div class="report-card-label">肌群推断 / Muscle Hypothesis</div>
        <div class="report-card-value">{dominant_targets[0] if dominant_targets else 'None'}</div>
    </div>
    <div class="report-card">
        <div class="report-card-label">动力链 / Chain</div>
        <div class="report-card-value">{len(kinetic_chain)} 条要点</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
        )

    st.markdown("### 标记骨架 / Annotated Views")
    img_col1, img_col2 = st.columns(2, gap="large")

    def show_pose_image(col, result_key: str, fallback_key: str, caption: str):
        with col:
            result = report.get(result_key, {})
            annotated_path = result.get("annotated_path")
            fallback_path = report.get(fallback_key, "")
            if annotated_path and os.path.exists(annotated_path):
                st.image(annotated_path, caption=caption, use_container_width=True)
            elif fallback_path and os.path.exists(fallback_path):
                st.image(fallback_path, caption=f"{caption} · 原始图片 / Original", use_container_width=True)
                st.caption("当前档案还没有标记图，重新分析一次后会生成。")
            else:
                st.info("暂无可显示图片")

    show_pose_image(img_col1, "front_result", "front_path", "Front annotated view")
    show_pose_image(img_col2, "side_result", "side_path", "Side annotated view")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown(f'<div class="report-markdown">{report.get("report_md", "")}</div>', unsafe_allow_html=True)
    with col_b:
        muscle_preview = "".join(
            f"<div class='archive-summary'><strong>{item.get('muscle', '')}</strong><br>{item.get('reason', '')}<br><span style='opacity:.8'>Priority: {item.get('priority', '')}</span></div>"
            for item in primary_muscles[:3]
        )
        st.markdown(
            f"""
<div class="report-card" style="margin-bottom:12px">
    <div class="report-card-label">{t['archive_latest']}</div>
    <div class="archive-summary">
        <div><strong>{report.get("patient_name") or report.get("patient_code") or "-"}</strong></div>
        <div>{report.get("created_at", "")}</div>
        <div style="margin-top:8px">{report.get("report_source", "")}</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
<div class="report-card" style="margin-bottom:12px">
    <div class="report-card-label">Muscle Targets</div>
    {muscle_preview or '<div class="archive-summary">No dominant muscle target.</div>'}
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
<div class="report-card">
    <div class="report-card-label">Summary</div>
    <div class="archive-summary">
        <div>{' | '.join(summary.get('summary_lines', []))}</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.download_button(
            t["report_download"],
            data=report.get("report_md", ""),
            file_name=f"{report.get('patient_key', 'report')}_rehab_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        html_report = build_report_html(report)
        pdf_report = build_report_pdf_bytes(report)
        st.download_button(
            "下载网页报告 / HTML",
            data=html_report,
            file_name=f"{report.get('patient_key', 'report')}_rehab_report.html",
            mime="text/html",
            use_container_width=True,
        )
        st.download_button(
            "下载 PDF 报告",
            data=pdf_report,
            file_name=f"{report.get('patient_key', 'report')}_rehab_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
else:
    st.markdown(f'<div class="status-box">{t["report_empty"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ──
st.markdown(f'<div class="footer">{t["footer"]}</div>', unsafe_allow_html=True)
