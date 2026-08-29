import html
import os

import streamlit as st
from app_pipeline import AssessmentInputError, commit_assessment, process_assessment
from recommendation_engine import apply_confirmed_plan
from report_export import build_report_html, build_report_pdf_bytes
from records_store import load_store, replace_assessment, save_store, search_patients, total_assessments

if "lang" not in st.session_state:
    st.session_state.lang = "zh"

LANG = {
    "zh": {
        "lang_btn": "EN",
        "subtitle": "上传站立、背面、侧面、前屈或其他体态照片，AI 自动识别照片类型并生成结构化评估。",
        "upload_title": "开始评估",
        "profile_title": "客户信息",
        "name_label": "姓名",
        "name_ph": "输入姓名",
        "code_label": "客户编号",
        "code_ph": "输入客户编号（可选）",
        "gender_label": "性别",
        "gender_options": ["未填写", "男", "女", "其他"],
        "age_label": "年龄",
        "height_label": "身高 (cm)",
        "weight_label": "体重 (kg)",
        "occupation_label": "职业 / 日常活动",
        "occupation_options": ["未填写", "久坐办公", "站立工作", "体力劳动", "学生", "运动相关", "自由职业", "其他"],
        "activity_label": "运动频率",
        "activity_options": ["未填写", "几乎不运动", "每周 1-2 次", "每周 3-5 次", "几乎每天"],
        "concerns_label": "主要诉求（可多选）",
        "concern_options": ["高低肩", "圆肩驼背", "骨盆前倾", "膝超伸", "扁平足", "颈肩不适", "腰背不适", "体态美观", "运动表现", "其他"],
        "pain_label": "疼痛或不适部位",
        "pain_ph": "如无请留空",
        "injury_label": "既往损伤 / 手术史",
        "injury_ph": "如无请留空",
        "front_label": "上传体态照片（可多选、角度和动作任意）",
        "upload_hint": "支持 JPG、PNG；单张不超过 15MB，单次最多 6 张",
        "front_tip": "支持正面、背面、侧面、斜向、前屈等姿态；尽量保持单人、主要关节清晰可见",
        "waiting": "请至少上传一张体态照片",
        "analyze": "立即开始 AI 姿态分析",
        "ready": "照片已就绪，点击按钮开始分析",
        "archive_title": "客户档案",
        "archive_search": "搜索客户姓名 / 编号",
        "archive_latest": "最新评估",
        "archive_load": "加载到主面板",
        "archive_empty": "暂无历史档案",
        "archive_stats_patients": "客户数",
        "archive_stats_assessments": "评估数",
        "api_title": "DeepSeek 设置",
        "api_hint": "支持通过环境变量 DEEPSEEK_API_KEY 或侧边栏输入密钥。",
        "report_section": "治疗师报告",
        "report_empty": "完成一次分析后，这里会显示专业报告。",
        "report_download": "下载报告",
        "report_generated": "报告已生成",
        "footer": "KinetiQ · 基于 2D 姿态关键点的体态康复筛查 · 非医学诊断 · Powered by MediaPipe + DeepSeek AI",
        "guide_title": "拍摄指南",
        "guide_sub": "标准照片，结果才可比较",
        "guide_front": "任意站立角度",
        "guide_side": "任意动作截图",
        "guide_rule_1": "头部到足部完整入镜",
        "guide_rule_2": "相机保持水平，避免俯拍",
        "guide_rule_3": "光线均匀，轮廓无遮挡",
        "guide_rule_4": "同一动作可上传多张便于交叉验证",
        "analysis_title": "分析说明",
        "analysis_sub": "本地筛查与证据边界",
        "analysis_item_1": "自动识别视角与动作类型",
        "analysis_item_2": "肩、髋、膝踝对线与动作对称性",
        "analysis_item_3": "相似关键点 RAG 参考",
        "analysis_item_4": "Markdown / HTML / PDF 报告",
        "privacy_title": "隐私保护",
        "privacy_text": "照片和档案保存在当前电脑，不向 Google 上传。静态照片不提供疾病诊断或 ACL 风险等级。",
        "improvement_title": "改善重点建议",
        "improvement_sub": "系统根据客观指标推荐，请选择 1–3 项作为下一阶段目标",
        "improvement_select": "选择希望优先改善的目标",
        "improvement_generate": "确认并生成训练计划",
        "improvement_saved": "改善计划已保存到本次评估",
    },
    "en": {
        "lang_btn": "中文",
        "subtitle": "Upload standing, back, side, forward-bend, or other pose photos. AI identifies each capture type and builds a structured report.",
        "upload_title": "Start Assessment",
        "profile_title": "Client Info",
        "name_label": "Name",
        "name_ph": "Enter name",
        "code_label": "Client ID",
        "code_ph": "Enter client ID (optional)",
        "gender_label": "Gender",
        "gender_options": ["Not provided", "Male", "Female", "Other"],
        "age_label": "Age",
        "height_label": "Height (cm)",
        "weight_label": "Weight (kg)",
        "occupation_label": "Occupation / Daily Activity",
        "occupation_options": ["Not provided", "Sedentary office", "Standing work", "Manual labor", "Student", "Sports-related", "Freelance", "Other"],
        "activity_label": "Activity Level",
        "activity_options": ["Not provided", "Rarely", "1-2 times/week", "3-5 times/week", "Almost daily"],
        "concerns_label": "Main Concerns (multi-select)",
        "concern_options": ["Uneven shoulders", "Rounded shoulders", "Anterior pelvic tilt", "Knee hyperextension", "Flat feet", "Neck/shoulder", "Low back", "Aesthetics", "Performance", "Other"],
        "pain_label": "Pain / Discomfort",
        "pain_ph": "Leave empty if none",
        "injury_label": "Injury / Surgery History",
        "injury_ph": "Leave empty if none",
        "front_label": "Upload Posture Photos (multiple angles or actions)",
        "upload_hint": "JPG or PNG; max 15MB each and 6 photos per assessment",
        "front_tip": "Front, back, side, oblique, and movement snapshots are supported; keep one person and the major joints visible when possible",
        "waiting": "Please upload at least one posture photo",
        "analyze": "Start AI Posture Analysis",
        "ready": "Photos ready — click to start analysis",
        "archive_title": "Client Archive",
        "archive_search": "Search client name / ID",
        "archive_latest": "Latest Assessment",
        "archive_load": "Load to Main Panel",
        "archive_empty": "No history yet",
        "archive_stats_patients": "Clients",
        "archive_stats_assessments": "Assessments",
        "api_title": "DeepSeek Settings",
        "api_hint": "You can use DEEPSEEK_API_KEY from the environment or enter it here.",
        "report_section": "Therapist Report",
        "report_empty": "Run one analysis to generate a professional report here.",
        "report_download": "Download Report",
        "report_generated": "Report generated",
        "footer": "KinetiQ · 2D posture landmark-based rehab screening · Not medical diagnosis · Powered by MediaPipe + DeepSeek AI",
        "guide_title": "Photo Guide",
        "guide_sub": "Standardized photos make results comparable",
        "guide_front": "Any standing angle",
        "guide_side": "Any movement frame",
        "guide_rule_1": "Keep head and feet fully visible",
        "guide_rule_2": "Keep the camera level",
        "guide_rule_3": "Use even light and a clear outline",
        "guide_rule_4": "Add repeated frames when you want cross-checking",
        "analysis_title": "What Is Analyzed",
        "analysis_sub": "Local screening with clear limits",
        "analysis_item_1": "Automatic angle and action recognition",
        "analysis_item_2": "Shoulder, hip, leg alignment and movement symmetry",
        "analysis_item_3": "Similar-landmark RAG references",
        "analysis_item_4": "Markdown / HTML / PDF reports",
        "privacy_title": "Privacy",
        "privacy_text": "Photos and records stay on this computer and are not uploaded to Google. Static photos do not diagnose disease or grade ACL risk.",
        "improvement_title": "Suggested Improvement Priorities",
        "improvement_sub": "Choose 1–3 objective priorities for the next phase",
        "improvement_select": "Select priorities",
        "improvement_generate": "Confirm and Build Plan",
        "improvement_saved": "The improvement plan was saved to this assessment",
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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

:root {
    --bg:       #e6e0e9;
    --surface:  #eee9f0;
    --card:     #e3dce6;
    --text:     #171319;
    --muted:    #514957;
    --accent:   #792f9b;
    --accent2:  #ae61cd;
    --accent3:  #e9bdfb;
    --accent-soft: #d9cce0;
    --green:    #30c963;
    --line:     #c8bdcd;
    --glow:     rgba(121,47,155,0.25);
    --shadow:   0 2px 8px rgba(45,35,51,0.05), 0 14px 36px rgba(45,35,51,0.08);
    --max:      1180px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Space Grotesk', -apple-system, 'PingFang SC', sans-serif;
    background: var(--bg);
    color: var(--text);
}

/* Keep all Streamlit-rendered content readable regardless of the active
   system/app theme. Individual dark buttons are restored below. */
.stApp,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] span,
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3,
.stApp [data-testid="stMarkdownContainer"] h4,
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] * {
    color: #171319 !important;
    -webkit-text-fill-color: #171319 !important;
    opacity: 1 !important;
}

.stApp {
    background:
        radial-gradient(1000px 480px at 85% -8%, rgba(174,97,205,0.08), transparent 60%),
        radial-gradient(800px 400px at -5% 8%, rgba(121,47,155,0.06), transparent 55%),
        var(--bg);
}

.block-container {
    padding: 0 32px 36px !important;
    max-width: 1580px !important;
}

[data-testid="stAppViewContainer"] > .main > div {
    padding-top: 0 !important;
}

[data-testid="stVerticalBlock"] {
    gap: 0.6rem !important;
}

/* ── TOPBAR ── */
.topbar {
    position: fixed;
    inset: 0 0 auto 0;
    z-index: 999;
    height: 62px;
    background: rgba(226,218,231,0.96);
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    border-bottom: 1px solid var(--line);
    display: flex;
    align-items: center;
    padding: 0 clamp(20px, 4vw, 64px);
    justify-content: space-between;
}

.topbar-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 21px;
    font-weight: 700;
    letter-spacing: -0.5px;
    background: linear-gradient(100deg, #792f9b, #ae61cd 55%, #e9bdfb 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

.topbar-logo em {
    font-style: normal;
    -webkit-text-fill-color: #ae61cd;
}

/* lang button */
.lang-wrap {
    position: fixed;
    top: 14px;
    right: clamp(20px, 4vw, 64px);
    z-index: 1001;
}

.lang-wrap div[data-testid="stButton"] > button {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    color: var(--muted) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 5px 13px !important;
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

/* ── WORKSPACE / 步骤卡片 ── */
.workspace-spacer { height: 82px; }

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #eee8f0;
    border: 1px solid var(--line);
    border-radius: 22px;
    box-shadow: var(--shadow);
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 6px 8px;
}

.rail-stack {
    position: sticky;
    top: 82px;
    display: flex;
    flex-direction: column;
    gap: 14px;
}

.rail-card {
    background: #e8e1eb;
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 10px 30px rgba(45,35,51,0.07);
}

.rail-card.accent {
    background: linear-gradient(160deg, #d6bee1 0%, #c59bd6 100%);
    color: #171319;
    border: 1px solid #b88bca;
}

.rail-kicker {
    color: var(--accent);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 7px;
}

.rail-card.accent .rail-kicker { color: #542365 !important; -webkit-text-fill-color: #542365 !important; }
.rail-title { color: #2d2333 !important; font-size: 19px; font-weight: 750; letter-spacing: -0.02em; }
.rail-card.accent .rail-title { color: #171319 !important; -webkit-text-fill-color: #171319 !important; }
.rail-sub { color: var(--muted); font-size: 12px; line-height: 1.55; margin: 6px 0 16px; }
.rail-card.accent .rail-sub { color: #3f3344 !important; -webkit-text-fill-color: #3f3344 !important; }

.rail-list { display: flex; flex-direction: column; gap: 11px; }
.rail-item { display: flex; align-items: flex-start; gap: 10px; font-size: 12px; line-height: 1.45; color: #5f5668; }
.rail-card.accent .rail-item { color: #241d27 !important; -webkit-text-fill-color: #241d27 !important; }
.rail-dot {
    width: 22px; height: 22px; flex: 0 0 22px;
    border-radius: 8px;
    display: grid; place-items: center;
    background: var(--accent-soft); color: var(--accent);
    font-size: 10px; font-weight: 800;
}
.rail-card.accent .rail-dot { background: #efe5f2; color: #542365 !important; -webkit-text-fill-color: #542365 !important; }

.pose-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; margin: 14px 0 16px; }
.pose-mini {
    min-height: 112px;
    border-radius: 14px;
    background: linear-gradient(180deg, #faf7fd, #f1e8f7);
    border: 1px solid #e5d8ed;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    color: var(--accent); font-size: 11px; font-weight: 700;
}
.pose-figure { position: relative; width: 42px; height: 70px; margin-bottom: 6px; }
.pose-head { position:absolute; width:13px; height:13px; border-radius:50%; background:var(--accent); left:14px; top:0; }
.pose-body { position:absolute; width:4px; height:29px; border-radius:4px; background:var(--accent); left:19px; top:14px; }
.pose-arm-l,.pose-arm-r,.pose-leg-l,.pose-leg-r { position:absolute; width:3px; border-radius:3px; background:var(--accent2); transform-origin:top; }
.pose-arm-l { height:27px; left:18px; top:18px; transform:rotate(32deg); }
.pose-arm-r { height:27px; left:22px; top:18px; transform:rotate(-32deg); }
.pose-leg-l { height:30px; left:19px; top:41px; transform:rotate(12deg); }
.pose-leg-r { height:30px; left:21px; top:41px; transform:rotate(-12deg); }
.pose-mini.side .pose-arm-l { transform:rotate(4deg); left:20px; }
.pose-mini.side .pose-arm-r { transform:rotate(-6deg); left:21px; opacity:.45; }
.pose-mini.side .pose-leg-l { transform:rotate(4deg); }
.pose-mini.side .pose-leg-r { transform:rotate(-4deg); opacity:.5; }

.privacy-note {
    margin-top: 14px; padding: 13px 14px;
    border-radius: 13px; background: rgba(255,255,255,0.34);
    font-size: 11px; line-height: 1.55; color: #241d27 !important;
}

.rail-card.accent *, .privacy-note * {
    color: #171319 !important;
    -webkit-text-fill-color: #171319 !important;
    opacity: 1 !important;
}

.step-head {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 22px;
    padding-bottom: 18px;
    border-bottom: 1px solid var(--line);
}

.step-num {
    width: 40px;
    height: 40px;
    border-radius: 14px;
    background: linear-gradient(135deg, #792f9b, #ae61cd);
    color: #fff;
    font-size: 18px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 14px rgba(121,47,155,0.30);
    flex-shrink: 0;
}
.step-num, .step-num * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

.step-title {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.03em;
    background: linear-gradient(100deg, #2d2333, #792f9b);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

.field-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 4px;
}

.field-tip {
    font-size: 12px;
    color: var(--muted);
    margin-top: 5px;
    line-height: 1.5;
}

.divider {
    height: 1px;
    background: var(--line);
    margin: 20px 0;
}

/* inputs */
/* Streamlit may inherit white widget labels from the active app theme. Force
   high-contrast text for every light form surface. */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stTextInput"] label,
[data-testid="stTextInput"] label p,
[data-testid="stNumberInput"] label,
[data-testid="stNumberInput"] label p,
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] label p,
[data-testid="stMultiSelect"] label,
[data-testid="stMultiSelect"] label p,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label p {
    color: #4b3e55 !important;
    -webkit-text-fill-color: #4b3e55 !important;
    opacity: 1 !important;
    font-weight: 650 !important;
}

div[data-testid="stTextInput"] input {
    background: var(--card) !important;
    border: 1px solid #d7ccdf !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-size: 14px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    padding: 11px 14px !important;
}
div[data-testid="stTextInput"] input::placeholder,
[data-testid="stNumberInput"] input::placeholder {
    color: #8b8093 !important;
    -webkit-text-fill-color: #8b8093 !important;
    opacity: 1 !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(121,47,155,0.12) !important;
}

[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input,
[data-testid="stMultiSelect"] > div > div {
    background: var(--card) !important;
    border-color: #d7ccdf !important;
    color: var(--text) !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] *,
[data-testid="stMultiSelect"] [data-baseweb="select"] *,
[data-testid="stNumberInput"] input {
    color: #2d2333 !important;
    -webkit-text-fill-color: #2d2333 !important;
    opacity: 1 !important;
}

[data-testid="stNumberInput"] button {
    background: #cec2d3 !important;
    border-color: #d7ccdf !important;
    color: #5a4268 !important;
}

[data-testid="stNumberInput"] button svg {
    fill: #5a4268 !important;
    color: #5a4268 !important;
}

[data-testid="stFileUploader"] section {
    background: #ddd5e1 !important;
    border: 2px dashed #d8cce4 !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section * {
    color: #4b3e55 !important;
    -webkit-text-fill-color: #4b3e55 !important;
    opacity: 1 !important;
}
[data-testid="stFileUploader"] section button {
    background: #cec1d4 !important;
    border: 1px solid #bfa9ce !important;
    color: #5f2b78 !important;
    font-weight: 700 !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--accent2) !important;
}

/* CTA button */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #792f9b, #ae61cd 60%, #c084fc) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 14px 28px !important;
    width: 100% !important;
    box-shadow: 0 6px 20px rgba(121,47,155,0.35) !important;
    letter-spacing: 0.2px !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 8px 28px rgba(121,47,155,0.45) !important;
    transform: translateY(-1px) !important;
    opacity: 1 !important;
}
div[data-testid="stButton"] > button[kind="primary"]:disabled {
    background: #e9e4f0 !important;
    color: #a89fb0 !important;
    box-shadow: none !important;
}

.status-box {
    padding: 13px 16px;
    border-radius: 12px;
    background: #d9cfe0;
    border: 1px solid #bdaec6;
    color: #241d27;
    font-size: 13px;
    line-height: 1.5;
    margin-bottom: 12px;
}

.status-box.ready {
    background: rgba(48,201,99,0.09);
    border-color: rgba(48,201,99,0.25);
    color: #1a9c4b;
}

/* ── REPORT ── */
.report-shell {
    max-width: var(--max);
    margin: 20px auto 0;
    background: #eee8f0;
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 26px 30px;
    box-shadow: var(--shadow);
}

.report-head {
    display: flex;
    justify-content: space-between;
    gap: 18px;
    align-items: flex-start;
    margin-bottom: 18px;
    padding-bottom: 15px;
    border-bottom: 1px solid var(--line);
}

.report-title {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.03em;
    background: linear-gradient(100deg, #2d2333, #792f9b);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

.report-sub {
    margin-top: 6px;
    color: var(--muted);
    line-height: 1.6;
    font-size: 13px;
    max-width: 64ch;
}

.report-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 14px;
}

.report-card {
    background: #dfd7e3;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 15px 16px;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.report-card:hover {
    border-color: rgba(121,47,155,0.35);
    box-shadow: 0 6px 18px rgba(121,47,155,0.10);
}

.report-card-label {
    color: var(--muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.report-card-value {
    color: var(--text);
    font-size: 20px;
    font-weight: 600;
    letter-spacing: -0.03em;
}

.improvement-card {
    height: 100%;
    min-height: 210px;
    background: linear-gradient(160deg, #e7e0ea 0%, #dbd1e0 100%);
    border: 1px solid #bfaec7;
    border-radius: 17px;
    padding: 17px;
    box-shadow: 0 8px 22px rgba(121,47,155,0.07);
}
.improvement-top { display:flex; justify-content:space-between; gap:8px; align-items:center; margin-bottom:10px; }
.improvement-priority { padding:4px 8px; border-radius:999px; background:var(--accent-soft); color:var(--accent); font-size:10px; font-weight:800; }
.improvement-title { font-size:16px; font-weight:750; color:var(--text); line-height:1.3; }
.improvement-summary { font-size:12px; line-height:1.6; color:var(--muted); margin-bottom:10px; }
.improvement-evidence { font-size:11px; line-height:1.55; color:#655a70; padding-top:9px; border-top:1px solid var(--line); }
.confirmed-plan { background:#ddd2e2; border:1px solid #bba9c5; border-radius:16px; padding:15px 17px; margin:8px 0 18px; }

.report-markdown {
    background: #ded6e2;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 18px 20px;
    color: var(--text);
    line-height: 1.8;
}

.report-markdown,
.report-markdown *,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] code {
    color: #171319 !important;
    -webkit-text-fill-color: #171319 !important;
    opacity: 1 !important;
}

.report-markdown h1,
.report-markdown h2,
.report-markdown h3 {
    margin-top: 0.8em;
    margin-bottom: 0.4em;
    color: #792f9b;
}

.archive-summary {
    font-size: 12px;
    color: var(--muted);
    line-height: 1.55;
    padding: 6px 0 0;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #ddd5e1;
    border-right: 1px solid var(--line);
}

/* ── FOOTER ── */
.footer {
    padding: 18px clamp(20px, 4vw, 64px) 26px;
    text-align: center;
    font-size: 12px;
    color: var(--muted);
    opacity: 0.7;
}

@media (max-width: 900px) {
    .topbar-nav { display: none; }
    .block-container { padding: 0 16px 28px !important; }
    .workspace-spacer { height: 74px; }
    .rail-stack { position: static; }
    .rail-card { padding: 16px; }
    .step-title { font-size: 19px; }
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
    [data-testid="column"] {
        flex: 1 1 100% !important;
        width: 100% !important;
        min-width: 100% !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ── TOPBAR ──
st.markdown(f"""
<div class="topbar">
    <div class="topbar-logo">Kineti<em>Q</em></div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    if st.button(t["lang_btn"], key="lang_switch", use_container_width=True):
        st.session_state.lang = "en" if st.session_state.lang == "zh" else "zh"
        st.rerun()
    st.markdown("---")
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

# ── THREE-COLUMN WORKSPACE ──
st.markdown('<div class="workspace-spacer"></div>', unsafe_allow_html=True)
guide_col, form_col, info_col = st.columns([1.05, 3.15, 1.05], gap="large", vertical_alignment="top")

with guide_col:
    st.markdown(
        f"""
<div class="rail-stack">
  <div class="rail-card">
    <div class="rail-kicker">01 · Capture</div>
    <div class="rail-title">{t['guide_title']}</div>
    <div class="rail-sub">{t['guide_sub']}</div>
    <div class="pose-pair">
      <div class="pose-mini">
        <div class="pose-figure"><i class="pose-head"></i><i class="pose-body"></i><i class="pose-arm-l"></i><i class="pose-arm-r"></i><i class="pose-leg-l"></i><i class="pose-leg-r"></i></div>
        {t['guide_front']}
      </div>
      <div class="pose-mini side">
        <div class="pose-figure"><i class="pose-head"></i><i class="pose-body"></i><i class="pose-arm-l"></i><i class="pose-arm-r"></i><i class="pose-leg-l"></i><i class="pose-leg-r"></i></div>
        {t['guide_side']}
      </div>
    </div>
    <div class="rail-list">
      <div class="rail-item"><span class="rail-dot">1</span><span>{t['guide_rule_1']}</span></div>
      <div class="rail-item"><span class="rail-dot">2</span><span>{t['guide_rule_2']}</span></div>
      <div class="rail-item"><span class="rail-dot">3</span><span>{t['guide_rule_3']}</span></div>
      <div class="rail-item"><span class="rail-dot">4</span><span>{t['guide_rule_4']}</span></div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

with info_col:
    st.markdown(
        f"""
<div class="rail-stack">
  <div class="rail-card accent">
    <div class="rail-kicker">02 · Insight</div>
    <div class="rail-title">{t['analysis_title']}</div>
    <div class="rail-sub">{t['analysis_sub']}</div>
    <div class="rail-list">
      <div class="rail-item"><span class="rail-dot">✓</span><span>{t['analysis_item_1']}</span></div>
      <div class="rail-item"><span class="rail-dot">✓</span><span>{t['analysis_item_2']}</span></div>
      <div class="rail-item"><span class="rail-dot">✓</span><span>{t['analysis_item_3']}</span></div>
      <div class="rail-item"><span class="rail-dot">✓</span><span>{t['analysis_item_4']}</span></div>
    </div>
    <div class="privacy-note"><strong>{t['privacy_title']}</strong><br>{t['privacy_text']}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

with form_col:
    with st.container(border=True):
        st.markdown(f'<div class="step-head"><div class="step-num">1</div><div class="step-title">{t["profile_title"]}</div></div>', unsafe_allow_html=True)

        col_name, col_code = st.columns([2, 1])
        with col_name:
            patient_name = st.text_input(t["name_label"], placeholder=t["name_ph"], key="member_name")
        with col_code:
            patient_code = st.text_input(t["code_label"], placeholder=t["code_ph"], key="member_code")

        col_g, col_a, col_h = st.columns([1, 1, 1])
        with col_g:
            gender = st.selectbox(t["gender_label"], t["gender_options"], key="gender")
        with col_a:
            age = st.number_input(t["age_label"], min_value=1, max_value=120, value=None, step=1, key="age")
        with col_h:
            height = st.number_input(t["height_label"], min_value=80, max_value=250, value=None, step=1, key="height")

        col_w, col_o, col_act = st.columns([1, 1, 1])
        with col_w:
            weight = st.number_input(t["weight_label"], min_value=20, max_value=300, value=None, step=1, key="weight")
        with col_o:
            occupation = st.selectbox(t["occupation_label"], t["occupation_options"], key="occupation")
        with col_act:
            activity = st.selectbox(t["activity_label"], t["activity_options"], key="activity")

        concerns = []
        col_pain, col_injury = st.columns([1, 1])
        with col_pain:
            pain_areas = st.text_input(t["pain_label"], placeholder=t["pain_ph"], key="pain")
        with col_injury:
            injury_history = st.text_input(t["injury_label"], placeholder=t["injury_ph"], key="injury")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="step-head"><div class="step-num">2</div><div class="step-title">{t["front_label"]}</div></div>', unsafe_allow_html=True)
        images = st.file_uploader(
            t["upload_hint"],
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="images",
        )
        st.markdown(f'<div class="field-tip">{t["front_tip"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        ready = bool(images)
        status_class = "ready" if ready else ""
        status_text = t["ready"] if ready else t["waiting"]
        st.markdown(f'<div class="status-box {status_class}">{status_text}</div>', unsafe_allow_html=True)

        if st.button(t["analyze"], key="cta", disabled=not ready, use_container_width=True, type="primary"):
            with st.spinner("AI 分析中..."):
                try:
                    result = process_assessment(
                        patient_name=patient_name,
                        patient_code=patient_code,
                        lang=st.session_state.lang,
                        images=images,
                        gender=gender,
                        age=age,
                        height=height,
                        weight=weight,
                        occupation=occupation,
                        activity=activity,
                        concerns=concerns,
                        pain_areas=pain_areas,
                        injury_history=injury_history,
                        deepseek_key=st.session_state.deepseek_key,
                    )
                except AssessmentInputError as exc:
                    st.error(str(exc))
                except Exception:
                    st.error("分析服务暂时不可用，请重启应用后重试。")
                else:
                    assessment = result["assessment"]
                    st.session_state.store = commit_assessment(result["profile"], assessment)
                    st.session_state.current_report = assessment
                    st.session_state.selected_patient_key = assessment["patient_key"]
                    st.success(t["report_generated"])

report = st.session_state.current_report
st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
st.markdown(f'<div class="step-head"><div class="step-num">3</div><div><div class="step-title">{t["report_section"]}</div><div class="report-sub">{t["subtitle"]}</div></div></div>', unsafe_allow_html=True)

if report:
    summary = report.get("summary", {})
    acl = summary.get("acl_risk", {})
    kinetic_chain = summary.get("kinetic_chain", [])
    recommendations = summary.get("recommendations", [])
    coverage_display = summary.get("view_coverage", {}).get("label_zh", "未知")
    movement_display = summary.get("movement_screening", {}).get("label_zh", "待评估")
    muscle_map = report.get("muscle_map", {})
    dominant_targets = muscle_map.get("dominant_targets", [])
    primary_muscles = muscle_map.get("primary_muscles", [])
    acl_display = acl.get("label_zh", "未评估（需动态测试）")

    st.markdown(
        f"""
<div class="report-grid">
    <div class="report-card">
        <div class="report-card-label">照片覆盖 / Capture Coverage</div>
        <div class="report-card-value">{coverage_display}</div>
    </div>
    <div class="report-card">
        <div class="report-card-label">主要观察 / Key Observation</div>
        <div class="report-card-value">{movement_display}</div>
    </div>
    <div class="report-card">
        <div class="report-card-label">验证优先级 / Priorities</div>
        <div class="report-card-value">{len(report.get('recommendation_options', []))} 项</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
        )

    improvement_options = report.get("recommendation_options", [])
    if improvement_options:
        st.markdown(f"### {t['improvement_title']}")
        st.caption(t["improvement_sub"])
        improvement_cols = st.columns(len(improvement_options), gap="medium")
        for option_col, option in zip(improvement_cols, improvement_options):
            evidence_html = "<br>".join(
                f"• {html.escape(str(item))}" for item in option.get("evidence", [])[:3]
            )
            with option_col:
                st.markdown(
                    f"""
<div class="improvement-card">
  <div class="improvement-top">
    <div class="improvement-title">{html.escape(option.get('title', ''))}</div>
    <span class="improvement-priority">{html.escape(option.get('priority', ''))}</span>
  </div>
  <div class="improvement-summary">{html.escape(option.get('summary', ''))}</div>
  <div class="improvement-evidence">{evidence_html}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

        option_ids = [item["id"] for item in improvement_options]
        option_titles = {item["id"]: item["title"] for item in improvement_options}
        confirmed_plan = report.get("confirmed_plan", {})
        default_ids = [item for item in confirmed_plan.get("selected_ids", []) if item in option_ids]
        selected_ids = st.multiselect(
            t["improvement_select"],
            option_ids,
            default=default_ids,
            format_func=lambda option_id: option_titles.get(option_id, option_id),
            max_selections=3,
            key=f"improvement_{report.get('record_id', 'current')}",
        )
        if st.button(
            t["improvement_generate"],
            key=f"confirm_improvement_{report.get('record_id', 'current')}",
            disabled=not selected_ids,
            use_container_width=True,
        ):
            report = apply_confirmed_plan(report, selected_ids, lang=st.session_state.lang)
            replace_assessment(st.session_state.store, report)
            save_store(st.session_state.store)
            st.session_state.current_report = report
            st.success(t["improvement_saved"])

        if report.get("confirmed_plan", {}).get("lines"):
            with st.container(border=True):
                st.markdown(f"#### {report['report_sections'].get('confirmed_plan_title', '已确认改善计划')}")
                for line in report["confirmed_plan"]["lines"]:
                    st.markdown(f"- {line}")

    st.markdown("### 标记骨架 / Annotated Views")
    image_results = report.get("image_results", [])
    if image_results:
        for row_start in range(0, len(image_results), 2):
            cols = st.columns(2, gap="large")
            for offset in range(2):
                index = row_start + offset
                if index >= len(image_results):
                    break
                with cols[offset]:
                    result = image_results[index]
                    annotated_path = result.get("annotated_path")
                    source_path = result.get("source_path", "")
                    detected_view = result.get("detected_view", "auto")
                    caption = f"图 {index + 1} · 识别视角 {detected_view}"
                    if annotated_path and os.path.exists(annotated_path):
                        st.image(annotated_path, caption=caption, use_container_width=True)
                    elif source_path and os.path.exists(source_path):
                        st.image(source_path, caption=f"{caption} · 原始图片 / Original", use_container_width=True)
                    else:
                        st.info("暂无可显示图片")
    else:
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

        img_col1, img_col2 = st.columns(2, gap="large")
        show_pose_image(img_col1, "front_result", "front_path", "Front annotated view")
        show_pose_image(img_col2, "side_result", "side_path", "Side annotated view")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        with st.container(border=True):
            st.markdown(report.get("report_md", ""))
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
    if st.session_state.lang == "zh":
        placeholders = [
            ("ACL 筛查", "需动态动作测试，静态照片不单独评估损伤风险"),
            ("肌群推断", "根据姿态偏移推测可能相关的肌群与动作模式"),
            ("动力链分析", "躯干、骨盆、下肢协同与代偿观察"),
        ]
    else:
        placeholders = [
            ("ACL Screening", "Requires dynamic testing, not inferred from static photos"),
            ("Muscle Hypothesis", "Possible related muscle groups from posture patterns"),
            ("Kinetic Chain", "Trunk, pelvis and lower-limb coordination"),
        ]
    pcols = st.columns(3, gap="medium")
    for pcol, (title, desc) in zip(pcols, placeholders):
        with pcol:
            st.markdown(
                f'<div class="report-card"><div class="report-card-label">{title}</div>'
                f'<div class="archive-summary">{desc}</div></div>',
                unsafe_allow_html=True,
            )

# ── FOOTER ──
st.markdown(f'<div class="footer">{t["footer"]}</div>', unsafe_allow_html=True)
