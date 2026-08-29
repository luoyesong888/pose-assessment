from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict

from analysis import summarize_case
from clinical_knowledge import (
    build_report_sections,
    infer_muscle_mapping,
    render_markdown_report,
)
from deepseek_client import generate_report
from pose import analyze_image_file, analyze_image_files
from posture_rag import rag_prompt_context, retrieve_assessment_context
from recommendation_engine import build_recommendation_options, recommendation_summary_lines
from records_store import load_store, save_store, upsert_assessment

DATA_DIR = Path(__file__).with_name("data")
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_IMAGES = 6
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class AssessmentInputError(ValueError):
    """照片或关键点质量不足，不能继续生成评估。"""


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def clean_text(value: str) -> str:
    return (value or "").strip()


def clean_optional_choice(value) -> str:
    text = clean_text(value)
    return "" if text in {"未填写", "Not provided"} else text


def patient_key(name: str, code: str = "") -> str:
    base = (code or name or "patient").strip().lower()
    base = "".join(ch if ch.isalnum() else "-" for ch in base)
    base = "-".join(filter(None, base.split("-")))
    return base or f"patient-{uuid.uuid4().hex[:8]}"


def save_uploaded_file(uploaded_file, key: str, view: str) -> Path:
    # Streamlit 会缓存已导入模块。如果 data/ 在应用运行期间被删除，
    # 模块顶部的建目录代码不会再次执行，因此每次保存前都要确保目录存在。
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = (Path(uploaded_file.name).suffix or ".png").lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise AssessmentInputError("仅支持 JPG、JPEG 或 PNG 图片")
    payload = bytes(uploaded_file.getbuffer())
    if not payload:
        raise AssessmentInputError(f"图片 {uploaded_file.name} 内容为空")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise AssessmentInputError(f"图片 {uploaded_file.name} 超过 15MB，请压缩后重试")
    target = UPLOAD_DIR / f"{key}_{view}_{uuid.uuid4().hex[:8]}{suffix}"
    target.write_bytes(payload)
    return target


def validate_upload_set(uploads: list) -> None:
    if len(uploads) > MAX_UPLOAD_IMAGES:
        raise AssessmentInputError(f"单次最多上传 {MAX_UPLOAD_IMAGES} 张照片")
    seen = set()
    for uploaded, *_ in uploads:
        payload = bytes(uploaded.getbuffer())
        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen:
            raise AssessmentInputError(f"检测到重复照片：{uploaded.name}")
        seen.add(digest)


def build_system_prompt(lang: str) -> str:
    if lang == "en":
        return (
            "You are a senior sports rehabilitation therapist writing for clinicians. "
            "Use clear, objective, professional language. "
            "Do not diagnose disease. "
            "The input may include front, back, side, forward-bend, or other pose photos. "
            "Translate only visible posture metrics into cautious observations, muscle function hypotheses, "
            "kinetic chain observations, rehab priorities, and follow-up guidance. "
            "Never assign ACL injury risk from static photos; state that dynamic testing is required. "
            "Use improvement_options as the bounded recommendation set; do not invent diagnoses or aggressive exercises. "
            "Mirror the supplied regional_sections and keep every evidence label (Visible observation, Functional hypothesis, "
            "Needs verification, Cannot be judged from photos). Then include ACL Screening Limits, Muscle Function Hypothesis, "
            "Kinetic Chain Analysis, Personalized Rehab Plan, Therapist Notes, and Follow-up Schedule."
        )
    return (
        "你是一名资深运动康复治疗师，请用面向治疗师的专业语言撰写 Markdown 报告。"
        "输入可以包含正面、背面、侧面、前屈或其他姿态照片。"
        "不要做疾病诊断，只根据姿态指标输出客观分析。"
        "请把姿态数据转化为谨慎的可见观察、可能相关的肌群功能假设、动力链分析、"
        "个性化康复重点和复评建议。"
        "不得根据静态照片给出 ACL 损伤风险等级，必须说明需要动态动作与病史测试。"
        "改善建议必须以 improvement_options 为边界，不得自行扩展疾病诊断、高风险动作或激进剂量。"
        "报告必须优先按 regional_sections 生成与示例类似的分区结构，且保留【可见观察】、【功能假设】、【需验证】、"
        "【不可由照片判断】等证据标签。同时包含：ACL 筛查边界、肌群功能推断、动力链分析、个性化康复建议、"
        "治疗师备注、复评计划。"
    )


def build_user_prompt(
    profile: dict,
    front_result: dict,
    side_result: dict,
    summary: dict,
    muscle_map: dict,
    sections: dict,
    lang: str,
    rag_context: dict | None = None,
    recommendation_options: list | None = None,
    image_results: list | None = None,
) -> str:
    payload = {
        "patient": profile,
        "front_metrics": front_result.get("metrics", {}),
        "front_issues": front_result.get("issues", []),
        "side_metrics": side_result.get("metrics", {}),
        "side_issues": side_result.get("issues", []),
        "summary": summary,
        "muscle_map": muscle_map,
        "report_sections": sections,
        "retrieved_posture_context": rag_prompt_context(rag_context or {}),
        "improvement_options": recommendation_options or [],
        "all_image_results": [
            {
                "detected_view": item.get("detected_view") or item.get("view"),
                "valid": item.get("valid"),
                "metrics": item.get("metrics", {}),
                "issues": item.get("issues", []),
            }
            for item in (image_results or [])
        ],
    }
    if lang == "en":
        return (
            "Generate a therapist-facing clinical report in Markdown based on the following structured pose data. "
            "Do not mention that this is a diagnosis. Emphasize cautious muscle hypotheses, screening limits, "
            "kinetic chain findings, and an actionable rehab plan. Follow the supplied report section names.\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
    return (
        "请根据以下结构化姿态数据，生成一份面向治疗师的专业 Markdown 报告。"
        "不要写成医学诊断，不得输出静态 ACL 风险等级；重点突出肌群功能假设、证据边界、动力链分析、"
        "可执行康复计划与复评建议。请严格按照给定的报告结构输出。\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def fallback_report(sections: dict, lang: str) -> str:
    return render_markdown_report(sections, lang=lang)


def pick_front_side(image_results: list) -> tuple[dict, dict | None]:
    """从自动视角结果里挑出正面类和侧面类各一个，供汇总使用。"""
    usable = [
        result
        for result in image_results
        if result.get("found") and result.get("valid") and result.get("metrics")
    ]

    def detected(result: dict) -> str:
        explicit = result.get("view")
        return result.get("detected_view") or (explicit if explicit in {"front", "side"} else "")

    front_candidates = [result for result in usable if detected(result) == "front"]
    side_candidates = [result for result in usable if detected(result) == "side"]
    return (front_candidates[0] if front_candidates else {}), (side_candidates[0] if side_candidates else None)


def ensure_analyzable(front_result: dict, side_result: dict | None, image_results: list | None = None) -> None:
    usable = [
        result
        for result in (image_results or [front_result, side_result])
        if result and result.get("found") and result.get("valid") and result.get("metrics")
    ]
    if usable:
        return
    details = []
    for index, result in enumerate(image_results or []):
        if not result.get("valid"):
            details.append(f"照片 {index + 1}：{'；'.join(result.get('issues', [])) or '未检测到可用人体'}")
    if not details:
        for label, result in (("正面", front_result), ("侧面", side_result)):
            if result:
                details.append(f"{label}：{'；'.join(result.get('issues', [])) or '未检测到人体'}")
    raise AssessmentInputError("未找到可用于体态分析的完整人体，请重新拍摄。\n" + "\n".join(details))


def process_assessment(
    *,
    patient_name: str,
    patient_code: str,
    lang: str,
    front_file=None,
    side_file=None,
    images=None,
    gender=None,
    age=None,
    height=None,
    weight=None,
    occupation=None,
    activity=None,
    concerns=None,
    pain_areas=None,
    injury_history=None,
    deepseek_key: str = "",
) -> Dict[str, object]:
    profile_name = clean_text(patient_name)
    profile_code = clean_text(patient_code)
    profile_key = patient_key(profile_name or profile_code, profile_code)
    profile = {
        "patient_key": profile_key,
        "patient_name": profile_name,
        "patient_code": profile_code,
        "language": lang,
        "created_at": now_iso(),
        "gender": clean_optional_choice(gender),
        "age": age,
        "height": height,
        "weight": weight,
        "occupation": clean_optional_choice(occupation),
        "activity": clean_optional_choice(activity),
        "concerns": [clean_text(item) for item in (concerns or []) if clean_text(item)],
        "pain_areas": clean_text(pain_areas),
        "injury_history": clean_text(injury_history),
    }

    uploads = []
    if front_file is not None:
        uploads.append((front_file, "front", "front"))
    if side_file is not None:
        uploads.append((side_file, "side", "side"))
    for index, uploaded in enumerate(images or []):
        uploads.append((uploaded, f"img{index}", "auto"))
    if not uploads:
        raise AssessmentInputError("请至少上传一张照片")
    validate_upload_set(uploads)

    image_paths = []
    analysis_requests = []
    for uploaded, tag, requested_view in uploads:
        path = save_uploaded_file(uploaded, profile_key, tag)
        image_paths.append(str(path))
        analysis_requests.append((str(path), requested_view))
    image_results = analyze_image_files(analysis_requests)

    front_result, side_result = pick_front_side(image_results)
    side_result = side_result or {}
    ensure_analyzable(front_result, side_result, image_results)
    summary = summarize_case(front_result, side_result, image_results=image_results)
    muscle_map = infer_muscle_mapping(front_result, side_result, summary, image_results=image_results)
    try:
        rag_context = retrieve_assessment_context(
            front_result,
            side_result,
            summary,
            image_results=image_results,
        )
    except Exception as exc:
        rag_context = {
            "knowledge": [],
            "similar_cases": [],
            "summary_lines": [f"本地姿态知识库暂不可用：{type(exc).__name__}"],
        }
    recommendation_options = build_recommendation_options(
        profile,
        front_result,
        side_result,
        summary,
        rag_context,
        lang=lang,
        image_results=image_results,
    )
    sections = build_report_sections(
        patient=profile,
        front_result=front_result,
        side_result=side_result,
        summary=summary,
        muscle_map=muscle_map,
        lang=lang,
        image_results=image_results,
    )
    sections["evidence_title"] = "RAG Evidence & Limits" if lang == "en" else "RAG 参考数据与证据边界"
    sections["evidence_lines"] = rag_context.get("summary_lines", [])
    sections["plan_lines"] = recommendation_summary_lines(recommendation_options)

    system_prompt = build_system_prompt(lang)
    user_prompt = build_user_prompt(
        profile,
        front_result,
        side_result,
        summary,
        muscle_map,
        sections,
        lang,
        rag_context=rag_context,
        recommendation_options=recommendation_options,
        image_results=image_results,
    )

    report_source = "deepseek"
    try:
        if deepseek_key:
            report_md = generate_report(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                api_key=deepseek_key,
            )
        else:
            raise RuntimeError("missing_api_key")
    except Exception as exc:
        report_source = f"fallback+local_rag: {exc}"
        report_md = fallback_report(sections, lang)

    assessment = {
        "record_id": uuid.uuid4().hex,
        "created_at": now_iso(),
        "patient_key": profile_key,
        "patient_name": profile_name,
        "patient_code": profile_code,
        "profile": profile,
        "front_path": front_result.get("source_path", ""),
        "side_path": (side_result or {}).get("source_path", ""),
        "image_paths": image_paths,
        "image_results": image_results,
        "front_result": front_result,
        "side_result": side_result,
        "summary": summary,
        "muscle_map": muscle_map,
        "report_sections": sections,
        "rag_context": rag_context,
        "recommendation_options": recommendation_options,
        "report_md": report_md,
        "report_source": report_source,
    }

    return {
        "profile": profile,
        "assessment": assessment,
    }


def commit_assessment(profile: dict, assessment: dict) -> dict:
    store = load_store()
    store = upsert_assessment(store, profile, assessment)
    save_store(store)
    return store


def ensure_visual_assets(report: dict | None) -> dict | None:
    if not report:
        return report

    def _ensure_one(result_key: str, source_key: str, view: str) -> None:
        result = report.get(result_key) or {}
        annotated_path = result.get("annotated_path")
        if annotated_path and Path(annotated_path).exists():
            return
        source_path = report.get(source_key) or result.get("source_path")
        if not source_path or not Path(source_path).exists():
            return
        refreshed = analyze_image_file(str(source_path), view=view)
        report[result_key] = {**result, **refreshed}

    _ensure_one("front_result", "front_path", "front")
    _ensure_one("side_result", "side_path", "side")
    refreshed_results = []
    for result in report.get("image_results", []):
        annotated_path = result.get("annotated_path")
        source_path = result.get("source_path")
        if annotated_path and Path(annotated_path).exists():
            refreshed_results.append(result)
        elif source_path and Path(source_path).exists():
            refreshed_results.append({**result, **analyze_image_file(str(source_path), view="auto")})
        else:
            refreshed_results.append(result)
    if refreshed_results:
        report["image_results"] = refreshed_results
    return report
