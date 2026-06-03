from __future__ import annotations

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
from pose import analyze_image_file
from records_store import load_store, save_store, upsert_assessment

DATA_DIR = Path(__file__).with_name("data")
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def clean_text(value: str) -> str:
    return (value or "").strip()


def patient_key(name: str, code: str = "") -> str:
    base = (code or name or "patient").strip().lower()
    base = "".join(ch if ch.isalnum() else "-" for ch in base)
    base = "-".join(filter(None, base.split("-")))
    return base or f"patient-{uuid.uuid4().hex[:8]}"


def save_uploaded_file(uploaded_file, key: str, view: str) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".png"
    target = UPLOAD_DIR / f"{key}_{view}_{uuid.uuid4().hex[:8]}{suffix}"
    target.write_bytes(uploaded_file.getbuffer())
    return target


def build_system_prompt(lang: str) -> str:
    if lang == "en":
        return (
            "You are a senior sports rehabilitation therapist writing for clinicians. "
            "Use clear, objective, professional language. "
            "Do not diagnose disease. "
            "Translate posture metrics into clinically relevant functional findings, muscle function hypotheses, "
            "ACL risk stratification, kinetic chain observations, rehab priorities, and follow-up guidance. "
            "Write in Markdown using sections: Assessment Summary, ACL Risk Assessment, Muscle Function Hypothesis, "
            "Kinetic Chain Analysis, Personalized Rehab Plan, Therapist Notes, Follow-up Schedule."
        )
    return (
        "你是一名资深运动康复治疗师，请用面向治疗师的专业语言撰写 Markdown 报告。"
        "不要做疾病诊断，只根据姿态指标输出客观分析。"
        "请把姿态数据转化为功能性结论、可能受影响的肌群、ACL 风险分层、动力链分析、"
        "个性化康复重点和复评建议。"
        "报告必须包含：评估概览、ACL 风险评估、肌群功能推断、动力链分析、个性化康复建议、"
        "治疗师备注、复评计划。"
    )


def build_user_prompt(profile: dict, front_result: dict, side_result: dict, summary: dict, muscle_map: dict, sections: dict, lang: str) -> str:
    payload = {
        "patient": profile,
        "front_metrics": front_result.get("metrics", {}),
        "front_issues": front_result.get("issues", []),
        "side_metrics": side_result.get("metrics", {}),
        "side_issues": side_result.get("issues", []),
        "summary": summary,
        "muscle_map": muscle_map,
        "report_sections": sections,
    }
    if lang == "en":
        return (
            "Generate a therapist-facing clinical report in Markdown based on the following structured pose data. "
            "Do not mention that this is a diagnosis. Emphasize muscle function hypotheses, risk stratification, "
            "kinetic chain findings, and an actionable rehab plan. Follow the supplied report section names.\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
    return (
        "请根据以下结构化姿态数据，生成一份面向治疗师的专业 Markdown 报告。"
        "不要写成医学诊断，重点突出可能受影响的肌群、风险分层、动力链分析、"
        "可执行康复计划与复评建议。请严格按照给定的报告结构输出。\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def fallback_report(sections: dict, lang: str) -> str:
    return render_markdown_report(sections, lang=lang)


def process_assessment(
    *,
    patient_name: str,
    patient_code: str,
    lang: str,
    front_file,
    side_file,
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
    }

    front_path = save_uploaded_file(front_file, profile_key, "front")
    side_path = save_uploaded_file(side_file, profile_key, "side")

    front_result = analyze_image_file(str(front_path), view="front")
    side_result = analyze_image_file(str(side_path), view="side")
    summary = summarize_case(front_result, side_result)
    muscle_map = infer_muscle_mapping(front_result, side_result, summary)
    sections = build_report_sections(
        patient=profile,
        front_result=front_result,
        side_result=side_result,
        summary=summary,
        muscle_map=muscle_map,
        lang=lang,
    )

    system_prompt = build_system_prompt(lang)
    user_prompt = build_user_prompt(profile, front_result, side_result, summary, muscle_map, sections, lang)

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
        report_source = f"fallback: {exc}"
        report_md = fallback_report(sections, lang)

    assessment = {
        "record_id": uuid.uuid4().hex,
        "created_at": now_iso(),
        "patient_key": profile_key,
        "patient_name": profile_name,
        "patient_code": profile_code,
        "front_path": str(front_path),
        "side_path": str(side_path),
        "front_result": front_result,
        "side_result": side_result,
        "summary": summary,
        "muscle_map": muscle_map,
        "report_sections": sections,
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
    return report
