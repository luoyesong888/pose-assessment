from __future__ import annotations

from typing import Any, Dict, List


def _unique(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _dedupe_muscles(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    result: List[Dict[str, str]] = []
    for item in items:
        key = (item.get("muscle", ""), item.get("reason", ""), item.get("priority", ""))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def infer_muscle_mapping(front_result: Dict[str, Any] | None, side_result: Dict[str, Any] | None, summary: Dict[str, Any] | None) -> Dict[str, Any]:
    """根据姿态指标推断可能受影响的肌群与动作控制问题。

    注意：这里只做“功能关联”推断，不做医学诊断。
    """
    front_metrics = (front_result or {}).get("metrics", {})
    side_metrics = (side_result or {}).get("metrics", {})
    front_issues = (front_result or {}).get("issues", [])
    side_issues = (side_result or {}).get("issues", [])
    acl = (summary or {}).get("acl_risk", {})
    acl_level = acl.get("level", "low")

    primary_muscles: List[Dict[str, str]] = []
    secondary_muscles: List[Dict[str, str]] = []
    movement_patterns: List[str] = []
    therapist_focus: List[str] = []

    if front_metrics.get("knee_alignment_pct", 0) > 0.045:
        primary_muscles.extend(
            [
                {
                    "muscle": "臀中肌 / 髋外展肌群",
                    "reason": "膝-踝对线偏移提示近端控制不足，常与髋外展控制下降相关。",
                    "priority": "高",
                },
                {
                    "muscle": "臀大肌 / 髋外旋肌群",
                    "reason": "落地或下蹲时膝内扣趋势常与髋外旋控制不足相关。",
                    "priority": "高",
                },
            ]
        )
        movement_patterns.append("下肢动力链近端控制不足，可能表现为膝内扣和单腿支撑不稳。")
        therapist_focus.append("优先观察单腿下蹲、台阶下落和变向动作中的膝-髋协调。")

    if front_metrics.get("trunk_shift_pct", 0) > 0.03:
        primary_muscles.extend(
            [
                {
                    "muscle": "腹横肌 / 内外腹斜肌",
                    "reason": "躯干中线偏移提示躯干抗旋转和骨盆稳定性不足。",
                    "priority": "高",
                },
                {
                    "muscle": "多裂肌 / 腰方肌",
                    "reason": "身体重心控制偏移常见于腰盆稳定不足。",
                    "priority": "中",
                },
            ]
        )
        movement_patterns.append("核心抗旋转与骨盆稳定性不足，可能影响上下注力传导。")
        therapist_focus.append("观察躯干是否需要通过腰椎代偿完成稳定。")

    if side_metrics.get("trunk_lean_deg", 0) > 6:
        primary_muscles.extend(
            [
                {
                    "muscle": "臀大肌",
                    "reason": "侧面前倾/前移常提示髋主导发力不足，臀部驱动力下降。",
                    "priority": "高",
                },
                {
                    "muscle": "腘绳肌群",
                    "reason": "髋铰链控制不足时常需要后侧链代偿。",
                    "priority": "中",
                },
            ]
        )
        movement_patterns.append("髋主导发力模式不足，可能转为躯干前移或腰椎代偿。")
        therapist_focus.append("重点看髋铰链动作是否由腰椎而不是髋关节完成。")

    if side_metrics.get("hip_hinge_left_deg", 180) < 165 or side_metrics.get("hip_hinge_right_deg", 180) < 165:
        secondary_muscles.extend(
            [
                {
                    "muscle": "腘绳肌群",
                    "reason": "髋铰链角度偏小提示后链参与不足或姿势补偿。",
                    "priority": "中",
                },
                {
                    "muscle": "竖脊肌 / 胸腰筋膜系统",
                    "reason": "躯干控制不足时常出现背伸代偿。",
                    "priority": "中",
                },
            ]
        )

    if acl_level in {"moderate", "high"}:
        therapist_focus.append("ACL 风险上升时应加强落地缓冲、髋外展和膝内扣控制。")

    if not primary_muscles:
        primary_muscles.append(
            {
                "muscle": "暂无明显高疑似肌群",
                "reason": "当前静态姿态未见明显偏移，可继续关注动态动作质量。",
                "priority": "低",
            }
        )

    parsed_primary = _dedupe_muscles(primary_muscles)
    parsed_secondary = _dedupe_muscles(secondary_muscles)

    return {
        "primary_muscles": parsed_primary,
        "secondary_muscles": parsed_secondary,
        "dominant_targets": _unique([item["muscle"] for item in parsed_primary]),
        "movement_patterns": _unique(movement_patterns),
        "therapist_focus": _unique(therapist_focus),
        "front_issues": front_issues,
        "side_issues": side_issues,
        "acl_level": acl_level,
    }


def build_report_sections(
    *,
    patient: Dict[str, Any],
    front_result: Dict[str, Any],
    side_result: Dict[str, Any],
    summary: Dict[str, Any],
    muscle_map: Dict[str, Any],
    lang: str = "zh",
) -> Dict[str, Any]:
    acl = summary.get("acl_risk", {})
    recommendations = summary.get("recommendations", [])

    if lang == "en":
        return {
            "title": f"Therapist Rehab Report: {patient.get('patient_name') or patient.get('patient_code') or 'Patient'}",
            "overview_title": "Assessment Summary",
            "risk_title": "ACL Risk Assessment",
            "chain_title": "Kinetic Chain Analysis",
            "muscle_title": "Muscle Function Hypothesis",
            "plan_title": "Personalized Rehab Plan",
            "notes_title": "Therapist Notes",
            "followup_title": "Follow-up Schedule",
            "overview_lines": [
                f"Patient: {patient.get('patient_name') or '-'}",
                f"ID: {patient.get('patient_code') or '-'}",
                f"Front findings: {', '.join(front_result.get('issues', []))}",
                f"Side findings: {', '.join(side_result.get('issues', []))}",
            ],
            "risk_lines": [
                f"ACL level: {acl.get('label_en', 'Low')} ({acl.get('score', 0):.2f})",
                f"Clinical note: {summary.get('summary_lines', [''])[0]}",
            ],
            "chain_lines": summary.get("kinetic_chain", []),
            "muscle_lines": [
                f"{item['muscle']} - {item['reason']} (Priority: {item['priority']})"
                for item in muscle_map.get("primary_muscles", [])
            ]
            + [
                f"{item['muscle']} - {item['reason']} (Priority: {item['priority']})"
                for item in muscle_map.get("secondary_muscles", [])
            ],
            "plan_lines": recommendations,
            "notes_lines": muscle_map.get("therapist_focus", []),
            "followup_lines": [
                "Reassess in 2-4 weeks or after a new training block.",
            ],
        }

    return {
        "title": f"治疗师康复报告：{patient.get('patient_name') or patient.get('patient_code') or '患者'}",
        "overview_title": "评估概览",
        "risk_title": "ACL 风险评估",
        "chain_title": "动力链分析",
        "muscle_title": "肌群功能推断",
        "plan_title": "个性化康复建议",
        "notes_title": "治疗师备注",
        "followup_title": "复评计划",
        "overview_lines": [
            f"患者：{patient.get('patient_name') or '-'}",
            f"编号：{patient.get('patient_code') or '-'}",
            f"正面结果：{', '.join(front_result.get('issues', []))}",
            f"侧面结果：{', '.join(side_result.get('issues', []))}",
        ],
        "risk_lines": [
            f"ACL 风险：{acl.get('label_zh', '低')}（{acl.get('score', 0):.2f}）",
            f"临床提示：{summary.get('summary_lines', [''])[0]}",
        ],
        "chain_lines": summary.get("kinetic_chain", []),
        "muscle_lines": [
            f"{item['muscle']} - {item['reason']}（优先级：{item['priority']}）"
            for item in muscle_map.get("primary_muscles", [])
        ]
        + [
            f"{item['muscle']} - {item['reason']}（优先级：{item['priority']}）"
            for item in muscle_map.get("secondary_muscles", [])
        ],
        "plan_lines": recommendations,
        "notes_lines": muscle_map.get("therapist_focus", []),
        "followup_lines": [
            "建议 2-4 周复评一次，或在新的训练周期后复查。",
        ],
    }


def render_markdown_report(sections: Dict[str, Any], lang: str = "zh") -> str:
    def bullets(items: List[str]) -> List[str]:
        return [f"- {item}" for item in items] if items else ["- 无"]

    return "\n".join(
        [
            f"# {sections['title']}",
            "",
            f"## {sections['overview_title']}",
            *bullets(sections.get("overview_lines", [])),
            "",
            f"## {sections['risk_title']}",
            *bullets(sections.get("risk_lines", [])),
            "",
            f"## {sections['chain_title']}",
            *bullets(sections.get("chain_lines", [])),
            "",
            f"## {sections['muscle_title']}",
            *bullets(sections.get("muscle_lines", [])),
            "",
            f"## {sections['plan_title']}",
            *bullets(sections.get("plan_lines", [])),
            "",
            f"## {sections['notes_title']}",
            *bullets(sections.get("notes_lines", [])),
            "",
            f"## {sections['followup_title']}",
            *bullets(sections.get("followup_lines", [])),
        ]
    )
