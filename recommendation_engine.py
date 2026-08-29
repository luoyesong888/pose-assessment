from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


def _meaningful(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(text and text not in {"无", "none", "no", "n/a", "未填写", "not provided"})


def _exercise(name: str, dosage: str, cue: str) -> Dict[str, str]:
    return {"name": name, "dosage": dosage, "cue": cue}


def _option(
    option_id: str,
    title: str,
    score: int,
    summary: str,
    evidence: List[str],
    validation: str,
    exercises: List[Dict[str, str]],
    precautions: List[str],
    rag_sources: List[str],
) -> Dict[str, Any]:
    return {
        "id": option_id,
        "title": title,
        "priority_score": score,
        "priority": "优先" if score >= 80 else ("建议" if score >= 50 else "基础"),
        "summary": summary,
        "evidence": evidence,
        "validation": validation,
        "exercises": exercises,
        "precautions": precautions,
        "rag_sources": rag_sources,
        "judgment_type": "screening_recommendation",
    }


def _rag_sources(rag_context: Dict[str, Any]) -> List[str]:
    result = []
    for item in rag_context.get("knowledge", []):
        source_id = item.get("source_id")
        if source_id and source_id not in result:
            result.append(source_id)
    return result[:5]


def _supported_sources(rag_context: Dict[str, Any], query: str) -> List[str]:
    from posture_rag import search_text

    sources = _rag_sources(rag_context)
    matches = search_text(
        query,
        top_k=3,
        source_kinds={"exercise_action_taxonomy", "fitness_topic_summary"},
    )
    for item in matches:
        source_id = item.get("source_id")
        if source_id and source_id not in sources:
            sources.append(source_id)
    return sources[:8]


def build_recommendation_options(
    profile: Dict[str, Any],
    front_result: Dict[str, Any],
    side_result: Dict[str, Any],
    summary: Dict[str, Any],
    rag_context: Dict[str, Any],
    lang: str = "zh",
    max_options: int = 3,
    image_results: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """以规则为主、RAG 为词汇支持，生成可解释且低风险的改善重点。"""
    front = (front_result or {}).get("metrics", {})
    side = (side_result or {}).get("metrics", {})
    coverage = summary.get("view_coverage", {})
    sources = _rag_sources(rag_context)
    stop_rule = "出现疼痛、麻木、眩晕或不稳定感时立即停止，并寻求专业评估。"
    options: List[Dict[str, Any]] = []
    all_metrics = [
        item.get("metrics", {}) for item in (image_results or [])
        if item.get("valid") and item.get("metrics")
    ]
    frontal_metrics = [item for item in all_metrics if item.get("view") in {"front", "back"}]
    if frontal_metrics:
        front = {
            key: max((float(item.get(key, 0) or 0) for item in frontal_metrics), default=0)
            for key in ("knee_alignment_pct", "trunk_shift_pct", "shoulder_tilt_pct", "hip_tilt_pct")
        }

    capture_types = coverage.get("types", [])
    if len(capture_types) == 1:
        options.append(
            _option(
                "complementary_capture",
                "增加补充角度或动作",
                35,
                f"已完成{coverage.get('label_zh', '单类照片')}分析；增加其他角度或动作可帮助交叉验证。",
                ["当前只有一类照片", "单帧结果只代表当前动作时点"],
                "可选择上传背面站立、侧面站立、前屈或下蹲截图，保持主要关节清晰可见。",
                [],
                ["不同角度不能直接比较像素距离，应比较各自视角下的归一化指标。"],
                sources,
            )
        )

    if _meaningful(profile.get("pain_areas")) or _meaningful(profile.get("injury_history")):
        options.append(
            _option(
                "professional_review",
                "先完成疼痛与伤史评估",
                95,
                "存在疼痛、不适或既往损伤信息，训练建议应先由治疗师结合症状和体格检查确认。",
                [f"疼痛 / 不适：{profile.get('pain_areas') or '未填写'}", f"既往损伤 / 手术史：{profile.get('injury_history') or '未填写'}"],
                "记录诱发动作、疼痛强度、持续时间和缓解因素，交由专业人员评估。",
                [],
                [stop_rule],
                sources,
            )
        )

    if front.get("knee_alignment_pct", 0) > 0.045:
        options.append(
            _option(
                "lower_limb_control",
                "验证膝髋踝动态协同",
                85,
                "静态膝踝对线差异超过当前观察阈值，优先通过低负荷动态动作确认控制模式。",
                [f"膝踝归一化对线差：{front['knee_alignment_pct']:.4f}", "静态差异不等于膝内扣或 ACL 风险"],
                "拍摄正面单腿下蹲或低台阶下落各 5 次，观察膝盖是否持续偏离足部方向。",
                [
                    _exercise("扶墙单腿平衡", "2 组 × 20–30 秒/侧", "骨盆保持水平，膝盖朝向第二脚趾"),
                    _exercise("高凳坐站", "2 组 × 6–8 次", "慢速起坐，保持髋膝足方向一致"),
                ],
                [stop_rule, "动作质量优先，不追求深度或负重。"],
                _supported_sources(rag_context, "squat lunge knee hip ankle single leg balance"),
            )
        )

    if front.get("trunk_shift_pct", 0) > 0.03:
        options.append(
            _option(
                "trunk_control",
                "提升躯干与骨盆控制",
                78,
                "肩髋中线存在相对位移，建议通过重复站姿与抗旋转动作验证稳定性。",
                [f"肩髋中线归一化位移：{front['trunk_shift_pct']:.4f}"],
                "自然站立复拍 3 次，再做四点跪姿对侧抬手抬腿，观察骨盆是否旋转。",
                [
                    _exercise("死虫式脚跟点地", "2 组 × 6 次/侧", "腰背保持舒适中立，缓慢呼气"),
                    _exercise("鸟狗式", "2 组 × 6 次/侧", "骨盆不旋转，动作幅度以稳定为准"),
                ],
                [stop_rule],
                _supported_sources(rag_context, "core plank bird dog dead bug trunk control"),
            )
        )

    if front.get("shoulder_tilt_pct", 0) > 0.03:
        options.append(
            _option(
                "shoulder_control",
                "验证肩胛与胸廓协同",
                70,
                "肩线可见高度差超过当前观察阈值，先验证其是否在重复站姿和抬臂时持续存在。",
                [f"肩线归一化高度差：{front['shoulder_tilt_pct']:.4f}"],
                "连续自然站立复拍 3 次，并完成正面缓慢抬臂 5 次。",
                [
                    _exercise("墙面滑手", "2 组 × 8 次", "肋骨保持自然，肩部不耸起"),
                    _exercise("四点跪姿胸椎旋转", "1–2 组 × 6 次/侧", "骨盆保持稳定，动作舒适即可"),
                ],
                [stop_rule, "不强行把两侧肩膀压到同一高度。"],
                _supported_sources(rag_context, "shoulder scapula upper back posture mobility stretch"),
            )
        )

    if front.get("hip_tilt_pct", 0) > 0.03:
        options.append(
            _option(
                "pelvic_control",
                "验证骨盆与髋外展控制",
                72,
                "骨盆线可见高度差超过当前观察阈值，建议先检查站位重复性与单腿支撑控制。",
                [f"骨盆线归一化高度差：{front['hip_tilt_pct']:.4f}"],
                "双脚位置固定后复拍 3 次，再进行扶墙单腿站立。",
                [
                    _exercise("臀桥", "2 组 × 8 次", "双侧均匀发力，不追求过高幅度"),
                    _exercise("扶墙髋外展", "2 组 × 8 次/侧", "躯干保持直立，动作缓慢"),
                ],
                [stop_rule],
                _supported_sources(rag_context, "hip glute balance single leg squat control"),
            )
        )

    hip_angle = side.get("hip_angle_deg", 180)
    if side.get("trunk_lean_deg", 0) > 6 or hip_angle < 165:
        options.append(
            _option(
                "hip_hinge",
                "重建髋铰链动作感",
                74,
                "侧面躯干相对垂线或髋角偏离直立观察范围，先排除摆拍后再练习髋主导动作。",
                [f"侧面躯干角：{side.get('trunk_lean_deg', 0):.1f}°", f"髋角：{hip_angle:.1f}°"],
                "使用木棍贴住头、胸椎和骶骨完成 5 次髋铰链，三点保持接触。",
                [
                    _exercise("四点跪姿后坐", "2 组 × 8 次", "保持脊柱舒适中立，髋部向后移动"),
                    _exercise("木棍髋铰链", "2 组 × 8 次", "三点接触，动作范围以稳定为准"),
                ],
                [stop_rule],
                _supported_sources(rag_context, "hip hinge squat mobility core training"),
            )
        )

    action_metrics = [item for item in all_metrics if item.get("view") in {"forward_bend", "other"}]
    knee_asymmetry = max((float(item.get("knee_flexion_asymmetry_deg", 0) or 0) for item in action_metrics), default=0)
    if knee_asymmetry > 8:
        options.append(
            _option(
                "movement_symmetry",
                "验证动作左右对称性",
                76,
                "动作截图中左右膝屈角度差超过当前观察阈值，先用连续视频确认。",
                [f"左右膝屈角度差：{knee_asymmetry:.1f}°", "单帧照片不能判断动作时序"],
                "从同一机位拍摄 5 次缓慢前屈或髋铰链动作，比较左右膝屈差是否重复出现。",
                [
                    _exercise("扶墙髋铰链", "2 组 × 6–8 次", "髋部向后，左右脚均匀承重"),
                    _exercise("高凳坐站", "2 组 × 6–8 次", "缓慢起坐，两侧膝盖方向保持一致"),
                ],
                [stop_rule],
                _supported_sources(rag_context, "hip hinge squat symmetry knee control"),
            )
        )

    if not options:
        options.append(
            _option(
                "maintenance",
                "维持活动与标准化复评",
                20,
                "静态关键点未见超过当前观察阈值的明显偏移，可把重点放在一般活动和动态动作质量。",
                ["当前静态观察未见超阈值项目"],
                "每 2–4 周在相同站位、光线和相机高度下复拍。",
                [
                    _exercise("轻松步行", "每周 3–5 次，每次 20–30 分钟", "保持可交谈强度"),
                    _exercise("扶墙单腿平衡", "2 组 × 20 秒/侧", "保持呼吸自然和身体稳定"),
                ],
                [stop_rule],
                _supported_sources(rag_context, "walking balance mobility general fitness"),
            )
        )

    options.sort(key=lambda item: item["priority_score"], reverse=True)
    return options[:max_options]


def recommendation_summary_lines(options: List[Dict[str, Any]]) -> List[str]:
    return [
        f"{item['priority']}：{item['title']}——{item['summary']}"
        for item in options
    ]


def build_confirmed_plan(
    options: List[Dict[str, Any]],
    selected_ids: List[str],
    lang: str = "zh",
) -> Dict[str, Any]:
    selected = [item for item in options if item["id"] in set(selected_ids)][:3]
    lines: List[str] = []
    for item in selected:
        lines.append(f"【{item['title']}】验证：{item['validation']}")
        for exercise in item.get("exercises", []):
            lines.append(f"{exercise['name']}：{exercise['dosage']}；提示：{exercise['cue']}")
        for precaution in item.get("precautions", []):
            lines.append(f"注意：{precaution}")
    return {
        "selected_ids": [item["id"] for item in selected],
        "selected_titles": [item["title"] for item in selected],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "lines": lines,
    }


def apply_confirmed_plan(
    assessment: Dict[str, Any],
    selected_ids: List[str],
    lang: str = "zh",
) -> Dict[str, Any]:
    plan = build_confirmed_plan(
        assessment.get("recommendation_options", []),
        selected_ids,
        lang=lang,
    )
    if not plan["selected_ids"]:
        return assessment
    title = "Confirmed Improvement Plan" if lang == "en" else "已确认改善计划"
    assessment["confirmed_plan"] = plan
    sections = assessment.setdefault("report_sections", {})
    sections["confirmed_plan_title"] = title
    sections["confirmed_plan_lines"] = plan["lines"]
    base_report = assessment.setdefault("base_report_md", assessment.get("report_md", ""))
    plan_markdown = "\n".join([f"## {title}", *[f"- {line}" for line in plan["lines"]]])
    assessment["report_md"] = f"{base_report.rstrip()}\n\n{plan_markdown}\n"
    return assessment
