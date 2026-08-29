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


def infer_muscle_mapping(
    front_result: Dict[str, Any] | None,
    side_result: Dict[str, Any] | None,
    summary: Dict[str, Any] | None,
    image_results: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """根据姿态指标推断可能受影响的肌群与动作控制问题。

    注意：这里只做“功能关联”推断，不做医学诊断。
    """
    front_metrics = (front_result or {}).get("metrics", {})
    side_metrics = (side_result or {}).get("metrics", {})
    front_issues = (front_result or {}).get("issues", [])
    side_issues = (side_result or {}).get("issues", [])
    acl = (summary or {}).get("acl_risk", {})
    acl_level = acl.get("level", "not_assessed")

    primary_muscles: List[Dict[str, str]] = []
    secondary_muscles: List[Dict[str, str]] = []
    movement_patterns: List[str] = []
    therapist_focus: List[str] = []

    all_metrics = [
        item.get("metrics", {}) for item in (image_results or [])
        if item.get("valid") and item.get("metrics")
    ]
    frontal_metrics = [
        metrics for metrics in all_metrics
        if metrics.get("view") in {"front", "back"}
    ]
    if frontal_metrics:
        front_metrics = {
            key: max((float(item.get(key, 0) or 0) for item in frontal_metrics), default=0)
            for key in ("knee_alignment_pct", "trunk_shift_pct", "shoulder_tilt_pct", "hip_tilt_pct")
        }

    if front_metrics.get("knee_alignment_pct", 0) > 0.045:
        primary_muscles.extend(
            [
                {
                    "muscle": "臀中肌 / 髋外展肌群",
                    "reason": "静态膝踝对线差异不能确认肌力不足；可在单腿下蹲中验证髋外展控制。",
                    "priority": "待验证",
                },
                {
                    "muscle": "臀大肌 / 髋外旋肌群",
                    "reason": "可通过落地或下蹲动作观察髋外旋控制，静态照片不作肌力结论。",
                    "priority": "待验证",
                },
            ]
        )
        movement_patterns.append("静态膝踝对线存在差异，是否伴随膝内扣或单腿支撑问题需动态验证。")
        therapist_focus.append("优先观察单腿下蹲、台阶下落和变向动作中的膝-髋协调。")

    if front_metrics.get("trunk_shift_pct", 0) > 0.03:
        primary_muscles.extend(
            [
                {
                    "muscle": "腹横肌 / 内外腹斜肌",
                    "reason": "肩髋中线相对位移可作为抗旋转动态测试的观察线索，不能由静态图确认无力。",
                    "priority": "待验证",
                },
                {
                    "muscle": "多裂肌 / 腰方肌",
                    "reason": "可在重复站姿和负重动作中验证腰盆控制，静态图仅提供线索。",
                    "priority": "待验证",
                },
            ]
        )
        movement_patterns.append("肩髋中线存在相对位移，建议验证躯干抗旋转和骨盆控制。")
        therapist_focus.append("观察躯干是否需要通过腰椎代偿完成稳定。")

    if side_metrics.get("trunk_lean_deg", 0) > 6:
        primary_muscles.extend(
            [
                {
                    "muscle": "臀大肌",
                    "reason": "侧面躯干偏移可作为髋铰链测试线索，不能由静态图确认臀部驱动力。",
                    "priority": "待验证",
                },
                {
                    "muscle": "腘绳肌群",
                    "reason": "建议在动态髋铰链中观察后侧链参与，静态站姿不足以判断代偿。",
                    "priority": "待验证",
                },
            ]
        )
        movement_patterns.append("侧面躯干偏移超过观察阈值，髋主导模式需通过动态动作验证。")
        therapist_focus.append("重点看髋铰链动作是否由腰椎而不是髋关节完成。")

    if side_metrics.get("hip_angle_deg", 180) < 165:
        therapist_focus.append("侧面髋角偏离直立位，先确认是否为自然站姿，再决定是否进行髋铰链动态测试。")

    action_metrics = [metrics for metrics in all_metrics if metrics.get("view") in {"forward_bend", "other"}]
    if any(item.get("knee_flexion_asymmetry_deg", 0) > 8 for item in action_metrics):
        primary_muscles.append(
            {
                "muscle": "髋膝伸展链 / 躯干稳定系统",
                "reason": "动作截图中可见左右膝屈角度差；需用连续视频区分动作时点与稳定代偿。",
                "priority": "待验证",
            }
        )
        therapist_focus.append("用连续前屈或髋铰链视频确认左右膝屈差是否可重复。")
    if any(item.get("view") == "forward_bend" for item in action_metrics):
        movement_patterns.append("已纳入前屈动作截图；可观察左右对称与膝屈差，但不能由单张 2D 图区分脊柱和髋关节的具体贡献。")

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


def build_regional_assessment_sections(
    image_results: List[Dict[str, Any]] | None,
    muscle_map: Dict[str, Any],
    patient: Dict[str, Any],
    lang: str = "zh",
) -> List[Dict[str, Any]]:
    """把关键点指标翻译成与示例类似的分区评估，并显式标注证据层级。"""
    usable = [
        item for item in (image_results or [])
        if item.get("valid") and item.get("metrics")
    ]
    frontal = [item for item in usable if item.get("metrics", {}).get("view") in {"front", "back"}]
    sides = [item for item in usable if item.get("metrics", {}).get("view") == "side"]
    actions = [item for item in usable if item.get("metrics", {}).get("view") in {"forward_bend", "other"}]

    def peak(items: List[Dict[str, Any]], key: str, default: float = 0.0) -> float:
        return max((float(item.get("metrics", {}).get(key, default) or default) for item in items), default=default)

    def peak_item(items: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
        return max(items, key=lambda item: float(item.get("metrics", {}).get(key, 0) or 0), default={})

    side_zh = {
        "left": "左侧", "right": "右侧", "level": "两侧接近水平",
        "centered": "接近居中", "equal": "左右接近",
    }

    def zh_sections() -> List[Dict[str, Any]]:
        basis: List[str] = [
            f"【照片覆盖】共上传 {len(image_results or [])} 张，其中 {len(usable)} 张关键点可用。",
        ]
        for index, item in enumerate(image_results or []):
            view = item.get("detected_view") or item.get("metrics", {}).get("view") or item.get("view", "unknown")
            quality = item.get("quality", {})
            confidence = quality.get("chain_confidence") or quality.get("landmarks", {}).get("chain_confidence")
            status = "可用" if item.get("valid") else "不可用"
            confidence_text = f"，关键链置信度 {float(confidence):.3f}" if confidence is not None else ""
            basis.append(f"【逐图识别】照片 {index + 1}：{view}，{status}{confidence_text}。")
        basis.extend([
            "【计算口径】高度差和中线偏移均按画面尺寸归一化，用于同一人在相似拍摄条件下复评。",
            "【适用范围】结果是 2D 体态与动作筛查，不是疾病诊断或结构性影像学结论。",
        ])

        pelvis: List[str] = []
        hip_tilt = peak(frontal + actions, "hip_tilt_pct")
        trunk_shift = peak(frontal + actions, "trunk_shift_pct")
        hip_tilt_metrics = peak_item(frontal + actions, "hip_tilt_pct").get("metrics", {})
        trunk_shift_metrics = peak_item(frontal + actions, "trunk_shift_pct").get("metrics", {})
        if frontal or actions:
            pelvis.append(
                f"【可见观察】骨盆线归一化高度差最大为 {hip_tilt:.4f}，"
                f"{side_zh.get(hip_tilt_metrics.get('hip_lower_side', ''), '方向不确定')}在画面中相对偏低；"
                + ("超过当前观察阈值。" if hip_tilt > 0.03 else "未超过当前观察阈值。")
            )
            pelvis.append(
                f"【可见观察】肩髋中线水平偏移最大为 {trunk_shift:.4f}，"
                f"肩线相对向{side_zh.get(trunk_shift_metrics.get('trunk_shift_side', ''), '不确定方向')}偏移；"
                + ("建议用重复拍摄确认。" if trunk_shift > 0.03 else "未见明显超阈值偏移。")
            )
        if actions:
            torso_length = peak(actions, "projected_torso_length_norm")
            pelvis.append(f"【动作观察】已纳入前屈/非标准动作截图；躯干投影长度指标为 {torso_length:.4f}。")
        if sides:
            pelvis.append(f"【可见观察】侧面躯干相对垂线偏移最大约 {peak(sides, 'trunk_lean_deg'):.1f}°。")
        pelvis.append("【需验证】单张 2D 照片不能确认髋关节、胸廓或椎体的三维旋转方向。")

        lower: List[str] = []
        knee_alignment = peak(frontal, "knee_alignment_pct")
        knee_metrics = peak_item(frontal, "knee_alignment_pct").get("metrics", {})
        if frontal:
            lower.append(
                f"【可见观察】膝踝归一化对线差最大为 {knee_alignment:.4f}，"
                f"{side_zh.get(knee_metrics.get('knee_alignment_dominant_side', ''), '方向不确定')}差异更明显；"
                + ("需进一步动态验证。" if knee_alignment > 0.045 else "未超过当前观察阈值。")
            )
        if actions:
            knee_diff = peak(actions, "knee_flexion_asymmetry_deg")
            action_knee_metrics = peak_item(actions, "knee_flexion_asymmetry_deg").get("metrics", {})
            lower.append(
                f"【动作观察】单帧中左右膝屈角度差最大约 {knee_diff:.1f}°，"
                f"{side_zh.get(action_knee_metrics.get('more_flexed_knee_side', ''), '方向不确定')}屈曲更多；应用连续视频确认是否重复。"
            )
        lower.extend([
            "【需验证】下肢稳定性需用单腿站、单腿蹲或台阶下落观察，不由静态照片直接定性。",
            "【证据边界】静态照片不评定 ACL 扭伤风险；需结合落地、变向、伤史、力量及专业检查。",
        ])

        upper: List[str] = []
        shoulder_tilt = peak(frontal + actions, "shoulder_tilt_pct")
        head_tilt = peak(frontal, "head_tilt_deg")
        shoulder_metrics = peak_item(frontal + actions, "shoulder_tilt_pct").get("metrics", {})
        head_metrics = peak_item(frontal, "head_tilt_deg").get("metrics", {})
        if frontal or actions:
            upper.append(
                f"【可见观察】肩线归一化高度差最大为 {shoulder_tilt:.4f}，"
                f"{side_zh.get(shoulder_metrics.get('shoulder_lower_side', ''), '方向不确定')}在画面中相对偏低；"
                + ("需重复站姿与抬臂动作确认。" if shoulder_tilt > 0.03 else "未超过当前观察阈值。")
            )
        if head_tilt:
            upper.append(
                f"【可见观察】头部在画面中的倾斜角最大约 {head_tilt:.1f}°，"
                f"{side_zh.get(head_metrics.get('head_lower_side', ''), '方向不确定')}偏低；需在相机水平条件下复拍。"
            )
        upper.extend([
            "【功能假设】如肩线或肩胛轮廓差异在多次拍摄中持续，可进一步检查抬臂时的肩胛运动控制。",
            "【需验证】肩胛不稳定或肩峰下疼痛需结合症状、抬臂视频、活动度和临床检查。",
            "【不可由照片判断】视力、听力、咀嚼功能及自主神经状态不从体态照片推断。",
        ])

        training: List[str] = []
        for item in muscle_map.get("primary_muscles", [])[:4]:
            training.append(f"【功能假设】{item.get('muscle', '')}：{item.get('reason', '')}")
        if not training:
            training.append("【功能假设】当前图像没有提供足够证据指向特定肌群。")
        training.extend([
            f"【主观输入】疼痛/不适：{patient.get('pain_areas') or '未填写'}；伤病史：{patient.get('injury_history') or '未填写'}。",
            "【需验证】左右发力差、背阔肌或胸肌强弱需用对称负重动作、徒手力量测试或肌电确认。",
        ])

        chain: List[str] = []
        if hip_tilt > 0.03 and trunk_shift > 0.03:
            chain.append("【组合模式】骨盆线高度差与肩髋中线偏移同时出现，可作为额状面躯干—骨盆控制测试的线索。")
        if shoulder_tilt > 0.03 and trunk_shift > 0.03:
            chain.append("【组合模式】肩线高度差与肩髋中线偏移同时出现，建议比较自然站立、抬臂和负荷时是否持续。")
        if actions and peak(actions, "knee_flexion_asymmetry_deg") > 8:
            chain.append("【组合模式】前屈/动作截图中左右膝屈差异较大，建议优先验证承重与髋膝协同的重复性。")
        if not chain:
            chain.append("【组合模式】当前未见两个以上超阈值指标组成的稳定模式。")
        chain.append("【解释规则】组合模式表示“同时出现的观察线索”，不表示已证明的因果或代偿链。")

        priorities: List[str] = []
        pain_text = str(patient.get("pain_areas") or "").strip()
        history_text = str(patient.get("injury_history") or "").strip()
        if pain_text or history_text:
            priorities.append("【优先级 1】先结合疼痛、麻木、既往损伤和临床检查，再决定训练负荷。")
        if knee_alignment > 0.045 or peak(actions, "knee_flexion_asymmetry_deg") > 8:
            priorities.append("【优先级 2】验证左右下肢承重与膝髋踝动态协同。")
        if hip_tilt > 0.03 or trunk_shift > 0.03:
            priorities.append("【优先级 3】验证躯干抗侧屈/抗旋转与骨盆控制。")
        if shoulder_tilt > 0.03 or head_tilt > 3:
            priorities.append("【优先级 4】验证头颈中立和抬臂时的肩胛控制。")
        if not priorities:
            priorities.append("【基础优先级】保持无痛活动，在相同拍摄条件下建立复评基线。")

        validation: List[str] = [
            "【标准复拍】相机水平、镜头高度与骨盆接近、头足完整入镜；自然站立连拍 3 张。",
            "【前屈复测】从同一机位录制 5 次缓慢前屈视频，观察肩髋中线、骨盆线与左右膝屈差是否重复。",
            "【下肢复测】扶墙单腿站 20 秒/侧，再做高凳坐站或低台阶下落 5 次；不追求深度。",
            "【上肢复测】正面与背面各录制 5 次缓慢抬臂，比较肩线、肩胛轮廓和症状。",
            "【停止条件】出现疼痛加重、麻木、无力、眩晕或明显不稳时停止，寻求医生或合格康复专业人员评估。",
        ]

        followup: List[str] = [
            f"【基线】肩线高度差 {shoulder_tilt:.4f}；骨盆线高度差 {hip_tilt:.4f}；肩髋中线偏移 {trunk_shift:.4f}。",
            f"【基线】膝踝对线差 {knee_alignment:.4f}；动作中左右膝屈差 {peak(actions, 'knee_flexion_asymmetry_deg'):.1f}°；头部倾斜 {head_tilt:.1f}°。",
            "【复评时间】建议 2–4 周后，或完成一个稳定训练周期后，在相同机位、光线、站位和动作速度下复测。",
            "【判定原则】优先看多次拍摄的趋势与症状变化，不用单次像素差作为训练成败结论。",
        ]
        return [
            {"title": "0. 评估依据与照片质量", "lines": basis},
            {"title": "1. 骨盆与脊柱", "lines": pelvis},
            {"title": "2. 下肢与左右侧身体", "lines": lower},
            {"title": "3. 肩胛、右侧身体与头部", "lines": upper},
            {"title": "4. 训练感受与肌肉发力", "lines": training},
            {"title": "5. 综合模式与处理优先级", "lines": chain + priorities},
            {"title": "6. 建议补充的验证测试", "lines": validation},
            {"title": "7. 复评基线与追踪方法", "lines": followup},
        ]

    if lang == "zh":
        return zh_sections()
    return [
        {"title": "1. Pelvis and spine", "lines": ["Visible observations and hypotheses are listed in the Chinese report schema."]},
        {"title": "2. Lower limbs and side-to-side comparison", "lines": ["Dynamic stability requires movement testing."]},
        {"title": "3. Scapulae, upper body, and head", "lines": ["Photos cannot assess vision, hearing, or a clinical shoulder diagnosis."]},
        {"title": "4. Training sensation and muscle recruitment", "lines": ["Muscle recruitment hypotheses require strength or movement testing."]},
    ]


def build_report_sections(
    *,
    patient: Dict[str, Any],
    front_result: Dict[str, Any],
    side_result: Dict[str, Any],
    summary: Dict[str, Any],
    muscle_map: Dict[str, Any],
    lang: str = "zh",
    image_results: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    acl = summary.get("acl_risk", {})
    recommendations = summary.get("recommendations", [])
    view_coverage = summary.get("view_coverage", {})
    front_issues = front_result.get("issues", []) or (
        ["未提供正面照片"] if lang == "zh" else ["No front photo provided"]
    )
    side_issues = side_result.get("issues", []) or (
        ["未提供侧面照片"] if lang == "zh" else ["No side photo provided"]
    )
    view_labels_zh = {
        "front": "正面站立", "back": "背面站立", "side": "侧面站立",
        "forward_bend": "前屈动作", "other": "其他姿态", "unknown": "未知",
    }
    image_lines_zh: List[str] = []
    image_lines_en: List[str] = []
    for index, item in enumerate(image_results or []):
        view = item.get("detected_view") or item.get("view") or "unknown"
        issues = item.get("issues", []) or ["无可用观察"]
        image_lines_zh.append(f"照片 {index + 1}（{view_labels_zh.get(view, view)}）：" + "、".join(issues))
        image_lines_en.append(f"Image {index + 1} ({view}): " + "; ".join(issues))
    regional_sections = build_regional_assessment_sections(image_results, muscle_map, patient, lang=lang)

    if lang == "en":
        return {
            "title": f"Posture Rehab Report: {patient.get('patient_name') or patient.get('patient_code') or 'Client'}",
            "overview_title": "Assessment Summary",
            "risk_title": "ACL Screening Limits",
            "chain_title": "Kinetic Chain Analysis",
            "muscle_title": "Muscle Function Hypothesis",
            "plan_title": "Personalized Rehab Plan",
            "notes_title": "Therapist Notes",
            "followup_title": "Follow-up Schedule",
            "regional_sections": regional_sections,
            "overview_lines": [
                f"Name: {patient.get('patient_name') or '-'}",
                f"ID: {patient.get('patient_code') or '-'}",
                f"Gender: {patient.get('gender') or '-'}",
                f"Age: {patient.get('age') or '-'}",
                f"Height: {patient.get('height') or '-'} cm",
                f"Weight: {patient.get('weight') or '-'} kg",
                f"Occupation / daily activity: {patient.get('occupation') or '-'}",
                f"Activity level: {patient.get('activity') or '-'}",
                f"Pain / discomfort: {patient.get('pain_areas') or 'None'}",
                f"Injury / surgery history: {patient.get('injury_history') or 'None'}",
                f"Capture coverage: {view_coverage.get('label_zh', 'Unknown')}",
                *(image_lines_en or [f"Front findings: {', '.join(front_issues)}", f"Side findings: {', '.join(side_issues)}"]),
            ],
            "risk_lines": [
                f"ACL status: {acl.get('label_en', 'Not assessed')}",
                f"Reason: {acl.get('reason', 'Dynamic testing is required.')}",
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
        "title": f"体态康复评估报告：{patient.get('patient_name') or patient.get('patient_code') or '客户'}",
        "overview_title": "评估概览",
        "risk_title": "ACL 筛查边界",
        "chain_title": "动力链分析",
        "muscle_title": "肌群功能推断",
        "plan_title": "个性化康复建议",
        "notes_title": "治疗师备注",
        "followup_title": "复评计划",
        "regional_sections": regional_sections,
        "overview_lines": [
            f"姓名：{patient.get('patient_name') or '-'}",
            f"编号：{patient.get('patient_code') or '-'}",
            f"性别：{patient.get('gender') or '-'}",
            f"年龄：{patient.get('age') or '-'}",
            f"身高：{patient.get('height') or '-'} cm",
            f"体重：{patient.get('weight') or '-'} kg",
            f"职业 / 日常活动：{patient.get('occupation') or '-'}",
            f"运动频率：{patient.get('activity') or '-'}",
            f"疼痛 / 不适：{patient.get('pain_areas') or '无'}",
            f"既往损伤 / 手术史：{patient.get('injury_history') or '无'}",
            f"视角覆盖：{view_coverage.get('label_zh', '未知')}",
            *(image_lines_zh or [f"正面结果：{', '.join(front_issues)}", f"侧面结果：{', '.join(side_issues)}"]),
        ],
        "risk_lines": [
            f"ACL 状态：{acl.get('label_zh', '未评估（需动态测试）')}",
            f"原因：{acl.get('reason', '静态照片不能单独评估 ACL 损伤风险')}",
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

    parts = [
        f"# {sections['title']}",
        "",
        f"## {sections['overview_title']}",
        *bullets(sections.get("overview_lines", [])),
        "",
    ]
    for regional in sections.get("regional_sections", []):
        parts.extend([f"## {regional.get('title', '')}", *bullets(regional.get("lines", [])), ""])
    parts.extend(
        [
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
        ]
    )
    if sections.get("evidence_lines"):
        parts.extend(
            [
                f"## {sections.get('evidence_title', 'RAG Evidence & Limits')}",
                *bullets(sections.get("evidence_lines", [])),
                "",
            ]
        )
    if sections.get("confirmed_plan_lines"):
        parts.extend(
            [
                f"## {sections.get('confirmed_plan_title', '已确认改善计划')}",
                *bullets(sections.get("confirmed_plan_lines", [])),
                "",
            ]
        )
    parts.extend(
        [
            f"## {sections['notes_title']}",
            *bullets(sections.get("notes_lines", [])),
            "",
            f"## {sections['followup_title']}",
            *bullets(sections.get("followup_lines", [])),
        ]
    )
    return "\n".join(parts)
