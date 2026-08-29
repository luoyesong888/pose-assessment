from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np


LANDMARKS = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

FRONT_REQUIRED = (
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

SIDE_CHAINS = {
    "left": ("left_shoulder", "left_hip", "left_knee", "left_ankle"),
    "right": ("right_shoulder", "right_hip", "right_knee", "right_ankle"),
}

VIEW_LABELS_ZH = {
    "front": "正面站立",
    "back": "背面站立",
    "side": "侧面站立",
    "forward_bend": "前屈动作",
    "other": "其他姿态",
}


def get_point(landmarks, name: str, w: int, h: int) -> np.ndarray:
    landmark = landmarks[LANDMARKS[name]]
    return np.array([landmark.x * w, landmark.y * h], dtype=float)


def _normalized_point(landmarks, name: str) -> np.ndarray:
    landmark = landmarks[LANDMARKS[name]]
    return np.array([landmark.x, landmark.y], dtype=float)


def landmark_confidence(landmark: Any) -> float:
    visibility = float(getattr(landmark, "visibility", 1.0) or 0.0)
    presence = float(getattr(landmark, "presence", 1.0) or 0.0)
    return min(visibility, presence)


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """计算三点之间的角度，b 是顶点。"""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cosine = np.dot(ba, bc) / denom
    return round(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))), 1)


def acl_not_assessed(reason: str = "静态照片不能单独评估 ACL 损伤风险") -> Dict[str, Any]:
    return {
        "score": None,
        "level": "not_assessed",
        "label_zh": "未评估（需动态测试）",
        "label_en": "Not assessed (dynamic testing required)",
        "valid": False,
        "reason": reason,
    }


def _chain_confidence(landmarks, names: Iterable[str]) -> float:
    values = [landmark_confidence(landmarks[LANDMARKS[name]]) for name in names]
    return sum(values) / len(values) if values else 0.0


def _capture_geometry(landmarks) -> Dict[str, float]:
    ls = _normalized_point(landmarks, "left_shoulder")
    rs = _normalized_point(landmarks, "right_shoulder")
    lh = _normalized_point(landmarks, "left_hip")
    rh = _normalized_point(landmarks, "right_hip")
    lk = _normalized_point(landmarks, "left_knee")
    rk = _normalized_point(landmarks, "right_knee")
    la = _normalized_point(landmarks, "left_ankle")
    ra = _normalized_point(landmarks, "right_ankle")
    shoulder_mid, hip_mid = (ls + rs) / 2, (lh + rh) / 2
    knee_mid, ankle_mid = (lk + rk) / 2, (la + ra) / 2
    torso_dx = float(shoulder_mid[0] - hip_mid[0])
    torso_dy = float(hip_mid[1] - shoulder_mid[1])
    torso_length = float(np.linalg.norm(shoulder_mid - hip_mid))
    shoulder_span = abs(float(ls[0] - rs[0]))
    hip_span = abs(float(lh[0] - rh[0]))
    span_ratio = shoulder_span / max(torso_length, 1e-6)
    trunk_angle = float(np.degrees(np.arctan2(abs(torso_dx), max(abs(torso_dy), 1e-6))))
    left_face = np.mean([landmark_confidence(landmarks[index]) for index in (1, 2, 3, 7, 9)])
    right_face = np.mean([landmark_confidence(landmarks[index]) for index in (4, 5, 6, 8, 10)])
    return {
        "torso_height_norm": round(torso_dy, 4),
        "torso_length_norm": round(torso_length, 4),
        "trunk_angle_deg": round(trunk_angle, 1),
        "shoulder_span_norm": round(shoulder_span, 4),
        "hip_span_norm": round(hip_span, 4),
        "shoulder_torso_ratio": round(span_ratio, 4),
        "face_visibility": round(float((left_face + right_face) / 2), 4),
        "face_asymmetry": round(abs(float(left_face - right_face)), 4),
        "upright_order": bool(shoulder_mid[1] < hip_mid[1] < knee_mid[1] < ankle_mid[1]),
        "legs_below_hips": bool(hip_mid[1] < knee_mid[1] < ankle_mid[1]),
    }


def landmark_quality(landmarks, view: str, min_confidence: float = 0.5) -> Dict[str, Any]:
    if not landmarks or len(landmarks) <= max(LANDMARKS.values()):
        return {"valid": False, "errors": ["人体关键点数量不足"], "warnings": [], "selected_side": None}

    if view in {"front", "back"}:
        low = [name for name in FRONT_REQUIRED if landmark_confidence(landmarks[LANDMARKS[name]]) < min_confidence]
        errors = [f"关键点置信度不足：{', '.join(low)}"] if low else []
        return {"valid": not errors, "errors": errors, "warnings": [], "selected_side": None}

    confidences = {side: _chain_confidence(landmarks, names) for side, names in SIDE_CHAINS.items()}
    selected_side = max(confidences, key=confidences.get)
    selected_confidence = confidences[selected_side]
    if view == "side":
        errors = [] if selected_confidence >= min_confidence else ["侧面肩、髋、膝、踝关键点置信度不足"]
    else:
        torso_names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        torso_confidence = _chain_confidence(landmarks, torso_names)
        errors = [] if torso_confidence >= min_confidence and selected_confidence >= min_confidence else ["姿态分析所需的肩、髋或下肢关键点置信度不足"]
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": [],
        "selected_side": selected_side,
        "chain_confidence": round(selected_confidence, 3),
    }


def validate_capture_geometry(landmarks, view: str) -> Dict[str, Any]:
    """保守检查标准自然站立视角；无法证明正面/侧面时拒绝生成报告。"""
    quality = landmark_quality(landmarks, view)
    errors = list(quality.get("errors", []))
    warnings = list(quality.get("warnings", []))
    if errors:
        return {**quality, "errors": errors, "warnings": warnings}

    geometry = _capture_geometry(landmarks)
    torso_height = float(geometry["torso_height_norm"])
    if torso_height < 0.10:
        errors.append("未检测到自然直立躯干，请使用站立照片而非前屈或坐姿照片")
    if not geometry["upright_order"]:
        errors.append("身体关键点顺序不符合自然站立姿势")

    span_ratio = float(geometry["shoulder_torso_ratio"])
    face_mean = float(geometry["face_visibility"])
    face_asymmetry = float(geometry["face_asymmetry"])

    if view == "front":
        if span_ratio < 0.28:
            errors.append("正面照片疑似侧身，双肩展开宽度不足")
        if face_mean < 0.45:
            errors.append("正面照片未可靠检测到面部，疑似背面或遮挡")
    else:
        if span_ratio > 0.70 and face_asymmetry < 0.12:
            errors.append("侧面照片疑似正面或背面，肩宽与面部特征不符合侧视图")
        if span_ratio > 0.90:
            errors.append("侧面照片双肩展开过宽，请保持标准侧身站立")

    return {
        **quality,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "geometry": geometry,
    }


def classify_capture_view(landmarks, frontal_face_count: int) -> Dict[str, Any]:
    """自动识别照片的主要观察类型。

    不再把背面、前屈或斜向姿态当作错误；只要关键点足够可见，
    就使用与该姿态匹配的描述性指标。
    """
    generic_quality = landmark_quality(landmarks, "other")
    if not generic_quality.get("valid"):
        return {
            "valid": False,
            "detected_view": "unknown",
            "quality": generic_quality,
        }

    geometry = _capture_geometry(landmarks)
    ratio = float(geometry.get("shoulder_torso_ratio", 1.0))
    torso_height = float(geometry.get("torso_height_norm", 0.0))
    trunk_angle = float(geometry.get("trunk_angle_deg", 0.0))
    upright = bool(geometry.get("upright_order"))

    if geometry.get("legs_below_hips") and (torso_height < 0.10 or trunk_angle >= 55):
        detected_view = "forward_bend"
    elif upright and trunk_angle >= 25:
        detected_view = "other"
    elif upright and ratio <= 0.34:
        detected_view = "side"
    elif upright and frontal_face_count > 0:
        detected_view = "front"
    elif upright and ratio > 0.34:
        detected_view = "back"
    else:
        detected_view = "other"

    quality = landmark_quality(landmarks, detected_view)
    quality["valid"] = bool(quality.get("valid"))
    quality["geometry"] = {**geometry, "frontal_face_count": frontal_face_count}
    quality.setdefault("warnings", [])
    if detected_view == "back":
        quality["warnings"].append("未检测到正脸，按背面姿态分析")
    elif detected_view == "other":
        quality["warnings"].append("拍摄角度或姿态非标准，仅输出当前可见关键点指标")
    return {"valid": quality["valid"], "detected_view": detected_view, "quality": quality}


def _invalid_measurement(view: str, quality: Dict[str, Any]) -> Dict[str, Any]:
    issues = quality.get("errors") or ["关键点质量不足，无法分析"]
    return {
        "valid": False,
        "view": view,
        "issues": issues,
        "metrics": {},
        "quality": quality,
        "acl_risk": acl_not_assessed("关键点或照片质量不足，未进行 ACL 筛查"),
    }


def measure_pose_metrics(landmarks, w: int, h: int, view: str = "front") -> Dict[str, Any]:
    """把静态照片关键点转为描述性指标，不输出 ACL 损伤概率。

    view 支持 front / back / side / forward_bend / other。
    自动视角需要同时读取像素级正脸信息，由 pose_worker 处理。
    """
    if view == "auto":
        return _invalid_measurement(
            view,
            {"valid": False, "errors": ["自动视角必须通过完整图片分析"], "warnings": []},
        )
    quality = landmark_quality(landmarks, view)
    if not quality["valid"]:
        return _invalid_measurement(view, quality)

    if view == "side":
        side = quality["selected_side"]
        shoulder = get_point(landmarks, f"{side}_shoulder", w, h)
        hip = get_point(landmarks, f"{side}_hip", w, h)
        knee = get_point(landmarks, f"{side}_knee", w, h)
        ankle = get_point(landmarks, f"{side}_ankle", w, h)
        trunk_dx = abs(float(shoulder[0] - hip[0]))
        trunk_dy = max(abs(float(shoulder[1] - hip[1])), 1.0)
        trunk_lean_deg = round(float(np.degrees(np.arctan2(trunk_dx, trunk_dy))), 1)
        hip_angle = calculate_angle(shoulder, hip, knee)
        knee_angle = calculate_angle(hip, knee, ankle)
        metrics = {
            "view": view,
            "selected_side": side,
            "trunk_lean_deg": trunk_lean_deg,
            "hip_angle_deg": hip_angle,
            "knee_angle_deg": knee_angle,
        }
        issues: List[str] = []
        if trunk_lean_deg > 6:
            issues.append("侧面躯干相对垂线偏移，需结合标准化复拍与动态动作确认")
        if knee_angle < 170:
            issues.append("侧面静态站姿可见膝关节屈曲，需确认是否为自然站姿")
        if not issues:
            issues.append("侧面静态关键点未见超过当前观察阈值的偏移")
        return {
            "valid": True, "view": view, "issues": issues, "metrics": metrics,
            "quality": quality, "acl_risk": acl_not_assessed(),
        }

    if view in {"forward_bend", "other"}:
        ls, rs = get_point(landmarks, "left_shoulder", w, h), get_point(landmarks, "right_shoulder", w, h)
        lh, rh = get_point(landmarks, "left_hip", w, h), get_point(landmarks, "right_hip", w, h)
        lk, rk = get_point(landmarks, "left_knee", w, h), get_point(landmarks, "right_knee", w, h)
        la, ra = get_point(landmarks, "left_ankle", w, h), get_point(landmarks, "right_ankle", w, h)
        shoulder_mid, hip_mid, knee_mid = (ls + rs) / 2, (lh + rh) / 2, (lk + rk) / 2
        torso_dx = abs(float(shoulder_mid[0] - hip_mid[0]))
        torso_dy = abs(float(shoulder_mid[1] - hip_mid[1]))
        trunk_angle = round(float(np.degrees(np.arctan2(torso_dx, max(torso_dy, 1.0)))), 1)
        projected_hip_angle = calculate_angle(shoulder_mid, hip_mid, knee_mid)
        left_knee_angle = calculate_angle(lh, lk, la)
        right_knee_angle = calculate_angle(rh, rk, ra)
        shoulder_tilt_pct = abs(float(ls[1] - rs[1])) / h if h else 0.0
        hip_tilt_pct = abs(float(lh[1] - rh[1])) / h if h else 0.0
        trunk_shift_pct = torso_dx / w if w else 0.0
        projected_torso_length = float(np.linalg.norm((shoulder_mid - hip_mid) / np.array([max(w, 1), max(h, 1)])))
        knee_flexion_asymmetry = abs(left_knee_angle - right_knee_angle)
        metrics = {
            "view": view,
            "projected_trunk_angle_deg": trunk_angle,
            "projected_hip_angle_deg": projected_hip_angle,
            "projected_torso_length_norm": round(projected_torso_length, 4),
            "left_knee_angle_deg": left_knee_angle,
            "right_knee_angle_deg": right_knee_angle,
            "knee_flexion_asymmetry_deg": round(knee_flexion_asymmetry, 1),
            "more_flexed_knee_side": "left" if left_knee_angle < right_knee_angle else ("right" if right_knee_angle < left_knee_angle else "equal"),
            "shoulder_tilt_pct": round(float(shoulder_tilt_pct), 4),
            "hip_tilt_pct": round(float(hip_tilt_pct), 4),
            "trunk_shift_pct": round(float(trunk_shift_pct), 4),
            "shoulder_lower_side": "left" if ls[1] > rs[1] else ("right" if rs[1] > ls[1] else "level"),
            "hip_lower_side": "left" if lh[1] > rh[1] else ("right" if rh[1] > lh[1] else "level"),
            "trunk_shift_side": "left" if shoulder_mid[0] < hip_mid[0] else ("right" if shoulder_mid[0] > hip_mid[0] else "centered"),
        }
        issues: List[str] = [
            "已识别为前屈动作，按动作截图分析" if view == "forward_bend" else
            "已按非标准角度姿态分析当前可见关键点"
        ]
        if view == "forward_bend":
            issues.append(f"躯干在画面中的归一化投影长度：{projected_torso_length:.4f}（不等于真实髋屈角度）")
        if shoulder_tilt_pct > 0.03:
            issues.append("动作中可见肩线高度差，建议多次重复确认")
        if hip_tilt_pct > 0.03:
            issues.append("动作中可见骨盆线高度差，建议多次重复确认")
        if trunk_shift_pct > 0.03:
            issues.append("动作中肩髋中线存在水平偏移")
        if knee_flexion_asymmetry > 8:
            issues.append("左右膝屈角度差较明显，需用视频或重复截图确认")
        issues.append("单张 2D 照片不能区分脊柱曲屈与髋铰链各自的贡献")
        return {
            "valid": True, "view": view, "issues": issues, "metrics": metrics,
            "quality": quality, "acl_risk": acl_not_assessed(),
        }

    ls, rs = get_point(landmarks, "left_shoulder", w, h), get_point(landmarks, "right_shoulder", w, h)
    lh, rh = get_point(landmarks, "left_hip", w, h), get_point(landmarks, "right_hip", w, h)
    lk, rk = get_point(landmarks, "left_knee", w, h), get_point(landmarks, "right_knee", w, h)
    la, ra = get_point(landmarks, "left_ankle", w, h), get_point(landmarks, "right_ankle", w, h)
    le, re = get_point(landmarks, "left_ear", w, h), get_point(landmarks, "right_ear", w, h)
    shoulder_mid, hip_mid, knee_mid = (ls + rs) / 2, (lh + rh) / 2, (lk + rk) / 2

    shoulder_diff_px, hip_diff_px = abs(ls[1] - rs[1]), abs(lh[1] - rh[1])
    trunk_shift_px, pelvic_shift_px = abs(shoulder_mid[0] - hip_mid[0]), abs(hip_mid[0] - knee_mid[0])
    left_knee_alignment_px, right_knee_alignment_px = abs(lk[0] - la[0]), abs(rk[0] - ra[0])
    shoulder_tilt_pct, hip_tilt_pct = shoulder_diff_px / h if h else 0.0, hip_diff_px / h if h else 0.0
    trunk_shift_pct, pelvic_shift_pct = trunk_shift_px / w if w else 0.0, pelvic_shift_px / w if w else 0.0
    knee_alignment_pct = max(left_knee_alignment_px, right_knee_alignment_px) / w if w else 0.0
    ears_visible = min(
        landmark_confidence(landmarks[LANDMARKS["left_ear"]]),
        landmark_confidence(landmarks[LANDMARKS["right_ear"]]),
    ) >= 0.5
    head_tilt_deg = (
        round(float(np.degrees(np.arctan2(abs(le[1] - re[1]), max(abs(le[0] - re[0]), 1.0)))), 1)
        if ears_visible else None
    )

    metrics: Dict[str, Any] = {
        "view": view,
        "shoulder_diff_px": round(float(shoulder_diff_px), 1),
        "hip_diff_px": round(float(hip_diff_px), 1),
        "trunk_shift_px": round(float(trunk_shift_px), 1),
        "pelvic_shift_px": round(float(pelvic_shift_px), 1),
        "left_knee_alignment_px": round(float(left_knee_alignment_px), 1),
        "right_knee_alignment_px": round(float(right_knee_alignment_px), 1),
        "shoulder_tilt_pct": round(float(shoulder_tilt_pct), 4),
        "hip_tilt_pct": round(float(hip_tilt_pct), 4),
        "trunk_shift_pct": round(float(trunk_shift_pct), 4),
        "pelvic_shift_pct": round(float(pelvic_shift_pct), 4),
        "knee_alignment_pct": round(float(knee_alignment_pct), 4),
        "head_tilt_deg": head_tilt_deg,
        "head_lower_side": ("left" if le[1] > re[1] else "right") if head_tilt_deg is not None else "",
        "shoulder_lower_side": "left" if ls[1] > rs[1] else ("right" if rs[1] > ls[1] else "level"),
        "hip_lower_side": "left" if lh[1] > rh[1] else ("right" if rh[1] > lh[1] else "level"),
        "trunk_shift_side": "left" if shoulder_mid[0] < hip_mid[0] else ("right" if shoulder_mid[0] > hip_mid[0] else "centered"),
        "knee_alignment_dominant_side": "left" if left_knee_alignment_px > right_knee_alignment_px else ("right" if right_knee_alignment_px > left_knee_alignment_px else "equal"),
    }

    issues: List[str] = []
    if shoulder_tilt_pct > 0.03:
        issues.append(f"肩线可见高度差：{'左' if ls[1] > rs[1] else '右'}侧偏低")
    if hip_tilt_pct > 0.03:
        issues.append(f"骨盆线可见高度差：{'左' if lh[1] > rh[1] else '右'}侧偏低")
    if trunk_shift_pct > 0.03:
        issues.append(f"肩髋中线存在相对位移：肩线相对向{'左' if shoulder_mid[0] < hip_mid[0] else '右'}侧")
    if knee_alignment_pct > 0.045:
        side = "左" if left_knee_alignment_px > right_knee_alignment_px else "右"
        issues.append(f"膝踝静态对线差异：{side}侧较明显，需动态测试确认")
    if head_tilt_deg is not None and head_tilt_deg > 3:
        side = "左" if le[1] > re[1] else "右"
        issues.append(f"头部在画面中向{side}侧倾斜约 {head_tilt_deg:.1f}°，需复拍确认")
    if not issues:
        label = "背面" if view == "back" else "正面"
        issues.append(f"{label}静态关键点未见超过当前观察阈值的偏移")

    return {
        "valid": True, "view": view, "issues": issues, "metrics": metrics,
        "quality": quality, "acl_risk": acl_not_assessed(),
    }


def analyze_posture(landmarks, w, h):
    return measure_pose_metrics(landmarks, w, h, view="front")["issues"]


def summarize_case(
    front_result: Dict[str, Any] | None,
    side_result: Dict[str, Any] | None,
    image_results: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    front_result, side_result = front_result or {}, side_result or {}
    front_metrics, side_metrics = front_result.get("metrics", {}), side_result.get("metrics", {})
    has_front = bool(front_result.get("valid", True) and front_metrics)
    has_side = bool(side_result.get("valid", True) and side_metrics)
    usable_results = [
        result for result in (image_results or [front_result, side_result])
        if result and result.get("valid") and result.get("metrics")
    ]
    assessment_valid = bool(usable_results)
    if not assessment_valid:
        return {
            "assessment_valid": False,
            "acl_risk": acl_not_assessed("照片或关键点质量不足"),
            "movement_screening": {"level": "invalid", "label_zh": "无法评估", "flags": []},
            "kinetic_chain": ["照片或关键点质量不足，不能生成动力链结论。"],
            "red_flags": ["请重新上传人体主要关节清晰可见的照片。"],
            "recommendations": ["完成合格复拍后再进行分析。"],
            "summary_lines": ["本次评估无效：照片或关键点质量不足。"],
        }

    flags: List[str] = []
    kinetic_patterns: List[str] = []
    detected_views: List[str] = []
    for result in usable_results:
        metrics = result.get("metrics", {})
        view = result.get("detected_view") or metrics.get("view") or result.get("view") or "other"
        if view not in detected_views:
            detected_views.append(view)
        if metrics.get("knee_alignment_pct", 0) > 0.045 and "knee_ankle_alignment" not in flags:
            flags.append("knee_ankle_alignment")
            kinetic_patterns.append("可见膝踝对线差异，需用单腿下蹲或落地动作确认。")
        if metrics.get("trunk_shift_pct", 0) > 0.03 and "trunk_shift" not in flags:
            flags.append("trunk_shift")
            kinetic_patterns.append("肩髋中线存在可见偏移，需用重复截图或动态躯干控制测试确认。")
        if metrics.get("trunk_lean_deg", 0) > 6 and "trunk_lean" not in flags:
            flags.append("trunk_lean")
            kinetic_patterns.append("侧向截图中躯干相对垂线偏移超过当前观察阈值。")
        if metrics.get("shoulder_tilt_pct", 0) > 0.03 and "shoulder_tilt" not in flags:
            flags.append("shoulder_tilt")
            kinetic_patterns.append("可见肩线高度差，需比较多次站姿或动作截图判断是否稳定存在。")
        if metrics.get("hip_tilt_pct", 0) > 0.03 and "hip_tilt" not in flags:
            flags.append("hip_tilt")
            kinetic_patterns.append("可见骨盆线高度差，需排除站位、镜头和动作时点影响。")
        if metrics.get("knee_flexion_asymmetry_deg", 0) > 8 and "knee_flexion_asymmetry" not in flags:
            flags.append("knee_flexion_asymmetry")
            kinetic_patterns.append("动作截图中左右膝屈角度不同，需用连续视频确认时序与重复性。")
    if not kinetic_patterns:
        kinetic_patterns.append("当前照片关键点未见超过观察阈值的明显差异；这不代表动态功能正常。")

    recommendations = ["ACL 风险需通过落地、变向、单腿下蹲、既往伤史和力量测试另行评估。"]
    if len(detected_views) == 1:
        recommendations.append("当前结论来自单一照片类型；可增加其他角度或动作以建立更完整的体态档案。")
    if "knee_ankle_alignment" in flags:
        recommendations.append("增加正面单腿下蹲和台阶下落录像，观察膝、髋、踝的动态协同。")
    if "trunk_shift" in flags or "trunk_lean" in flags:
        recommendations.append("在治疗师监督下复测自然站姿、髋铰链和躯干抗旋转控制。")
    if not flags:
        recommendations.append("保持标准化拍摄条件，2-4 周后用相同站位复评静态指标。")

    movement_label = "存在需复测项目" if flags else "未见超阈值项目"
    coverage_label = " + ".join(VIEW_LABELS_ZH.get(view, view) for view in detected_views)
    return {
        "assessment_valid": True,
        "view_coverage": {
            "front": has_front,
            "side": has_side,
            "back": "back" in detected_views,
            "forward_bend": "forward_bend" in detected_views,
            "other": "other" in detected_views,
            "types": detected_views,
            "label_zh": coverage_label,
        },
        "acl_risk": acl_not_assessed(),
        "movement_screening": {"level": "attention" if flags else "observed", "label_zh": movement_label, "flags": flags},
        "kinetic_chain": kinetic_patterns,
        "red_flags": [
            "静态图仅用于描述可见对线，不能预测损伤或替代临床检查。",
            f"当前照片覆盖：{coverage_label}；结论仅适用于已提供的角度与动作时点。",
        ],
        "recommendations": recommendations,
        "summary_lines": [
            "ACL 风险：未评估（需动态测试）",
            f"视角覆盖：{coverage_label}",
            f"静态对线观察：{movement_label}",
            f"动力链：{kinetic_patterns[0]}",
        ],
    }
