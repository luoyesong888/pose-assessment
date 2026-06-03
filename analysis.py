from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

# MediaPipe 关节点编号定义
LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


def get_point(landmarks, name, w, h):
    idx = LANDMARKS[name]
    l = landmarks[idx]
    return np.array([l.x * w, l.y * h], dtype=float)


def calculate_angle(a, b, c):
    """计算三点之间的角度，b 是顶点。"""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cosine = np.dot(ba, bc) / denom
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return round(float(angle), 1)


def _risk_level(score: float) -> str:
    if score < 2.0:
        return "low"
    if score < 4.0:
        return "moderate"
    return "high"


def _risk_label(level: str, lang: str = "zh") -> str:
    labels = {
        "zh": {"low": "低", "moderate": "中等", "high": "较高"},
        "en": {"low": "Low", "moderate": "Moderate", "high": "High"},
    }
    return labels["en" if lang == "en" else "zh"].get(level, level)


def measure_pose_metrics(landmarks, w: int, h: int, view: str = "front") -> Dict[str, Any]:
    """把单张图的姿态关键点整理成给治疗师看的结构化指标。"""
    ls = get_point(landmarks, "left_shoulder", w, h)
    rs = get_point(landmarks, "right_shoulder", w, h)
    lh = get_point(landmarks, "left_hip", w, h)
    rh = get_point(landmarks, "right_hip", w, h)
    lk = get_point(landmarks, "left_knee", w, h)
    rk = get_point(landmarks, "right_knee", w, h)
    la = get_point(landmarks, "left_ankle", w, h)
    ra = get_point(landmarks, "right_ankle", w, h)

    shoulder_mid = (ls + rs) / 2
    hip_mid = (lh + rh) / 2
    knee_mid = (lk + rk) / 2
    ankle_mid = (la + ra) / 2

    shoulder_diff_px = abs(ls[1] - rs[1])
    hip_diff_px = abs(lh[1] - rh[1])
    trunk_shift_px = abs(shoulder_mid[0] - hip_mid[0])
    pelvic_shift_px = abs(hip_mid[0] - knee_mid[0])

    left_knee_alignment_px = abs(lk[0] - la[0])
    right_knee_alignment_px = abs(rk[0] - ra[0])

    shoulder_tilt_pct = shoulder_diff_px / h if h else 0.0
    hip_tilt_pct = hip_diff_px / h if h else 0.0
    trunk_shift_pct = trunk_shift_px / w if w else 0.0
    pelvic_shift_pct = pelvic_shift_px / w if w else 0.0
    knee_alignment_pct = max(left_knee_alignment_px, right_knee_alignment_px) / w if w else 0.0

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
    }

    issues: List[str] = []
    if shoulder_tilt_pct > 0.03:
        side = "左" if ls[1] > rs[1] else "右"
        issues.append(f"肩线不平衡：{side}侧偏低")
    if hip_tilt_pct > 0.03:
        side = "左" if lh[1] > rh[1] else "右"
        issues.append(f"骨盆不平衡：{side}侧偏低")
    if trunk_shift_pct > 0.03:
        side = "左" if shoulder_mid[0] < hip_mid[0] else "右"
        issues.append(f"躯干中线偏移：向{side}侧")
    if view == "front" and knee_alignment_pct > 0.045:
        side = "左" if left_knee_alignment_px > right_knee_alignment_px else "右"
        issues.append(f"膝-踝对线偏移：{side}侧控制不足")

    if not issues:
        issues.append("可见姿态对称性较好")

    if view == "front":
        acl_score = (
            knee_alignment_pct * 55
            + shoulder_tilt_pct * 18
            + hip_tilt_pct * 17
            + trunk_shift_pct * 10
        )
        acl_level = _risk_level(acl_score)
    else:
        trunk_lean_pct = abs(shoulder_mid[0] - hip_mid[0]) / w if w else 0.0
        trunk_lean_deg = round(float(np.degrees(np.arctan2(abs(shoulder_mid[0] - hip_mid[0]), max(abs(shoulder_mid[1] - hip_mid[1]), 1.0)))), 1)
        hip_hinge_left = calculate_angle(lh, lk, la)
        hip_hinge_right = calculate_angle(rh, rk, ra)
        side_score = trunk_lean_pct * 40 + max(abs(180 - hip_hinge_left), abs(180 - hip_hinge_right)) / 60
        metrics["trunk_lean_deg"] = trunk_lean_deg
        metrics["hip_hinge_left_deg"] = hip_hinge_left
        metrics["hip_hinge_right_deg"] = hip_hinge_right
        metrics["side_score"] = round(float(side_score), 2)
        acl_score = side_score
        acl_level = _risk_level(side_score)

    metrics["acl_proxy_score"] = round(float(acl_score), 2)
    metrics["acl_risk_level"] = acl_level
    metrics["acl_risk_label_zh"] = _risk_label(acl_level, "zh")
    metrics["acl_risk_label_en"] = _risk_label(acl_level, "en")

    return {
        "view": view,
        "issues": issues,
        "metrics": metrics,
        "acl_risk": {
            "score": metrics["acl_proxy_score"],
            "level": acl_level,
            "label_zh": metrics["acl_risk_label_zh"],
            "label_en": metrics["acl_risk_label_en"],
        },
    }


def analyze_posture(landmarks, w, h):
    """兼容旧接口，返回简短问题列表。"""
    result = measure_pose_metrics(landmarks, w, h, view="front")
    return result["issues"]


def summarize_case(front_result: Dict[str, Any] | None, side_result: Dict[str, Any] | None) -> Dict[str, Any]:
    front_metrics = (front_result or {}).get("metrics", {})
    side_metrics = (side_result or {}).get("metrics", {})

    acl_level = front_metrics.get("acl_risk_level", "low")
    if acl_level == "moderate" and side_metrics.get("acl_risk_level") == "high":
        acl_level = "high"

    kinetic_patterns: List[str] = []
    if front_metrics.get("knee_alignment_pct", 0) > 0.045:
        kinetic_patterns.append("下肢矢状面控制不足，疑似近端稳定性不足导致膝-踝对线偏移。")
    if front_metrics.get("trunk_shift_pct", 0) > 0.03:
        kinetic_patterns.append("躯干中线偏移提示核心与骨盆带控制不足。")
    if side_metrics.get("trunk_lean_deg", 0) > 6:
        kinetic_patterns.append("侧面存在明显躯干前移/前倾趋势，建议强化髋主导发力模式。")
    if not kinetic_patterns:
        kinetic_patterns.append("整体动力链协同较稳定，未见明显失衡模式。")

    red_flags: List[str] = []
    if acl_level in {"moderate", "high"}:
        red_flags.append("需重点关注单腿落地、变向和深蹲位移时的膝内扣控制。")
    if side_metrics.get("trunk_lean_deg", 0) > 8:
        red_flags.append("需要监测腰骶-髋段代偿，避免腰椎过度负荷。")
    if not red_flags:
        red_flags.append("当前未见明显高风险红旗，仍建议动态复评。")

    recommendations: List[str] = []
    if acl_level == "high":
        recommendations.extend([
            "优先进行髋外展与外旋肌群激活训练。",
            "加入镜像反馈的单腿下蹲与落地控制训练。",
        ])
    elif acl_level == "moderate":
        recommendations.extend([
            "采用低负荷的神经肌肉控制训练，重点优化膝-踝对线。",
            "增加髋主导动作模式训练和核心抗旋转训练。",
        ])
    else:
        recommendations.extend([
            "维持现有训练并加入预防性稳定性训练。",
            "每 2-4 周复评一次姿态与动作控制变化。",
        ])

    if side_metrics.get("trunk_lean_deg", 0) > 6:
        recommendations.append("补充髋铰链模式和胸椎活动度训练。")

    return {
        "acl_risk": {
            "level": acl_level,
            "label_zh": _risk_label(acl_level, "zh"),
            "label_en": _risk_label(acl_level, "en"),
            "score": front_metrics.get("acl_proxy_score", side_metrics.get("side_score", 0)),
        },
        "kinetic_chain": kinetic_patterns,
        "red_flags": red_flags,
        "recommendations": recommendations,
        "summary_lines": [
            f"ACL 风险：{_risk_label(acl_level, 'zh')}",
            f"动力链：{kinetic_patterns[0]}",
        ],
    }
