from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# 避免第三方库把缓存写到不可写的用户目录。
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name("data") / ".matplotlib"))

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

from analysis import (  # noqa: E402
    acl_not_assessed,
    classify_capture_view,
    measure_pose_metrics,
    validate_capture_geometry,
)


MODEL_PATH = Path(__file__).with_name("pose_landmarker.task")
ANNOTATED_DIR = Path(__file__).with_name("data") / "annotated"
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
]


def failed(view: str, message: str) -> Dict[str, Any]:
    return {
        "found": False,
        "valid": False,
        "view": view,
        "issues": [message],
        "metrics": {},
        "quality": {"valid": False, "errors": [message], "warnings": []},
        "acl_risk": acl_not_assessed(message),
    }


def create_detector():
    errors = []
    preferred = os.getenv("KINETIQ_MEDIAPIPE_DELEGATE", "CPU").strip().upper()
    delegates = (
        (("GPU", python.BaseOptions.Delegate.GPU), ("CPU", python.BaseOptions.Delegate.CPU))
        if preferred == "GPU"
        else (("CPU", python.BaseOptions.Delegate.CPU),)
    )
    for name, delegate in delegates:
        try:
            base = python.BaseOptions(model_asset_path=str(MODEL_PATH), delegate=delegate)
            options = vision.PoseLandmarkerOptions(
                base_options=base,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=4,
                min_pose_detection_confidence=0.6,
                min_pose_presence_confidence=0.6,
            )
            return vision.PoseLandmarker.create_from_options(options), name
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def render_annotated(frame, landmarks, image_path: Path, tag: str) -> str:
    """在图片上绘制骨架连线与关键点，保存后返回路径（失败返回空串）。"""
    h, w = frame.shape[:2]
    annotated = frame.copy()
    for start, end in CONNECTIONS:
        p1, p2 = landmarks[start], landmarks[end]
        if min(float(p1.visibility), float(p2.visibility)) < 0.5:
            continue
        cv2.line(
            annotated,
            (int(p1.x * w), int(p1.y * h)),
            (int(p2.x * w), int(p2.y * h)),
            (46, 204, 113),
            3,
        )
    for landmark in landmarks:
        if float(landmark.visibility) < 0.5:
            continue
        cv2.circle(annotated, (int(landmark.x * w), int(landmark.y * h)), 5, (231, 76, 60), -1)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    annotated_path = ANNOTATED_DIR / f"{image_path.stem}_{tag}_annotated.jpg"
    if not cv2.imwrite(str(annotated_path), annotated):
        return ""
    return str(annotated_path)


def detect_frontal_faces(frame, landmarks) -> int:
    """只在 MediaPipe 头部区域检测正脸，避免背景物体造成误报。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade_path = Path(__file__).with_name("face_detector.xml")
    face_detector = cv2.CascadeClassifier(str(cascade_path))
    if face_detector.empty():
        raise RuntimeError("正脸检测模型不可用")
    h, w = frame.shape[:2]
    face_landmarks = landmarks[:11]
    x_values = [float(item.x) for item in face_landmarks]
    y_values = [float(item.y) for item in face_landmarks]
    shoulder_y = min(float(landmarks[11].y), float(landmarks[12].y))
    x1 = max(0, int((min(x_values) - 0.06) * w))
    x2 = min(w, int((max(x_values) + 0.06) * w))
    y1 = max(0, int((min(y_values) - 0.06) * h))
    y2 = min(h, int(max(max(y_values) + 0.05, shoulder_y) * h))
    face_roi = gray[y1:y2, x1:x2]
    if not face_roi.size:
        return 0
    min_face = max(20, min(face_roi.shape[:2]) // 5)
    faces = face_detector.detectMultiScale(
        face_roi,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(min_face, min_face),
    )
    return len(faces)


def analyze_auto(frame, image_path: Path, landmarks, delegate: str) -> Dict[str, Any]:
    """自动识别正面、背面、侧面、前屈或其他姿态并按类型分析。"""
    h, w = frame.shape[:2]
    try:
        frontal_face_count = detect_frontal_faces(frame, landmarks)
    except RuntimeError as exc:
        return failed("auto", str(exc))
    classification = classify_capture_view(landmarks, frontal_face_count)
    detected_view = classification["detected_view"]
    capture_quality = classification["quality"]
    if not classification["valid"]:
        result = failed("auto", "；".join(capture_quality.get("errors", [])) or "无法判断照片视角")
        result["found"] = True
        result["detected_view"] = detected_view
        result["quality"] = capture_quality
        result["engine_delegate"] = delegate
        return result

    structured = measure_pose_metrics(landmarks, w, h, view=detected_view)
    if not structured.get("valid"):
        result = failed("auto", "；".join(structured.get("issues", [])) or "无法分析该图片")
        result["found"] = True
        result["detected_view"] = detected_view
        result["quality"] = structured.get("quality", capture_quality)
        result["engine_delegate"] = delegate
        return result

    annotated_path = render_annotated(frame, landmarks, image_path, detected_view)
    if not annotated_path:
        return failed("auto", "标记图片保存失败")

    quality = dict(capture_quality)
    quality["landmarks"] = structured.get("quality", {})
    quality["detected_view"] = detected_view
    quality["downgraded"] = False

    return {
        "found": True,
        "valid": True,
        "view": "auto",
        "detected_view": detected_view,
        "issues": structured["issues"],
        "metrics": structured["metrics"],
        "quality": quality,
        "acl_risk": structured["acl_risk"],
        "annotated_path": annotated_path,
        "source_path": str(image_path),
        "engine_delegate": delegate,
    }


def analyze_with_detector(image_path: Path, view: str, detector, delegate: str) -> Dict[str, Any]:
    if not image_path.is_file():
        return failed(view, "上传图片不存在")
    frame = cv2.imread(str(image_path))
    if frame is None:
        return failed(view, "图片读取失败")

    rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba_frame)
    detection = detector.detect(image)
    if not detection.pose_landmarks:
        return failed(view, "未检测到完整人体关键点，请确保头部至足部均在画面内")
    if len(detection.pose_landmarks) > 1:
        return failed(view, f"检测到 {len(detection.pose_landmarks)} 个人，请上传仅包含一人的照片")

    landmarks = detection.pose_landmarks[0]
    if view == "auto":
        return analyze_auto(frame, image_path, landmarks, delegate)

    capture_quality = validate_capture_geometry(landmarks, view)
    if view == "front" and capture_quality["valid"]:
        try:
            frontal_face_count = detect_frontal_faces(frame, landmarks)
        except RuntimeError:
            return failed(view, "正脸检测模型不可用，请重新安装项目资源")
        capture_quality.setdefault("geometry", {})["frontal_face_count"] = frontal_face_count
        if frontal_face_count == 0:
            capture_quality["valid"] = False
            capture_quality["errors"].append("正面照片未检测到正脸，请确认不是背面照且面部无遮挡")
    if not capture_quality["valid"]:
        result = failed(view, "；".join(capture_quality["errors"]))
        result["found"] = True
        result["quality"] = capture_quality
        result["engine_delegate"] = delegate
        return result

    h, w = frame.shape[:2]
    structured = measure_pose_metrics(landmarks, w, h, view=view)
    if not structured.get("valid"):
        result = failed(view, "；".join(structured.get("issues", [])))
        result["found"] = True
        result["quality"] = structured.get("quality", capture_quality)
        result["engine_delegate"] = delegate
        return result

    annotated = frame.copy()
    for start, end in CONNECTIONS:
        p1, p2 = landmarks[start], landmarks[end]
        if min(float(p1.visibility), float(p2.visibility)) < 0.5:
            continue
        cv2.line(
            annotated,
            (int(p1.x * w), int(p1.y * h)),
            (int(p2.x * w), int(p2.y * h)),
            (46, 204, 113),
            3,
        )
    for landmark in landmarks:
        if float(landmark.visibility) < 0.5:
            continue
        cv2.circle(annotated, (int(landmark.x * w), int(landmark.y * h)), 5, (231, 76, 60), -1)

    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    annotated_path = ANNOTATED_DIR / f"{image_path.stem}_{view}_annotated.jpg"
    if not cv2.imwrite(str(annotated_path), annotated):
        return failed(view, "标记图片保存失败")
    return {
        "found": True,
        "valid": True,
        "view": view,
        "issues": structured["issues"],
        "metrics": structured["metrics"],
        "quality": {**capture_quality, "landmarks": structured.get("quality", {})},
        "acl_risk": structured["acl_risk"],
        "annotated_path": str(annotated_path),
        "source_path": str(image_path),
        "engine_delegate": delegate,
    }


def analyze(image_path: Path, view: str) -> Dict[str, Any]:
    detector, delegate = create_detector()
    try:
        return analyze_with_detector(image_path, view, detector, delegate)
    finally:
        detector.close()


def analyze_batch(items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    if not items:
        return []
    detector, delegate = create_detector()
    try:
        return [
            analyze_with_detector(
                Path(item.get("image", "")).resolve(),
                item.get("view", "auto"),
                detector,
                delegate,
            )
            for item in items
        ]
    finally:
        detector.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--view", choices=("front", "side", "auto"), default="auto")
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()
    if args.batch:
        payload = json.loads(sys.stdin.read() or "[]")
        result = analyze_batch(payload)
    elif args.image:
        result = analyze(Path(args.image).resolve(), args.view)
    else:
        parser.error("--image or --batch is required")
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
