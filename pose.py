import cv2
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from analysis import analyze_posture, measure_pose_metrics

# 初始化
model_path = "pose_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(base_options=base_options)
detector = vision.PoseLandmarker.create_from_options(options)

DATA_DIR = Path(__file__).with_name("data")
ANNOTATED_DIR = DATA_DIR / "annotated"
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

# 骨架连线定义
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)
]


def analyze_image_file(image_path, view="front"):
    """返回单张图片的结构化姿态结果，供 Streamlit 页面调用。"""
    image = mp.Image.create_from_file(image_path)
    result = detector.detect(image)

    if not result.pose_landmarks:
        return {
            "found": False,
            "view": view,
            "issues": ["未检测到人体关键点"],
            "metrics": {},
            "acl_risk": {"level": "low", "score": 0},
        }

    img = cv2.imread(image_path)
    if img is None:
        return {
            "found": False,
            "view": view,
            "issues": ["图片读取失败"],
            "metrics": {},
            "acl_risk": {"level": "low", "score": 0},
        }

    h, w = img.shape[:2]
    landmarks = result.pose_landmarks[0]
    structured = measure_pose_metrics(landmarks, w, h, view=view)

    annotated = img.copy()
    for start, end in CONNECTIONS:
        x1 = int(landmarks[start].x * w)
        y1 = int(landmarks[start].y * h)
        x2 = int(landmarks[end].x * w)
        y2 = int(landmarks[end].y * h)
        cv2.line(annotated, (x1, y1), (x2, y2), (46, 204, 113), 3)

    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(annotated, (x, y), 5, (231, 76, 60), -1)

    for idx, issue in enumerate(structured["issues"][:4]):
        cv2.putText(
            annotated,
            issue,
            (20, 40 + idx * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 215, 0),
            2,
            cv2.LINE_AA,
        )

    annotated_path = ANNOTATED_DIR / f"{Path(image_path).stem}_{view}_annotated.jpg"
    cv2.imwrite(str(annotated_path), annotated)
    return {
        "found": True,
        "view": view,
        "issues": structured["issues"],
        "metrics": structured["metrics"],
        "acl_risk": structured["acl_risk"],
        "annotated_path": str(annotated_path),
    }

def analyze_image(image_path):
    image = mp.Image.create_from_file(image_path)
    result = detector.detect(image)

    if result.pose_landmarks:
        print("✅ 关节点识别成功")
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        landmarks = result.pose_landmarks[0]

        # 画连线
        for start, end in CONNECTIONS:
            x1 = int(landmarks[start].x * w)
            y1 = int(landmarks[start].y * h)
            x2 = int(landmarks[end].x * w)
            y2 = int(landmarks[end].y * h)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 画关节点
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        # 姿态分析
        issues = analyze_posture(landmarks, w, h)

        # 把分析结果显示在图片上
        for i, issue in enumerate(issues):
            cv2.putText(img, issue, (20, 40 + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 打印到终端
        print("\n📋 姿态分析结果：")
        for issue in issues:
            print(issue)

        cv2.imshow("姿态分析", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ 没有检测到人体，请换一张图片")

if __name__ == "__main__":
    image_path = input("请输入图片路径：")
    analyze_image(image_path)
