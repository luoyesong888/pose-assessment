import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 初始化
model_path = "pose_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(base_options=base_options)
detector = vision.PoseLandmarker.create_from_options(options)

# 骨架连线定义
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)
]

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

        cv2.imshow("姿态分析", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ 没有检测到人体，请换一张图片")

if __name__ == "__main__":
    image_path = input("请输入图片路径：")
    analyze_image(image_path)