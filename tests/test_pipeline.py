from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app_pipeline


class Upload:
    def __init__(self, name="test.jpg", payload=b"test-image"):
        self.name = name
        self.payload = payload

    def getbuffer(self):
        return self.payload


def invalid_result(view: str):
    return {
        "found": False,
        "valid": False,
        "view": view,
        "issues": ["未检测到完整人体关键点"],
        "metrics": {},
    }


def valid_result(view: str):
    metrics = {
        "view": view,
        "shoulder_tilt_pct": 0.01,
        "hip_tilt_pct": 0.01,
        "trunk_shift_pct": 0.01,
        "knee_alignment_pct": 0.01,
    }
    if view == "side":
        metrics = {"view": view, "trunk_lean_deg": 2.0, "hip_angle_deg": 178.0, "knee_angle_deg": 179.0}
    return {
        "found": True,
        "valid": True,
        "view": view,
        "issues": ["静态关键点未见超过当前观察阈值的偏移"],
        "metrics": metrics,
    }


def flexible_result(view: str):
    result = valid_result("front" if view == "back" else view)
    result["view"] = "auto"
    result["detected_view"] = view
    result["metrics"]["view"] = view
    result["issues"] = [f"已识别：{view}"]
    if view == "forward_bend":
        result["metrics"] = {
            "view": view,
            "shoulder_tilt_pct": 0.02,
            "hip_tilt_pct": 0.01,
            "trunk_shift_pct": 0.02,
            "left_knee_angle_deg": 176.0,
            "right_knee_angle_deg": 164.0,
            "knee_flexion_asymmetry_deg": 12.0,
        }
    return result


class PipelineTests(unittest.TestCase):
    def test_duplicate_uploads_are_rejected(self):
        uploads = [
            (Upload("a.jpg", b"same"), "img0", "auto"),
            (Upload("b.jpg", b"same"), "img1", "auto"),
        ]
        with self.assertRaises(app_pipeline.AssessmentInputError):
            app_pipeline.validate_upload_set(uploads)

    def test_more_than_six_uploads_are_rejected(self):
        uploads = [
            (Upload(f"{index}.jpg", str(index).encode()), f"img{index}", "auto")
            for index in range(7)
        ]
        with self.assertRaises(app_pipeline.AssessmentInputError):
            app_pipeline.validate_upload_set(uploads)

    def test_single_side_result_is_not_reused_as_front(self):
        side = valid_result("side")
        front_result, side_result = app_pipeline.pick_front_side([side])
        self.assertEqual(front_result, {})
        self.assertIs(side_result, side)

    def test_detection_failure_stops_report_generation(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(app_pipeline, "UPLOAD_DIR", Path(temp_dir) / "uploads"), patch.object(
            app_pipeline, "analyze_image_files", side_effect=lambda requests: [invalid_result(view) for _, view in requests]
        ):
            with self.assertRaises(app_pipeline.AssessmentInputError):
                app_pipeline.process_assessment(
                    patient_name="测试", patient_code="T1", lang="zh",
                    front_file=Upload("front.jpg", b"front"), side_file=Upload("side.jpg", b"side"), deepseek_key="",
                )

    def test_valid_assessment_uses_rag_and_no_acl_grade(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(app_pipeline, "UPLOAD_DIR", Path(temp_dir) / "uploads"), patch.object(
            app_pipeline, "analyze_image_files", side_effect=lambda requests: [valid_result(view) for _, view in requests]
        ):
            result = app_pipeline.process_assessment(
                patient_name="测试", patient_code="T2", lang="zh",
                front_file=Upload("front.jpg", b"front"), side_file=Upload("side.jpg", b"side"), deepseek_key="",
            )
        assessment = result["assessment"]
        self.assertEqual(assessment["summary"]["acl_risk"]["level"], "not_assessed")
        self.assertIn("## ACL 筛查边界", assessment["report_md"])
        self.assertIn("## RAG 参考数据与证据边界", assessment["report_md"])
        self.assertTrue(assessment["report_source"].startswith("fallback+local_rag"))
        self.assertTrue(assessment["recommendation_options"])
        self.assertEqual(assessment["recommendation_options"][0]["id"], "maintenance")

    def test_back_and_forward_bend_generate_a_complete_report(self):
        results = [flexible_result("back"), flexible_result("forward_bend")]
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            app_pipeline, "UPLOAD_DIR", Path(temp_dir) / "uploads"
        ), patch.object(app_pipeline, "analyze_image_files", return_value=results):
            output = app_pipeline.process_assessment(
                patient_name="背面前屈测试",
                patient_code="FLEX-1",
                lang="zh",
                images=[Upload("back.jpg", b"back"), Upload("bend.jpg", b"bend")],
                deepseek_key="",
            )

        assessment = output["assessment"]
        coverage = assessment["summary"]["view_coverage"]
        self.assertTrue(coverage["back"])
        self.assertTrue(coverage["forward_bend"])
        self.assertIn("照片 1（背面站立）", assessment["report_md"])
        self.assertIn("照片 2（前屈动作）", assessment["report_md"])
        self.assertIn("## 1. 骨盆与脊柱", assessment["report_md"])
        self.assertIn("## 2. 下肢与左右侧身体", assessment["report_md"])
        self.assertIn("## 3. 肩胛、右侧身体与头部", assessment["report_md"])
        self.assertIn("## 5. 综合模式与处理优先级", assessment["report_md"])
        self.assertIn("## 6. 建议补充的验证测试", assessment["report_md"])
        self.assertIn("## 7. 复评基线与追踪方法", assessment["report_md"])
        self.assertIn("【不可由照片判断】", assessment["report_md"])
        self.assertEqual(assessment["summary"]["acl_risk"]["level"], "not_assessed")


if __name__ == "__main__":
    unittest.main()
