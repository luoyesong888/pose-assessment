from __future__ import annotations

import unittest
from types import SimpleNamespace

from analysis import (
    classify_capture_view,
    measure_pose_metrics,
    summarize_case,
    validate_capture_geometry,
)


def make_landmarks(view: str = "front", confidence: float = 1.0):
    points = [
        SimpleNamespace(x=0.5, y=0.10, visibility=confidence, presence=confidence)
        for _ in range(33)
    ]
    if view == "front":
        values = {
            0: (0.50, 0.10), 11: (0.40, 0.25), 12: (0.60, 0.25),
            23: (0.44, 0.50), 24: (0.56, 0.50),
            25: (0.44, 0.70), 26: (0.56, 0.70),
            27: (0.44, 0.90), 28: (0.56, 0.90),
        }
    else:
        values = {
            0: (0.50, 0.10), 11: (0.49, 0.25), 12: (0.51, 0.25),
            23: (0.49, 0.50), 24: (0.51, 0.50),
            25: (0.49, 0.70), 26: (0.51, 0.70),
            27: (0.49, 0.90), 28: (0.51, 0.90),
        }
        for index in (4, 5, 6, 8, 10, 12, 24, 26, 28):
            points[index].visibility = 0.2
            points[index].presence = 0.2
    for index, (x, y) in values.items():
        points[index].x = x
        points[index].y = y
    return points


class AnalysisTests(unittest.TestCase):
    def test_zero_visibility_is_rejected(self):
        result = measure_pose_metrics(make_landmarks(confidence=0.0), 1000, 1000, "front")
        self.assertFalse(result["valid"])
        self.assertEqual(result["metrics"], {})
        self.assertEqual(result["acl_risk"]["level"], "not_assessed")

    def test_standard_front_geometry_passes(self):
        quality = validate_capture_geometry(make_landmarks("front"), "front")
        self.assertTrue(quality["valid"], quality)

    def test_back_or_occluded_face_is_rejected_for_front(self):
        landmarks = make_landmarks("front")
        for index in range(11):
            landmarks[index].visibility = 0.1
            landmarks[index].presence = 0.1
        quality = validate_capture_geometry(landmarks, "front")
        self.assertFalse(quality["valid"])
        self.assertTrue(any("面部" in item for item in quality["errors"]))

    def test_forward_fold_is_rejected_as_non_standing_capture(self):
        landmarks = make_landmarks("front")
        for index in (11, 12):
            landmarks[index].y = 0.54
        for index in (23, 24):
            landmarks[index].y = 0.50

        quality = validate_capture_geometry(landmarks, "front")

        self.assertFalse(quality["valid"])
        self.assertTrue(any("前屈" in item for item in quality["errors"]))

    def test_frontal_pose_is_rejected_for_side_slot(self):
        quality = validate_capture_geometry(make_landmarks("front"), "side")
        self.assertFalse(quality["valid"])
        self.assertTrue(any("侧面照片" in item for item in quality["errors"]))

    def test_side_angles_use_hip_and_knee_vertices(self):
        result = measure_pose_metrics(make_landmarks("side"), 1000, 1000, "side")
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["metrics"]["selected_side"], "left")
        self.assertEqual(result["metrics"]["hip_angle_deg"], 180.0)
        self.assertEqual(result["metrics"]["knee_angle_deg"], 180.0)
        self.assertNotIn("hip_hinge_left_deg", result["metrics"])

    def test_auto_view_distinguishes_front_and_back(self):
        front = classify_capture_view(make_landmarks("front"), frontal_face_count=1)
        self.assertTrue(front["valid"])
        self.assertEqual(front["detected_view"], "front")

        back = classify_capture_view(make_landmarks("front"), frontal_face_count=0)
        self.assertTrue(back["valid"])
        self.assertEqual(back["detected_view"], "back")

    def test_auto_view_accepts_forward_fold_and_measures_it(self):
        landmarks = make_landmarks("front")
        for index in (11, 12):
            landmarks[index].y = 0.54
        classification = classify_capture_view(landmarks, frontal_face_count=0)
        self.assertTrue(classification["valid"], classification)
        self.assertEqual(classification["detected_view"], "forward_bend")

        result = measure_pose_metrics(landmarks, 1000, 1000, "forward_bend")
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["metrics"]["view"], "forward_bend")
        self.assertTrue(any("前屈" in item for item in result["issues"]))

    def test_auto_view_accepts_clear_side_without_frontal_face(self):
        side = classify_capture_view(make_landmarks("side"), frontal_face_count=0)
        self.assertTrue(side["valid"], side)
        self.assertEqual(side["detected_view"], "side")

    def test_large_oblique_trunk_is_routed_to_other_not_front(self):
        landmarks = make_landmarks("front")
        for index in (11, 12):
            landmarks[index].x -= 0.16
        result = classify_capture_view(landmarks, frontal_face_count=1)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["detected_view"], "other")

    def test_static_summary_never_assigns_acl_grade(self):
        front = measure_pose_metrics(make_landmarks("front"), 1000, 1000, "front")
        side = measure_pose_metrics(make_landmarks("side"), 1000, 1000, "side")
        summary = summarize_case(front, side)
        self.assertTrue(summary["assessment_valid"])
        self.assertEqual(summary["acl_risk"]["level"], "not_assessed")
        self.assertIsNone(summary["acl_risk"]["score"])

    def test_single_view_summary_is_explicitly_partial(self):
        side = measure_pose_metrics(make_landmarks("side"), 1000, 1000, "side")
        summary = summarize_case({}, side)
        self.assertTrue(summary["assessment_valid"])
        self.assertFalse(summary["view_coverage"]["front"])
        self.assertTrue(summary["view_coverage"]["side"])
        self.assertIn("侧面站立", " ".join(summary["summary_lines"]))

    def test_back_and_forward_fold_are_both_in_summary(self):
        back = measure_pose_metrics(make_landmarks("front"), 1000, 1000, "back")
        back["detected_view"] = "back"
        bent_landmarks = make_landmarks("front")
        for index in (11, 12):
            bent_landmarks[index].y = 0.54
        bent = measure_pose_metrics(bent_landmarks, 1000, 1000, "forward_bend")
        bent["detected_view"] = "forward_bend"

        summary = summarize_case({}, {}, image_results=[back, bent])

        self.assertTrue(summary["assessment_valid"])
        self.assertTrue(summary["view_coverage"]["back"])
        self.assertTrue(summary["view_coverage"]["forward_bend"])
        self.assertIn("背面站立", summary["view_coverage"]["label_zh"])
        self.assertIn("前屈动作", summary["view_coverage"]["label_zh"])


if __name__ == "__main__":
    unittest.main()
