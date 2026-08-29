from __future__ import annotations

import unittest

from posture_rag import RAG_DB, retrieve_assessment_context


@unittest.skipUnless(RAG_DB.exists(), "local RAG database is not available")
class RagTests(unittest.TestCase):
    def test_side_only_assessment_retrieves_side_cases(self):
        side = {
            "valid": True,
            "detected_view": "side",
            "metrics": {"view": "side", "trunk_lean_deg": 5.0, "hip_angle_deg": 170.0, "knee_angle_deg": 175.0},
            "issues": [],
        }
        context = retrieve_assessment_context(
            {},
            side,
            {"acl_risk": {"level": "not_assessed"}},
            image_results=[side],
        )
        self.assertTrue(context["similar_cases"])
        self.assertTrue(all(item["query_capture_type"] == "side" for item in context["similar_cases"]))
        self.assertTrue(all(item["capture_type"] == "side" for item in context["similar_cases"]))

    def test_back_and_forward_bend_each_retrieve_matching_cases(self):
        back = {
            "valid": True,
            "detected_view": "back",
            "metrics": {
                "view": "back", "shoulder_tilt_pct": 0.04, "hip_tilt_pct": 0.03,
                "trunk_shift_pct": 0.035, "knee_alignment_pct": 0.05,
            },
            "issues": [],
        }
        bend = {
            "valid": True,
            "detected_view": "forward_bend",
            "metrics": {
                "view": "forward_bend", "shoulder_tilt_pct": 0.03, "hip_tilt_pct": 0.02,
                "trunk_shift_pct": 0.02, "knee_flexion_asymmetry_deg": 10.0,
                "projected_torso_length_norm": 0.08, "projected_trunk_angle_deg": 20.0,
                "projected_hip_angle_deg": 95.0,
            },
            "issues": [],
        }
        context = retrieve_assessment_context(
            {}, {}, {"acl_risk": {"level": "not_assessed"}}, image_results=[back, bend]
        )
        groups = {item["capture_type"]: item["matches"] for item in context["similar_by_image"]}
        self.assertTrue(groups["back"])
        self.assertTrue(all(item["capture_type"] == "frontal_plane" for item in groups["back"]))
        self.assertTrue(groups["forward_bend"])
        self.assertTrue(all(item["capture_type"] == "forward_bend" for item in groups["forward_bend"]))
        self.assertNotIn("未提供有效正面照片", " ".join(context["summary_lines"]))


if __name__ == "__main__":
    unittest.main()
