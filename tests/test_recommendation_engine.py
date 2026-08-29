from __future__ import annotations

import unittest

from recommendation_engine import (
    apply_confirmed_plan,
    build_recommendation_options,
)


class RecommendationEngineTests(unittest.TestCase):
    def test_knee_alignment_generates_explainable_low_load_option(self):
        options = build_recommendation_options(
            {},
            {"metrics": {"knee_alignment_pct": 0.06}},
            {"metrics": {}},
            {"view_coverage": {"front": True, "side": True, "label_zh": "正面 + 侧面"}},
            {"knowledge": [{"source_id": "test-source"}]},
        )
        option = next(item for item in options if item["id"] == "lower_limb_control")
        self.assertTrue(option["evidence"])
        self.assertTrue(option["validation"])
        self.assertTrue(option["exercises"])
        self.assertIn("test-source", option["rag_sources"])

    def test_pain_history_prioritizes_professional_review(self):
        options = build_recommendation_options(
            {"pain_areas": "膝痛"},
            {"metrics": {}},
            {"metrics": {}},
            {"view_coverage": {"front": True, "side": True, "label_zh": "正面 + 侧面"}},
            {"knowledge": []},
        )
        self.assertEqual(options[0]["id"], "professional_review")
        self.assertEqual(options[0]["exercises"], [])

    def test_confirmed_plan_updates_report_without_duplicate_sections(self):
        assessment = {
            "report_md": "# Report\n",
            "report_sections": {},
            "recommendation_options": [
                {
                    "id": "maintenance",
                    "title": "维持活动",
                    "validation": "复拍",
                    "exercises": [{"name": "步行", "dosage": "20 分钟", "cue": "轻松"}],
                    "precautions": ["疼痛时停止"],
                }
            ],
        }
        updated = apply_confirmed_plan(assessment, ["maintenance"])
        self.assertIn("已确认改善计划", updated["report_md"])
        self.assertEqual(updated["confirmed_plan"]["selected_ids"], ["maintenance"])
        updated = apply_confirmed_plan(updated, ["maintenance"])
        self.assertEqual(updated["report_md"].count("## 已确认改善计划"), 1)


if __name__ == "__main__":
    unittest.main()
