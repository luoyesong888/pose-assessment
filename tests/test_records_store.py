from __future__ import annotations

import unittest

from records_store import upsert_assessment


class RecordsStoreTests(unittest.TestCase):
    def test_extended_profile_fields_are_persisted_and_updated(self):
        profile = {
            "patient_key": "p1",
            "patient_name": "测试",
            "patient_code": "001",
            "created_at": "2026-01-01",
            "gender": "女",
            "age": 30,
            "height": 165,
            "weight": 55,
            "occupation": "久坐办公",
            "activity": "每周 1-2 次",
            "concerns": ["高低肩"],
            "pain_areas": "无",
            "injury_history": "无",
        }
        store = upsert_assessment({"patients": []}, profile, {"created_at": "2026-01-02"})
        patient = store["patients"][0]
        self.assertEqual(patient["age"], 30)
        self.assertEqual(patient["concerns"], ["高低肩"])

        updated = {**profile, "age": 31, "occupation": "站立工作"}
        store = upsert_assessment(store, updated, {"created_at": "2026-02-01"})
        patient = store["patients"][0]
        self.assertEqual(patient["age"], 31)
        self.assertEqual(patient["occupation"], "站立工作")
        self.assertEqual(len(patient["assessments"]), 2)


if __name__ == "__main__":
    unittest.main()
