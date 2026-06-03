from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).with_name("data")
STORE_FILE = DATA_DIR / "patient_records.json"


def ensure_store_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_store() -> Dict[str, Any]:
    ensure_store_dir()
    if not STORE_FILE.exists():
        return {"patients": []}
    try:
        return json.loads(STORE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"patients": []}


def save_store(store: Dict[str, Any]) -> None:
    ensure_store_dir()
    STORE_FILE.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def _find_patient_index(store: Dict[str, Any], patient_key: str) -> int:
    for idx, patient in enumerate(store.get("patients", [])):
        if patient.get("patient_key") == patient_key:
            return idx
    return -1


def upsert_assessment(store: Dict[str, Any], patient_profile: Dict[str, Any], assessment: Dict[str, Any]) -> Dict[str, Any]:
    patient_key = patient_profile["patient_key"]
    patient_idx = _find_patient_index(store, patient_key)
    patient_entry = {
        "patient_key": patient_key,
        "patient_name": patient_profile.get("patient_name", ""),
        "patient_code": patient_profile.get("patient_code", ""),
        "created_at": patient_profile.get("created_at"),
        "updated_at": assessment.get("created_at"),
        "assessments": [],
    }

    if patient_idx >= 0:
        patient_entry = store["patients"][patient_idx]
        patient_entry["patient_name"] = patient_profile.get("patient_name", patient_entry.get("patient_name", ""))
        patient_entry["patient_code"] = patient_profile.get("patient_code", patient_entry.get("patient_code", ""))
        patient_entry["updated_at"] = assessment.get("created_at")
    else:
        store.setdefault("patients", []).append(patient_entry)

    patient_entry.setdefault("assessments", []).insert(0, assessment)
    return store


def search_patients(store: Dict[str, Any], query: str = "") -> List[Dict[str, Any]]:
    patients = store.get("patients", [])
    if not query:
        return patients

    q = query.lower().strip()
    result: List[Dict[str, Any]] = []
    for patient in patients:
        haystack = " ".join(
            [
                str(patient.get("patient_name", "")),
                str(patient.get("patient_code", "")),
                str(patient.get("patient_key", "")),
            ]
        ).lower()
        if q in haystack:
            result.append(patient)
    return result


def total_assessments(store: Dict[str, Any]) -> int:
    return sum(len(patient.get("assessments", [])) for patient in store.get("patients", []))
