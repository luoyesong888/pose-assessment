from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


RAG_DB = Path(__file__).with_name("data") / "modelscope_posture" / "catalog.sqlite3"
EMBEDDING_DIM = 384

POSE_FEATURE_SCALES = {
    "shoulder_tilt": 0.03,
    "hip_tilt": 0.03,
    "trunk_shift": 0.03,
    "knee_alignment": 0.045,
    "knee_flexion_asymmetry": 10.0,
    "projected_torso_length": 0.20,
    "projected_trunk_angle": 12.0,
    "hip_angle": 18.0,
    "knee_angle": 15.0,
}

CAPTURE_TYPE_MAP = {
    "front": "frontal_plane",
    "back": "frontal_plane",
    "side": "side",
    "forward_bend": "forward_bend",
    "other": "other",
}


def _tokens(text: str) -> Iterable[str]:
    normalized = " ".join((text or "").lower().split())[:6000]
    for token in re.findall(r"[a-z0-9_.%+-]+", normalized):
        yield f"w:{token}"
    for sequence in re.findall(r"[\u3400-\u9fff]+", normalized):
        for size in (1, 2, 3):
            for index in range(max(0, len(sequence) - size + 1)):
                yield f"c{size}:{sequence[index:index + size]}"


def embed_text(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        index = value % dim
        sign = 1.0 if value & (1 << 63) else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    if norm:
        vector /= norm
    return vector


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(RAG_DB)
    conn.row_factory = sqlite3.Row
    return conn


@lru_cache(maxsize=2)
def _load_vector_index(db_mtime_ns: int) -> tuple[np.ndarray, List[str], List[str]]:
    del db_mtime_ns
    if not RAG_DB.exists():
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32), [], []
    with _connect() as conn:
        rows = conn.execute(
            "SELECT chunk_id, source_kind, embedding FROM rag_chunks WHERE embedding IS NOT NULL"
        ).fetchall()
    chunk_ids: List[str] = []
    source_kinds: List[str] = []
    vectors: List[np.ndarray] = []
    for row in rows:
        vector = np.frombuffer(row["embedding"], dtype=np.float32)
        if vector.size != EMBEDDING_DIM:
            continue
        chunk_ids.append(row["chunk_id"])
        source_kinds.append(row["source_kind"])
        vectors.append(vector)
    if not vectors:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32), [], []
    return np.vstack(vectors), chunk_ids, source_kinds


def search_text(query: str, top_k: int = 5, source_kinds: Iterable[str] | None = None) -> List[Dict[str, Any]]:
    if not RAG_DB.exists():
        return []
    matrix, chunk_ids, indexed_kinds = _load_vector_index(RAG_DB.stat().st_mtime_ns)
    if not chunk_ids:
        return []
    query_vector = embed_text(query)
    allowed = set(source_kinds or [])
    eligible = np.array(
        [index for index, kind in enumerate(indexed_kinds) if not allowed or kind in allowed],
        dtype=np.int64,
    )
    if eligible.size == 0:
        return []
    eligible_scores = matrix[eligible] @ query_vector
    candidate_count = min(len(eligible), max(top_k * 4, 20))
    if candidate_count == len(eligible):
        local_indexes = np.argsort(eligible_scores)[::-1]
    else:
        local_indexes = np.argpartition(eligible_scores, -candidate_count)[-candidate_count:]
        local_indexes = local_indexes[np.argsort(eligible_scores[local_indexes])[::-1]]
    indexes = eligible[local_indexes]
    scores = matrix @ query_vector

    selected: List[tuple[str, float]] = []
    with _connect() as conn:
        for index in indexes:
            chunk_id = chunk_ids[int(index)]
            selected.append((chunk_id, float(scores[int(index)])))
            if len(selected) >= top_k:
                break
        results: List[Dict[str, Any]] = []
        for chunk_id, score in selected:
            row = conn.execute(
                """
                SELECT chunk_id, source_kind, source_id, title, content, metadata_json
                FROM rag_chunks WHERE chunk_id=?
                """,
                (chunk_id,),
            ).fetchone()
            if not row:
                continue
            results.append(
                {
                    "chunk_id": row["chunk_id"],
                    "source_kind": row["source_kind"],
                    "source_id": row["source_id"],
                    "title": row["title"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                    "score": round(score, 4),
                }
            )
    return results


def _query_pose_features(metrics: Dict[str, Any]) -> Dict[str, float]:
    aliases = {
        "shoulder_tilt": ("shoulder_tilt_pct",),
        "hip_tilt": ("hip_tilt_pct",),
        "trunk_shift": ("trunk_shift_pct",),
        "knee_alignment": ("knee_alignment_pct",),
        "knee_flexion_asymmetry": ("knee_flexion_asymmetry_deg",),
        "projected_torso_length": ("projected_torso_length_norm",),
        "projected_trunk_angle": ("projected_trunk_angle_deg", "trunk_lean_deg"),
        "hip_angle": ("projected_hip_angle_deg", "hip_angle_deg"),
        "knee_angle": ("knee_angle_deg",),
    }
    result: Dict[str, float] = {}
    for feature, names in aliases.items():
        for name in names:
            if name in metrics and metrics.get(name) is not None:
                result[feature] = float(metrics[name])
                break
    if "knee_angle" not in result:
        angles = [
            float(metrics[name]) for name in ("left_knee_angle_deg", "right_knee_angle_deg")
            if metrics.get(name) is not None
        ]
        if angles:
            result["knee_angle"] = sum(angles) / len(angles)
    return result


def search_similar_pose_cases(
    metrics: Dict[str, Any],
    capture_type: str | None = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    if not RAG_DB.exists():
        return []
    query = _query_pose_features(metrics)
    if len(query) < 2:
        return []
    query_capture = CAPTURE_TYPE_MAP.get(capture_type or metrics.get("view", ""), capture_type or "other")
    columns = ", ".join(POSE_FEATURE_SCALES)
    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT f.chunk_id, f.capture_type, {columns},
                   f.activity, f.person_count,
                   c.source_id, c.title, c.content, c.metadata_json
            FROM rag_pose_features f
            JOIN rag_chunks c USING(chunk_id)
            WHERE f.feature_count >= 3
              AND f.person_count = 1
              AND f.local_file_exists = 1
            """
        ).fetchall()
    if not rows:
        return []
    ranked = []
    for row in rows:
        shared = [name for name in query if row[name] is not None]
        if len(shared) < 2:
            continue
        squared = [
            ((float(row[name]) - query[name]) / POSE_FEATURE_SCALES[name]) ** 2
            for name in shared
        ]
        distance = float(np.sqrt(sum(squared) / len(squared)))
        candidate_capture = row["capture_type"] or "other"
        type_penalty = 0.0 if candidate_capture == query_capture else (0.65 if candidate_capture == "other" else 1.5)
        ranked.append((distance + type_penalty, distance, type_penalty, shared, row))
    ranked.sort(key=lambda item: item[0])
    results = []
    for total_distance, geometry_distance, type_penalty, shared, row in ranked[:top_k]:
        results.append(
            {
                "chunk_id": row["chunk_id"],
                "source_kind": "mpii_pose_case",
                "source_id": row["source_id"],
                "title": row["title"],
                "content": row["content"],
                "metadata": json.loads(row["metadata_json"] or "{}"),
                "capture_type": row["capture_type"] or "other",
                "query_capture_type": query_capture,
                "matched_features": shared,
                "distance": round(total_distance, 4),
                "geometry_distance": round(geometry_distance, 4),
                "type_penalty": round(type_penalty, 4),
                "activity": row["activity"] or "",
                "person_count": row["person_count"] or 0,
            }
        )
    return results


def required_guardrails() -> List[Dict[str, Any]]:
    if not RAG_DB.exists():
        return []
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT chunk_id, source_kind, source_id, title, content, metadata_json
            FROM rag_chunks
            WHERE source_kind='clinical_guardrail'
            ORDER BY chunk_id
            """
        ).fetchall()
    return [
        {
            "chunk_id": row["chunk_id"],
            "source_kind": row["source_kind"],
            "source_id": row["source_id"],
            "title": row["title"],
            "content": row["content"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "score": 1.0,
        }
        for row in rows
    ]


def _case_feature_range(cases: List[Dict[str, Any]], name: str) -> str:
    values = [
        item.get("metadata", {}).get("features", {}).get(name)
        for item in cases
    ]
    parsed = [float(value) for value in values if value is not None]
    if not parsed:
        return "缺失"
    return f"{min(parsed):.4f}-{max(parsed):.4f}"


def retrieve_assessment_context(
    front_result: Dict[str, Any],
    side_result: Dict[str, Any],
    summary: Dict[str, Any],
    top_k: int = 6,
    image_results: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    candidates = image_results or [front_result or {}, side_result or {}]
    usable = [item for item in candidates if item and item.get("valid", True) and item.get("metrics")]
    image_queries = []
    for index, item in enumerate(usable):
        capture_type = item.get("detected_view") or item.get("metrics", {}).get("view") or item.get("view") or "other"
        image_queries.append(
            {
                "image_index": index,
                "capture_type": capture_type,
                "issues": item.get("issues", []),
                "metrics": item.get("metrics", {}),
            }
        )
    query_payload = {
        "images": image_queries,
        "acl_level": (summary or {}).get("acl_risk", {}).get("level", ""),
        "intent": "多视角体态筛查、背面、侧面、前屈、其他动作、关键点几何、动作控制与证据边界",
    }
    query_text = json.dumps(query_payload, ensure_ascii=False, sort_keys=True)
    knowledge = required_guardrails()
    knowledge.extend(search_text(
        query_text,
        top_k=4,
        source_kinds={
            "dataset_card",
            "sensor_summary",
            "exercise_action_taxonomy",
            "fitness_topic_summary",
        },
    ))
    similar_cases = []
    similar_by_image = []
    seen = set()
    per_image_k = max(1, min(3, top_k))
    for query in image_queries:
        matches = search_similar_pose_cases(
            query["metrics"],
            capture_type=query["capture_type"],
            top_k=per_image_k,
        )
        similar_by_image.append(
            {
                "image_index": query["image_index"],
                "capture_type": query["capture_type"],
                "matches": matches,
            }
        )
        for match in matches:
            if match["chunk_id"] in seen:
                continue
            seen.add(match["chunk_id"])
            similar_cases.append({**match, "query_image_index": query["image_index"]})
    similar_cases = similar_cases[: max(top_k, per_image_k)]
    type_counts: Dict[str, int] = {}
    for group in similar_by_image:
        key = group["capture_type"]
        type_counts[key] = type_counts.get(key, 0) + len(group["matches"])
    candidate_match_count = sum(type_counts.values())
    return {
        "query": query_payload,
        "knowledge": knowledge,
        "similar_cases": similar_cases,
        "similar_by_image": similar_by_image,
        "summary_lines": [
            f"本地 RAG 已对 {len(image_queries)} 张有效照片分别检索，逐图命中 {candidate_match_count} 个候选，去重后选取 {len(similar_cases)} 个 MPII 样本用于报告。",
            "逐图分类命中：" + ("，".join(f"{name} {count} 个" for name, count in type_counts.items()) or "无足够共享特征的样本") + "。",
            "检索会优先匹配相同投影类型；正面与背面共用 frontal_plane 几何库，侧面、前屈和其他姿态分别建立索引。",
            "相似案例只用于关键点模式参照；活动标签不参与病因、功能或损伤风险判断。",
            "非标准角度只使用当前可见关键点，不对被遮挡结构作结论。",
        ],
    }


def rag_prompt_context(context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "usage_rule": (
            "RAG 内容仅作为姿态关键点与动作模式参考。不得把相似案例当成诊断，"
            "不得从静态图推断视力、听力、疼痛、韧带损伤或自主神经状态。"
        ),
        "knowledge": [
            {
                "title": item.get("title"),
                "content": item.get("content"),
                "source_id": item.get("source_id"),
                "score": item.get("score"),
            }
            for item in context.get("knowledge", [])
        ],
        "similar_pose_cases": [
            {
                "source_id": item.get("source_id"),
                "distance": item.get("distance"),
                "query_image_index": item.get("query_image_index"),
                "query_capture_type": item.get("query_capture_type"),
                "matched_capture_type": item.get("capture_type"),
                "matched_features": item.get("matched_features", []),
                "normalized_features": item.get("metadata", {}).get("features", {}),
                "usage": "仅用于关键点几何相似参照，不用于诊断或活动推断",
            }
            for item in context.get("similar_cases", [])
        ],
    }
