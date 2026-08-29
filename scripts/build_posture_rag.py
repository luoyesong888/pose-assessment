#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from posture_rag import EMBEDDING_DIM, RAG_DB, embed_text  # noqa: E402


WAREHOUSE_DIR = RAG_DB.parent
DATASETS_DIR = WAREHOUSE_DIR / "datasets"
MPII_ID = "Voxel51/MPII_Human_Pose_Dataset"
MPII_DIR = DATASETS_DIR / "Voxel51" / "MPII_Human_Pose_Dataset"
SENSOR_ID = "ennmmmmm/Action-Recognition-Dataset-Based-on-Smartphones"
SENSOR_DIR = DATASETS_DIR / "ennmmmmm" / "Action-Recognition-Dataset-Based-on-Smartphones"
ACTION_LIBRARY_ID = "iic/3DHuman_action_dataset"
ACTION_LIBRARY_DIR = DATASETS_DIR / "iic" / "3DHuman_action_dataset"
FITNESS_COMMUNITY_ID = "mlfoundations-dev/stackexchange_fitness"
FITNESS_COMMUNITY_DIR = DATASETS_DIR / "mlfoundations-dev" / "stackexchange_fitness"

EXERCISE_ACTION_GROUPS = {
    "squat": ["AirSquat", "BackSquat", "OverheadSquat"],
    "core": ["Plank", "BicycleCrunch", "Situps"],
    "conditioning": ["Burpee", "JumpingJacks", "JumpingRope", "StandardWalk"],
    "mobility": ["ArmStretching", "WalkingUpTheStairs"],
}

FITNESS_TOPIC_PATTERNS = {
    "shoulder_posture": ("shoulder", "scapula", "upper back", "posture"),
    "trunk_control": ("core", "plank", "bird dog", "dead bug", "lower back"),
    "hip_knee_control": ("squat", "lunge", "hip", "knee", "glute"),
    "ankle_balance": ("ankle", "calf", "balance", "single leg"),
    "mobility": ("mobility", "stretch", "flexibility", "range of motion"),
}


GUARDRAILS = [
    {
        "id": "static-photo-boundary",
        "title": "静态体态照片的证据边界",
        "content": (
            "正面或侧面静态照片可以描述可见关键点、肩线、骨盆线、躯干相对位移和膝踝对线。"
            "它不能单独确认疼痛来源、肌力不足、关节不稳定、韧带损伤、视力、听力、咀嚼功能或自主神经状态。"
            "报告应使用观察到、提示、可能相关、需要动态测试验证等措辞。"
        ),
    },
    {
        "id": "acl-boundary",
        "title": "ACL 风险筛查边界",
        "content": (
            "ACL 风险不能由一张静态站姿照片直接预测。静态膝踝对线只能作为代理观察，"
            "需要结合单腿下蹲、落地、变向、既往伤史、关节松弛度和力量测试。"
        ),
    },
    {
        "id": "view-validation",
        "title": "照片视角与动作验证",
        "content": (
            "系统可分析正面、背面、侧面、前屈和其他姿态，但必须先识别拍摄类型，"
            "再与相同投影或动作类型的样本比较。每张照片只能描述当前可见关键点，"
            "背面不推断前侧结构，前屈单帧不把画面投影角当成真实髋屈角。"
        ),
    },
    {
        "id": "rag-case-boundary",
        "title": "相似姿态案例的使用规则",
        "content": (
            "MPII 相似案例来自日常活动和体育动作，适合验证关键点结构与动作多样性。"
            "相似度不代表相同病因、症状或受伤风险，不能把案例活动标签转化为患者临床结论。"
        ),
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(RAG_DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS rag_chunks (
            chunk_id TEXT PRIMARY KEY,
            source_kind TEXT NOT NULL,
            source_id TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            embedding BLOB NOT NULL,
            embedding_dim INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS rag_pose_features (
            chunk_id TEXT PRIMARY KEY,
            capture_type TEXT,
            shoulder_tilt REAL,
            hip_tilt REAL,
            trunk_shift REAL,
            knee_alignment REAL,
            knee_flexion_asymmetry REAL,
            projected_torso_length REAL,
            projected_trunk_angle REAL,
            hip_angle REAL,
            knee_angle REAL,
            feature_count INTEGER NOT NULL,
            person_count INTEGER NOT NULL,
            local_file_exists INTEGER NOT NULL DEFAULT 0,
            activity TEXT,
            FOREIGN KEY (chunk_id) REFERENCES rag_chunks(chunk_id)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts USING fts5(
            chunk_id UNINDEXED,
            title,
            content,
            tokenize='unicode61'
        );

        CREATE TABLE IF NOT EXISTS rag_builds (
            build_id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            pose_case_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            error TEXT
        );
        """
    )
    feature_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(rag_pose_features)").fetchall()
    }
    if "local_file_exists" not in feature_columns:
        conn.execute(
            "ALTER TABLE rag_pose_features ADD COLUMN local_file_exists INTEGER NOT NULL DEFAULT 0"
        )
    extra_columns = {
        "capture_type": "TEXT",
        "knee_flexion_asymmetry": "REAL",
        "projected_torso_length": "REAL",
        "projected_trunk_angle": "REAL",
        "hip_angle": "REAL",
        "knee_angle": "REAL",
    }
    for name, sql_type in extra_columns.items():
        if name not in feature_columns:
            conn.execute(f"ALTER TABLE rag_pose_features ADD COLUMN {name} {sql_type}")
    return conn


def insert_chunk(
    conn: sqlite3.Connection,
    *,
    chunk_id: str,
    source_kind: str,
    source_id: str,
    title: str,
    content: str,
    metadata: Dict[str, Any],
) -> None:
    embedding = embed_text(f"{title}\n{content}")
    conn.execute(
        """
        INSERT INTO rag_chunks(
            chunk_id, source_kind, source_id, title, content, metadata_json,
            embedding, embedding_dim, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chunk_id,
            source_kind,
            source_id,
            title,
            content,
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
            embedding.tobytes(),
            EMBEDDING_DIM,
            utc_now(),
        ),
    )
    conn.execute(
        "INSERT INTO rag_chunks_fts(chunk_id, title, content) VALUES (?, ?, ?)",
        (chunk_id, title, content),
    )


def distill_guardrails(conn: sqlite3.Connection) -> None:
    for item in GUARDRAILS:
        insert_chunk(
            conn,
            chunk_id=f"guardrail:{item['id']}",
            source_kind="clinical_guardrail",
            source_id="local-safety-policy",
            title=item["title"],
            content=item["content"],
            metadata={"priority": "required", "language": "zh"},
        )


def distill_dataset_cards(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT dataset_id, display_name, description, license, tasks_json,
               tags_json, relevance_score, source_url, download_status,
               local_file_count, local_size_bytes
        FROM datasets WHERE relevance_score > 0
        """
    ).fetchall()
    for row in rows:
        tasks = json.loads(row["tasks_json"] or "[]")
        content = (
            f"数据集：{row['display_name'] or row['dataset_id']}。"
            f"任务：{', '.join(tasks) if tasks else '未标注'}。"
            f"描述：{row['description'] or '未提供'}。"
            f"许可证：{row['license'] or '未知'}。"
            f"本地状态：{row['download_status']}，有效文件 {row['local_file_count']} 个。"
            "使用时必须遵循原始许可证；数据集规模和标签不等同于临床证据。"
        )
        insert_chunk(
            conn,
            chunk_id=f"dataset:{row['dataset_id']}",
            source_kind="dataset_card",
            source_id=row["dataset_id"],
            title=row["display_name"] or row["dataset_id"],
            content=content,
            metadata={
                "license": row["license"],
                "tasks": tasks,
                "source_url": row["source_url"],
                "download_status": row["download_status"],
                "local_file_count": row["local_file_count"],
                "local_size_bytes": row["local_size_bytes"],
            },
        )


def joint_map(person: Dict[str, Any]) -> Dict[str, tuple[float, float]]:
    names = person.get("joints_id") or []
    points = person.get("points") or []
    return {
        str(name): (float(point[0]), float(point[1]))
        for name, point in zip(names, points)
        if isinstance(point, list) and len(point) >= 2
    }


def point_angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float | None:
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    denom = math.hypot(*ba) * math.hypot(*bc)
    if not denom:
        return None
    cosine = max(-1.0, min(1.0, (ba[0] * bc[0] + ba[1] * bc[1]) / denom))
    return math.degrees(math.acos(cosine))


def pose_features(joints: Dict[str, tuple[float, float]]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    if "l_shoulder" in joints and "r_shoulder" in joints:
        values["shoulder_tilt"] = abs(joints["l_shoulder"][1] - joints["r_shoulder"][1])
    if "l_hip" in joints and "r_hip" in joints:
        values["hip_tilt"] = abs(joints["l_hip"][1] - joints["r_hip"][1])
    if all(name in joints for name in ("l_shoulder", "r_shoulder", "l_hip", "r_hip")):
        shoulder_mid = (joints["l_shoulder"][0] + joints["r_shoulder"][0]) / 2
        hip_mid = (joints["l_hip"][0] + joints["r_hip"][0]) / 2
        values["trunk_shift"] = abs(shoulder_mid - hip_mid)
    alignments = []
    for side in ("l", "r"):
        knee = joints.get(f"{side}_knee")
        ankle = joints.get(f"{side}_ankle")
        if knee and ankle:
            alignments.append(abs(knee[0] - ankle[0]))
    if alignments:
        values["knee_alignment"] = max(alignments)

    torso_names = ("l_shoulder", "r_shoulder", "l_hip", "r_hip")
    if all(name in joints for name in torso_names):
        shoulder_mid = (
            (joints["l_shoulder"][0] + joints["r_shoulder"][0]) / 2,
            (joints["l_shoulder"][1] + joints["r_shoulder"][1]) / 2,
        )
        hip_mid = (
            (joints["l_hip"][0] + joints["r_hip"][0]) / 2,
            (joints["l_hip"][1] + joints["r_hip"][1]) / 2,
        )
        torso_dx = shoulder_mid[0] - hip_mid[0]
        torso_dy = hip_mid[1] - shoulder_mid[1]
        torso_length = math.hypot(torso_dx, torso_dy)
        values["projected_torso_length"] = torso_length
        values["projected_trunk_angle"] = math.degrees(
            math.atan2(abs(torso_dx), max(abs(torso_dy), 1e-6))
        )
        shoulder_span = abs(joints["l_shoulder"][0] - joints["r_shoulder"][0])
        ratio = shoulder_span / max(torso_length, 1e-6)
        if all(name in joints for name in ("l_knee", "r_knee", "l_ankle", "r_ankle")):
            knee_mid_y = (joints["l_knee"][1] + joints["r_knee"][1]) / 2
            ankle_mid_y = (joints["l_ankle"][1] + joints["r_ankle"][1]) / 2
            legs_below_hips = hip_mid[1] < knee_mid_y < ankle_mid_y
            upright_order = shoulder_mid[1] < hip_mid[1] < knee_mid_y < ankle_mid_y
        else:
            legs_below_hips = False
            upright_order = torso_dy > 0.10
        if legs_below_hips and (torso_dy < 0.10 or values["projected_trunk_angle"] >= 55):
            values["capture_type"] = "forward_bend"
        elif upright_order and ratio <= 0.34:
            values["capture_type"] = "side"
        elif upright_order:
            values["capture_type"] = "frontal_plane"
        else:
            values["capture_type"] = "other"

    knee_angles = []
    hip_angles = []
    for side in ("l", "r"):
        hip = joints.get(f"{side}_hip")
        knee = joints.get(f"{side}_knee")
        ankle = joints.get(f"{side}_ankle")
        shoulder = joints.get(f"{side}_shoulder")
        if hip and knee and ankle:
            angle = point_angle(hip, knee, ankle)
            if angle is not None:
                knee_angles.append(angle)
        if shoulder and hip and knee:
            angle = point_angle(shoulder, hip, knee)
            if angle is not None:
                hip_angles.append(angle)
    if knee_angles:
        values["knee_angle"] = sum(knee_angles) / len(knee_angles)
        if len(knee_angles) == 2:
            values["knee_flexion_asymmetry"] = abs(knee_angles[0] - knee_angles[1])
    if hip_angles:
        values["hip_angle"] = sum(hip_angles) / len(hip_angles)
    return values


def average_feature(person_features: List[Dict[str, Any]], name: str) -> float | None:
    values = [item[name] for item in person_features if name in item]
    return sum(values) / len(values) if values else None


def describe_feature(name: str, value: float | None) -> str:
    labels = {
        "shoulder_tilt": "肩线高度差",
        "hip_tilt": "骨盆线高度差",
        "trunk_shift": "肩髋中线相对位移",
        "knee_alignment": "膝踝水平对线差",
        "knee_flexion_asymmetry": "左右膝屈角差",
        "projected_torso_length": "躯干投影长度",
        "projected_trunk_angle": "躯干画面投影角",
        "hip_angle": "髋角投影均值",
        "knee_angle": "膝角投影均值",
    }
    return f"{labels[name]} {value:.4f}" if value is not None else f"{labels[name]}缺失"


def distill_mpii_cases(conn: sqlite3.Connection) -> int:
    manifest = MPII_DIR / "samples.json"
    if not manifest.exists():
        return 0
    samples = json.loads(manifest.read_text(encoding="utf-8")).get("samples", [])
    count = 0
    for index, sample in enumerate(samples):
        persons = ((sample.get("annopoints") or {}).get("keypoints") or [])
        if not persons:
            continue
        labels = [
            item.get("label")
            for item in ((sample.get("activity") or {}).get("classifications") or [])
            if item.get("label")
        ]
        person_features = [pose_features(joint_map(person)) for person in persons]
        numeric_names = (
            "shoulder_tilt", "hip_tilt", "trunk_shift", "knee_alignment",
            "knee_flexion_asymmetry", "projected_torso_length",
            "projected_trunk_angle", "hip_angle", "knee_angle",
        )
        averaged = {
            name: average_feature(person_features, name)
            for name in numeric_names
        }
        capture_type = (
            str(person_features[0].get("capture_type") or "other")
            if len(person_features) == 1 else "multi_person"
        )
        feature_count = sum(value is not None for value in averaged.values())
        if feature_count < 2:
            continue
        sample_id = ((sample.get("_id") or {}).get("$oid") or str(index))
        filepath = str(sample.get("filepath") or "")
        local_file = MPII_DIR / filepath
        content = (
            f"MPII 人体姿态案例，活动标签：{', '.join(labels) if labels else '未提供'}；"
            f"画面标注人数：{len(persons)}；投影类型：{capture_type}；"
            + "；".join(describe_feature(name, value) for name, value in averaged.items())
            + "。这些是归一化关键点几何量，只能用于姿态结构相似检索，不能推断症状或损伤。"
        )
        chunk_id = f"mpii:{sample_id}"
        metadata = {
            "filepath": filepath,
            "local_file_exists": local_file.is_file() and local_file.stat().st_size > 0,
            "activities": labels,
            "person_count": len(persons),
            "features": averaged,
            "capture_type": capture_type,
            "split": sample.get("tags") or [],
            "video_id": sample.get("video_id"),
            "frame_sec": sample.get("frame_sec"),
            "license": "bsd-2-clause; images restricted to research use by dataset card",
        }
        insert_chunk(
            conn,
            chunk_id=chunk_id,
            source_kind="mpii_pose_case",
            source_id=f"{MPII_ID}:{sample_id}",
            title=f"MPII 姿态案例：{', '.join(labels) if labels else sample_id}",
            content=content,
            metadata=metadata,
        )
        conn.execute(
            """
            INSERT INTO rag_pose_features(
                chunk_id, capture_type, shoulder_tilt, hip_tilt, trunk_shift, knee_alignment,
                knee_flexion_asymmetry, projected_torso_length, projected_trunk_angle,
                hip_angle, knee_angle, feature_count, person_count, local_file_exists, activity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                capture_type,
                averaged["shoulder_tilt"],
                averaged["hip_tilt"],
                averaged["trunk_shift"],
                averaged["knee_alignment"],
                averaged["knee_flexion_asymmetry"],
                averaged["projected_torso_length"],
                averaged["projected_trunk_angle"],
                averaged["hip_angle"],
                averaged["knee_angle"],
                feature_count,
                len(persons),
                int(metadata["local_file_exists"]),
                ", ".join(labels),
            ),
        )
        count += 1
        if count % 1000 == 0:
            print(f"Distilled MPII cases: {count}", flush=True)
    return count


def numeric_summary(csv_path: Path) -> Dict[str, Any]:
    count = 0
    labels: Counter[str] = Counter()
    totals: Dict[str, float] = {}
    totals_sq: Dict[str, float] = {}
    numeric_counts: Dict[str, int] = {}
    with csv_path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            count += 1
            label = str(row.get("labels") or "")
            labels[label] += 1
            for name in ("ACCx(m/s^2)", "ACCy(m/s^2)", "ACCz(m/s^2)", "GYROx/(rad/s)", "GYROy/(rad/s)", "GYROz/(rad/s)"):
                try:
                    value = float(row.get(name) or "")
                except ValueError:
                    continue
                totals[name] = totals.get(name, 0.0) + value
                totals_sq[name] = totals_sq.get(name, 0.0) + value * value
                numeric_counts[name] = numeric_counts.get(name, 0) + 1
    stats = {}
    for name, total in totals.items():
        n = numeric_counts[name]
        mean = total / n
        variance = max(0.0, totals_sq[name] / n - mean * mean)
        stats[name] = {"mean": round(mean, 5), "std": round(math.sqrt(variance), 5)}
    return {"rows": count, "labels": dict(labels), "stats": stats}


def distill_sensor_files(conn: sqlite3.Connection) -> int:
    label_dir = SENSOR_DIR / "label_data"
    if not label_dir.exists():
        return 0
    count = 0
    for path in sorted(label_dir.glob("*.csv"), key=lambda item: int(item.stem)):
        summary = numeric_summary(path)
        content = (
            f"智能手机动作传感器文件 {path.name}，共 {summary['rows']} 行。"
            f"动作标签分布：{json.dumps(summary['labels'], ensure_ascii=False)}。"
            f"加速度与角速度统计：{json.dumps(summary['stats'], ensure_ascii=False)}。"
            "传感器数据可用于动作识别与动态变化参照，不能替代图像体态或临床检查。"
        )
        insert_chunk(
            conn,
            chunk_id=f"sensor:{path.stem}",
            source_kind="sensor_summary",
            source_id=f"{SENSOR_ID}:{path.name}",
            title=f"动作传感器摘要 {path.stem}",
            content=content,
            metadata={"relative_path": path.relative_to(SENSOR_DIR).as_posix(), **summary},
        )
        count += 1
    return count


def distill_exercise_action_taxonomy(conn: sqlite3.Connection) -> int:
    action_file = ACTION_LIBRARY_DIR / "action_ids.txt"
    if not action_file.exists():
        return 0
    available = {line.strip() for line in action_file.read_text(encoding="utf-8").splitlines() if line.strip()}
    count = 0
    for group, expected in EXERCISE_ACTION_GROUPS.items():
        matched = [name for name in expected if name in available]
        if not matched:
            continue
        content = (
            f"三维动作库中的 {group} 动作词汇：{', '.join(matched)}。"
            "该数据只证明动作名称与运动场景存在，不证明康复适应证、剂量或临床有效性。"
        )
        insert_chunk(
            conn,
            chunk_id=f"action-taxonomy:{group}",
            source_kind="exercise_action_taxonomy",
            source_id=f"{ACTION_LIBRARY_ID}:{group}",
            title=f"运动动作词汇：{group}",
            content=content,
            metadata={
                "group": group,
                "actions": matched,
                "license": "Apache License 2.0",
                "usage": "vocabulary_only",
            },
        )
        count += 1
    return count


def distill_fitness_community_topics(conn: sqlite3.Connection) -> int:
    parquet_path = FITNESS_COMMUNITY_DIR / "data" / "train-00000-of-00001.parquet"
    if not parquet_path.exists():
        return 0
    import pyarrow.parquet as pq

    counts = Counter()
    examples: Dict[str, List[str]] = {topic: [] for topic in FITNESS_TOPIC_PATTERNS}
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(columns=["instruction"], batch_size=2048):
        for instruction in batch.column(0).to_pylist():
            question = " ".join(str(instruction or "").split())
            lowered = question.lower()
            for topic, patterns in FITNESS_TOPIC_PATTERNS.items():
                if not any(pattern in lowered for pattern in patterns):
                    continue
                counts[topic] += 1
                if len(examples[topic]) < 12:
                    examples[topic].append(question[:240])
    count = 0
    for topic, total in counts.items():
        content = (
            f"Fitness 社区中与 {topic} 相关的问题共 {total} 条。"
            f"常见问题示例：{' | '.join(examples[topic][:5])}。"
            "这些是未经临床审核的社区问题，只用于识别用户语言、训练目标和动作同义词；"
            "不得使用社区回答生成诊断、禁忌证或训练剂量。"
        )
        insert_chunk(
            conn,
            chunk_id=f"fitness-topic:{topic}",
            source_kind="fitness_topic_summary",
            source_id=f"{FITNESS_COMMUNITY_ID}:{topic}",
            title=f"Fitness 常见训练主题：{topic}",
            content=content,
            metadata={
                "topic": topic,
                "question_count": total,
                "examples": examples[topic],
                "usage": "terminology_and_intent_only",
                "answers_used": False,
            },
        )
        count += 1
    return count


def main() -> int:
    conn = connect()
    started_at = utc_now()
    build_id = conn.execute(
        "INSERT INTO rag_builds(started_at, status) VALUES (?, 'running')",
        (started_at,),
    ).lastrowid
    conn.commit()
    try:
        conn.execute("DELETE FROM rag_pose_features")
        conn.execute("DELETE FROM rag_chunks")
        conn.execute("DELETE FROM rag_chunks_fts")
        distill_guardrails(conn)
        distill_dataset_cards(conn)
        pose_case_count = distill_mpii_cases(conn)
        sensor_count = distill_sensor_files(conn)
        action_taxonomy_count = distill_exercise_action_taxonomy(conn)
        fitness_topic_count = distill_fitness_community_topics(conn)
        chunk_count = conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
        conn.execute(
            """
            UPDATE rag_builds SET finished_at=?, chunk_count=?, pose_case_count=?, status='complete'
            WHERE build_id=?
            """,
            (utc_now(), chunk_count, pose_case_count, build_id),
        )
        conn.commit()
        print(
            json.dumps(
                {
                    "database": str(RAG_DB),
                    "chunks": chunk_count,
                    "pose_cases": pose_case_count,
                    "sensor_summaries": sensor_count,
                    "exercise_action_taxonomies": action_taxonomy_count,
                    "fitness_topic_summaries": fitness_topic_count,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        conn.execute(
            "UPDATE rag_builds SET finished_at=?, status='failed', error=? WHERE build_id=?",
            (utc_now(), f"{type(exc).__name__}: {exc}", build_id),
        )
        conn.commit()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
