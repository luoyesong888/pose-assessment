#!/usr/bin/env python3
"""同步魔搭社区的人体姿态数据集目录，并为已下载文件建立 SQLite 索引。"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import dataset_snapshot_download


PROJECT_DIR = Path(__file__).resolve().parents[1]
WAREHOUSE_DIR = PROJECT_DIR / "data" / "modelscope_posture"
DATABASE_PATH = WAREHOUSE_DIR / "catalog.sqlite3"
DATASETS_DIR = WAREHOUSE_DIR / "datasets"

SEARCH_TERMS = [
    "体态",
    "人体姿态",
    "姿态估计",
    "人体关键点",
    "人体行为",
    "动作识别",
    "骨骼动作",
    "步态识别",
    "跌倒检测",
    "健身动作",
    "瑜伽姿态",
    "human pose",
    "pose estimation",
    "human keypoints",
    "body keypoints",
    "skeleton action",
    "action recognition",
    "gait recognition",
    "fall detection",
    "fitness exercise",
    "运动康复",
    "康复训练",
    "健身",
    "训练计划",
    "physical therapy",
    "fitness",
    "workout plan",
    "corrective exercise",
]

# 首批只选择公开、无需登录、许可证清楚，并且直接包含可用数据的仓库。
DEFAULT_DOWNLOADS = [
    "modelscope/body_2d_keypoints_test_dataset",
    "modelscope/cv_body-3d-keypoints_video_dataset",
    "DatatangBeijing/15People-22LandmarksAnnotationDataof3DHumanBody",
    "DatatangBeijing/18880Imagesof466People-3DInstanceSegmentationand22LandmarksAnnotationDataofHumanBody",
    "DatatangBeijing/100PeopleGaitRecognitionData",
    "DatatangBeijing/3903People-GaitRecognitionDataInSurveillanceScenes",
    "DatatangBeijing/1472People_GaitRecognitionDataInSurveillanceScenes",
    "DatatangBeijing/5808People-HumanPoseRecognitionData",
    "ennmmmmm/Action-Recognition-Dataset-Based-on-Smartphones",
    "MAYI666/HumanActivityRecognitionDataset",
    "Voxel51/MPII_Human_Pose_Dataset",
]

POSITIVE_TERMS = {
    "体态": 4,
    "人体姿态": 5,
    "姿态估计": 4,
    "人体关键点": 5,
    "人体行为": 3,
    "动作识别": 3,
    "骨骼": 3,
    "步态": 3,
    "跌倒": 3,
    "健身": 2,
    "瑜伽": 2,
    "human pose": 5,
    "body-2d-keypoints": 5,
    "body-3d-keypoints": 5,
    "pose-estimation": 4,
    "action-recognition": 3,
    "skeleton": 3,
    "gait": 3,
    "fall detection": 3,
}

NEGATIVE_TERMS = {
    "camera pose": 8,
    "object pose": 7,
    "robot": 5,
    "机器人": 5,
    "hand pose": 4,
    "手势": 3,
    "face": 4,
    "人脸": 4,
    "speech": 4,
    "语音": 4,
    "driver": 3,
    "驾驶": 3,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, tuple, set)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if is_dataclass(value):
        return json_value(asdict(value))
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return json_value(enum_value)
    return str(value)


def repo_dict(repo: Any) -> dict[str, Any]:
    if is_dataclass(repo):
        return json_value(asdict(repo))
    return {
        key: json_value(getattr(repo, key))
        for key in dir(repo)
        if not key.startswith("_") and not callable(getattr(repo, key))
    }


def relevance_score(payload: dict[str, Any]) -> int:
    searchable = " ".join(
        str(payload.get(key) or "")
        for key in ("id", "display_name", "description", "tasks", "tags")
    ).lower()
    score = sum(weight for term, weight in POSITIVE_TERMS.items() if term in searchable)
    score -= sum(weight for term, weight in NEGATIVE_TERMS.items() if term in searchable)
    return score


def connect_database() -> sqlite3.Connection:
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id TEXT PRIMARY KEY,
            owner TEXT,
            name TEXT,
            display_name TEXT,
            description TEXT,
            license TEXT,
            tasks_json TEXT NOT NULL,
            tags_json TEXT NOT NULL,
            downloads INTEGER,
            likes INTEGER,
            advertised_size_bytes INTEGER,
            private INTEGER NOT NULL,
            gated INTEGER NOT NULL,
            login_required INTEGER NOT NULL,
            relevance_score INTEGER NOT NULL,
            source_url TEXT NOT NULL,
            created_at TEXT,
            last_modified TEXT,
            discovered_at TEXT NOT NULL,
            downloaded_at TEXT,
            local_path TEXT,
            local_file_count INTEGER NOT NULL DEFAULT 0,
            local_size_bytes INTEGER NOT NULL DEFAULT 0,
            download_status TEXT NOT NULL DEFAULT 'cataloged',
            error TEXT
        );

        CREATE TABLE IF NOT EXISTS search_hits (
            search_term TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            discovered_at TEXT NOT NULL,
            PRIMARY KEY (search_term, dataset_id),
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
        );

        CREATE TABLE IF NOT EXISTS local_files (
            dataset_id TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            extension TEXT,
            size_bytes INTEGER NOT NULL,
            modified_at TEXT,
            PRIMARY KEY (dataset_id, relative_path),
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
        );

        CREATE TABLE IF NOT EXISTS pose_samples (
            dataset_id TEXT NOT NULL,
            sample_id TEXT NOT NULL,
            filepath TEXT,
            media_type TEXT,
            tags_json TEXT NOT NULL,
            keypoint_count INTEGER NOT NULL,
            local_file_exists INTEGER NOT NULL,
            local_size_bytes INTEGER NOT NULL,
            annotation_json TEXT NOT NULL,
            PRIMARY KEY (dataset_id, sample_id),
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
        );

        CREATE TABLE IF NOT EXISTS sync_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            search_terms_json TEXT NOT NULL,
            catalog_count INTEGER NOT NULL DEFAULT 0,
            downloaded_count INTEGER NOT NULL DEFAULT 0,
            local_file_count INTEGER NOT NULL DEFAULT 0,
            local_size_bytes INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            error TEXT
        );
        """
    )
    return conn


def search_catalog(api: HubApi) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    catalog: dict[str, dict[str, Any]] = {}
    hits: dict[str, set[str]] = {}
    for term in SEARCH_TERMS:
        page_number = 1
        while True:
            page = api.list_repos(
                "dataset",
                search=term,
                page_size=50,
                page_number=page_number,
            )
            items = list(page)
            for repo in items:
                payload = repo_dict(repo)
                dataset_id = str(payload["id"])
                payload["relevance_score"] = relevance_score(payload)
                catalog[dataset_id] = payload
                hits.setdefault(term, set()).add(dataset_id)
            if len(items) < 50:
                break
            page_number += 1
    return catalog, hits


def save_catalog(
    conn: sqlite3.Connection,
    catalog: dict[str, dict[str, Any]],
    hits: dict[str, set[str]],
) -> None:
    discovered_at = utc_now()
    for dataset_id, item in catalog.items():
        conn.execute(
            """
            INSERT INTO datasets (
                dataset_id, owner, name, display_name, description, license,
                tasks_json, tags_json, downloads, likes, advertised_size_bytes,
                private, gated, login_required, relevance_score, source_url,
                created_at, last_modified, discovered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET
                owner=excluded.owner,
                name=excluded.name,
                display_name=excluded.display_name,
                description=excluded.description,
                license=excluded.license,
                tasks_json=excluded.tasks_json,
                tags_json=excluded.tags_json,
                downloads=excluded.downloads,
                likes=excluded.likes,
                advertised_size_bytes=excluded.advertised_size_bytes,
                private=excluded.private,
                gated=excluded.gated,
                login_required=excluded.login_required,
                relevance_score=excluded.relevance_score,
                source_url=excluded.source_url,
                created_at=excluded.created_at,
                last_modified=excluded.last_modified,
                discovered_at=excluded.discovered_at
            """,
            (
                dataset_id,
                item.get("owner"),
                item.get("name"),
                item.get("display_name"),
                item.get("description"),
                item.get("license"),
                json.dumps(item.get("tasks") or [], ensure_ascii=False),
                json.dumps(item.get("tags") or [], ensure_ascii=False),
                item.get("downloads") or 0,
                item.get("likes") or 0,
                item.get("file_size") or 0,
                int(bool(item.get("private"))),
                int(bool(item.get("gated"))),
                int(bool(item.get("login_required"))),
                item["relevance_score"],
                f"https://www.modelscope.cn/datasets/{dataset_id}",
                item.get("created_at"),
                item.get("last_modified"),
                discovered_at,
            ),
        )
    for term, dataset_ids in hits.items():
        conn.executemany(
            """
            INSERT INTO search_hits(search_term, dataset_id, discovered_at)
            VALUES (?, ?, ?)
            ON CONFLICT(search_term, dataset_id) DO UPDATE SET
                discovered_at=excluded.discovered_at
            """,
            [(term, dataset_id, discovered_at) for dataset_id in dataset_ids],
        )
    conn.commit()


def local_dataset_dir(dataset_id: str) -> Path:
    owner, name = dataset_id.split("/", 1)
    return DATASETS_DIR / owner / name


def download_dataset(
    conn: sqlite3.Connection,
    dataset_id: str,
    *,
    max_workers: int,
) -> None:
    target = local_dataset_dir(dataset_id)
    target.mkdir(parents=True, exist_ok=True)
    conn.execute(
        "UPDATE datasets SET download_status='downloading', error=NULL WHERE dataset_id=?",
        (dataset_id,),
    )
    conn.commit()
    try:
        dataset_snapshot_download(
            dataset_id=dataset_id,
            local_dir=str(target),
            max_workers=max_workers,
        )
        index_local_files(conn, dataset_id, target)
    except Exception as exc:
        conn.execute(
            "UPDATE datasets SET download_status='failed', error=? WHERE dataset_id=?",
            (f"{type(exc).__name__}: {exc}", dataset_id),
        )
        conn.commit()
        raise


def iter_local_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and "/.cache/" not in path.as_posix():
            yield path


def index_local_files(
    conn: sqlite3.Connection,
    dataset_id: str,
    root: Path,
    *,
    status: str = "downloaded",
) -> None:
    conn.execute("DELETE FROM local_files WHERE dataset_id=?", (dataset_id,))
    rows = []
    total_bytes = 0
    invalid_empty_files = 0
    for path in iter_local_files(root):
        stat = path.stat()
        if stat.st_size == 0 and path.suffix.lower() in {
            ".csv",
            ".jpg",
            ".jpeg",
            ".json",
            ".mp4",
            ".png",
            ".zip",
        }:
            invalid_empty_files += 1
            continue
        total_bytes += stat.st_size
        rows.append(
            (
                dataset_id,
                path.relative_to(root).as_posix(),
                path.suffix.lower(),
                stat.st_size,
                datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
            )
        )
    conn.executemany(
        """
        INSERT INTO local_files(dataset_id, relative_path, extension, size_bytes, modified_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    effective_status = "partial" if invalid_empty_files else status
    error = (
        f"{invalid_empty_files} empty data file(s) failed remote hash validation"
        if invalid_empty_files
        else None
    )
    conn.execute(
        """
        UPDATE datasets SET
            downloaded_at=?, local_path=?, local_file_count=?, local_size_bytes=?,
            download_status=?, error=?
        WHERE dataset_id=?
        """,
        (
            utc_now(),
            str(root),
            len(rows),
            total_bytes,
            effective_status,
            error,
            dataset_id,
        ),
    )
    import_pose_samples(conn, dataset_id, root)
    conn.commit()


def count_keypoints(value: Any) -> int:
    if isinstance(value, dict):
        total = 0
        if isinstance(value.get("keypoints"), list):
            total += sum(
                len(item.get("points") or [])
                for item in value["keypoints"]
                if isinstance(item, dict)
            )
        return total + sum(count_keypoints(item) for item in value.values())
    if isinstance(value, list):
        return sum(count_keypoints(item) for item in value)
    return 0


def import_pose_samples(conn: sqlite3.Connection, dataset_id: str, root: Path) -> None:
    manifest = root / "samples.json"
    if not manifest.exists():
        return
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    samples = payload.get("samples", []) if isinstance(payload, dict) else []
    conn.execute("DELETE FROM pose_samples WHERE dataset_id=?", (dataset_id,))
    rows = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue
        identifier = sample.get("_id")
        if isinstance(identifier, dict):
            identifier = identifier.get("$oid")
        sample_id = str(identifier or index)
        filepath = str(sample.get("filepath") or "")
        local_file = root / filepath if filepath else None
        exists = bool(local_file and local_file.is_file() and local_file.stat().st_size > 0)
        size_bytes = local_file.stat().st_size if exists and local_file else 0
        rows.append(
            (
                dataset_id,
                sample_id,
                filepath,
                sample.get("_media_type"),
                json.dumps(sample.get("tags") or [], ensure_ascii=False),
                count_keypoints(sample),
                int(exists),
                size_bytes,
                json.dumps(sample, ensure_ascii=False, separators=(",", ":")),
            )
        )
    conn.executemany(
        """
        INSERT INTO pose_samples(
            dataset_id, sample_id, filepath, media_type, tags_json,
            keypoint_count, local_file_exists, local_size_bytes, annotation_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def refresh_all_local_indexes(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT dataset_id, download_status FROM datasets"
    ).fetchall()
    for dataset_id, download_status in rows:
        path = local_dataset_dir(dataset_id)
        if path.exists():
            status = "downloaded" if download_status == "downloaded" else "partial"
            index_local_files(conn, dataset_id, path, status=status)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-only", action="store_true", help="只同步目录，不下载数据文件")
    parser.add_argument("--dataset", action="append", help="只下载指定 owner/name，可重复传入")
    parser.add_argument("--reindex", action="store_true", help="仅重新扫描已有本地文件")
    parser.add_argument("--max-workers", type=int, default=32, help="并发下载线程数（默认 32）")
    args = parser.parse_args()

    conn = connect_database()
    started_at = utc_now()
    conn.execute(
        "UPDATE sync_runs SET finished_at=?, status='interrupted' WHERE status='running'",
        (started_at,),
    )
    run_id = conn.execute(
        "INSERT INTO sync_runs(started_at, search_terms_json, status) VALUES (?, ?, 'running')",
        (started_at, json.dumps(SEARCH_TERMS, ensure_ascii=False)),
    ).lastrowid
    conn.commit()

    try:
        if args.reindex:
            refresh_all_local_indexes(conn)
        else:
            api = HubApi()
            catalog, hits = search_catalog(api)
            save_catalog(conn, catalog, hits)
            if not args.catalog_only:
                requested = args.dataset or DEFAULT_DOWNLOADS
                for dataset_id in requested:
                    if dataset_id not in catalog:
                        repo = repo_dict(api.get_dataset(dataset_id))
                        repo["relevance_score"] = relevance_score(repo)
                        save_catalog(conn, {dataset_id: repo}, {"explicit": {dataset_id}})
                    print(f"Downloading {dataset_id}", flush=True)
                    download_dataset(conn, dataset_id, max_workers=max(1, args.max_workers))

        stats = conn.execute(
            """
            SELECT
                COUNT(*),
                SUM(CASE WHEN download_status='downloaded' THEN 1 ELSE 0 END),
                COALESCE(SUM(local_file_count), 0),
                COALESCE(SUM(local_size_bytes), 0)
            FROM datasets
            """
        ).fetchone()
        conn.execute(
            """
            UPDATE sync_runs SET finished_at=?, catalog_count=?, downloaded_count=?,
                local_file_count=?, local_size_bytes=?, status='complete'
            WHERE run_id=?
            """,
            (utc_now(), *stats, run_id),
        )
        conn.commit()
        print(
            json.dumps(
                {
                    "database": str(DATABASE_PATH),
                    "catalog_count": stats[0],
                    "downloaded_count": stats[1],
                    "local_file_count": stats[2],
                    "local_size_bytes": stats[3],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        conn.execute(
            "UPDATE sync_runs SET finished_at=?, status='failed', error=? WHERE run_id=?",
            (utc_now(), f"{type(exc).__name__}: {exc}", run_id),
        )
        conn.commit()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
