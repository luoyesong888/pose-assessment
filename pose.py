from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from analysis import acl_not_assessed


WORKER = Path(__file__).with_name("pose_worker.py")


def _failed_result(view: str, message: str, technical_error: str = "") -> Dict[str, Any]:
    return {
        "found": False,
        "valid": False,
        "view": view,
        "issues": [message],
        "metrics": {},
        "quality": {"valid": False, "errors": [message], "warnings": []},
        "acl_risk": acl_not_assessed(message),
        "technical_error": technical_error,
    }


def analyze_image_file(image_path, view: str = "front") -> Dict[str, Any]:
    """在隔离子进程中运行 MediaPipe，避免 GPU/Metal 崩溃拖垮网页进程。"""
    command = [sys.executable, str(WORKER), "--image", str(image_path), "--view", view]
    try:
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).parent),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _failed_result(view, "姿态引擎运行超时，请缩小图片后重试")
    except OSError as exc:
        return _failed_result(view, "无法启动姿态分析进程", f"{type(exc).__name__}: {exc}")

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "unknown worker error")[-2000:]
        return _failed_result(
            view,
            "姿态引擎初始化失败，请重启应用或检查 MediaPipe 运行环境",
            detail,
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return _failed_result(view, "姿态引擎返回了无效结果", f"{exc}: {completed.stdout[-500:]}")
    return result


def analyze_image_files(items: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """一次子进程批量分析多张图片，并在图片之间复用同一个 MediaPipe 模型。"""
    if not items:
        return []
    payload = [{"image": str(path), "view": view} for path, view in items]
    command = [sys.executable, str(WORKER), "--batch"]
    try:
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).parent),
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=max(120, 45 * len(payload)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [_failed_result(item["view"], "姿态引擎批量运行超时，请减少图片后重试") for item in payload]
    except OSError as exc:
        detail = f"{type(exc).__name__}: {exc}"
        return [_failed_result(item["view"], "无法启动姿态分析进程", detail) for item in payload]

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "unknown worker error")[-2000:]
        return [
            _failed_result(item["view"], "姿态引擎初始化失败，请重启应用或检查 MediaPipe 运行环境", detail)
            for item in payload
        ]
    try:
        results = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        detail = f"{exc}: {completed.stdout[-500:]}"
        return [_failed_result(item["view"], "姿态引擎返回了无效结果", detail) for item in payload]
    if not isinstance(results, list) or len(results) != len(payload):
        return [_failed_result(item["view"], "姿态引擎返回的批量结果数量不一致") for item in payload]
    return results


def analyze_image(image_path):
    """兼容旧命令行接口。"""
    result = analyze_image_file(image_path, view="front")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_image(sys.argv[1])
    else:
        analyze_image(input("请输入图片路径：").strip())
