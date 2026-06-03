from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")


def _get_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    return explicit_key or os.getenv("DEEPSEEK_API_KEY")


def generate_report(
    *,
    system_prompt: str,
    user_prompt: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.3,
    timeout: int = 90,
) -> str:
    key = _get_api_key(api_key)
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing")

    payload: Dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()
