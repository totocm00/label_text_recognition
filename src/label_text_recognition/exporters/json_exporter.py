# ==========================================================
# OCR 결과를 JSON 파일로 저장하는 가장 단순한 Exporter입니다.
# 나중에 CSV, 이미지 오버레이, API 응답 등으로 확장할 수 있도록
# 별도 디렉터리로 분리해두었습니다.
# ==========================================================

import os
import json
from typing import Any


def export_to_json(results: list[dict], output_path: str) -> None:
    """
    OCR 결과 리스트를 JSON 파일로 저장합니다.
    :param results: [{"line_index": 1, "text": "...", "avg_conf": 0.91}, ...]
    :param output_path: 저장할 파일 경로
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
