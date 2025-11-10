# ==========================================================
# json_exporter.py
# ----------------------------------------------------------
# 역할:
#   - OCR 결과를 JSON 파일로 저장하는 가장 기본적인 Exporter입니다.
#   - 저장 경로는 호출하는 쪽(camera_loop 등)에서 넘겨줍니다.
#
# 추가된 내용:
#   ✅ ocr_config.yaml 의 enable_save_output 값을 읽어서
#      저장을 할지 말지를 여기서도 한 번 더 확인합니다.
#      (카메라 쪽에서 깜빡하고 저장을 호출해도 여기서 막힙니다.)
#   ✅ 저장이 비활성화되어 있으면 폴더도 만들지 않고,
#      파일도 생성하지 않으며 안내 메시지만 출력합니다.
#
# 확장 포인트:
#   - 이후 CSV, DB, REST API 연동 등으로 확장할 때 이 파일을 기준으로
#     같은 인터페이스(export_to_xxx) 형태로 추가하면 됩니다.
# ==========================================================

import os
import json
from typing import Any, List

# 프로젝트 공통 설정을 가져오기 위해 사용
from label_text_recognition.config.loader import load_ocr_config


def export_to_json(results: List[dict], output_path: str) -> None:
    """
    OCR 결과 리스트를 JSON 파일로 저장합니다.

    Parameters
    ----------
    results : list[dict]
        OCR 한 줄/한 박스마다의 결과가 들어 있는 리스트입니다.
        예: [{"text": "시험일", "avg_conf": 0.94, "box": [[x1,y1], ...]}, ...]
    output_path : str
        저장할 JSON 파일의 전체 경로입니다.
        예: assets/json/capture_20251111_150845.json
    """
    # 1) 설정을 불러와서 저장 기능이 켜져 있는지 확인
    cfg = load_ocr_config()
    enable_save_output = cfg.get("enable_save_output", True)

    if not enable_save_output:
        # 저장을 하지 않기로 한 환경이라면 여기서 바로 종료
        # (카메라 코드에서 저장 호출을 해도 여기서 한 번 더 안전장치)
        print("💾 JSON 저장이 비활성화되어 있어 파일을 생성하지 않습니다. "
              "(enable_save_output: false)")
        return

    # 2) 저장할 디렉터리가 없으면 생성
    #    예: output_path = "assets/json/capture_xxx.json" 이면
    #        "assets/json" 폴더를 만들어 줍니다.
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # 3) JSON 파일로 덤프
    #    ensure_ascii=False 를 꼭 넣어야 한글이 "????"가 아닌
    #    실제 한글로 저장됩니다.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"✅ JSON 저장 완료: {output_path}")