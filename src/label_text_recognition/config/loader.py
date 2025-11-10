# ==========================================================
# YAML 설정 파일(ocr_config.yaml)을 읽어서 파이썬 dict로 반환합니다.
# 카메라 인덱스, 프레임 크기, OCR 언어, 출력 경로 등 자주 바꾸는 값은
# 전부 이 파일을 통해 가져오도록 해서 "매직 넘버"를 코드에 직접 쓰지 않게 합니다.
# ==========================================================

import os
import yaml


def load_ocr_config() -> dict:
    """
    ocr_config.yaml 파일을 읽어서 dict로 반환합니다.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "ocr_config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data
