# ==========================================================
# 카메라를 켜서 실시간으로 OCR을 돌려보는 데모 스크립트입니다.
# 실제 로직은 src/label_text_recognition/camera/camera_loop.py 안에 있고
# 여기서는 그 함수를 불러와서 실행만 합니다.
# ==========================================================

import os
import sys

# src/ 경로를 파이썬 경로에 추가 (로컬 실행 편의용)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from label_text_recognition.camera.camera_loop import start_camera_ocr


if __name__ == "__main__":
    start_camera_ocr()
