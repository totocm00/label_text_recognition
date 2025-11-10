# ==========================================================
# 카메라 초기화 전담 유틸리티.
# 설정 파일의 camera_index 값을 해석(auto / 숫자 / None)하고,
# 실제로 열 수 있는 장치를 확인하여 반환합니다.
# ==========================================================

import cv2
import os
from label_text_recognition.camera.auto_camera_finder import resolve_camera_index

def init_camera(cfg):
    """
    설정 파일(cfg)을 기반으로 카메라를 초기화하고 VideoCapture 객체를 반환합니다.
    1. camera_index: auto → 자동 탐색
    2. camera_index: int → 해당 번호 사용
    3. 예외 발생 시 None 반환
    """

    camera_id = resolve_camera_index(cfg.get("camera_index", "auto"))
    if camera_id is None:
        print("❌ 카메라 인덱스를 결정할 수 없습니다. 설정을 확인하세요.")
        return None

    frame_w = cfg.get("frame_width", 960)
    frame_h = cfg.get("frame_height", 540)

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

    if not cap.isOpened():
        print(f"❌ 카메라 {camera_id} 를 열 수 없습니다.")
        print("⚙️ 카메라 연결 상태나 YAML의 camera_index 값을 확인하세요.")
        return None

    print(f"✅ Camera {camera_id} opened successfully.")
    return cap
