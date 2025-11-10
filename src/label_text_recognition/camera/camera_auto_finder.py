# ==========================================================
# 이 모듈은
# "어떤 PC에서 실행하더라도 카메라를 자동으로 잡게" 해주는
# 유틸리티입니다.
#
# 배경
# - 노트북: 대부분 0번 카메라
# - 데스크톱/외장캠: 1, 2, 4, 6번 등 제각각
# - 배포용 코드에서는 사용자가 일일이 camera_index를 바꾸기 번거로움
#
# 그래서 이 파일은 두 가지 모드를 제공합니다.
# 1) 자동 탐색 모드: camera_index가 "auto"로 설정된 경우
#    → 0~max_index까지 순회하며 열리는 첫 번째 카메라를 반환
# 2) 수동 고정 모드: camera_index에 숫자를 넣은 경우
#    → 그 숫자를 그대로 카메라 인덱스로 사용
#
# 추가 기능
# ----------------------------------------------------------
# - cv2.VideoCapture() 접근 시 OS 오류나 장치 문제로 예외가 발생하더라도
#   try/except로 안전하게 처리되어 프로그램이 중단되지 않습니다.
# - YAML 설정(camera_index)에 숫자가 아닌 값이 들어와도
#   자동으로 경고 후 None을 반환하여 호출부에서 안전하게 종료됩니다.
#
# 사용 예시
# ----------------------------------------------------------
# from label_text_recognition.camera.camera_auto_finder import resolve_camera_index
# cfg = load_ocr_config()
# camera_id = resolve_camera_index(cfg.get("camera_index", "auto"))
# cap = cv2.VideoCapture(camera_id)
#
# YAML에서의 설정 예시
# ----------------------------------------------------------
# camera_index: auto        # → 자동으로 사용 가능한 카메라 찾기
# camera_index: 0           # → 0번 카메라로 고정
# camera_index: 5           # → 5번 카메라로 고정
#
# 주의
# ----------------------------------------------------------
# - 자동 탐색은 0부터 순서대로 찾으므로, 여러 개 연결된 환경에서는
#   "가장 먼저 열리는" 장치를 선택합니다.
# - 아무 장치도 열리지 않으면 None을 반환하므로, 호출부에서 예외처리를
#   한 번 더 해주는 게 안전합니다.
# ==========================================================

import cv2


def find_available_camera(max_index: int = 10) -> int | None:
    """
    0 ~ max_index 범위 내에서 '열리는' 첫 번째 카메라 인덱스를 탐색합니다.

    Parameters
    ----------
    max_index : int
        탐색할 최대 인덱스 번호. 기본값은 10으로, 0~10번까지 확인합니다.

    Returns
    -------
    int | None
        사용 가능한 카메라 인덱스를 찾으면 그 인덱스를 반환하고,
        찾지 못하면 None을 반환합니다.
    """
    for i in range(max_index + 1):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                print(f"✅ Available camera found at index {i}")
                return i
            cap.release()
        except Exception as e:
            print(f"⚠️ Camera index {i} check failed: {e}")
            continue

    print("❌ No available camera detected.")
    return None


def resolve_camera_index(cfg_value) -> int | None:
    """
    YAML 설정값을 해석해서 실제로 사용할 카메라 인덱스를 결정합니다.

    동작 규칙
    ----------
    1) cfg_value가 문자열이고, 그 값이 'auto' 이면
       → 자동 탐색 모드로 전환하여 사용 가능한 첫 번째 카메라를 찾습니다.
    2) cfg_value가 숫자거나 숫자로 변환 가능한 문자열이면
       → 그 값을 그대로 인덱스로 사용합니다.
    3) 그 외의 값이 들어오면
       → 잘못된 설정이므로 None을 반환하고 호출부에서 처리하도록 합니다.

    Parameters
    ----------
    cfg_value :
        YAML에서 읽어온 camera_index 값. 'auto' 또는 정수/정수형 문자열을 예상합니다.

    Returns
    -------
    int | None
        실제로 사용할 카메라 인덱스. 잘못된 값이면 None.
    """
    # 1) 'auto'로 설정된 경우 → 자동 탐색
    if isinstance(cfg_value, str) and cfg_value.lower() == "auto":
        print("🔍 Auto camera detection enabled.")
        return find_available_camera()

    # 2) 숫자 또는 숫자형 문자열인 경우 → 그대로 사용
    try:
        return int(cfg_value)
    except (ValueError, TypeError):
        # 3) 그 외의 데이터가 들어온 경우 → 잘못된 설정
        print(f"⚠️ Invalid camera_index value in config: {cfg_value}")
        return None
