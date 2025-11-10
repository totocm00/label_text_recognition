# ==========================================================
# 카메라 초기화 유틸리티
# YAML 설정의 camera_index(auto / 숫자 / None)를 해석하고,
# 실제 연결 가능한 카메라를 열어 VideoCapture 객체를 반환합니다.
# auto일 때는 감지된 카메라 인덱스들을 같이 출력합니다.
# ==========================================================

import cv2
from label_text_recognition.camera.camera_auto_finder import resolve_camera_index


def scan_available_cameras(max_index: int = 10):
    """
    0 ~ max_index-1 범위에서 열리는 카메라만 수집해서 리스트로 반환.
    auto 모드일 때 콘솔에 보여주려고 쓰는 보조 함수.
    """
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def init_camera(cfg):
    """
    설정 파일(cfg)을 기반으로 카메라를 초기화합니다.

    1. camera_index: "auto" → 연결 가능한 첫 카메라 자동 탐색
    2. camera_index: int → 해당 번호 사용
    3. 실패 시 None 반환
    """

    # ----------------------------------------------------------
    # 1️⃣ 카메라 인덱스 해석
    # ----------------------------------------------------------
    raw_index = cfg.get("camera_index", "auto")

    # auto로 설정된 경우, 어떤 카메라들이 잡히는지 먼저 보여줌
    available = None
    if isinstance(raw_index, str) and raw_index.lower() == "auto":
        available = scan_available_cameras(max_index=10)
        if not available:
            print("❌ 연결 가능한 카메라가 없습니다.")
            print("⚙️ 카메라 케이블/노트북 웹캠을 확인하거나 YAML에서 숫자로 지정해보세요.")
            return None
        print(f"🔍 감지된 카메라 인덱스: {available}")

    # 실제로 사용할 인덱스 결정 (auto든 숫자든 여기서 최종 결정)
    camera_id = resolve_camera_index(raw_index)

    if camera_id is None:
        print(f"❌ 카메라 인덱스를 결정할 수 없습니다. (입력값: {raw_index})")
        print("⚙️ ocr_config.yaml에서 camera_index를 숫자로 직접 지정해보세요. (예: 0)")
        return None

    # ----------------------------------------------------------
    # 2️⃣ VideoCapture 생성 및 해상도 설정
    # ----------------------------------------------------------
    frame_w = cfg.get("frame_width", 960)
    frame_h = cfg.get("frame_height", 540)

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

    # ----------------------------------------------------------
    # 3️⃣ 정상 오픈 여부 확인
    # ----------------------------------------------------------
    if not cap.isOpened():
        print(f"❌ 카메라 {camera_id} 를 열 수 없습니다.")
        # 만약 auto 스캔 결과가 있으면 힌트도 같이 출력
        if available:
            print(f"💡 참고: 방금 감지된 카메라 인덱스는 {available} 였습니다.")
            print("   그 중 하나를 ocr_config.yaml에 숫자로 넣어서 다시 시도해보세요.")
        else:
            print("⚙️ 장치 연결 상태 또는 YAML 설정(camera_index)을 확인하세요.")
        return None

    print(f"✅ Camera {camera_id} opened successfully ({frame_w}x{frame_h})")
    return cap
