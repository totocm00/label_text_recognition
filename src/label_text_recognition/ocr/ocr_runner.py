# ==========================================================
# "이미지 한 장을 받아서 → OCR 엔진으로 돌리고 → 후처리해서
# 결과(list[dict])와 시각화된 이미지(ndarray)를 돌려주는"
# 단일 진입점 모듈입니다.
# demos/ 쪽에서는 이 함수 하나만 호출하면 됩니다.
#
# 개선 사항
# ----------------------------------------------------------
# - try/except로 예외를 안전하게 처리 (엔진 내부 오류 방어)
# - 결과가 비었을 경우 원인을 함께 반환하도록 구조 확장
# - 로그 및 UI 확장을 고려해 message(상태 설명 문자열) 추가
# - cls_enable 인자를 추가해서 YAML에서 "방향 보정(cls)"을 ON/OFF 가능하게 함
#   * cls_enable=True  → 기울기/방향 자동 보정 → 정확도↑, 처리시간/CPU/GPU 사용량 소폭↑
#   * cls_enable=False → 보정 없이 빠르게 → 실시간/저사양 환경에 유리
# ==========================================================

from typing import Any, Tuple
from .ocr_utils import merge_words_with_boxes


def run_ocr_on_image(
    image_bgr,
    ocr_engine,
    conf_threshold: float = 0.5,
    cls_enable: bool = True,
) -> Tuple[list[dict], Any, str]:
    """
    단일 이미지에 대해 OCR을 실행하고 후처리된 결과, 시각화 이미지, 상태 메시지를 반환합니다.

    Parameters
    ----------
    image_bgr : ndarray
        입력 이미지 (BGR)
    ocr_engine :
        PaddleOCR 인스턴스
    conf_threshold : float
        이 값보다 낮은 confidence는 필터링됩니다.
    cls_enable : bool
        True  → 텍스트 방향/기울기 보정까지 수행 (정확도 우선 모드)
        False → 보정 단계 생략 (속도/자원 우선 모드)

    Returns
    -------
    (merged_results, vis_image, message)
        merged_results : list[dict] - 인식 결과 (텍스트, bbox, avg_conf 등)
        vis_image      : ndarray - 시각화된 이미지
        message        : str - 상태나 원인 정보 ("OK", "EMPTY ...", "ERROR: ...")
    """
    try:
        # ----------------------------------------------------------
        # ① OCR 실행
        #    cls_enable이 True면 PaddleOCR가 한 줄 한 줄 방향을 먼저 보정한 뒤 인식합니다.
        #    이 단계는 살짝 느려지지만 기울어진 사진/카메라 캡처에서는 인식률이 올라갑니다.
        #    False로 두면 이 보정 단계가 생략되어 더 빠르고 GPU/CPU 부담이 줄어듭니다.
        # ----------------------------------------------------------
        ocr_result = ocr_engine.ocr(image_bgr, cls=cls_enable)

        if not ocr_result or not ocr_result[0]:
            # 결과 자체가 비었을 때
            return [], image_bgr, "EMPTY: OCR 결과 없음 (글자 영역 미검출)"

        # ----------------------------------------------------------
        # ② Confidence 필터링
        # ----------------------------------------------------------
        filtered = []
        for box, (text, conf) in ocr_result[0]:
            try:
                if float(conf) >= conf_threshold:
                    filtered.append((box, (text, conf)))
            except (ValueError, TypeError):
                # conf가 숫자로 변환 안 되면 그냥 건너뜀
                continue

        if not filtered:
            return [], image_bgr, f"EMPTY: 모든 결과의 confidence가 threshold({conf_threshold}) 미만"

        # ----------------------------------------------------------
        # ③ 후처리 및 결과 병합
        # ----------------------------------------------------------
        merged_results, vis_img = merge_words_with_boxes(image_bgr, filtered)

        if not merged_results:
            return [], vis_img, "EMPTY: 후처리 병합 결과 없음"

        # 정상적으로 끝난 경우
        return merged_results, vis_img, "OK"

    except Exception as e:
        # ----------------------------------------------------------
        # ④ 예외 발생 시 안전하게 반환
        # ----------------------------------------------------------
        print(f"⚠️ run_ocr_on_image 예외 발생: {e}")
        return [], image_bgr, f"ERROR: {str(e)}"