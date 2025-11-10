# ==========================================================
# ocr_runner.py
# ----------------------------------------------------------
# "이미지 한 장을 받아서 → OCR 엔진으로 돌리고 → 후처리해서
# 결과(list[dict])와 시각화된 이미지(ndarray)를 돌려주는"
# 단일 진입점 모듈입니다.
#
# 추가 개선 (2025-11-11):
#   ✅ ocr_config.yaml 의 enable_* 토글 상태를 message 안에 포함
#      → UI / 로그 쪽에서 현재 세션이 저장 비활성화인지(SAVE_OFF),
#        콘솔 로그 모드인지(CONSOLE_ON) 등을 한 번에 파악 가능
#   ✅ 함수 시그니처는 유지 (demos/ 쪽 수정 필요 없음)
# ==========================================================

from typing import Any, Tuple

from label_text_recognition.config.loader import load_ocr_config
from .ocr_utils import merge_words_with_boxes


def _build_mode_suffix(cfg: dict) -> str:
    """
    YAML에서 가져온 enable_* 값들을 간단한 문자열로 표현해주는 헬퍼.
    예) "MODE: SAVE_ON, CONSOLE_OFF, REDRAW_ON"
    UI에서 이 문자열만 파싱해서 뱃지로 표시해도 됨.
    """
    save_on = cfg.get("enable_save_output", True)
    console_on = cfg.get("enable_console_log", True)
    redraw_on = cfg.get("enable_redraw_from_json", True)

    parts = []
    parts.append("SAVE_ON" if save_on else "SAVE_OFF")
    parts.append("CONSOLE_ON" if console_on else "CONSOLE_OFF")
    parts.append("REDRAW_ON" if redraw_on else "REDRAW_OFF")

    return "MODE: " + ", ".join(parts)


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
        vis_image      : ndarray    - 시각화된 이미지
        message        : str        - 상태나 원인 정보 + 현재 모드 정보
                                      예) "OK | MODE: SAVE_OFF, CONSOLE_ON, REDRAW_ON"
    """
    # 여기서 한 번 설정을 읽어두면 이 함수만 봐도 현재 세션 모드를 알 수 있음
    cfg = load_ocr_config()
    mode_suffix = _build_mode_suffix(cfg)

    try:
        # ----------------------------------------------------------
        # ① OCR 실행
        # ----------------------------------------------------------
        ocr_result = ocr_engine.ocr(image_bgr, cls=cls_enable)

        if not ocr_result or not ocr_result[0]:
            # 결과 자체가 비었을 때
            return [], image_bgr, f"EMPTY: OCR 결과 없음 (글자 영역 미검출) | {mode_suffix}"

        # ----------------------------------------------------------
        # ② Confidence 필터링
        # ----------------------------------------------------------
        filtered = []
        for box, (text, conf) in ocr_result[0]:
            try:
                if float(conf) >= conf_threshold:
                    filtered.append((box, (text, conf)))
            except (ValueError, TypeError):
                # confidence가 숫자 변환이 안 되는 경우는 조용히 스킵
                continue

        if not filtered:
            return [], image_bgr, (
                f"EMPTY: 모든 결과의 confidence가 threshold({conf_threshold}) 미만 | {mode_suffix}"
            )

        # ----------------------------------------------------------
        # ③ 후처리 및 결과 병합
        # ----------------------------------------------------------
        merged_results, vis_img = merge_words_with_boxes(image_bgr, filtered)

        if not merged_results:
            return [], vis_img, f"EMPTY: 후처리 병합 결과 없음 | {mode_suffix}"

        # ----------------------------------------------------------
        # ④ 정상 종료
        # ----------------------------------------------------------
        return merged_results, vis_img, f"OK | {mode_suffix}"

    except Exception as e:
        # 예외가 나더라도 이미지 원본과 상태 메시지를 돌려줍니다.
        # UI에서는 message.startswith("ERROR") 만으로 판단 가능.
        print(f"⚠️ run_ocr_on_image 예외 발생: {e}")
        return [], image_bgr, f"ERROR: {str(e)} | {mode_suffix}"