# ==========================================================
# "이미지 한 장을 받아서 → OCR 엔진으로 돌리고 → 후처리해서
# 결과(list[dict])와 시각화된 이미지(ndarray)를 돌려주는"
# 단일 진입점 모듈입니다.
# demos/ 쪽에서는 이 함수 하나만 호출하면 됩니다.
# ==========================================================

from typing import Any
from .ocr_utils import merge_words_with_boxes


def run_ocr_on_image(image_bgr, ocr_engine, conf_threshold: float = 0.5) -> tuple[list[dict], Any]:
    """
    단일 이미지에 대해 OCR을 실행하고 후처리된 결과와 시각화 이미지를 반환합니다.
    :param image_bgr: BGR 이미지(ndarray)
    :param ocr_engine: PaddleOCR 인스턴스
    :param conf_threshold: 이 값보다 낮은 confidence는 버립니다.
    :return: (merged_results, vis_image)
    """
    ocr_result = ocr_engine.ocr(image_bgr, cls=False)

    if not ocr_result or not ocr_result[0]:
        return [], image_bgr

    # 1차 confidence 필터링
    filtered = []
    for box, (text, conf) in ocr_result[0]:
        if float(conf) >= conf_threshold:
            filtered.append((box, (text, conf)))

    merged_results, vis_img = merge_words_with_boxes(image_bgr, filtered)
    return merged_results, vis_img
