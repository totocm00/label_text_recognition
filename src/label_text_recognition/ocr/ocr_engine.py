# ==========================================================
# 여러 언어를 한 번에 사용할 수 있도록 PaddleOCR 엔진을
# 언어별로 초기화해서 dict로 반환하는 모듈입니다.
# 예) {"en": <PaddleOCR>, "korean": <PaddleOCR>}
# ==========================================================

from paddleocr import PaddleOCR


def build_ocr_engines(lang_list: list[str]) -> dict[str, PaddleOCR]:
    """
    주어진 언어 목록을 바탕으로 PaddleOCR 엔진을 여러 개 생성합니다.
    :param lang_list: ["en", "korean"] 이런 식의 언어코드 리스트
    :return: {"en": ocr_en, "korean": ocr_kr}
    """
    engines: dict[str, PaddleOCR] = {}
    for lang in lang_list:
        engines[lang] = PaddleOCR(lang=lang)
    return engines
