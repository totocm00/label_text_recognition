# ==========================================================
# 설정 관련 모듈을 묶어주는 패키지 초기화 파일입니다.
# 외부에서는 from label_text_recognition.config import load_ocr_config
# 정도로만 사용하도록 설계합니다.
# ==========================================================

from .loader import load_ocr_config

__all__ = ["load_ocr_config"]
