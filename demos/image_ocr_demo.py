# ==========================================================
# 이미지 한 장에 대해 OCR을 수행해보는 데모 스크립트입니다.
# 카메라 없이 테스트하거나, assets/pictures 안의 샘플로 빠르게 보고 싶을 때 사용합니다.
# ==========================================================

import os
import sys
import argparse
import time
import cv2

# src 경로 추가
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from label_text_recognition.config.loader import load_ocr_config
from label_text_recognition.ocr.ocr_engine import build_ocr_engines
from label_text_recognition.ocr.ocr_runner import run_ocr_on_image
from label_text_recognition.exporters.json_exporter import export_to_json


def main():
    parser = argparse.ArgumentParser(description="Run OCR on a single image.")
    parser.add_argument("--image", "-i", required=True, help="path to image file")
    args = parser.parse_args()

    cfg = load_ocr_config()
    ocr_langs = cfg.get("ocr_langs", ["en"])
    conf_threshold = cfg.get("conf_threshold", 0.5)
    output_json_dir = cfg.get("output_dir_json", "assets/json")
    os.makedirs(output_json_dir, exist_ok=True)

    # 엔진 생성
    engines = build_ocr_engines(ocr_langs)
    main_engine = engines[ocr_langs[0]]

    # 이미지 읽기
    img = cv2.imread(args.image)
    if img is None:
        print(f"❌ 이미지 파일을 읽을 수 없습니다: {args.image}")
        return

    # OCR 실행
    results, vis_img = run_ocr_on_image(img, main_engine, conf_threshold)

    # JSON 저장
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(output_json_dir, f"image_{ts}.json")
    export_to_json(results, out_json)

    print(f"✅ OCR 완료, 결과 JSON: {out_json}")
    for r in results:
        print(f"- {r['text']} ({r['avg_conf']:.2f})")


if __name__ == "__main__":
    main()
