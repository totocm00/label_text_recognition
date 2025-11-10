# ==========================================================
# redraw_from_json.py
# ----------------------------------------------------------
# 저장된 OCR JSON 결과를 불러와서
# 원본 이미지 위에 박스와 텍스트를 다시 그려주는 스크립트입니다.
# ==========================================================

import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import argparse

from label_text_recognition.config.loader import load_ocr_config


# ----------------------------------------------------------
# 1️⃣ JSON 기반 한글 텍스트 재시각화 함수
# ----------------------------------------------------------
def redraw_from_json(img_path: str, json_path: str, cfg) -> str:
    """OCR JSON 결과를 기반으로 이미지 위에 한글 텍스트를 다시 그림"""

    # 1. YAML에서 폰트 설정 및 출력 여부 확인
    font_path = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
    enable_redraw = cfg.get("enable_redraw_from_json", True)
    if not enable_redraw:
        print("🔕 redraw_from_json 기능이 비활성화되어 있습니다. (YAML 설정 확인)")
        return None

    # 2. 원본 이미지 로드
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {img_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    # 3. 폰트 로드
    try:
        font = ImageFont.truetype(font_path, 20)
    except OSError:
        print("⚠️ NotoSansCJK 폰트를 찾을 수 없습니다. 기본 폰트로 대체합니다.")
        font = ImageFont.load_default()

    # 4. JSON 불러오기
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 5. OCR 결과 반복하면서 박스+텍스트 그리기
    for item in data:
        text = item.get("text", "")
        box = item.get("box", [])
        if not box:
            continue

        # 좌표 계산
        x1, y1 = int(box[0][0]), int(box[0][1])
        x2, y2 = int(box[2][0]), int(box[2][1])

        # 사각형 박스
        draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 0), width=2)

        # 텍스트 (박스 위쪽에 표시)
        text_y = y1 - 22 if y1 - 22 > 0 else y1 + 2
        draw.text((x1, text_y), text, font=font, fill=(255, 0, 0))

    # 6. 결과 저장
    today = datetime.now().strftime("%Y%m%d")
    base_name = os.path.basename(img_path)
    file_name = os.path.splitext(base_name)[0] + "_redraw.jpg"
    out_dir = os.path.join("assets", "redraw", today)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, file_name)

    out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, out_bgr)
    print(f"✅ 재시각화 완료 → {out_path}")

    return out_path


# ----------------------------------------------------------
# 2️⃣ CLI 실행부
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON 기반 OCR 결과 다시 그리기 (PIL 한글 지원)")
    parser.add_argument("--img", required=True, help="원본 이미지 경로 (예: assets/pictures_origin/capture_XXXX.jpg)")
    parser.add_argument("--json", required=True, help="OCR 결과 JSON 경로 (예: assets/json/capture_XXXX.json)")
    args = parser.parse_args()

    cfg = load_ocr_config()
    redraw_from_json(args.img, args.json, cfg)