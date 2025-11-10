# ==========================================================
# src/label_text_recognition/ocr/ocr_utils.py
# ----------------------------------------------------------
# OCR 결과를 사람이 보기 좋고, 후처리·시각화하기 쉽게 가공하는
# 유틸리티 함수들이 들어있는 모듈입니다.
#
# 기능 요약
# ----------------------------------------------------------
# ✅ paddleocr의 원시 결과(단어 단위)를 한 줄 단위로 병합
# ✅ 각 줄마다 평균 confidence 계산
# ✅ 이미지 위에 사각형 박스 및 텍스트를 시각화
# ✅ OpenCV + Pillow 혼합 사용 → 한글 텍스트도 깨짐 없이 표시
#
# 사용 예시
# ----------------------------------------------------------
# from label_text_recognition.ocr.ocr_utils import merge_words_with_boxes
# merged_results, vis_image = merge_words_with_boxes(frame, ocr_result)
#
# 결과 예시:
# merged_results = [
#   {"line_index": 1, "text": "시험일 2025.11.11", "avg_conf": 0.93},
#   {"line_index": 2, "text": "성명 홍길동", "avg_conf": 0.95}
# ]
# ==========================================================

from typing import Any
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


def merge_words_with_boxes(image, ocr_result, y_thresh=20, x_gap_thresh=30):
    """
    OCR 결과(box, text, conf)를 받아서 같은 줄의 단어를 묶고,
    이미지 위에 한글이 포함된 텍스트까지 정상적으로 시각화합니다.

    Parameters
    ----------
    image : np.ndarray
        원본 이미지 (BGR 형식)
    ocr_result : list
        paddleocr.ocr(...) 호출 결과 중 하나의 프레임 데이터
    y_thresh : int
        두 단어의 y좌표 차이가 이 값 이하이면 같은 줄로 판단
    x_gap_thresh : int
        단어 간의 x 간격이 이 값 이하이면 같은 문장으로 이어붙임

    Returns
    -------
    merged_results : list[dict]
        {"line_index": 1, "text": "...", "avg_conf": 0.91, ...} 형태의 결과 리스트
    vis_image : np.ndarray
        박스와 텍스트가 표시된 BGR 이미지
    """
    # ------------------------------------------------------
    # 1️⃣ OCR 결과를 (텍스트 + 위치 + 신뢰도) 구조로 정리
    # ------------------------------------------------------
    lines = []
    for box_info in ocr_result:
        box, (text, conf) = box_info
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        cx, cy = np.mean(x_coords), np.mean(y_coords)
        lines.append({
            "text": text.strip(),
            "conf": float(conf),
            "cx": cx,
            "cy": cy,
            "x_min": min(x_coords),
            "x_max": max(x_coords),
            "box": np.array(box).astype(int).tolist()
        })

    if not lines:
        return [], image

    # ------------------------------------------------------
    # 2️⃣ Y좌표를 기준으로 정렬 후 같은 줄끼리 그룹화
    # ------------------------------------------------------
    lines.sort(key=lambda t: (t["cy"], t["cx"]))

    grouped_lines = []
    current_line = [lines[0]]
    for i in range(1, len(lines)):
        if abs(lines[i]["cy"] - current_line[-1]["cy"]) <= y_thresh:
            current_line.append(lines[i])
        else:
            grouped_lines.append(current_line)
            current_line = [lines[i]]
    grouped_lines.append(current_line)

    # ------------------------------------------------------
    # 3️⃣ 시각화 설정
    # ------------------------------------------------------
    vis_img = image.copy()
    colors = [
        (0, 255, 0),
        (255, 255, 0),
        (0, 255, 255),
        (255, 128, 0),
        (255, 0, 255),
        (0, 128, 255),
    ]

    # PIL로 텍스트 렌더링 (OpenCV는 한글 깨짐)
    pil_img = Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_path = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
    try:
        font = ImageFont.truetype(font_path, 20)
    except OSError:
        print("⚠️ 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        font = ImageFont.load_default()

    merged_results = []

    # ------------------------------------------------------
    # 4️⃣ 각 줄(line)에 대해 단어 병합 및 시각화
    # ------------------------------------------------------
    for line_idx, line in enumerate(grouped_lines, start=1):
        # 줄 내 단어를 x_min 순으로 정렬
        line.sort(key=lambda t: t["x_min"])
        merged_line_words = []
        current_word = line[0]["text"]

        for j in range(1, len(line)):
            gap = line[j]["x_min"] - line[j - 1]["x_max"]
            if gap < x_gap_thresh:
                current_word += " " + line[j]["text"]
            else:
                merged_line_words.append(current_word)
                current_word = line[j]["text"]
        merged_line_words.append(current_word)

        merged_text = " ".join(merged_line_words)

        # 각 단어별 박스(OpenCV)
        for word in line:
            pts = np.array(word["box"], np.int32)
            cv2.polylines(
                vis_img,
                [pts],
                isClosed=True,
                color=colors[line_idx % len(colors)],
                thickness=2,
            )

        # 텍스트(PIL, 한글 지원)
        y_pos = int(line[0]["cy"]) - 25
        x_pos = int(line[0]["x_min"])
        draw.text(
            (x_pos, y_pos),
            f"{line_idx}. {merged_text}",
            font=font,
            fill=(255, 0, 0),
        )

        merged_results.append({
            "line_index": line_idx,
            "text": merged_text,
            "avg_conf": float(np.mean([w["conf"] for w in line])),
        })

    # ------------------------------------------------------
    # 5️⃣ PIL 이미지를 다시 OpenCV BGR로 변환 후 반환
    # ------------------------------------------------------
    vis_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return merged_results, vis_img