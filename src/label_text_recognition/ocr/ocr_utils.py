# src/label_text_recognition/ocr/ocr_utils.py
# ==========================================================
# OCR 결과를 사람이 보기 좋게, 그리고 후처리·시각화하기 좋게
# 가공하는 유틸리티 함수들이 들어있는 모듈입니다.
# 여기서는 "비슷한 y좌표에 있는 단어들을 한 줄로 묶는" 단순 버전을 제공합니다.
# ==========================================================

from typing import Any
import numpy as np
import cv2


def merge_words_with_boxes(image, ocr_result, y_thresh=20, x_gap_thresh=30):
    """
    paddleocr가 반환한 결과를 받아서 같은 줄에 있는 단어를 합치고
    이미지에 박스를 그려주는 함수입니다.
    :param image: 원본 이미지 (ndarray, BGR)
    :param ocr_result: paddleocr.ocr(...) 호출 결과 중 한 프레임
    :param y_thresh: 같은 줄로 볼 y 간격
    :param x_gap_thresh: 단어를 붙여서 한 문장으로 볼 x 간격
    :return: (merged_results, vis_image)
    """
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

    # Y좌표 기준 정렬
    lines.sort(key=lambda t: (t["cy"], t["cx"]))

    # 같은 줄끼리 묶기
    grouped_lines = []
    current_line = [lines[0]]
    for i in range(1, len(lines)):
        if abs(lines[i]["cy"] - current_line[-1]["cy"]) <= y_thresh:
            current_line.append(lines[i])
        else:
            grouped_lines.append(current_line)
            current_line = [lines[i]]
    grouped_lines.append(current_line)

    merged_results = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = [
        (0, 255, 0),
        (255, 255, 0),
        (0, 255, 255),
        (255, 128, 0),
        (255, 0, 255),
        (0, 128, 255),
    ]

    for line_idx, line in enumerate(grouped_lines, start=1):
        # 같은 줄 안에서는 x 순으로
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

        # 시각화
        for word in line:
            pts = np.array(word["box"], np.int32)
            cv2.polylines(
                image,
                [pts],
                isClosed=True,
                color=colors[line_idx % len(colors)],
                thickness=2,
            )

        y_pos = int(line[0]["cy"]) - 10
        cv2.putText(
            image,
            f"{line_idx}. {merged_text}",
            (int(line[0]["x_min"]), y_pos),
            font,
            0.7,
            (0, 0, 255),
            2,
        )

        merged_results.append({
            "line_index": line_idx,
            "text": merged_text,
            "avg_conf": float(np.mean([w["conf"] for w in line])),
        })

    return merged_results, image
