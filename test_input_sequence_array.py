import cv2
import numpy as np
import os
from paddleocr import PaddleOCR

# OCR 초기화
ocr = PaddleOCR(lang='en')
IMG_PATH = "test.jpg"  # 입력 이미지

# 출력 경로 설정
OUTPUT_DIR = os.path.join("..", "assets", "samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "test.jpeg")

def merge_words_with_boxes(image, ocr_result, y_thresh=20, x_gap_thresh=30):
    """
    OCR 결과 정렬 + 단어 병합 + 시각화
    """
    lines = []
    for box_info in ocr_result:
        box, (text, conf) = box_info
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        cx, cy = np.mean(x_coords), np.mean(y_coords)
        lines.append({
            "text": text.strip(),
            "conf": conf,
            "cx": cx,
            "cy": cy,
            "x_min": min(x_coords),
            "x_max": max(x_coords),
            "box": np.array(box).astype(int)
        })

    # Y 좌표 기준 정렬
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

    merged_texts = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    color_list = [
        (0, 255, 0), (255, 255, 0), (0, 255, 255),
        (255, 128, 0), (255, 0, 255), (0, 128, 255)
    ]

    for line_idx, line in enumerate(grouped_lines, start=1):
        line.sort(key=lambda t: t["x_min"])
        merged_line = []
        current_word = line[0]["text"]

        for i in range(1, len(line)):
            gap = line[i]["x_min"] - line[i - 1]["x_max"]
            if gap < x_gap_thresh:
                current_word += " " + line[i]["text"]
            else:
                merged_line.append(current_word)
                current_word = line[i]["text"]
        merged_line.append(current_word)
        merged_text = " ".join(merged_line)
        merged_texts.append(merged_text)

        # ----- 시각화 -----
        for word in line:
            cv2.polylines(image, [word["box"]], isClosed=True, color=color_list[line_idx % len(color_list)], thickness=2)
        # 줄 단위 표시
        y_pos = int(line[0]["cy"]) - 10
        cv2.putText(image, f"{line_idx}. {merged_text}", (int(line[0]["x_min"]), y_pos),
                    font, 0.7, (0, 0, 255), 2)

    return merged_texts, image


if __name__ == "__main__":
    img = cv2.imread(IMG_PATH)
    result = ocr.ocr(IMG_PATH, cls=False)

    merged_texts, vis_img = merge_words_with_boxes(img, result[0])

    print("🧩 인식 순서대로 정렬된 텍스트:")
    for idx, line_text in enumerate(merged_texts, start=1):
        print(f"{idx}. {line_text}")

    # 출력 파일 저장
    cv2.imwrite(OUTPUT_PATH, vis_img)
    print(f"✅ 시각화 결과 저장 완료: {OUTPUT_PATH}")
