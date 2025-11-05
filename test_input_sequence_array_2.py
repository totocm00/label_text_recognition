import cv2
import numpy as np
import os
import json
from paddleocr import PaddleOCR

# ------------------- 설정 -------------------
ocr = PaddleOCR(lang='en')
IMG_PATH = "test.jpg"

# 출력 폴더 및 파일 경로
OUTPUT_DIR = os.path.join("..", "assets", "samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_IMG_PATH = os.path.join(OUTPUT_DIR, "test.jpeg")
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "test_result.json")


# ------------------- 함수 정의 -------------------
def merge_words_with_boxes(image, ocr_result, y_thresh=20, x_gap_thresh=30):
    """
    OCR 결과를 줄 단위로 정렬 및 병합하고 시각화 이미지 생성
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

    # Y좌표 기준 정렬
    lines.sort(key=lambda t: (t["cy"], t["cx"]))

    # 줄 그룹화
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
    color_list = [
        (0, 255, 0), (255, 255, 0), (0, 255, 255),
        (255, 128, 0), (255, 0, 255), (0, 128, 255)
    ]

    # 줄 단위 병합 및 시각화
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

        # 시각화
        for word in line:
            pts = np.array(word["box"], np.int32)
            cv2.polylines(image, [pts], isClosed=True,
                          color=color_list[line_idx % len(color_list)], thickness=2)

        y_pos = int(line[0]["cy"]) - 10
        cv2.putText(image, f"{line_idx}. {merged_text}",
                    (int(line[0]["x_min"]), y_pos),
                    font, 0.7, (0, 0, 255), 2)

        # 결과 리스트 추가
        merged_results.append({
            "line_index": line_idx,
            "text": merged_text,
            "avg_conf": float(np.mean([w["conf"] for w in line]))
        })

    return merged_results, image


# ------------------- 실행 -------------------
if __name__ == "__main__":
    print("🔍 OCR 분석 시작...")

    img = cv2.imread(IMG_PATH)
    result = ocr.ocr(IMG_PATH, cls=False)

    merged_results, vis_img = merge_words_with_boxes(img, result[0])

    # 결과 출력
    print("\n🧩 인식 순서대로 정렬된 텍스트:")
    for line in merged_results:
        print(f"{line['line_index']}. {line['text']} (정확도: {line['avg_conf']:.2f})")

    # 시각화 이미지 저장
    cv2.imwrite(OUTPUT_IMG_PATH, vis_img)
    print(f"\n✅ 시각화 결과 저장 완료: {OUTPUT_IMG_PATH}")

    # JSON 저장
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=4)
    print(f"✅ JSON 결과 저장 완료: {OUTPUT_JSON_PATH}")
