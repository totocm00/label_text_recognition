import cv2
import numpy as np
import os
import json
import time
from paddleocr import PaddleOCR

# ------------------- 상수 설정 -------------------
# 카메라 장치 번호 (매직 넘버)
CAMERA_ID = 0      # 0 또는 4 사용, 필요하면 여기만 바꿔

# 해상도
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# 출력 루트 디렉터리 (여기만 바꾸면 전체 경로가 바뀜)
OUTPUT_ROOT = os.path.join("..", "assets", "test_index_output_json_and_images")

# 파일 이름 기본값
OUTPUT_IMG_BASENAME = "test_camera.jpeg"
OUTPUT_JSON_BASENAME = "test_camera_result.json"

# OCR
OCR_CONF_THRESH = 0.5

# 디렉터리 생성
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# PaddleOCR 초기화
ocr = PaddleOCR(lang="en")


# ------------------- OCR 병합 함수 -------------------
def merge_words_with_boxes(image, ocr_result, y_thresh=20, x_gap_thresh=30):
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

        merged_results.append({
            "line_index": line_idx,
            "text": merged_text,
            "avg_conf": float(np.mean([w["conf"] for w in line]))
        })

    return merged_results, image


# ------------------- 메인 카메라 루프 -------------------
def main():
    print(f"🎥 카메라 {CAMERA_ID} 번 장치 시작 중...")
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"❌ 카메라 {CAMERA_ID}를 열 수 없습니다.")
        return

    print("✅ 실시간 OCR 시작 (캡처: SPACE / 종료: Q)")
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 프레임을 읽을 수 없습니다.")
            break

        display = frame.copy()
        cv2.putText(display, f"Camera {CAMERA_ID} | Press [SPACE] to OCR, [Q] to Quit",
                    (10, 30), font, 0.6, (255, 255, 255), 2)
        cv2.imshow("Camera OCR Live", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == 32:  # Spacebar
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            print(f"\n📸 이미지 캡처 ({timestamp}) → OCR 시작 중...")
            frame_copy = frame.copy()

            ocr_result = ocr.ocr(frame_copy, cls=False)
            merged_results, vis_img = merge_words_with_boxes(frame_copy, ocr_result[0])

            # 타임스탬프 붙인 출력 경로 만들기
            img_out = os.path.join(
                OUTPUT_ROOT,
                OUTPUT_IMG_BASENAME.replace("test_camera", f"capture_{timestamp}")
            )
            json_out = os.path.join(
                OUTPUT_ROOT,
                OUTPUT_JSON_BASENAME.replace("test_camera", f"capture_{timestamp}")
            )

            cv2.imwrite(img_out, vis_img)
            with open(json_out, "w", encoding="utf-8") as f:
                json.dump(merged_results, f, ensure_ascii=False, indent=4)

            print(f"✅ 결과 저장 완료:\n- 이미지: {img_out}\n- JSON: {json_out}")
            for idx, line in enumerate(merged_results, start=1):
                print(f"{idx}. {line['text']} (정확도: {line['avg_conf']:.2f})")

    cap.release()
    cv2.destroyAllWindows()
    print("🟢 종료되었습니다.")


if __name__ == "__main__":
    main()
