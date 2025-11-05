# pcb_orc_camera.py
import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
import time

# ------------------- 설정 -------------------
CAM_ID = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

LOWER_GREEN = np.array([35, 40, 40])
UPPER_GREEN = np.array([95, 255, 255])
MIN_CONTOUR_AREA = 15000
MAX_CONTOUR_AREA = FRAME_WIDTH * FRAME_HEIGHT * 0.95

OCR_CONF_THRESH = 0.5
SERIAL_REGEX = re.compile(r"[A-Z0-9\-]{4,20}", re.I)

ocr = PaddleOCR(lang="en")

# ------------------- 함수 -------------------
def crop_rotated_rect(img, rect, pad_ratio=0.08):
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    width, height = int(rect[1][0]), int(rect[1][1])
    if width <= 0 or height <= 0:
        return None
    src_pts = box.astype("float32")
    dst_pts = np.array(
        [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))
    if pad_ratio:
        h, w = warped.shape[:2]
        padx, pady = int(w * pad_ratio), int(h * pad_ratio)
        warped = cv2.copyMakeBorder(
            warped, pady, pady, padx, padx, cv2.BORDER_REPLICATE
        )
    return warped


def is_valid_text(txt):
    return bool(SERIAL_REGEX.search(txt))


# ------------------- 메인 루프 -------------------
def main():
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    print("✅ PCB OCR 실시간 감지 시작... (종료: q / 일시정지: f)")

    last_time = time.time()
    fps = 0
    detected_texts = set()
    freeze_mode = False        # f키로 토글
    last_detect_time = None    # 마지막 감지 시각
    auto_reset_delay = 10      # 감지 후 재시작까지 시간(초)

    while True:
        if not freeze_mode:
            ret, frame = cap.read()
            if not ret:
                break
        vis = frame.copy()

        # ---------------- 감지 후 10초 지나면 리셋 ----------------
        if last_detect_time and time.time() - last_detect_time > auto_reset_delay:
            print("🔁 자동 리셋: 감지 상태 초기화")
            detected_texts.clear()
            last_detect_time = None

        # ---------------- 색상 기반 감지 ----------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue
            rect = cv2.minAreaRect(cnt)
            cropped = crop_rotated_rect(frame, rect)
            if cropped is None:
                continue

            # OCR 전 3채널 보장
            if len(cropped.shape) == 2:
                cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

            ocr_res = ocr.ocr(cropped, cls=False)
            for line in ocr_res[0]:
                txt, conf = line[1][0], line[1][1]
                if conf < OCR_CONF_THRESH:
                    continue
                if is_valid_text(txt):
                    if txt not in detected_texts:
                        detected_texts.add(txt)
                        last_detect_time = time.time()  # 마지막 감지 시각 기록
                        print(f"📘 감지됨: {txt} (정확도: {conf:.2f})")
                    box = cv2.boxPoints(rect).astype(int)
                    cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)
                    cv2.putText(
                        vis, txt, (int(rect[0][0]), int(rect[0][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )

        # ---------------- FPS / 상태 표시 ----------------
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (now - last_time)) if now != last_time else fps
        last_time = now
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if freeze_mode:
            cv2.putText(vis, "PAUSED (Press F to Resume)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("PCB OCR", vis)

        # ---------------- 키보드 입력 ----------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            freeze_mode = not freeze_mode
            print("⏸ 일시정지" if freeze_mode else "▶ 재개")

    cap.release()
    cv2.destroyAllWindows()
    print("🟢 종료되었습니다.")


if __name__ == "__main__":
    main()
