# ==========================================================
# 📸 camera_loop 모듈
# ----------------------------------------------------------
# 기능 요약:
#   - 실시간으로 웹캠 화면을 띄우며, [SPACE] 누를 때 OCR을 수행합니다.
#   - OCR 결과는 박스와 텍스트가 그려진 이미지 + JSON 파일로 저장됩니다.
#   - [q]를 누르면 종료합니다.
#
# 주요 특징:
#   ✅ 한글 깨짐(????) 문제 완전 해결 (Pillow 기반 draw_korean_text 적용)
#   ✅ 선명도(Definition) 계산 및 시각 표시
#   ✅ 카메라 자동 감지(auto) 지원
#   ✅ YAML의 enable_* / visualize / export_options.* 옵션으로
#      - 저장 여부
#      - 실시간 B박스 표시
#      - 실시간 B박스 좌표 표시
#      - 디버그용 B박스 이미지 저장
#      를 한 번에 ON/OFF 제어
#
# 사용법:
#   1. ocr_config.yaml 설정값을 조정합니다.
#      - enable_save_output: false → 결과 파일 저장 안 함
#      - enable_console_log: true  → 터미널에 OCR 로그 표시
#      - visualize.draw_bbox_on_live: true  → 실시간 화면에 B박스 표시
#      - visualize.show_bbox_coords_on_live: true → 실시간 화면에 좌표 표시
#      - export_options.debug_image.enabled: true → B박스가 그려진 디버그 이미지 저장
#   2. 터미널에서 실행:
#        python demos/camera_ocr_demo.py
#   3. 실행 중:
#        [SPACE] → 캡처 및 OCR 실행
#        [q]     → 종료
#
# 작성 목적:
#   - 현장용 "OCR 확인용 카메라 데모"로 안정적 테스트를 수행하기 위함.
#   - open_vision_factory의 label_text_recognition 서브모듈 기반 데모.
# ==========================================================

import os
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from label_text_recognition.config.loader import load_ocr_config
from label_text_recognition.ocr.ocr_engine import build_ocr_engines
from label_text_recognition.ocr.ocr_runner import run_ocr_on_image
from label_text_recognition.exporters.json_exporter import export_to_json
from label_text_recognition.camera.camera_initializer import init_camera


# ==========================================================
# 🧩 1️⃣ 한글 텍스트 렌더링 함수
# ----------------------------------------------------------
# OpenCV(cv2.putText)는 기본 폰트만 지원하기 때문에 한글이 깨집니다.
# Pillow(PIL)을 이용하여 한글 폰트를 로드하고, 텍스트를 이미지에 그립니다.
# ==========================================================
def draw_korean_text(
    img_bgr,
    text,
    x,
    y,
    font_path="/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    font_size=20,
    color=(0, 255, 0),
):
    """
    OpenCV가 한글을 지원하지 않아 PIL로 텍스트를 표시하는 함수.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        print("⚠️ NotoSans 폰트를 찾지 못했습니다. 기본 폰트를 사용합니다.")
        font = ImageFont.load_default()

    draw.text((x, y), text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ==========================================================
# 🧮 2️⃣ 선명도(Definition) 계산 함수
# ----------------------------------------------------------
# 이미지의 라플라시안 분산(Laplacian Variance)을 이용해 초점 흐림을 측정합니다.
# 값이 높을수록 선명하고, 낮을수록 흐립니다.
# 화면 상단의 Definition 표시와 품질 경고 기준으로 사용됩니다.
# ==========================================================
def get_definition_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


# ==========================================================
# 🚀 3️⃣ 메인 함수: start_camera_ocr()
# ----------------------------------------------------------
# 프로그램 진입점.
#   [SPACE] → OCR 수행 및 결과 저장
#   [q]     → 종료
#
# YAML 설정값을 불러와 enable_* / visualize / export_options.* 토글을
# 기반으로 기능을 제어합니다.
# ----------------------------------------------------------
# 사용 예시:
#   - enable_save_output: false → 폴더 미생성 및 저장 비활성화
#   - enable_console_log: false → 터미널 로그 최소화
#   - visualize.draw_bbox_on_live: true → 실시간 B박스 모드 (테스트용)
#   - visualize.show_bbox_coords_on_live: true → 실시간 좌표 표시 모드 (테스트용)
# ==========================================================
def start_camera_ocr() -> None:
    """실시간 카메라 OCR 데모 실행"""

    # ------------------------------------------------------
    # 1️⃣ 설정 로드 및 기본 파라미터
    # ------------------------------------------------------
    cfg = load_ocr_config()
    conf_threshold = cfg.get("conf_threshold", 0.5)
    definition_threshold = cfg.get("definition_threshold", 200)
    cls_enable = cfg.get("ocr_cls_enable", True)

    # YAML 기반 기능 토글 (기존)
    enable_definition_overlay = cfg.get("enable_definition_overlay", True)
    enable_console_log = cfg.get("enable_console_log", True)
    enable_save_output = cfg.get("enable_save_output", True)
    enable_retry_on_error = cfg.get("enable_retry_on_error", False)

    # 시각화 옵션(visualize 섹션) - 없으면 기본값 사용
    visualize_cfg = cfg.get("visualize", {})
    show_live_preview = visualize_cfg.get("show_live_preview", True)
    draw_bbox_on_live = visualize_cfg.get("draw_bbox_on_live", False)
    show_bbox_coords_on_live = visualize_cfg.get("show_bbox_coords_on_live", False)
    # show_definition_on_live 는 기존 enable_definition_overlay 와 호환되도록 구성
    show_definition_on_live = visualize_cfg.get(
        "show_definition_on_live", enable_definition_overlay
    )

    # 디버그용 B박스 이미지 저장 옵션 (export_options.debug_image)
    export_options = cfg.get("export_options", {})
    debug_image_cfg = export_options.get("debug_image", {})
    debug_image_enabled = debug_image_cfg.get("enabled", False)
    debug_image_dir = debug_image_cfg.get("path", "assets/debug_images")
    debug_image_pattern = debug_image_cfg.get("filename_pattern", "debug_{ts}.png")

    # 출력 경로 설정 (기존 기본 경로)
    out_img_dir = cfg.get("output_dir_images", "assets/pictures")
    out_img_origin_dir = cfg.get("output_dir_images_origin", "assets/pictures-origin")
    out_json_dir = cfg.get("output_dir_json", "assets/json")

    # 저장 기능이 켜져 있을 때만 폴더 생성
    if enable_save_output:
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_img_origin_dir, exist_ok=True)
        os.makedirs(out_json_dir, exist_ok=True)
        if debug_image_enabled:
            os.makedirs(debug_image_dir, exist_ok=True)
    else:
        print("💾 [비활성화] enable_save_output: false → 폴더 생성/저장 비활성화")

    # ------------------------------------------------------
    # 2️⃣ OCR 엔진 초기화
    # ------------------------------------------------------
    ocr_langs = cfg.get("ocr_langs", ["en"])
    ocr_engines = build_ocr_engines(ocr_langs)
    main_engine = ocr_engines[ocr_langs[0]]

    # ------------------------------------------------------
    # 3️⃣ 카메라 열기
    # ------------------------------------------------------
    cap = init_camera(cfg)
    if cap is None:
        print("❌ 카메라를 열 수 없습니다.")
        return

    print("✅ Camera OCR ready")
    print("   [SPACE] → OCR 실행 / [q] → 종료")

    font = cv2.FONT_HERSHEY_SIMPLEX

    # 직전 OCR 결과를 저장해두는 변수
    # → 실시간 화면에서 B박스/좌표를 다시 그릴 때 사용
    last_results = []
    last_def_score = 0.0

    # ------------------------------------------------------
    # 4️⃣ 메인 루프: 실시간 영상 처리
    # ------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 프레임을 읽을 수 없습니다. 카메라 연결을 확인하세요.")
            break

        # 현재 프레임 선명도 계산
        live_def = get_definition_score(frame)
        display = frame.copy()

        # 4-1) 화면 안내 문구
        cv2.putText(display, "Press [SPACE] to OCR, [q] to quit",
                    (10, 30), font, 0.6, (255, 255, 255), 2)

        # 4-2) Definition 표시 (시각화 옵션 기반)
        if show_definition_on_live:
            color = (0, 255, 0) if live_def >= definition_threshold else (0, 0, 255)
            cv2.putText(display,
                        f"Definition: {live_def:.1f} (th={definition_threshold})",
                        (10, 60), font, 0.55, color, 2)

        # 4-3) 실시간 B박스 + 좌표 표시 (테스트/디버깅용)
        #  - last_results 는 마지막으로 SPACE 눌렀을 때의 OCR 결과입니다.
        #  - draw_bbox_on_live: B박스 폴리라인 표시 여부
        #  - show_bbox_coords_on_live: 각 박스의 중심 좌표를 텍스트로 표시 여부
        if draw_bbox_on_live and last_results:
            for idx, item in enumerate(last_results):
                box = item.get("box", None)
                if not box or len(box) < 4:
                    continue

                # box: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                if show_bbox_coords_on_live:
                    # 중심점 계산 (4점 평균)
                    cx = int(sum(p[0] for p in box) / len(box))
                    cy = int(sum(p[1] for p in box) / len(box))

                    label = f"#{idx} ({cx}, {cy})"
                    cv2.putText(
                        display,
                        label,
                        (cx, cy - 5),
                        font,
                        0.4,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

        # 카메라 화면 표시 (visualize.show_live_preview)
        if show_live_preview:
            cv2.imshow("Label Text Recognition - Camera", display)
        else:
            # 실험적으로 화면 표시를 끄고 싶을 경우를 대비한 옵션
            # (여기서는 imshow를 호출하지 않음)
            pass

        key = cv2.waitKey(1) & 0xFF

        # 종료
        if key == ord("q"):
            break

        # --------------------------------------------------
        # 🟢 [SPACE] 누르면 OCR 실행
        # --------------------------------------------------
        if key == 32:  # space
            ts = time.strftime("%Y%m%d_%H%M%S")
            print(f"\n📸 {ts} - OCR 실행 중...")
            def_score = live_def

            # 1) OCR 수행
            results, vis_img, msg = run_ocr_on_image(
                frame.copy(), main_engine, conf_threshold, cls_enable
            )

            # 2) 오류 시 재시도 (토글)
            if msg.startswith("ERROR") and enable_retry_on_error:
                print("⚠️ OCR 오류 발생 → 1회 재시도")
                results, vis_img, msg = run_ocr_on_image(
                    frame.copy(), main_engine, conf_threshold, cls_enable
                )

            # 3) 결과 시각화 (박스 + 텍스트)
            #    - vis_img 위에 B박스를 그리고, 한글 텍스트 + 신뢰도를 함께 표시
            for r in results:
                box = r.get("box", [])
                text = r.get("text", "")
                avg_conf = r.get("avg_conf", 0.0)

                if box:
                    x1, y1 = int(box[0][0]), int(box[0][1])
                    x2, y2 = int(box[2][0]), int(box[2][1])
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2),
                                  (0, 255, 255), 2)
                    vis_img = draw_korean_text(
                        vis_img,
                        f"{text} ({avg_conf:.2f})",
                        x1, y1 - 22,
                        font_size=20, color=(255, 0, 0)
                    )

            # 마지막 결과를 저장해 두었다가
            # 실시간 화면에서 B박스/좌표를 다시 그릴 때 사용
            last_results = results
            last_def_score = def_score

            # 4) 저장 경로 지정 (기존 + JSON 경로)
            img_path_origin = os.path.join(out_img_origin_dir, f"capture_{ts}.jpg")
            img_path = os.path.join(out_img_dir, f"capture_{ts}.jpg")
            json_path = os.path.join(out_json_dir, f"capture_{ts}.json")

            # 5) 저장 (enable_save_output 기반)
            if enable_save_output:
                # 원본 이미지 저장
                cv2.imwrite(img_path_origin, frame)
                # vis_img (B박스 + 텍스트 그려진 결과) 저장
                cv2.imwrite(img_path, vis_img)
                # JSON 저장 (export_to_json 은 내부에서 config 기반 export_all_json 호출)
                export_to_json(results, json_path)

                # 디버그용 B박스 이미지 저장 (선택 사항)
                if debug_image_enabled:
                    # vis_img 위에 좌표/인덱스를 추가로 그려서 저장해도 되고,
                    # frame 기준으로 다시 그려도 됨. 여기서는 vis_img 기준으로 저장.
                    debug_frame = vis_img.copy()
                    for idx, r in enumerate(results):
                        box = r.get("box", [])
                        if not box or len(box) < 4:
                            continue
                        cx = int(sum(p[0] for p in box) / len(box))
                        cy = int(sum(p[1] for p in box) / len(box))
                        label = f"#{idx} ({cx},{cy})"
                        cv2.putText(
                            debug_frame,
                            label,
                            (cx, cy - 5),
                            font,
                            0.4,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )

                    debug_filename = debug_image_pattern.replace("{ts}", ts)
                    debug_path = os.path.join(debug_image_dir, debug_filename)
                    cv2.imwrite(debug_path, debug_frame)
                    print(f"🟩 디버그 B박스 이미지 저장: {debug_path}")

                print(
                    "✅ 결과 저장 완료:\n"
                    f"   - {img_path_origin}\n"
                    f"   - {img_path}\n"
                    f"   - {json_path}"
                )
            else:
                print("💾 저장 비활성화 상태이므로 파일은 생성되지 않습니다.")

            # 6) 콘솔 로그 (enable_console_log)
            if not results:
                if enable_console_log:
                    print(f"⚠️ OCR 결과 없음. Definition={def_score:.2f}")
                continue

            confs = [r.get("avg_conf", 0.0) for r in results]
            overall_conf = sum(confs) / len(confs)

            if enable_console_log:
                for r in results:
                    print(f"- {r.get('text', '')} ({r.get('avg_conf', 0.0):.2f})")
                print(f"📈 평균 신뢰도: {overall_conf:.2f}")

                if def_score < definition_threshold:
                    print("⚠️ 이미지가 다소 흐립니다.")
                elif overall_conf < conf_threshold:
                    print("⚠️ 인식은 되었으나 신뢰도가 낮습니다.")
                else:
                    print("✅ 선명도와 인식률 모두 양호합니다.")

    # ------------------------------------------------------
    # 5️⃣ 종료 처리
    # ------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()
    print("🟢 OCR 세션을 정상 종료했습니다.")