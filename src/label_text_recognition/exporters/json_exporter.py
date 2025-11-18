# ==========================================================
# json_exporter.py  (개선/확장 버전)
# ----------------------------------------------------------
# 역할:
#   - OCR 결과를 JSON 파일로 저장하는 Exporter.
#   - ocr_config.yaml 의 export_options.* 값을 반영하여
#       1) 텍스트 JSON 저장 여부
#       2) bbox JSON 저장 여부
#       3) merge_with_text_json 여부
#       4) 저장 경로(path), 파일명 패턴(filename_pattern)
#     을 모두 제어함.
#
# 특징:
#   - camera_loop / ocr_runner 등이 어떤 경로를 넘겨줘도,
#     이 파일 내부에서 다시 config 정보를 기준으로
#     '최종 저장 위치와 파일명'을 결정함.
#
#   - enable_save_output: false 이면
#       → 어떤 JSON도 생성하지 않고 안내 메시지만 출력.
#
#   - merge_with_text_json: true 이면
#       → bbox 데이터를 텍스트 JSON 내부에 통합하여
#         하나의 JSON 파일로 저장함.
#
#   - export_options.debug_image 등 다른 옵션 확장에도 대비됨.
#
# ==========================================================

import os
import json
from datetime import datetime
from typing import Any, List, Dict

# 프로젝트 공통 설정 로더
from label_text_recognition.config.loader import load_ocr_config


# ----------------------------------------------------------
# (도우미) 타임스탬프 생성기
# ----------------------------------------------------------
def _timestamp() -> str:
    """
    현재 시각을 'YYYYMMDD_HHMMSS' 형식으로 반환합니다.
    JSON 파일 이름 패턴에서 {ts}를 치환할 때 사용됩니다.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ----------------------------------------------------------
# (핵심) 텍스트 JSON 저장 함수
# ----------------------------------------------------------
def _save_text_json(results: List[Dict[str, Any]], cfg: dict) -> str:
    """
    텍스트 JSON을 저장합니다.

    Parameters
    ----------
    results : list
        OCR 결과 리스트. (text / avg_conf / box 포함)
    cfg : dict
        전체 OCR 설정 객체 (ocr_config.yaml 내용)

    Returns
    -------
    output_path : str
        저장된 텍스트 JSON의 전체 경로. (merge 시 bbox 병합용)
    """

    text_cfg = cfg["export_options"]["text_json"]
    enabled = text_cfg.get("enabled", True)
    if not enabled:
        print("💾 텍스트 JSON 저장이 비활성화되어 있어 생성하지 않습니다.")
        return ""

    # enable_save_output 이 false면 어떤 JSON도 생성하지 않음
    if not cfg.get("enable_save_output", True):
        print("💾 enable_save_output=false → 텍스트 JSON 생성 취소")
        return ""

    # 저장 경로/파일명 결정
    ts = _timestamp()
    out_dir = text_cfg["path"]
    filename_pattern = text_cfg.get("filename_pattern", "capture_{ts}.json")
    filename = filename_pattern.replace("{ts}", ts)
    output_path = os.path.join(out_dir, filename)

    # 폴더 생성
    os.makedirs(out_dir, exist_ok=True)

    # JSON dump
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"✅ 텍스트 JSON 저장 완료: {output_path}")
    return output_path


# ----------------------------------------------------------
# (핵심) bbox JSON 저장 함수
# ----------------------------------------------------------
def _save_bbox_json(results: List[Dict[str, Any]], cfg: dict) -> str:
    """
    바운딩 박스 JSON을 저장합니다.

    Parameters
    ----------
    results : list
        OCR 결과 리스트. 각 항목에 ["text", "avg_conf", "box"]가 포함되어야 함.
    cfg : dict
        전체 OCR 설정 객체

    Returns
    -------
    output_path : str
        저장된 bbox JSON 경로
    """

    bbox_cfg = cfg["export_options"]["bbox_json"]
    enabled = bbox_cfg.get("enabled", True)
    if not enabled:
        print("🟦 bbox_json.enabled=false → 바운딩 박스 JSON 생성하지 않음.")
        return ""

    # enable_save_output 확인
    if not cfg.get("enable_save_output", True):
        print("💾 enable_save_output=false → bbox JSON 생성 취소")
        return ""

    # 저장 경로/파일명
    ts = _timestamp()
    out_dir = bbox_cfg["path"]
    filename_pattern = bbox_cfg.get("filename_pattern", "bbox_{ts}.json")
    filename = filename_pattern.replace("{ts}", ts)
    output_path = os.path.join(out_dir, filename)

    # bbox 데이터만 추출
    bbox_only = []
    for idx, item in enumerate(results):
        bbox_only.append({
            "id": idx,
            "text": item.get("text", ""),
            "confidence": item.get("avg_conf", 0.0),
            "bbox": item.get("box", []),   # [[x1,y1],...]
        })

    # 폴더 생성
    os.makedirs(out_dir, exist_ok=True)

    # JSON 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bbox_only, f, ensure_ascii=False, indent=4)

    print(f"🟦 바운딩 박스 JSON 저장 완료: {output_path}")
    return output_path


# ----------------------------------------------------------
# (메인 API) export_all_json
# ----------------------------------------------------------
def export_all_json(results: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    텍스트 JSON, 바운딩 박스 JSON을 config 기반으로 처리하여 저장합니다.

    Parameters
    ----------
    results : list
        OCR 결과 리스트.
        예: [{"text": "...", "avg_conf": 0.92, "box": [[x1,y1], ...]}, ...]

    Returns
    -------
    dict
        {
          "text_json": "assets/json/capture_....json",
          "bbox_json": "assets/json_bbox/bbox_....json"
        }
        (해당 항목이 비활성화된 경우 빈 문자열 반환)
    """

    cfg = load_ocr_config()

    # 저장 OFF면 전체 JSON 동작을 막음
    if not cfg.get("enable_save_output", True):
        print("💾 enable_save_output=false → 모든 JSON 저장 비활성화")
        return {"text_json": "", "bbox_json": ""}

    text_cfg = cfg["export_options"]["text_json"]
    bbox_cfg = cfg["export_options"]["bbox_json"]

    merge = bbox_cfg.get("merge_with_text_json", False)

    # ------------------------------------------------------
    # 1) 텍스트 JSON 먼저 저장
    # ------------------------------------------------------
    txt_json_path = ""
    if text_cfg.get("enabled", True):
        txt_json_path = _save_text_json(results, cfg)

    # ------------------------------------------------------
    # 2) bbox JSON (단독 또는 merge)
    # ------------------------------------------------------
    bbox_json_path = ""

    if bbox_cfg.get("enabled", True):

        if merge and txt_json_path:
            # --------------------------------------------------
            # 🔗 merge_with_text_json = true
            # → 텍스트 JSON 내부에 bbox 데이터만 append
            # --------------------------------------------------
            print("🔗 merge_with_text_json=true → 텍스트 JSON 안에 bbox 데이터 병합")

            # bbox_only 구성
            bbox_only = []
            for idx, item in enumerate(results):
                bbox_only.append({
                    "id": idx,
                    "text": item.get("text", ""),
                    "confidence": item.get("avg_conf", 0.0),
                    "bbox": item.get("box", []),
                })

            # 텍스트 JSON 읽기 → 병합 → 다시 저장
            with open(txt_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            data = {
                "results": data,      # 기존 텍스트 결과
                "bbox": bbox_only     # 추가 bbox 결과
            }

            with open(txt_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            bbox_json_path = txt_json_path  # 하나의 파일로 통합
            print(f"🔗 bbox 데이터가 텍스트 JSON에 병합되었습니다 → {txt_json_path}")

        else:
            # --------------------------------------------------
            # 별도 저장
            # --------------------------------------------------
            bbox_json_path = _save_bbox_json(results, cfg)

    return {
        "text_json": txt_json_path,
        "bbox_json": bbox_json_path
    }
