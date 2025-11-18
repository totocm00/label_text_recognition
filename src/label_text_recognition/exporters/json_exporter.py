# ==========================================================
# json_exporter.py  (개선/확장 + 하위 호환 버전)
# ----------------------------------------------------------
# 역할:
#   - OCR 결과를 JSON 파일로 저장하는 Exporter 모듈입니다.
#   - 이 모듈은 "텍스트/좌표/신뢰도" 같은 **데이터(JSON)** 만 담당하고,
#     B박스가 그려진 실제 이미지 저장은 camera_loop.py 에서 처리합니다.
#
#   - ocr_config.yaml 의 export_options.* 값을 반영하여:
#       1) 텍스트 JSON 저장 여부 (text_json.enabled)
#       2) bbox JSON 저장 여부 (bbox_json.enabled)
#       3) bbox를 텍스트 JSON 안에 병합할지 여부 (bbox_json.merge_with_text_json)
#       4) 저장 경로(path), 파일명 패턴(filename_pattern)
#     을 모두 제어할 수 있습니다.
#
# 특징:
#   - camera_loop / ocr_runner 등에서 이 모듈을 호출할 때,
#     예전 함수명(export_to_json)과 새 함수명(export_all_json)을
#     모두 지원합니다.
#
#   - enable_save_output: false 이면
#       → 어떤 JSON도 생성하지 않고 안내 메시지만 출력합니다.
#
#   - merge_with_text_json: true 이면
#       → bbox 데이터를 텍스트 JSON 내부에 통합하여
#         하나의 JSON 파일로 저장합니다.
#
#   - 기존 코드:
#       from label_text_recognition.exporters.json_exporter import export_to_json
#     이 그대로 동작하도록 **하위 호환 래퍼 함수(export_to_json)** 를 제공합니다.
#
# 주의:
#   - 이 파일은 "JSON 데이터"만 다루며,
#     "B박스가 그려진 이미지 저장"은 export_options.debug_image 를 읽어서
#     camera_loop.py 에서 처리하는 구조로 분리하는 것을 권장합니다.
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

    예:
        filename_pattern: "capture_{ts}.json"
        → "capture_20251119_143501.json"
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
        예: [{"text": "시험일", "avg_conf": 0.94, "box": [[x1,y1], ...]}, ...]
    cfg : dict
        전체 OCR 설정 객체 (ocr_config.yaml 내용)

    Returns
    -------
    output_path : str
        저장된 텍스트 JSON의 전체 경로. (merge 시 bbox 병합용)
        저장이 비활성화된 경우 빈 문자열("")을 반환합니다.
    """

    # ocr_config.yaml → export_options.text_json 섹션 읽기
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
    out_dir = text_cfg["path"]  # 예: "assets/json"
    filename_pattern = text_cfg.get("filename_pattern", "capture_{ts}.json")
    filename = filename_pattern.replace("{ts}", ts)
    output_path = os.path.join(out_dir, filename)

    # 폴더 생성 (없으면 자동 생성)
    os.makedirs(out_dir, exist_ok=True)

    # JSON dump
    # - ensure_ascii=False: 한글이 "????"가 아니라 실제 한글로 저장되도록 함
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"✅ 텍스트 JSON 저장 완료: {output_path}")
    return output_path


# ----------------------------------------------------------
# (핵심) bbox JSON 저장 함수
# ----------------------------------------------------------
def _save_bbox_json(results: List[Dict[str, Any]], cfg: dict) -> str:
    """
    바운딩 박스 전용 JSON을 저장합니다.

    Parameters
    ----------
    results : list
        OCR 결과 리스트. 각 항목에 ["text", "avg_conf", "box"]가 포함되어야 함.
        예: [{"text": "시험일", "avg_conf": 0.94, "box": [[x1,y1], ...]}, ...]
    cfg : dict
        전체 OCR 설정 객체

    Returns
    -------
    output_path : str
        저장된 bbox JSON 경로.
        저장이 비활성화되면 빈 문자열("")을 반환합니다.
    """

    bbox_cfg = cfg["export_options"]["bbox_json"]
    enabled = bbox_cfg.get("enabled", True)
    if not enabled:
        print("🟦 bbox_json.enabled=false → 바운딩 박스 JSON 생성하지 않음.")
        return ""

    # enable_save_output 확인 (전역 세이브 스위치)
    if not cfg.get("enable_save_output", True):
        print("💾 enable_save_output=false → bbox JSON 생성 취소")
        return ""

    # 저장 경로/파일명
    ts = _timestamp()
    out_dir = bbox_cfg["path"]  # 예: "assets/json_bbox"
    filename_pattern = bbox_cfg.get("filename_pattern", "bbox_{ts}.json")
    filename = filename_pattern.replace("{ts}", ts)
    output_path = os.path.join(out_dir, filename)

    # bbox 데이터만 추출해서 별도의 구조로 저장
    bbox_only = []
    for idx, item in enumerate(results):
        bbox_only.append({
            "id": idx,
            "text": item.get("text", ""),
            "confidence": item.get("avg_conf", 0.0),
            "bbox": item.get("box", []),   # [[x1,y1], ...]
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

    # ocr_config.yaml 전체 설정 불러오기
    cfg = load_ocr_config()

    # 전역 스위치: enable_save_output=false 이면 모든 JSON 저장을 막음
    if not cfg.get("enable_save_output", True):
        print("💾 enable_save_output=false → 모든 JSON 저장 비활성화")
        return {"text_json": "", "bbox_json": ""}

    text_cfg = cfg["export_options"]["text_json"]
    bbox_cfg = cfg["export_options"]["bbox_json"]

    # bbox_json.merge_with_text_json 옵션
    merge = bbox_cfg.get("merge_with_text_json", False)

    # ------------------------------------------------------
    # 1) 텍스트 JSON 먼저 저장
    # ------------------------------------------------------
    txt_json_path = ""
    if text_cfg.get("enabled", True):
        txt_json_path = _save_text_json(results, cfg)

    # ------------------------------------------------------
    # 2) bbox JSON (단독 저장 또는 텍스트 JSON과 merge)
    # ------------------------------------------------------
    bbox_json_path = ""

    if bbox_cfg.get("enabled", True):

        if merge and txt_json_path:
            # --------------------------------------------------
            # 🔗 merge_with_text_json = true
            # → 텍스트 JSON 내부에 bbox 데이터만 append
            # --------------------------------------------------
            print("🔗 merge_with_text_json=true → 텍스트 JSON 안에 bbox 데이터 병합")

            # bbox_only 구성 (텍스트 + 좌표만 깔끔하게)
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

            # 기존 결과 형태가 list일 수도 있어서 감싸주는 형태로 구성
            # 예전 버전: [ {...}, {...}, ... ]
            # 새 버전: { "results": [...], "bbox": [...] }
            if isinstance(data, list):
                data = {"results": data}

            data["bbox"] = bbox_only

            with open(txt_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            bbox_json_path = txt_json_path  # 하나의 파일로 통합
            print(f"🔗 bbox 데이터가 텍스트 JSON에 병합되었습니다 → {txt_json_path}")

        else:
            # --------------------------------------------------
            # 별도 파일로 bbox JSON 저장
            # --------------------------------------------------
            bbox_json_path = _save_bbox_json(results, cfg)

    return {
        "text_json": txt_json_path,
        "bbox_json": bbox_json_path
    }


# ----------------------------------------------------------
# (하위 호환용) export_to_json
# ----------------------------------------------------------
def export_to_json(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    [하위 호환 래퍼]

    기존 코드에서 사용 중인:
        export_to_json(results, output_path)
    인터페이스를 그대로 유지하기 위해 제공되는 래퍼 함수입니다.

    - 새 구조에서는 output_path 대신
      ocr_config.yaml 내 export_options.text_json.* 설정을 우선 사용합니다.
    - output_path 인자는 현재는 무시되며, 향후 필요시
      fallback 경로로 사용할 수 있습니다.

    Parameters
    ----------
    results : list[dict]
        OCR 결과 리스트.
    output_path : str
        예전 인터페이스에서 사용하던 JSON 저장 경로.
        현재 구현에서는 사용하지 않습니다.
    """

    print(
        "ℹ️ export_to_json() 하위 호환 래퍼 호출됨 "
        "(실제 저장은 export_all_json() 및 ocr_config.yaml 설정을 따름)"
    )

    # 새 config 기반 시스템으로 실제 저장 처리
    export_all_json(results)