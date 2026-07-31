import os
import sys
import json
import datetime
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# 같은 폴더의 모듈 import
sys.path.insert(0, os.path.dirname(__file__))
from detector import load_detector, detect_plates
from preprocessor import preprocess_plate
from recognizer import load_ocr, recognize_text
from postprocessor import postprocess, format_plate_text

# =========================================
# 페이지 설정
# =========================================
st.set_page_config(page_title="번호판 인식 시스템", layout="wide")
st.title("🚗 한국 자동차 번호판 인식 (ALPR)")

# =========================================
# 모델 로드 (캐싱)
# =========================================
@st.cache_resource
def init_detector():
    """YOLOv8 모델 로드."""
    return load_detector()

@st.cache_resource
def init_ocr():
    """PaddleOCR 한국어 모델 로드."""
    return load_ocr()

with st.spinner("모델 로딩 중..."):
    yolo_model = init_detector()
    ocr_model = init_ocr()

# =========================================
# 세션 상태 초기화 (입/출차 기록용)
# =========================================
if "parking_records" not in st.session_state:
    st.session_state.parking_records = []

# =========================================
# 공통 파이프라인 함수
# =========================================
def run_pipeline(image: np.ndarray) -> list[dict]:
    """이미지 → 번호판 검출 → 전처리 → OCR → 후처리 전체 파이프라인."""
    results = []

    # Stage 1: 번호판 검출
    detections = detect_plates(yolo_model, image)

    for det in detections:
        # Stage 2: 전처리
        preprocess_result = preprocess_plate(det.cropped)

        # Stage 3: OCR (원본 크롭으로 시도)
        ocr_result = recognize_text(ocr_model, det.cropped)

        # 원본으로 인식 실패 시 전처리된 이미지로 재시도
        if not ocr_result.text:
            ocr_result = recognize_text(ocr_model, preprocess_result.denoised)

        # Stage 4: 후처리
        post_result = postprocess(ocr_result.text)

        results.append({
            "detection": det,
            "preprocess": preprocess_result,
            "ocr": ocr_result,
            "post": post_result,
        })

    return results

def draw_detections(image: np.ndarray, results: list[dict]) -> np.ndarray:
    """원본 이미지에 검출 결과를 시각화한다."""
    annotated = image.copy()
    for r in results:
        x1, y1, x2, y2 = r["detection"].bbox
        text = r["post"]["cleaned"] or "인식 실패"
        conf = r["detection"].confidence

        # 바운딩박스
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 라벨 배경
        label = f"{text} ({conf:.0%})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return annotated

# =========================================
# 탭 구성
# =========================================
tab1, tab2, tab3 = st.tabs(["🔍 번호판 인식", "📋 샘플 테스트", "🅿️ 입/출차 기록"])

# =========================================
# 탭 1: 번호판 인식
# =========================================
with tab1:
    st.markdown("차량 이미지를 업로드하면 번호판을 검출하고 텍스트를 인식합니다.")

    upload_col, camera_col = st.columns(2)
    with upload_col:
        uploaded = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png", "webp"])
    with camera_col:
        camera = st.camera_input("카메라 촬영")

    # 입력 이미지 결정
    input_source = uploaded or camera

    if input_source is not None:
        # 이미지 로드
        file_bytes = np.frombuffer(input_source.read(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("이미지를 읽을 수 없습니다.")
            st.stop()

        # 파이프라인 실행
        with st.spinner("번호판 분석 중..."):
            results = run_pipeline(image)

        if not results:
            st.warning("번호판을 검출하지 못했습니다. 다른 이미지를 시도해주세요.")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="원본 이미지")
            st.stop()

        # ----- 검출 결과 시각화 -----
        annotated = draw_detections(image, results)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="검출 결과", use_container_width=True)

        # ----- 각 번호판별 상세 결과 -----
        for i, r in enumerate(results):
            st.divider()
            st.subheader(f"번호판 #{i+1}")

            # 인식 결과
            post = r["post"]
            ocr = r["ocr"]

            if post["valid"]:
                formatted = format_plate_text(post["parts"], post["format"])
                st.success(f"**인식 결과: {formatted}**  (신뢰도: {ocr.confidence:.1%}, 포맷: {post['format']})")
            elif post["cleaned"]:
                st.warning(f"**인식 결과: {post['cleaned']}**  (포맷 검증 실패, 신뢰도: {ocr.confidence:.1%})")
            else:
                st.error("텍스트를 인식하지 못했습니다.")

            method = r["detection"].method
            if method == "yolo":
                st.caption("검출 경로: 🎯 YOLO (번호판 전용 모델)")
            else:
                st.caption("검출 경로: 🔧 OpenCV 폴백 (YOLO가 못 찾아서 에지 검출로 대체)")

            # 파이프라인 단계별 시각화
            with st.expander("🔬 파이프라인 단계별 보기"):
                prep = r["preprocess"]
                cols = st.columns(5)

                steps = [
                    ("1. 원본 크롭", prep.original),
                    ("2. 리사이즈", prep.resized),
                    ("3. 그레이스케일", prep.gray),
                    ("4. 이진화", prep.binary),
                    ("5. 노이즈 제거", prep.denoised),
                ]
                for col, (label, img) in zip(cols, steps):
                    with col:
                        display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
                        st.image(display, caption=label, use_container_width=True)

            # OCR 상세
            if ocr.details:
                with st.expander("📝 OCR 상세 결과"):
                    for bbox, text, conf in ocr.details:
                        st.write(f"- `{text}` (신뢰도: {conf:.1%})")

    else:
        st.info("이미지를 업로드하거나 카메라로 촬영하세요.")

# =========================================
# 탭 2: 샘플 테스트
# =========================================
with tab2:
    st.markdown("샘플 이미지로 인식률을 테스트합니다.")

    sample_dir = os.path.join(os.path.dirname(__file__), "samples")
    answers_path = os.path.join(sample_dir, "answers.json")
    # 번들 샘플은 answers.json에 정답을 담는다(파일명은 Windows 한글 인코딩 문제를
    # 피하려 영문으로만 구성). answers.json에 없는 파일(직접 추가한 샘플)은 기존
    # 방식대로 파일명 자체를 정답으로 쓴다 — 두 방식을 병행 지원한다.
    known_answers = {}
    if os.path.exists(answers_path):
        with open(answers_path, "r", encoding="utf-8") as f:
            known_answers = json.load(f)

    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir, exist_ok=True)
        st.info(f"📁 `plate/samples/` 폴더에 번호판 이미지를 넣어주세요.")
        st.markdown("""
        **샘플 이미지 준비 방법:**
        1. 번호판이 보이는 차량 사진을 `plate/samples/` 폴더에 저장
        2. 파일명에 정답을 포함 (예: `123가4567.jpg`, `서울12나3456.png`)
           — 또는 `samples/answers.json`에 `{"파일명.jpg": "정답"}` 형태로 추가
        3. 이 탭에서 "테스트 실행" 클릭
        """)
    else:
        sample_files = [f for f in os.listdir(sample_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

        if not sample_files:
            st.info("📁 `plate/samples/` 폴더에 번호판 이미지를 넣어주세요.")
        else:
            st.write(f"**{len(sample_files)}개** 샘플 이미지 발견")

            if st.button("🚀 전체 샘플 테스트 실행"):
                total = len(sample_files)
                progress = st.progress(0)
                results_summary = []

                for idx, fname in enumerate(sorted(sample_files)):
                    fpath = os.path.join(sample_dir, fname)
                    img = cv2.imread(fpath)
                    if img is None:
                        continue

                    results = run_pipeline(img)
                    recognized = results[0]["post"]["cleaned"] if results else ""

                    # answers.json에 있으면 그 정답을 쓰고, 없으면 기존처럼 파일명에서 추출
                    answer = known_answers.get(fname, os.path.splitext(fname)[0])
                    answer_clean = answer.replace(" ", "").replace("-", "")

                    is_correct = recognized == answer_clean

                    results_summary.append({
                        "파일": fname,
                        "정답": answer,
                        "인식결과": recognized or "(실패)",
                        "일치": "✅" if is_correct else "❌",
                    })

                    progress.progress((idx + 1) / total)

                # 결과 요약
                df = pd.DataFrame(results_summary)
                correct_count = sum(1 for r in results_summary if r["일치"] == "✅")
                st.metric("인식률", f"{correct_count}/{total} ({correct_count/total*100:.0f}%)")
                st.dataframe(df, use_container_width=True)

# =========================================
# 탭 3: 입/출차 기록
# =========================================
with tab3:
    st.markdown("번호판 인식 결과를 입/출차 기록으로 저장합니다.")

    record_col1, record_col2 = st.columns([2, 1])

    with record_col1:
        plate_input = st.text_input("번호판 번호 (직접 입력 또는 탭 1에서 인식)")

    with record_col2:
        action = st.selectbox("구분", ["입차", "출차"])

    if st.button("📝 기록 추가"):
        if plate_input:
            now = datetime.datetime.now()
            record = {
                "번호판": plate_input,
                "구분": action,
                "시간": now.strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state.parking_records.append(record)
            st.success(f"{action} 기록 완료: {plate_input}")
        else:
            st.warning("번호판 번호를 입력하세요.")

    # 기록 테이블
    if st.session_state.parking_records:
        st.subheader("📊 주차 기록")
        df = pd.DataFrame(st.session_state.parking_records)

        # 주차 시간 계산 (같은 번호판의 입차-출차 매칭)
        display_records = []
        entries = {}  # 번호판별 입차 시간

        for record in st.session_state.parking_records:
            plate = record["번호판"]
            time = record["시간"]

            if record["구분"] == "입차":
                entries[plate] = time
                display_records.append({
                    "번호판": plate,
                    "입차": time,
                    "출차": "-",
                    "주차시간": "-",
                })
            elif record["구분"] == "출차" and plate in entries:
                entry_time = datetime.datetime.strptime(entries[plate], "%Y-%m-%d %H:%M:%S")
                exit_time = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
                duration = exit_time - entry_time
                minutes = int(duration.total_seconds() / 60)

                # 기존 입차 기록 업데이트
                for dr in display_records:
                    if dr["번호판"] == plate and dr["출차"] == "-":
                        dr["출차"] = time
                        dr["주차시간"] = f"{minutes}분"
                        break

                del entries[plate]

        st.dataframe(pd.DataFrame(display_records), use_container_width=True)

        if st.button("🗑️ 기록 초기화"):
            st.session_state.parking_records = []
            st.rerun()
    else:
        st.info("아직 기록이 없습니다.")
