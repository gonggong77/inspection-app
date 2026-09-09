"""
app.py  ─  [수업 2단계] Streamlit UI로 .keras 모델 추론하기
════════════════════════════════════════════════════════════════════
infer_keras.py 의 [모델 로드 / 전처리 / 추론] 로직은 그대로 유지하고,
Streamlit 웹 UI(파일 업로드 / 카메라 촬영 / 결과 시각화)를 추가한 버전.

실행 방법:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow import keras


# ─────────────────────────────────────────────────────────────────
# Streamlit 버전 호환 헬퍼
#   st.image()의 "컨테이너 너비에 맞추기" 옵션 이름이 버전마다 다르다.
#     - 구버전 : use_column_width=True
#     - 중간버전: use_container_width=True
#     - 최신버전: width="stretch"  (use_container_width는 deprecated)
#   설치된 streamlit 버전에 관계없이 동작하도록 순서대로 시도한다.
# ─────────────────────────────────────────────────────────────────
def show_image_full_width(pil_img, caption=None):
    try:
        st.image(pil_img, caption=caption, width="stretch")
    except TypeError:
        try:
            st.image(pil_img, caption=caption, use_container_width=True)
        except TypeError:
            try:
                st.image(pil_img, caption=caption, use_column_width=True)
            except TypeError:
                st.image(pil_img, caption=caption)

# ── 설정 ─────────────────────────────────────────────────────────
MODEL_PATH     = "./weights/leather_model.keras"   # .keras 모델 경로
INPUT_IMG_SIZE = (224, 224)
CLASSES        = ["정상", "불량"]


# ─────────────────────────────────────────────────────────────────
# 1. 모델 로드
#    @st.cache_resource → 앱이 다시 실행(rerun)돼도 모델을 매번 새로
#    불러오지 않고, 세션 간에 캐시된 모델 객체를 재사용한다.
#    (모델처럼 직렬화가 안 되는 리소스는 st.cache_data가 아니라
#     st.cache_resource를 쓰는 것이 Streamlit 공식 권장 방식이다.)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="모델을 불러오는 중입니다...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


# ─────────────────────────────────────────────────────────────────
# 2. 이미지 전처리 (기존 로직 그대로)
#    VGG16 학습 때 쓴 preprocess_input 과 동일하게 맞춰야 예측이 정확하다.
# ─────────────────────────────────────────────────────────────────
def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(INPUT_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = keras.applications.vgg16.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


# ─────────────────────────────────────────────────────────────────
# 3. 추론 (기존 로직 그대로)
#    출력은 sigmoid 단일값 → 0에 가까우면 정상, 1에 가까우면 불량
# ─────────────────────────────────────────────────────────────────
def predict(model, pil_img):
    arr   = preprocess(pil_img)
    prob  = float(model.predict(arr, verbose=0)[0][0])
    label = CLASSES[1 if prob > 0.5 else 0]
    return label, prob


# ─────────────────────────────────────────────────────────────────
# 4. Streamlit UI
# ─────────────────────────────────────────────────────────────────
def main():
    # 1) 페이지 설정
    st.set_page_config(page_title="가죽 이상 탐지", page_icon="🧵", layout="centered")
    st.title("🧵 가죽 이상 탐지 (Leather Defect Detection)")
    st.caption("사진을 업로드하거나 카메라로 촬영하면, VGG16 기반 모델이 정상/불량 여부를 판별합니다.")

    # 모델 로드 (캐시됨 - 최초 1회만 실제 로드)
    try:
        model = load_model()
    except FileNotFoundError as e:
        st.error(f"모델을 불러올 수 없습니다: {e}")
        st.stop()

    st.divider()

    # 2) 이미지 입력 방식 선택
    input_method = st.radio("이미지 입력 방식을 선택하세요", ["파일 업로드", "카메라 촬영"], horizontal=True)

    pil_img = None

    if input_method == "파일 업로드":
        uploaded_file = st.file_uploader("이미지 파일을 업로드하세요 (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")

    else:  # 카메라 촬영
        camera_file = st.camera_input("카메라로 가죽 이미지를 촬영하세요")
        if camera_file is not None:
            pil_img = Image.open(camera_file).convert("RGB")

    # 미리보기
    if pil_img is not None:
        show_image_full_width(pil_img, caption="입력 이미지 미리보기")

    st.divider()

    # 3) 검사 실행
    run = st.button("🔍 검사 시작", type="primary", disabled=(pil_img is None))

    if run and pil_img is not None:
        with st.spinner("추론 중입니다..."):
            label, prob = predict(model, pil_img)

        normal_prob = 1 - prob
        defect_prob = prob

        # 4) 결과 표시
        if label == "정상":
            st.success(f"✅ 판정 결과: {label}  (불량 확률 {defect_prob:.1%})")
        else:
            st.error(f"🚨 판정 결과: {label}  (불량 확률 {defect_prob:.1%})")

        col1, col2 = st.columns(2)
        col1.markdown(
            f"<p style='margin-bottom:0;'>정상 확률</p>"
            f"<p style='color:#1f77ff; font-size:2rem; font-weight:700; margin-top:0;'>{normal_prob:.1%}</p>",
            unsafe_allow_html=True,
        )
        col2.markdown(
            f"<p style='margin-bottom:0;'>불량 확률</p>"
            f"<p style='color:#e60000; font-size:2rem; font-weight:700; margin-top:0;'>{defect_prob:.1%}</p>",
            unsafe_allow_html=True,
        )

        # 막대 그래프: 정상=파란색, 불량=빨간색 / X축 라벨은 가로로 표시
        chart_df = pd.DataFrame({
            "구분":  ["정상", "불량"],
            "확률":  [normal_prob, defect_prob],
            "색상":  ["#1f77ff", "#e60000"],
        })

        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("구분:N", title=None, sort=["정상", "불량"], axis=alt.Axis(labelAngle=0)),
                y=alt.Y("확률:Q", title="확률", axis=alt.Axis(format="%")),
                color=alt.Color("색상:N", scale=None, legend=None),
                tooltip=[alt.Tooltip("구분:N"), alt.Tooltip("확률:Q", format=".1%")],
            )
            .properties(height=300)
        )
        try:
            st.altair_chart(chart, width="stretch")
        except TypeError:
            st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
    main()