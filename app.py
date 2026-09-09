"""
app.py  ─  [수업 2단계] Streamlit UI로 .keras 모델 추론하기
════════════════════════════════════════════════════════════════════
infer_keras.py 의 [모델 로드 / 전처리 / 추론] 로직은 그대로 유지하고,
Streamlit 웹 UI(파일 업로드 / 카메라 촬영 / 결과 시각화)를 추가한 버전.

실행 방법:
    streamlit run app.py
가죽 이상 탐지 모델을 사용하는 Streamlit 웹 앱.
"""

# import os

# import numpy as np
# import pandas as pd
# import streamlit as st
# import tensorflow as tf
# from PIL import Image
# from tensorflow import keras
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm


# # ── 페이지 설정 ──────────────────────────────────────────────────
# # 이모지나 외부 이미지 파일 없이 사용할 수 있는 간단한 갈색 아이콘이다.
# PAGE_ICON = Image.new("RGB", (32, 32), color=(105, 73, 48))

# st.set_page_config(
#     page_title="가죽 이상 탐지",
#     page_icon=PAGE_ICON,
#     layout="centered",
# )

# st.title("가죽 이상 탐지")
# st.caption("가죽 이미지를 입력하면 AI 모델이 정상 여부와 예측 확률을 표시합니다.")


# # ── 모델 및 추론 설정 ────────────────────────────────────────────
# MODEL_PATH = "./weights/leather_model.keras"
# INPUT_IMG_SIZE = (224, 224)
# CLASSES = ["정상", "불량"]

# FONT_PATH = "fonts/NanumGothic-Regular.ttf"
# font_prop = fm.FontProperties(fname=FONT_PATH)

# plt.title("검사 결과", fontproperties=font_prop)
# plt.xlabel("분류", fontproperties=font_prop)
# plt.ylabel("확률", fontproperties=font_prop)


# # ─────────────────────────────────────────────────────────────────
# # 1. 모델 로드
# #    앱이 입력 변경 등으로 다시 실행되어도 캐시에 저장된 모델을 재사용한다.
# # ─────────────────────────────────────────────────────────────────
# @st.cache_resource
# def load_model():
#     if not os.path.exists(MODEL_PATH):
#         raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")
#     return tf.keras.models.load_model(MODEL_PATH)


# # ─────────────────────────────────────────────────────────────────
# # 2. 이미지 전처리
# #    기존 코드와 동일하게 RGB 변환, 224×224 크기 조정,
# #    VGG16 전처리, 배치 차원 추가를 수행한다.
# # ─────────────────────────────────────────────────────────────────
# def preprocess(pil_img):
#     img = pil_img.convert("RGB").resize(INPUT_IMG_SIZE)
#     arr = np.array(img, dtype=np.float32)
#     arr = keras.applications.vgg16.preprocess_input(arr)
#     return np.expand_dims(arr, axis=0)


# # ─────────────────────────────────────────────────────────────────
# # 3. 추론
# #    기존 코드와 동일하게 sigmoid 출력값이 0.5보다 크면 불량으로 판정한다.
# # ─────────────────────────────────────────────────────────────────
# def predict(model, pil_img):
#     arr = preprocess(pil_img)
#     prob = float(model.predict(arr, verbose=0)[0][0])
#     label = CLASSES[1 if prob > 0.5 else 0]
#     return label, prob


# # ─────────────────────────────────────────────────────────────────
# # 4. 이미지 입력
# # ─────────────────────────────────────────────────────────────────
# input_mode = st.radio(
#     "이미지 입력 방식",
#     ["파일 업로드", "카메라 촬영"],
#     horizontal=True,
# )

# if input_mode == "파일 업로드":
#     image_file = st.file_uploader(
#         "가죽 이미지 선택",
#         type=["jpg", "jpeg", "png"],
#     )
# else:
#     image_file = st.camera_input("가죽 이미지를 촬영하세요")

# pil_img = None
# if image_file is not None:
#     try:
#         pil_img = Image.open(image_file).convert("RGB")
#         st.image(pil_img, caption="입력 이미지")
#     except Exception as error:
#         st.error(f"이미지를 읽을 수 없습니다: {error}")


# # ─────────────────────────────────────────────────────────────────
# # 5. 검사 실행 및 결과 표시
# # ─────────────────────────────────────────────────────────────────
# if st.button("검사 시작", type="primary", disabled=pil_img is None):
#     try:
#         with st.spinner("가죽 이미지를 검사하고 있습니다."):
#             model = load_model()
#             label, defect_prob = predict(model, pil_img)

#         normal_prob = 1 - defect_prob

#         if label == "정상":
#             st.success("검사 결과: 정상입니다.")
#         else:
#             st.error("검사 결과: 불량입니다.")

#         normal_column, defect_column = st.columns(2)
#         normal_column.metric("정상 확률", f"{normal_prob:.1%}")
#         defect_column.metric("불량 확률", f"{defect_prob:.1%}")

#         chart_data = pd.DataFrame(
#             {"확률": [defect_prob]},
#             index=["불량"],
#         )
#         st.subheader("불량 확률")
#         st.bar_chart(chart_data, y="확률", height=220)

#     except FileNotFoundError as error:
#         st.error(str(error))
#     except Exception as error:
#         st.error(f"검사 중 오류가 발생했습니다: {error}")


import os
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow import keras


# set_page_config()는 스크립트에서 가장 먼저 호출되는 Streamlit 명령이어야 한다.
st.set_page_config(page_title="가죽 이상 탐지", page_icon="🧵", layout="centered")


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