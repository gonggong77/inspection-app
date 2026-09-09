# 🧵 가죽 이상 탐지 (Leather Defect Detection)

VGG16 기반 전이학습 모델(`.keras`)로 가죽 표면의 정상/불량 여부를 판별하는 Streamlit 웹 앱입니다.
`infer_keras.py`(CLI 버전)의 모델 로드 / 전처리 / 추론 로직을 그대로 이어받아, 파일 업로드와 웹캠 촬영을 지원하는 웹 UI(`app.py`)로 구현했습니다.

## 주요 기능

- **이미지 입력**: 파일 업로드(jpg/jpeg/png) 또는 카메라 촬영 중 선택
- **미리보기**: 입력한 이미지를 검사 전 화면에서 확인
- **추론**: "🔍 검사 시작" 버튼 클릭 시 모델 추론 실행
- **결과 표시**
  - 정상 → 성공 메시지(초록), 불량 → 에러 메시지(빨강)
  - 정상 확률(파란색) / 불량 확률(빨간색) 숫자로 표시
  - 정상·불량 순서로 정렬된 막대 그래프(정상=파란색, 불량=빨간색, X축 라벨 가로 표시)
- **모델 캐싱**: `@st.cache_resource`로 앱이 재실행(rerun)돼도 모델을 매번 새로 불러오지 않음

## 프로젝트 구조

```
.
├── app.py                  # Streamlit 웹 앱 (메인 실행 파일)
├── infer_keras.py          # UI 없이 모델 로직만 확인하는 원본 CLI 스크립트
├── weights/
│   └── leather_model.keras # 학습된 .keras 모델 파일 (직접 준비 필요)
└── README.md
```

## 요구 사항

- Python 3.9 이상 권장
- 패키지: `streamlit`, `tensorflow`, `pillow`, `numpy`, `pandas`, `altair`
  (`altair`, `pandas`는 대부분 Streamlit 설치 시 함께 설치됩니다)

```bash
pip install streamlit tensorflow pillow numpy pandas altair
```

> 카메라 촬영 기능은 Streamlit 내장 `st.camera_input`을 사용하므로 `opencv-python` 설치가 필요 없습니다. (원본 `infer_keras.py`의 웹캠 모드만 `opencv-python`이 필요합니다.)

## 모델 파일 준비

`app.py`의 `MODEL_PATH` 설정에 맞춰 학습된 모델 파일을 아래 경로에 위치시켜야 합니다.

```
./weights/leather_model.keras
```

경로를 변경하려면 `app.py` 상단의 `MODEL_PATH` 값을 수정하세요.

```python
MODEL_PATH = "./weights/leather_model.keras"
```

## 실행 방법

```bash
streamlit run app.py
```

실행 후 브라우저(기본 `http://localhost:8501`)가 자동으로 열립니다.

1. "파일 업로드" 또는 "카메라 촬영" 중 이미지 입력 방식 선택
2. 이미지를 업로드하거나 촬영하면 미리보기 표시
3. "🔍 검사 시작" 버튼 클릭
4. 판정 결과(정상/불량), 확률, 막대 그래프 확인

## 모델 입출력 규격

| 항목 | 값 |
|---|---|
| 입력 이미지 크기 | 224 × 224 (RGB) |
| 전처리 | `keras.applications.vgg16.preprocess_input` |
| 출력 | Sigmoid 단일 확률값 (0~1) |
| 판정 기준 | `prob > 0.5` → 불량, 그 외 → 정상 |

## 참고: infer_keras.py와의 차이

| 구분 | infer_keras.py (원본) | app.py (Streamlit 버전) |
|---|---|---|
| 실행 방식 | 콘솔 스크립트 (`python infer_keras.py`) | 웹 앱 (`streamlit run app.py`) |
| 입력 전환 | 코드 상단 `INPUT_MODE` 변수 수정 | 화면에서 라디오 버튼으로 선택 |
| 카메라 | OpenCV로 직접 창 제어 | `st.camera_input` |
| 결과 확인 | 콘솔 출력 + `matplotlib` 창 | 웹 UI(성공/에러 메시지, 확률, 막대 그래프) |
| 모델 재로드 | 실행마다 새로 로드 | `st.cache_resource`로 최초 1회만 로드 |

## 문제 해결(Troubleshooting)

- **`st.image()` / `st.altair_chart()`에서 `use_container_width` 관련 `TypeError`가 발생하는 경우**
  설치된 Streamlit 버전에 따라 너비 지정 파라미터명이 다릅니다(`use_column_width` → `use_container_width` → `width`). `app.py`에는 버전에 관계없이 동작하도록 호환 처리가 되어 있으니, 계속 오류가 발생하면 `pip show streamlit`으로 버전을 확인해 주세요.
- **모델 파일을 찾을 수 없다는 오류가 나는 경우**
  `weights/leather_model.keras` 경로에 모델 파일이 있는지, `MODEL_PATH` 설정이 실제 경로와 일치하는지 확인하세요.
