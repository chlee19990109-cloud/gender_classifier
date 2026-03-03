# AI 성별 분류기 (AI Gender Classifier)

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white) 
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white) 
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)

**작성자**: 이충환

본 프로젝트는 OpenCV의 하르 분류기(Haar Cascade Classifier)와 2차원 히스토그램 비교 기술을 활용하여 이미지 내 인물의 성별을 예측하는 AI 분류기입니다. **Python 3.13** 및 **Streamlit**을 기반으로 웹 인터페이스(UI)가 설계되었습니다.

🔗 **배포된 웹 애플리케이션 접속하기**: [https://genderclassifier-c6sgqtwxb5ciwsjy2vs9xh.streamlit.app/](https://genderclassifier-c6sgqtwxb5ciwsjy2vs9xh.streamlit.app/)

---

## 🎯 주요 기능 및 파일 구성
- `06.detect_face.py`: 하르 분류기를 이용한 기본 얼굴 및 눈 검출 기능.
- `07.detect_hair_lip.py`: 눈의 중심좌표를 이용해 얼굴 각도를 보정한 후, 입술/윗머리/귀밑머리 영역(ROI) 검출.
- `08.compare_hist.py`: 마스크를 이용해 검출된 각 영역(입술, 머리)의 2차원 히스토그램을 산출하고 비교하여 유사도 계산.
- `09.gender_classifier.py`: **Streamlit 웹 애플리케이션**. 앞선 기능들을 모두 결합하여 사용자에게 직관적인 UI로 성별 분류 결과를 시각화하여 제공합니다.

---

## 🔬 기술 요소 (Tech Stack)

### 1. 웹 프레임워크: Streamlit
- **디자인**: 반응형 웹 및 커스텀 CSS를 활용한 직관적인 대시보드 형태의 UI 제어.
- **버전**: Python 3.13 환경에서 최적화.

### 2. 하르 분류기 (Haar-based Cascade Classifier)
- OpenCV의 사전 학습된 XML 분류기를 사용합니다.
- 복잡한 신경망보다 가벼우며 빠르고 효율적으로 얼굴 객체를 검출할 수 있습니다.
- `haarcascade_frontalface_alt2.xml`: 정면 얼굴 검출.
- `haarcascade_eye.xml`: 얼굴 내 눈 검출.

### 3. 영상 보정 및 히스토그램 비교 (OpenCV)
- **얼굴 기울기 보정**: 검출된 두 눈의 좌표 차분을 이용해 아크탄젠트(역탄젠트)를 계산하여 기울기를 보정합니다 (`cv2.warpAffine`).
- **영역 지정 및 마스크**: 입술, 윗머리, 귀밑머리의 위치 비례를 추정해 관심 영역(ROI)으로 지정하고 마스크를 생성합니다 (`cv2.ellipse`).
- **성별 분류 원리 (2차원 히스토그램)**: 
  - 각 지정된 영역(입술-얼굴, 귀밑-윗머리)에 대한 HSV 색상 공간의 2차원 히스토그램을 `cv2.calcHist()`로 산출합니다.
  - `cv2.compareHist()`로 상호상관(Correlation) 분석을 수행하여 남성/여성 간의 입술 색상 대비, 머리길이 대비 등을 점수화하고 2단계 임계값 판별을 거쳐 최종 결과를 내립니다.

---

## 💻 환경 설정 및 실행 방법 (Environment Setup)

이 애플리케이션은 **Python 3.13** 환경에서 실행 및 배포하는 것을 기준으로 작성되었습니다.

### 1. 가상환경 설정 및 패키지 설치
Python 3.13이 설치된 상태에서 터미널(명령 프롬프트)을 열고 아래의 명령어를 순서대로 실행합니다.

```bash
# 가상환경 생성 (.venv)
python -m venv .venv

# 가상환경 활성화
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 필수 라이브러리 설치
pip install -r requirements.txt
```
*(요구되는 패키지: `opencv-python`, `numpy`, `streamlit`)*

### 2. 애플리케이션 실행
패키지 설치가 완료되면, 앱을 실행합니다.

```bash
streamlit run 09.gender_classifier.py
```
명령어를 입력하면 로컬 웹 서버가 실행되고, 브라우저가 자동으로 열리면서 `http://localhost:8501` 경로를 통해 성별 분류기에 접속할 수 있습니다.

### 3. 사용 방법
1. 좌측 설정 탭에서 '내 PC에서 이미지 업로드'를 선택하여 본인의 이미지를 사용하거나 '샘플 이미지 사용'을 통해 제공된 이미지를 고릅니다.
2. 메인 페이지의 원본 이미지와 분석 결과를 확인합니다. 
3. (주의사항: 눈, 입술, 머리 영역을 기반으로 분석하므로 *정면 사진*일수록 정확도가 높게 나옵니다.)
