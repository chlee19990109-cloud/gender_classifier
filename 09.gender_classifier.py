import cv2
import numpy as np
import streamlit as st
import os
from header.haar_utils import preprocessing, correct_image, detect_object
from header.haar_classify import classify, display
from header.haar_histogram import make_masks, calc_histo

# 페이지 설정
st.set_page_config(page_title="인공지능 성별 분류기", page_icon="👩‍🦰👨‍🦱", layout="wide")

# CSS 스타일링 추가
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 20px;
        font-weight: 800;
    }
    .sub-text {
        text-align: center;
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">✨ 인공지능 성별 분류기 ✨</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">얼굴 이미지를 분석하여 성별을 예측합니다. 이미지를 업로드하거나 샘플 이미지를 선택해보세요!</p>', unsafe_allow_html=True)

st.sidebar.title("🛠️ 옵션 설정")
st.sidebar.markdown("---")

image_source = st.sidebar.radio("이미지 소스 선택", ("내 PC에서 업로드", "샘플 이미지 사용"))

img = None

if image_source == "내 PC에서 업로드":
    st.sidebar.markdown("### 📤 이미지 업로드")
    uploaded_file = st.sidebar.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

elif image_source == "샘플 이미지 사용":
    st.sidebar.markdown("### 🖼️ 샘플 이미지 선택")
    sample_dir = "images/face"
    if os.path.exists(sample_dir):
        sample_images = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if sample_images:
            selected_sample = st.sidebar.selectbox("테스트할 샘플을 선택하세요", sample_images)
            sample_path = os.path.join(sample_dir, selected_sample)
            img = cv2.imread(sample_path)
            if img is None:
                st.sidebar.error("샘플 이미지를 읽을 수 없습니다.")
        else:
            st.sidebar.warning(f"'{sample_dir}' 폴더에 이미지가 없습니다.")
    else:
        st.sidebar.warning(f"'{sample_dir}' 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")

# 이미지 처리 로직
if img is not None:
    # 레이아웃 나누기
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📷 원본 이미지")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

    image, gray = preprocessing(img)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100))
    
    with col2:
        st.markdown("### 🔍 분석 결과")
        if len(faces) > 0:
            x, y, w, h = faces[0].tolist()
            face_image = image[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25, 20))
            if len(eyes) == 2:
                with st.spinner('얼굴, 눈, 입술 특징을 분석 중입니다...'):
                    face_center = (x + w // 2, y + h // 2)
                    eye_centers = [(x + ex + ew // 2, y + ey + eh // 2) for ex, ey, ew, eh in eyes]
                    
                    corr_image, corr_centers = correct_image(image, face_center, eye_centers)
                    sub_roi = detect_object(face_center, faces[0])
                    masks = make_masks(sub_roi, corr_image.shape[:2])
                    sims = calc_histo(corr_image, sub_roi, masks)
                    
                    text, result = classify(corr_image, sims)
                    disp_image = display(corr_image, face_center, corr_centers, sub_roi)
                    
                st.image(cv2.cvtColor(disp_image, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # 결과 강조 표시
                if text == "Woman":
                    st.success(f"### 🎉 예측 성별: 여자 (Woman)")
                else:
                    st.info(f"### 🎉 예측 성별: 남자 (Man)")
                    
                st.markdown(f"**세부 지표:** `{result}`")
            else:
                st.error("⚠️ 눈을 정확히 검출하지 못했습니다. (정면 얼굴 이미지를 사용해주세요)")
        else:
            st.error("⚠️ 얼굴을 검출하지 못했습니다. (배경이 복잡하거나 얼굴이 너무 작을 수 있습니다)")
else:
    st.info("👈 왼쪽 사이드바에서 이미지를 업로드하거나 샘플을 선택해주세요.")