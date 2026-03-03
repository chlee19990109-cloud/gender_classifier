import cv2
import numpy as np
import streamlit as st
import os
from header.haar_utils import preprocessing, correct_image, detect_object
from header.haar_classify import classify, display
from header.haar_histogram import make_masks, calc_histo

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI 성별 분류기",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 커스텀 CSS 적용 (UI 디자인 에셋) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Pretendard', sans-serif;
    }
    
    .stApp {
        background-color: #f8fafc;
    }
    
    .main-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 2.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    
    .sub-title {
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    .card-box {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        transition: transform 0.2s;
    }
    .card-box:hover {
        transform: translateY(-2px);
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        height: 3rem;
        font-weight: 600;
        background-color: #4f46e5;
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #4338ca;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.4);
    }
    
    .result-female {
        background-color: #fdf2f8;
        border-left: 4px solid #ec4899;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }

    .result-male {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    
    /* 사이드바 스타일링 */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# --- 헤더 영역 ---
st.markdown("""
<div class="main-header">
    <div class="main-title">✨ AI 성별 분류기 (Gender Classifier)</div>
    <div class="sub-title">OpenCV Haar Cascade 및 히스토그램 분석 기반 성별 예측 시스템</div>
</div>
""", unsafe_allow_html=True)

# --- 사이드바 설정 영역 ---
with st.sidebar:
    st.markdown("### ⚙️ 설정창")
    st.markdown("분석할 이미지 소스를 선택하세요.")
    image_source = st.radio("이미지 가져오기", ("💾 내 PC에서 업로드", "🖼️ 샘플 이미지 사용"), label_visibility="collapsed")
    
    st.markdown("---")
    
    img = None
    if image_source == "💾 내 PC에서 업로드":
        uploaded_file = st.file_uploader("이미지 파일 선택", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    elif image_source == "🖼️ 샘플 이미지 사용":
        sample_dir = "images/face"
        if os.path.exists(sample_dir):
            sample_images = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if sample_images:
                selected_sample = st.selectbox("샘플을 선택하세요", sample_images)
                sample_path = os.path.join(sample_dir, selected_sample)
                img = cv2.imread(sample_path)
                if img is None:
                    st.error("이미지를 읽어오는데 실패했습니다.")
            else:
                st.warning(f"'{sample_dir}' 폴더 내에 이미지가 없습니다.")
        else:
            st.error(f"'{sample_dir}' 경로를 찾을 수 없습니다.")

# --- 메인 컨텐츠 영역 ---
if img is not None:
    col1, padding, col2 = st.columns([1, 0.1, 1])
    
    # 1. 왼쪽: 원본 이미지
    with col1:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📷 원본 이미지</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. 이미지 분석 (백그라운드 처리)
    image, gray = preprocessing(img)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100))
    
    # 3. 오른쪽: 분석 결과
    with col2:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔍 분석 결과</div>', unsafe_allow_html=True)
        
        if len(faces) > 0:
            x, y, w, h = faces[0].tolist()
            face_image = image[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25, 20))
            
            if len(eyes) == 2:
                with st.spinner('얼굴 특징 및 히스토그램을 분석하는 중입니다...'):
                    face_center = (x + w // 2, y + h // 2)
                    eye_centers = [(x + ex + ew // 2, y + ey + eh // 2) for ex, ey, ew, eh in eyes]
                    
                    # 전처리 및 검출
                    corr_image, corr_centers = correct_image(image, face_center, eye_centers)
                    sub_roi = detect_object(face_center, faces[0])
                    masks = make_masks(sub_roi, corr_image.shape[:2])
                    sims = calc_histo(corr_image, sub_roi, masks)
                    
                    # 성별 판별 로직
                    text, result = classify(corr_image, sims)
                    disp_image = display(corr_image, face_center, corr_centers, sub_roi)
                    
                st.image(cv2.cvtColor(disp_image, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                if text == "Woman":
                    st.markdown(f'<div class="result-female">🎉 <b>예측 성별: 여자 (Woman)</b><br><small>{result}</small></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-male">🎉 <b>예측 성별: 남자 (Man)</b><br><small>{result}</small></div>', unsafe_allow_html=True)
            else:
                st.error("⚠️ 눈을 정확히 검출하지 못했습니다. (정면 얼굴 이미지를 사용해주세요)")
        else:
            st.error("⚠️ 얼굴을 검출하지 못했습니다. (배경이 복잡하거나 이목구비가 가려졌을 수 있습니다)")
            
        st.markdown('</div>', unsafe_allow_html=True)
else:
    # 이미지가 없을 때 대기 화면
    st.info("👈 왼쪽 사이드바에서 이미지를 업로드하거나 샘플을 선택하여 분석을 시작하세요.")
