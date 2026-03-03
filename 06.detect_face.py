import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="얼굴 및 눈 검출", layout="centered")
st.title("06. 하르 분류기 - 얼굴 및 눈 검출")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Uploaded file conversion
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("영상 파일 읽기 에러")
    else:
        st.subheader("원본 및 검출 결과")
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="업로드된 원본 이미지", use_container_width=True)
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100))
        
        if len(faces) > 0:
            for x, y, w, h in faces:
                face_image = image[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25, 20))
                
                if len(eyes) == 2:
                    for ex, ey, ew, eh in eyes:
                        center = (x + ex + ew // 2, y + ey + eh // 2)
                        cv2.circle(image, center, 10, (0, 255, 0), 2)
                else:
                    st.warning("눈 미검출")
                
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
            with col2:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="검출 결과", use_container_width=True)
        else:
            st.warning("얼굴 미검출")