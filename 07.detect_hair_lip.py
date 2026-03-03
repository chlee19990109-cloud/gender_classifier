import cv2
import numpy as np
import streamlit as st
from header.haar_utils import preprocessing, correct_image, detect_object

st.set_page_config(page_title="머리 및 입술 영역 검출", layout="centered")
st.title("07. 하르 분류기 - 머리 및 입술 영역 검출")
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is not None:
        image, gray = preprocessing(img)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100))
        if len(faces) > 0:
            x, y, w, h = faces[0].tolist()
            face_image = image[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25, 20))
            if len(eyes) == 2:
                face_center = (x + w//2, y + h//2)
                eye_centers = [[x+ex+ew//2, y+ey+eh//2] for ex,ey,ew,eh in eyes]
                
                corr_image, corr_center = correct_image(image, face_center, eye_centers)
                rois = detect_object(face_center, faces[0])
                
                cv2.rectangle(corr_image, rois[0], (255, 0, 255), 2)
                cv2.rectangle(corr_image, rois[1], (255, 0, 255), 2)
                cv2.rectangle(corr_image, rois[2], (255, 0, 0), 2)
                cv2.circle(corr_image, tuple(corr_center[0]), 5, (0, 255, 0), 2)
                cv2.circle(corr_image, tuple(corr_center[1]), 5, (0, 255, 0), 2)
                cv2.circle(corr_image, face_center, 3, (0, 0, 255), 2)
                
                st.image(cv2.cvtColor(corr_image, cv2.COLOR_BGR2RGB), caption="보정 및 검출 결과", use_container_width=True)
            else:
                st.warning("눈 미검출")
        else:
            st.warning("얼굴 미검출")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)