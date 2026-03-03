import cv2
import numpy as np
import streamlit as st
from header.haar_utils import preprocessing, correct_image, detect_object
from header.haar_histogram import make_masks, calc_histo

st.set_page_config(page_title="히스토그램 비교", layout="centered")
st.title("08. 히스토그램 비교 (입술, 머리)")
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
                masks = make_masks(rois, corr_image.shape[:2])
                sim = calc_histo(corr_image, rois, masks)
                
                st.image(cv2.cvtColor(corr_image, cv2.COLOR_BGR2RGB), caption="보정된 이미지", use_container_width=True)
                st.success(f"입술-얼굴 유사도: {sim[0]:4.2f}")
                st.success(f"윗-귀밑머리 유사도: {sim[1]:4.2f}")
            else:
                st.warning("눈 미검출")
        else:
            st.warning("얼굴 미검출")