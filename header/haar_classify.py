import cv2
from header.haar_histogram import draw_ellipse

def classify(image, sims):
    criteria = 0.25 if sims[0] > 0.2 else 0.1            # 얼굴-입술 유사도: 비슷하면(큰값) 남자
    value = sims[1] > criteria

    text = "Woman" if value else "Man"                   # value: sims[1] 여자, sims[0] 남자
    result = "유사도 [입술-얼굴: %4.3f 윗-귀밑머리: %4.3f]" % (sims)

    # 텍스트를 영상에 직접 출력하지 않고 반환하여 Streamlit UI에서 표시하도록 함
    return text, result

def display(image, face_center, centers, sub):
    cv2.circle(image, face_center, 3, (0, 0, 255), 2)	    # 얼굴 중심점 표시
    cv2.circle(image, tuple(centers[0]), 10, (0, 255, 0), 2)	    # 눈 표시
    cv2.circle(image, tuple(centers[1]), 10, (0, 255, 0), 2)
    draw_ellipse(image, sub[2], 0.35,(0, 0, 255),  2)	            # 얼굴 타원
    draw_ellipse(image, sub[3], 0.45,(255, 100, 0), 2)              # 입술 타원
    return image                                                    # Streamlit에서 표시하기 위해 이미지 반환