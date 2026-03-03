import numpy as np, cv2

# 타원 그리기 함수
def draw_ellipse(image, roi, ratio, color, thickness=cv2.FILLED):
    x, y, w, h = roi
    center = (x + w // 2, y + h // 2)                   # 타원 중심
    size = (int(w * ratio), int(h * ratio))             # 타원 크기 (그려질 타원 비율)
    cv2.ellipse(image, center, size, 0, 0, 360, color, thickness)
    return image

# 각 마스크 생성 함수
def make_masks(rois, correct_shape):                                # 영역별 마스크 생성 함수
    base_mask = np.full(correct_shape, 255, np.uint8)
    hair_mask = draw_ellipse(base_mask, rois[3], 0.45, 0,  -1)      # 얼굴 타원 그리기  # 전체 머리 영역의 45%를 타원 반지름으로 사용
    # 기본 마스크에 입술영역 타원 그림
    lip_mask = draw_ellipse(np.copy(base_mask), rois[2], 0.45, 255) # 입력 타원 그리기  # 기본 마스크에 입술영역 타원 그리기

    masks = [hair_mask, hair_mask, lip_mask, ~lip_mask]             # 4개 마스크 구성   # ~lip mask: 입술 마스크 반전하여 구성
    masks = [mask[y:y+h,x:x+w] for mask,(x,y, w,h) in zip(masks, rois)]

    # for i, mask in enumerate(masks):                              # 마스크 영상 윈도우 표시
    #     cv2.imshow('mask'+str(i), mask)
    # cv2.waitKey()

    return masks

# 마스크 이용하여 각 서브영역의 히스토그램 생성
# 기울기 보정 영상에 각 서브 영역 및 마스크로 히스토그램 계산
def calc_histo(image, rois, masks):
    bsize = (64, 64,64)                                            # 히스토그램 계급 개수
    ranges = (0,256, 0,256, 0,256)                                 # 각 채널 빈도 범위

    subs = [image[y:y+h, x:x+w] for x, y, w, h in rois]            # 관심 영역 참조로 영상 생성
    hists = [cv2.calcHist([sub], [0,1,2], mask, bsize, ranges)
             for sub, mask in zip(subs, masks)]   # 관심 영역 영상 히스토그램
    hists = [ h/np.sum(h) for h in hists]           # 히스토그램값 정규화

    sim1 = cv2.compareHist(hists[2], hists[3], cv2.HISTCMP_CORREL)  # 입술-얼굴 유사도
    sim2 = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL)  # 윗-귀밑머리 유사도
    return  sim1, sim2