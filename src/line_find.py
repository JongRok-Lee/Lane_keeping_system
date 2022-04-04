#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

# cap = cv2.VideoCapture('/home/jrvm/xycar_ws/src/week7/line_drive/src/base_camera_dark.avi')
cap = cv2.VideoCapture('/home/jrvm/xycar_ws/src/week7/line_drive/src/xycar_track1.mp4')
# cap = cv2.VideoCapture('/home/jrvm/xycar_ws/src/week7/hough_drive/src/hough_track.avi')
if cap.isOpened() == False:
    print("Video load failed!")
    exit(1)

# HSV 색깔 영역의 임계 값
low = np.array([0, 0, 100])
high = np.array([131, 255, 255])
# 영상의 너비 (640)
width_640 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# ROI를 자를 가로 세로 값 (200, 20)
scan_width_200, scan_height_20 = 200, 20
# ROI가 아닌 부분의 왼쪽, 오른쪽 가로 값 (200, 440)
lmid_200, rmid_440 = scan_width_200, width_640 - scan_width_200
# 녹색 사각형의 가로 세로 값 (20, 10)
area_width_20, area_height_10 = 20, 10
# ROI의 시작 높이 값 (430)
vertical_430 = 430
# 녹색 사각형의 ROI내에서 높이 시작과, 끝 값 (5, 15)
row_begin_5 = (scan_height_20 - area_height_10) // 2
row_end_15 = row_begin_5 + area_height_10
# 차선이 있는 녹색 사각형 검출을 위한 흰색 픽셀 임계값
pixel_threshold_160 = 0.8 * area_width_20 * area_height_10

while cv2.waitKey(10) != 27:
    ret, frame = cap.read()
    if not ret:
        print("Image load failed!")
        break
    # 가우시안 필터
    blur = cv2.GaussianBlur(frame, (3, 3), 0)
    # HSV로 변환
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # 색깔 영역 추출
    bin = cv2.inRange(hsv, low, high)
    bin_roi = bin[vertical_430:vertical_430 + scan_height_20, :]
    
    # imshow를 위해 bin 이미지 BGR로 변환
    view_bin = cv2.cvtColor(bin, cv2.COLOR_GRAY2BGR)
    view_roi = view_bin[vertical_430:vertical_430 + scan_height_20, :]
    # bin 영상에 ROI 빨간색으로 그리기
    cv2.rectangle(view_bin,
                  (0, vertical_430),
                  (width_640 - 1, vertical_430 + scan_height_20),
                  (0, 0, 255), 3)
    
    # 차선 검출 녹색 사각형의 왼쪽 픽셀 값 변수 선언
    left, right = -1, -1
    # 왼쪽 부분의 차선 검출하기 (l = 0~180)
    for l in range (0, lmid_200 - area_width_20):
        # 사각형은 mask(l:l + 20, 5:15)영역. 즉, 0:20~180:200을 1픽셀 씩 탐색
        area = bin_roi[row_begin_5:row_end_15, l : l + area_width_20]
        # 사각형의 차선 픽셀 수가 80%면 탐색 중지후 left에 사각형 왼쪽 값 반환
        if cv2.countNonZero(area) > pixel_threshold_160:
            left = l
            break
        
    # 오른쪽 부분의 차선 검출하기 (r = 620~440)
    for r in range(width_640 - area_width_20, rmid_440, -1):
        # 사각형은 mask(r:r+20, 5:15)영역. 즉, 620:640~440:460을 1픽셀 씩 탐색
        area = bin_roi[row_begin_5:row_end_15, r:r + area_width_20]
        # 사각형의 차선 픽셀 수가 80%면 탐색 중지후 right에 사각형 왼쪽 값 반환
        if cv2.countNonZero(area) > pixel_threshold_160:
            right = r
            break
    # 왼쪽 차선이 검출되면
    if left != -1:
        # 마스크 영역의 왼쪽 차선에 녹색 사각형 그리기
        cv2.rectangle(view_roi,
                      (left, row_begin_5),
                      (left + area_width_20, row_end_15),
                      (0, 255, 0), 3)
    else:
        print("Lost left line")
    
    # 오른쪽 차선이 검출되면
    if right != -1:
        # 마스크 영역의 오른쪽 차선에 녹색 사각형 그리기
        cv2.rectangle(view_roi,
                      (right, row_begin_5),
                      (right + area_width_20, row_end_15),
                      (0, 255, 0), 3)
    else:
        print("Lost right 1ine")
    
    dst = cv2.hconcat([frame, view_bin])
    cv2.imshow("origin", dst)
    #cv2.imshow("binary", view_bin)
    
cap.release()
cv2.destroyAllWindows()