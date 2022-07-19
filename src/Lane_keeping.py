#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2, rospy
import numpy as np

from xycar_msgs.msg import xycar_motor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# speed: 20, PID: (0.36, 0.0006, 0.08)
################## 상수 ########################################
width, height = 640, 480            # 영상의 크기
init_l, init_r = 320, 320           # 좌우 차선 차선 탐색을 위한 픽셀 시작 점
roi_w, roi_h = 200, 20              # 좌우 차선 ROI 영역의 크기 값
lane_width = 440                    # 차선 너비 픽셀
kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
no_l, no_r = 0, 0
speed = 0
################## 변수 ########################################
box_w, box_h = 15, 20               # 차선 체크박스 크기
l_pos, r_pos = box_w / 2, width - box_w /2      # 위치 초기화
offset = 340                        # ROI 창의 높이
pixel_threshold = int(0.4 * box_w * box_h)   # 차선 체크박스 임계값
#################### PID ##############################
class PID():
  def __init__(self,kp,ki,kd):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.p_error = 0.0
    self.i_error = 0.0
    self.d_error = 0.0
  def pid_control(self, cte):
    self.d_error = cte-self.p_error
    # if abs(cte) >= 120
    self.p_error = cte
    self.i_error += cte
    return self.kp*self.p_error + self.ki*self.i_error + self.kd*self.d_error
################# 이동평균 필터 상수 ######################
k_l, k_r = 0, 0                     # k번 째 수 의미
preAvg_l, preAvg_r = 0, 0           # 이전의 평균 값
N = 10                              # 슬라이딩 윈도우 크기
l_buf, r_buf = np.zeros(N + 1), np.zeros(N + 1) # 슬라이딩 윈도우
################# 고정 함수 ######################
# cv 브릿지
bridge = CvBridge()
cv_img = np.empty(shape=[0])

# 콜백 함수 선언
def img_callback(data):
    global cv_img
    cv_img = bridge.imgmsg_to_cv2(data, "bgr8")

# 기본 로스 함수
rospy.init_node('line_follow')
rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback)
pub = rospy.Publisher("xycar_motor", xycar_motor, queue_size=1)
motor_control = xycar_motor()
################# 자율 주행 함수 #############################
# 이미지 처리 함수
def process_image(cv_img):
    global roi_h, offset
    show = cv_img.copy()
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    conv = 255 - gray
    _, output = cv2.threshold(conv, 115, 255, cv2.THRESH_BINARY)
    erode = cv2.erode(output, kernal)
    bin = cv2.dilate(erode, kernal)
    bin_roi = bin[offset : (offset + roi_h) + 1, :].copy()
    return show, bin, bin_roi

# 차선 좌표 검출 함수
def get_line_pos(bin_roi):
    global box_w, box_h, pixel_threshold, width, lane_width, l_pos, r_pos, init_l, init_r, no_l, no_r
    # 왼쪽 차선 검출
    for l in range(init_l, 0 - 1, -1):                  # init_l부터 왼쪽으로 탐색            
        box = bin_roi[0:box_h + 1, l - box_w:l + 1]     # 박스 크기 설정
        if cv2.countNonZero(box) > pixel_threshold:     # 차선 탐색 조건문
            l_pos = l                                   # 왼쪽 차선 검출
            no_l = 0
            # init_l = l_pos + 50                         # init_l 재설정
            break
    # 왼쪽 차선이 검출되지 않았다면 
    else:
        l_pos = r_pos - lane_width                      # 왼쪽 차선을 오른쪽 차선 기준으로 보정
        if l_pos < - 190:
            no_l = 1
        # init_l = 50                                     # init_l을 50으로 설정

    # 오른쪽 차선 검출
    for r in range(init_r, width - 1):                  # init_r부터 오른쪽으로 탐색            
        box = bin_roi[0:box_h + 1, r:r + box_w + 1]     # 박스 크기 설정
        if cv2.countNonZero(box) > pixel_threshold:     # 차선 탐색 조건문
            r_pos = r                                   # 오른쪽 차선 검출
            no_r = 0
            # init_r = r_pos - 50                         # init_r 재설정
            break
    # 오른쪽 차선이 검출되지 않았다면 
    else:
        r_pos = l_pos + lane_width                      # 오른쪽 차선을 왼쪽 차선 기준으로 보정
        if r_pos > width +  190:
            no_r = 1
        # init_r = width - 50                             # init_r을 50으로 설정
    # 중간 픽셀 계산
    # c_pos = (l_pos + r_pos) /2
    return l_pos, r_pos#, c_pos

    
# 이동 평균 필터
def movAvgFilter(pos, k, preAvg, buf):
    global N
    if k == 0:
        buf = pos*np.ones(N + 1)
        k, preAvg = 1, pos
        
    for i in range(0, N):
        buf[i] = buf[i + 1]
    
    buf[N] = pos
    avg = preAvg + (pos - buf[0]) / N
    preAvg = avg
    return k, preAvg, buf, int(round(avg))
        
# 박스, 그리기 함수
# def draw_box(show, bin, l_pos, r_pos, c_pos):
#     global box_w, box_h, width, offset
#     show_bin = cv2.cvtColor(bin, cv2.COLOR_GRAY2BGR)
#     cv2.rectangle(show, (l_pos - box_w, offset), (l_pos, offset + box_h), (255, 0, 0), 1)
#     cv2.rectangle(show, (r_pos, offset), (r_pos + box_w, offset + box_h), (0, 0, 255), 1)
#     cv2.rectangle(show, (c_pos - box_w/2, offset), (c_pos + box_w/2, offset + box_h), (0, 255, 0), 1)
#     cv2.rectangle(show, (width/2 - box_w/2, offset), (width/2 + box_w/2, offset + box_h), (255, 255, 255), 2)
    
#     cv2.rectangle(show_bin, (l_pos - box_w, offset), (l_pos, offset + box_h), (255, 0, 0), 1)
#     cv2.rectangle(show_bin, (r_pos, offset), (r_pos + box_w, offset + box_h), (0, 0, 255), 1)
#     cv2.rectangle(show_bin, (c_pos - box_w/2, offset), (c_pos + box_w/2, offset + box_h), (0, 255, 0), 1)
#     cv2.rectangle(show_bin, (width/2 - box_w/2, offset), (width/2 + box_w/2, offset + box_h), (255, 255, 255), 2)
    
#     dst = cv2.hconcat([show, show_bin])
#     return dst

# 모터 퍼블리시 함수
def pub_motor(angle, speed):
    global pub
    global motor_control
    motor_control.angle = angle
    motor_control.speed = speed

    pub.publish(motor_control)

while not rospy.is_shutdown():
    if cv_img.size != (width*height*3):
        continue
    show, bin, bin_roi = process_image(cv_img)
    l_pos, r_pos = get_line_pos(bin_roi)
    k_l, preAvg_l, l_buf, l_pos = movAvgFilter(l_pos, k_l, preAvg_l, l_buf)
    k_r, preAvg_r, r_buf, r_pos = movAvgFilter(r_pos, k_r, preAvg_r, r_buf)
    init_l = l_pos + 50                         # init_l, init_r 재설정
    init_r = r_pos - 50

    if init_l < 0:
        init_l = 50
    if init_r > width:
        init_r = width - 50

    c_pos = (l_pos + r_pos) / 2
    error = c_pos - width/2
    pid = PID(0.35, 0.0006, 0.02)

    #  Back Drive
    if no_l == 1:
        angle, speed = 50, -10
    elif no_r == 1:
        angle, speed = -50, -10
    else:
         angle = pid.pid_control(error)
        #  speed = 30

    #  slow and fast
    if no_l ==  0 and no_r ==0:
        if abs(angle) < 22:
            speed += 0.05
            speed = min(speed, 35)
        else:
            speed -= 0.1
            speed = max(speed, 17)

    

    pub_motor(angle, speed)
    print("speed: {}, angle: {}, error: {}".format(speed, angle,error))
    # dst = draw_box(show, bin, l_pos, r_pos, c_pos)
    # print("l_pos: {0}, init_l: {1}, r_pos: {2}, init_r: {3}, angle: {4}".format(l_pos, init_l, r_pos, init_r, angle))

    # cv2.imshow("dst", dst)
    # cv2.waitKey(1)
