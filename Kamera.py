import cv2 
import numpy as np 
import imutils 
import math 
from abc import ABC, abstractmethod 
 
class tennis_ball_detect(ABC): 
    def num_tennisball(self): 
        pass 
    def ballcoordinates(self): 
        pass 
    def centroid(self): 
        pass 
 
class region_number(tennis_ball_detect): 
    def num_tennisball(self): 
        self.__numtennisball = len(contours) 
    def get_numtennisball(self): 
        return self.__numtennisball 
    def centroid(self): 
        self.__coordinates, self.__radc = cv2.minEnclosingCircle(ctr) 
    def get_centroid(self): 
        return self.__coordinates, self.__radc 
 
class general_control(tennis_ball_detect): 
    def centroid(self): 
        momentsctr = cv2.moments(mask) 
        m00 = momentsctr['m00'] 
        self.__center_x, self.__center_y = None, None 
        if m00 != 0: 
            self.__center_x = int(momentsctr['m10'] / m00) 
            self.__center_y = int(momentsctr['m01'] / m00) 
            ctr = (-1, -1) 
    def get_centroid(self): 
        return self.__center_x , self.__center_y 
def callback_val(x): 
    pass 
 
cap= cv2.VideoCapture(0) 
cv2.namedWindow("Term Project") 
cv2.createTrackbar("LH", "Term Project", 29, 255, callback_val) 
cv2.createTrackbar("UH", "Term Project", 62, 255, callback_val) 
cv2.createTrackbar("LS", "Term Project", 47, 255, callback_val) 
cv2.createTrackbar("US", "Term Project", 255, 255, callback_val) 
cv2.createTrackbar("LV", "Term Project", 121, 255, callback_val) 
cv2.createTrackbar("UV", "Term Project", 255, 255, callback_val) 
 
#video_file = ''  # proje"  # if given frames are read from file 
WIDTH =600# width of the windows 
ONLY_MAX = False  # if True only the max circle is drawn 
 
kamera = cv2.VideoCapture(0)  # default web camera=0 
 
cv2.namedWindow('proje') 
cv2.moveWindow('proje', 120, 120)  # 'frame' window position 
 
# Write some Text 
#I add different fonts 
font = cv2.FONT_HERSHEY_SIMPLEX 
font2 = cv2.FONT_HERSHEY_COMPLEX 
font3 = cv2.FONT_HERSHEY_COMPLEX_SMALL 
 
Region1 = range (0, 119) 
Region2 = range (120, 239) 
Region3 = range (240, 359) 
Region4 = range (360, 479) 
Region5 = range (480, 600) 
 
#Define objects 
obj_region = region_number() 
obj_general = general_control() 
 
while True: 
    (ok, frame) = kamera.read() 
 
    l_h = cv2.getTrackbarPos("LH", "Term Project") 
    l_s = cv2.getTrackbarPos("LS", "Term Project") 
    l_v = cv2.getTrackbarPos("LV", "Term Project") 
 
    u_h = cv2.getTrackbarPos("UH", "Term Project") 
    u_s = cv2.getTrackbarPos("US", "Term Project") 
    u_v = cv2.getTrackbarPos("UV", "Term Project") 
 
    GREEN_RANGE = ((l_h, l_s, l_v), (u_h, u_s, u_v)) 
 
    colorLower, colorUpper = GREEN_RANGE  # select color range 
 
    frame = imutils.resize(frame, WIDTH) 
    # blur = cv2.GaussianBlur(frame, (1, 1), 0) 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    mask = cv2.inRange(hsv, colorLower, colorUpper) 
    mask = cv2.erode(mask, None, iterations=3) 
    mask = cv2.dilate(mask, None, iterations=3) 
 
    # kernel = np.ones((5, 5), np.float32) / 25 
    # mask = cv2.filter2D(mask, -1, kernel) 
    # mask = cv2.blur(mask, (5, 5)) 
    # mask = cv2.GaussianBlur(mask, (5, 5), 0) 
    # mask = cv2.medianBlur(mask, 5) 
    # mask = cv2.bilateralFilter(mask, 9, 75, 75) 
 
    result = cv2.bitwise_and(frame, frame, mask=mask) 
    mask_copy = mask.copy() 
    contours = cv2.findContours(mask_copy, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2] 
 
    obj_region.num_tennisball() 
    cont = obj_region.get_numtennisball() 
    if cont > 0: 
        for ctr in contours: 
            if ONLY_MAX: 
                cmax = max(contours, key=cv2.contourArea) 
                obj_region.centroid() 
                (x, y), radius = obj_region.get_centroid() 
            else: 
                obj_region.centroid() 
                (x, y), radius = obj_region.get_centroid() 
 
            if radius >= 10:  # draw circle if radius>50 px 
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 5) 
                if int(x) in Region1: 
                    region = '1. REGION' 

 
                if int(x) in Region2: 
                    region = '2. REGION' 

 
                if int(x) in Region3: 
                    region = '3. REGION' 

 
                if int(x) in Region4: 
                    region = '4. REGION' 

 
                if int(x) in Region5: 
                    region = '5. REGION' 
 
                strXY = str(int(x)) + ',' + str(int(y)) 
                stringName = 'hiii' 
                greenPixel = np.pi * ( radius * radius) 
                stringPixel = str(int(greenPixel)) + 'pixel' 
                stringCounter = str(cont) 
                stringRegion = str(region) 
                cv2.putText(frame, strXY, (int(x), int(y)), font, 1, (255, 255, 0), 2) 
                cv2.putText(frame, stringName, (95, 450), font3, 1, (0, 0, 0), 2) 
                cv2.putText(frame, stringCounter, (50,50), font3, 2, (0,255,0),3) 
                cv2.putText(frame, stringPixel, (int(x), int(y + 60)), font2, 1, (0, 255, 0), 2) 
                cv2.putText(frame, stringRegion, (int(x), int(y + 30)), font, 1, (0, 0, 0), 3) 
 
    obj_general.centroid() 
    (centroid_x,centroid_y)= obj_general.get_centroid() 
 
    if centroid_x != None and centroid_y != None: 
        ctr = (centroid_x, centroid_y) 
        cv2.circle(frame, ctr, 10, (0, 0, 500)) 
 
    cv2.imshow("proje", frame) 
    cv2.imshow("mask", mask) 
    cv2.imshow("result", result) 
    key = cv2.waitKey(2 
                      ) & 0xFF 
    if key == ord('q') or key == 1: 
        break 
kamera.release() 
cv2.destroyAllWindows() 

 

 
