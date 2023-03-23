# -*- coding: utf-8 -*-
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sn
import time
#from utils import *
# import pandas
# import matplotlib.pyplot as plt
#from object_detection import ObjectDetection
#nvidia-smi #ekran kartı için force komutu

#fps için
t0 = time.time()
startTime = 0
#


points=[]
def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)            
    
#path='C:\Windows\System32\HawkEye\deneme\track.py'           

#model = torch.hub.load('C:\Windows\System32\HawkEye', 'custom', path='C:\Windows\System32\HawkEye\yolov5n.pt', source='local', force_reload=True, autoshape=False)
model = torch.hub.load('C:\Windows\System32\HawkEye', 'custom','yolov5n', source='local')
model.conf=0.15

#print(model)
cap=cv2.VideoCapture(0) #'carvideo.mp4'
count=0
#count2=0

area=[(160,140),(470,140),(470,350),(95,350)] #polyline için dört noktanın koordinatlarını gir
#area=[(x1,y1)(x2,y2)(x3,y3)(x4,y4)]
#area=[(225,125),(747,125),(747,400),(145,400)] #1020,600 için

while True:
    ret,frame=cap.read() #ret, img
    if not ret:
        break
    
    # count += 1 #to skip 4 frames, means faster
    # if count % 4 != 0:
    #     continue
    
    frame=cv2.resize(frame,(640,480)) #1020,600
    
    #fps için:
    currentTime = time.time()

    fps = 1/(currentTime - startTime)
    startTime = currentTime

    cv2.putText(frame, "FPS: " + str(int(fps)), (10,50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
    
    #
    
    results=model(frame)
    #results=utils.non_max_suppression(results, 80, 0.2, 0.4)
    list=[]
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d=(row['name'])
        cx=int(x1+x2)//2
        cy=int(y1+y2)//2
        
        if 'sports ball' in d: # nesne adı, datasette nasıl yazıldıysa
           results=cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
           #cv2.circle(frame, ((int(x1+x2/2), int(y1+y2/2))), 5, (255,0,0), 2)
           if results == 1:
               print('IN')
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
               cv2.putText(frame,str(d),(x1,y1),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2) #2,2 default
               cv2.putText(frame,str('IN'),(185,50),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2) #(185,50)
               #cv2.circle(frame, (cx,cy), 5,(255,0,0),2) #normalde nesnenin çerçevesine orta nokta koymak için ama bu hali çalışmıyor
               
               #list.append([cx])
           if results != 1:
               print('OUT')
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
               cv2.putText(frame,str(d),(x1,y1),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
               cv2.putText(frame,str('OUT'),(185,50),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)                 
               
        # if 'sports ball' not in d:
        #     results2=cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
        #     if results2 == 1:
        #         print('OUT')
        #         cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
        #         cv2.putText(frame,str(d),(x1,y1),cv2.FONT_HERSHEY_DUPLEX,2,(255,0,0),2)
        #         cv2.putText(frame,str('OUT'),(185,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
                
           # if results>=0: 
           #     print(results)
           #     cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
           #     cv2.putText(frame,str(d),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
           #     list.append([cx])
    cv2.polylines(frame, [np.array(area, np.int32)], True, (0,255,0), 2) #yeşil ve kalınlığı iki olan poly çiz
    #a=print(len(list))
    #cv2.putText(frame,str(a),(40,40),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2) #area içindeki nesnelerin sayısını (40,40) pixeline yaz
    cv2.imshow("FRAME",frame)
    cv2.setMouseCallback("FRAME",POINTS)
    # b=results.pandas().xyxy[0]
    # print(b)

    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

