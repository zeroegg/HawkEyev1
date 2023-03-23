# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 00:00:52 2023

@author: Onur
"""

import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
#import detect
import time
#from PIL import ImageGrab #save img için: line 90
#nvidia-smi
#from  hubconf import custom #yolov7 için

#model = custom(path_or_model='C:\Windows\System32\YOLOv7\yolov7.pt')
model = torch.hub.load('C:\Windows\System32\HawkEye', 'custom','yolov5s', source='local')
model.conf=0.15 #confidence threshold 0.15
model.iou = 0.35 #iou threshold'intersection over union region'

#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Open video stream
cap = cv2.VideoCapture(0)

t0 = time.time()
startTime = 0

# Define region of interest
# roi = (np.array([[10, 50], [400, 50], [400, 200], [10, 200]], np.int32))

# pts = np.array([ [50,50], [150,50], [200,200], [25,200] ], np.int32)
# pts = pts.reshape((-1, 1, 2))
# isClosed = True
# color = (255, 0, 0)

#frame = cv2.polylines(frame, [pts], isClosed, color, thickness=4)


# #
count=0           
area=[(65,223),(496,218),(628,248),(47,320)]

# ##
while True:
    
    # Read frame from video stream
    ret, frame = cap.read()
    if not ret:
        break
    
    # count += 1 #to skip 4 frames
    # if count % 4 != 0:
    #       continue
      
    #frame = cv2.polylines(frame, [pts], isClosed, color, thickness=4)
    frame=cv2.resize(frame,(640,480)) #1020,600-640,480
    # Crop region of interest
    #x1, y1, x2, y2 = roi
    #frame = cv2.polylines(frame, [pts], isClosed, color, thickness=4)
    #frame = frame[y1:y2, x1:x2]
    
    currentTime = time.time()
    
    fps = 1/(currentTime - startTime)
    startTime = currentTime
    
    cv2.putText(frame, "FPS: " + str(int(fps)), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
    # Convert to PIL image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #img = Image.fromarray(img)

    # Run detection
    results = model(frame, size=640) #frame=results
    
    print(results.s) #shell'e size yazdır
    
    #labels, cord_thres = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1] #confidence ve label verilerini çeker
    #results.xyxy[0][:, :-1]
    #c=results.pandas().xyxy[0].confidence
    #results=utils.non_max_suppression(results, 80, 0.2, 0.4)
    
    #list=[]
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d=(row['name'])
        c=(row['confidence'])
        cx=int(x1+x2)//2
        cy=int(y1+y2)//2
        
        # #to save detected image
        # im = ImageGrab.grab()
        # results.save() #pandaya bağlı olanlarda float olduğu için hata veriyo tekrar bak
        #print(dir(frame)) #objenin type'ını yazdırır
        
        if 'sports ball' in d: # aranan nesne adı, datasette nasıl yazıldıysa
            results=cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
            #cv2.circle(frame, ((int(x1+x2/2), int(y1+y2/2))), 5, (255,0,0), 2)
            if results == 1:
                print('IN')
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
                cv2.putText(frame,str(d),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2) #2,2 default
                cv2.putText(frame,str('IN'),(185,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2) #(185,50)
                #cv2.circle(frame, (cx,cy), 5,(255,0,0),2) #normalde nesnenin çerçevesine orta nokta koymak için ama bu hali çalışmıyor
                cv2.putText(frame,str("%.2f" % c),(x2,y2),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2) #2,2 default cord_thres[:,-1]
                
            if results != 1:
                print('OUT')
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
                cv2.putText(frame,str(d),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                cv2.putText(frame,str('OUT'),(185,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)           
                cv2.putText(frame,str("%.2f" % c),(x2,y2),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                
    cv2.polylines(frame, [np.array(area, np.int32)], True, (0,255,0), 2) #yeşil ve kalınlığı iki olan poly çiz
    #results.print()
    # Draw bounding boxes on image
    #results.render() #frame=results
    #frame = cv2.cvtColor(np.array(frame.ims[0]), cv2.COLOR_RGB2BGR) #ims değiştir

    # Show image
    cv2.imshow("FRAME",frame)
    #cv2.setMouseCallback("FRAME",POINTS)
    
    
    
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video stream and close window
cap.release()
cv2.destroyAllWindows()
