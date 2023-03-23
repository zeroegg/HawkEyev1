# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 01:23:34 2023

@author: Onur
"""
import cv2
import numpy as np
import torch

model = torch.hub.load('C:\Windows\System32\HawkEye', 'custom','yolov5s', source='local')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for det in results.pred:
        if det is not None and len(det):
            for *xyxy, conf, cls in reversed(det):
                label = f'{model.names[int(cls)]} {conf:.2f}'
                #cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 255, 0), thickness=2)
                cv2.putText(frame, label, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

    cv2.imshow('Object Tracking', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
