#Make camera working ✔️
#detect person ✔️
#and unusal object like knife or bottle 
#when unsual object is detect make sound and take photo/video
#Display the warning sign

###
#Add timestamp when detected
#Record video when person detect
#night vision mode with ir level 
###

import cv2
import time
from ultralytics import YOLO

cap=cv2.VideoCapture(0)
model=YOLO("yolov8n.pt")
ALERT_CLASSES = {"fork", "bottle", "knife"}
while True:
    ret,frame=cap.read()
    
    results=model.predict(source=frame,stream=False,verbose=False,iou=0.5,conf=0.5)
    annotated = results[0].plot()
    
    cv2.imshow("Webcam",annotated)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()