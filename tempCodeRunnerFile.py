import cv2
import time
from ultralytics import YOLO

cap=cv2.VideoCapture(0)
model=YOLO("yolov8n.pt")

while True:
    ret,frame=cap.read()
    
    results=model.predict(source=frame,stream=True,verbose=False,iou=0.5,conf=0.5)
    annotated = results[0].plot()
    
    cv2.imshow("Webcam",annotated)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()