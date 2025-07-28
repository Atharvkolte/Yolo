#Make camera working ✔️
#detect person ✔️
#and unusal object like knife or bottle ✔️
#when unsual object is detect make sound and take photo/video✔️
#Display the warning sign✔️

###
#Add timestamp when detected
#Record video when person detect
#night vision mode with ir level 
###

import cv2
import time
from ultralytics import YOLO
import winsound
import time
from datetime import datetime
import os

def alertSound():
    winsound.Beep(1000, 500)  # 1000 Hz for 500 ms
    time.sleep(0.2)           # Short pause

def alertTriggered(img):
    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if not os.path.exists("CheckingDetection"):
        os.makedirs("CheckingDetection")
    filename=f"CheckingDetection/detected_{timestamp}.jpg"
    cv2.imwrite(filename,img)

cap=cv2.VideoCapture(0)
model=YOLO("yolov8n.pt")
ALERT_CLASSES = {"fork", "bottle", "knife"}
while True:
    ret,frame=cap.read()
    
    results=model.predict(source=frame,stream=False,verbose=False,iou=0.5,conf=0.5)
    annotated = results[0].plot()
    
    alert_triggered=False
    for result in results:
        boxes=result.boxes
        for box in boxes:
            class_id=int(box.cls)
            class_name=model.names[class_id]
            conf=float(box.conf)
            
            if class_name in ALERT_CLASSES:
                alertSound()
                alert_triggered=True
                print(f"Alert:{class_name} Detected")
                cv2.putText(annotated,f"detected:{class_name}",(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(225,0,225),2)
                
    if alert_triggered:
        alertTriggered(annotated)
        

    cv2.imshow("Webcam",annotated)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()