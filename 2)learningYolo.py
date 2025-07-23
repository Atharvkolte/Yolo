#Open laptop or mobile camera ✔️ 
#show fps ✔️
#with Yolo detect obect live ✔️
#count each type of object✔️


import cv2
import time
from ultralytics import YOLO
from collections import Counter

model=YOLO("yolov8n.pt")
cap=cv2.VideoCapture(0)
pTime=0

while True:
    ret,frame=cap.read()
    if not ret:
        print("Camera not opened")
        break
    
    #FPS
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(frame,f"FPS:{int(fps)}",(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(225,0,225),2)
    
    result=model.predict(source=frame,stream=True,verbose=False,iou=0.5,conf=0.5)
    for r in result:
        annoted=r.plot()
        detect_count=Counter(model.names[int(c)] for c in r.boxes.cls)
        y=100
        for label,n in detect_count.items():
            cv2.putText(annoted,f"{label}={n}",(20,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(225,0,225),2)
            y+=30
        cv2.imshow("Yolo8n",annoted)

    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()