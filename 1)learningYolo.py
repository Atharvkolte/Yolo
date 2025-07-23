from ultralytics import YOLO
from collections import Counter
import time 
import cv2
import os
import sys

class YoloClassification:
    ### It initalize the Path
    def __init__(self,path,model="yolov8n.pt",output_dir="detectedImage"):
        self.path=path
        if not os.path.exists(self.path):
            print("File does not exist:", self.path)
            sys.exit(1)
        # Load a pre-trained YOLOv8 model
        self.model = YOLO(model)  # 'n' means nano model - very small and fast
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir,exist_ok=True)
        
        # Extract the original filename (without extension)
        original_filename=os.path.splitext(os.path.basename(self.path))[0]
        
        # Define the output filename (e.g., "detected_1img.jpg")
        output_filename=f"detected_{original_filename}.png"
        self.output_path= os.path.join(output_dir,output_filename)
    
    ### Displays all the names in the yolo model
    def canBepredict(self):
        print(self.model.names)
    
    ### It detect and store the data
    def detectAndSave(self,output_dir=None):
        # Load a pre-trained YOLOv8 model
        #model = YOLO('yolov8n.pt')  # 'n' means nano model - very small and fast
        results = self.model(self.path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            original_filename = os.path.splitext(os.path.basename(self.path))[0]
            output_filename = f"detected_{original_filename}.png"
            output_path = os.path.join(output_dir, output_filename)
        else:
            output_path=self.output_path

        # Show the detected objects on the image
        results[0].show()
         
        # Save the result
        results[0].save(filename=output_path)

    ### Detecting and counting
    def detectCount(self,class_id=None):
        if class_id:
            result=self.model(self.path,classes=[class_id])### there pretrained model in yolo class 0 stand for person u can detect on cat dog or anything
        else:
            result=self.model(self.path)
        result[0].show()
        detect_count=Counter(self.model.names[int(c)] for c in result[0].boxes.cls)
        print(detect_count)
        return detect_count

    ### Real time seeing
    def realTimeDetection(self):
        cap=cv2.VideoCapture(0)
        while True:
            ret,frame=cap.read()
            if not ret:
                print("Something not working")
                return
        
            #results = self.model.predict(source=frame, show=True)
            # Detect objects 
            # run inference (stream=True keeps the model in memory => faster)
            results = self.model.predict(source=frame, stream=True,verbose=False)
            
            
            for r in results: # one result per frame
                annoted=r.plot()  # Ultralytics helper draws boxes/labels
                cv2.imshow("YOLOv8n",annoted)
            
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    ### Real time detection and count
    def realTimeDetectionCount(self):
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
            
            result=self.model.predict(source=frame,stream=True,verbose=False,iou=0.5,conf=0.5)
            for r in result:
                annoted=r.plot()
                detect_count=Counter(self.model.names[int(c)] for c in r.boxes.cls)
                y=100
                for label,n in detect_count.items():
                    cv2.putText(annoted,f"{label}={n}",(20,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(225,0,225),2)
                    y+=30
                cv2.imshow("Yolo8n",annoted)

            
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()



def main():
    ###################################
    path="C:\\DS\\Atharva\\CV\\learning process\\YOLO\\IMAGES\\5)img.png"
    model="yolov8n.pt"
    detector=YoloClassification(path,model)
    ###################################
    
    detector.canBepredict()
    #detector.detectAndSave()
    #detector.realTimeDetection()
    #detector.detectCount()
    #detector.realTimeDetectionCount()

if __name__=="__main__":
    main()
