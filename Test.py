from ultralytics import YOLO
from collections import Counter

model=YOLO("yolov8n.pt")

img_path="C:\\DS\\Atharva\\CV\\learning process\\YOLO\\IMAGES\\4)img.png"

results=model(img_path)

results[0].show()

num_objects = len(results[0].boxes)        # total detections
print("Total objects detected:", num_objects)

class_counts = Counter(model.names[int(c)] for c in results[0].boxes.cls)
print(class_counts)      # e.g. Counter({'person': 3, 'bottle': 2})


#detected_obj=[model.names[int(cls)] for cls in result[0].boxes.cls.unique()]
#print(detected_obj)

