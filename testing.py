from ultralytics import YOLO
from collections import Counter

model=YOLO("yolov8n.pt")

# img_path="C:\\DS\\Atharva\\CV\\learning process\\YOLO\\IMAGES\\4)img.png"

# results=model(img_path)

# # for c in results[0].boxes.cls:
# #     print(model.names[int(c)])

# print(results.names())

print(model.names)