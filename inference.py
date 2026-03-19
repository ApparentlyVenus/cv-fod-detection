from ultralytics import YOLO
import cv2
import os

model = YOLO('runs/detect/runs/fod/weights/best.pt')

results = model(
    source = 'dataset/test/images',
    conf = 0.25, # only show detections with confidence level above 25
    iou = 0.45, # overlapp ratio, filters duplocates
    save = True,
    project = 'runs',
    name = 'fod_inference',
    exist_ok = True
)

for r in results:
    print(f"Image: {os.path.basename(r.path)} - {len(r.boxes)} detections")