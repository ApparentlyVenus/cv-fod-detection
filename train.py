from ultralytics import YOLO
import torch

model = YOLO('yolov8n.pt')

results = model.train(
    data = 'dataset/data.yaml',
    epochs = 50, # every training image is seen 50 times
    imgsz = 640, # image size is 640 x 640
    batch = 16, # how many images go simultaneously
    project = 'runs', # output file
    name = 'fod', # suboutput file
    exist_ok = True, 
    verbose= True
)