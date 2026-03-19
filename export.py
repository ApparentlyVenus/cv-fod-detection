from ultralytics import YOLO
import time
import torch
import os


model = YOLO('runs/detect/runs/fod/weights/best.pt')

model.export(format='onnx', imgsz=640, simplify=True)
print('Export complete')

print("\nBenchmarking PyTorch vs ONNX inference speed...")

img = torch.zeros(1, 3, 640, 640)

# benchmark pytorch
start = time.perf_counter()
for _ in range(100):
    model(img, verbose=False)
pytorch_ms = (time.perf_counter() - start) / 100 * 1000

# benchmark onnx
onnx_model = YOLO('runs/detect/runs/fod/weights/best.onnx')
start = time.perf_counter()
for _ in range(100):
    onnx_model(img, verbose=False)
onnx_ms = (time.perf_counter() - start) / 100 * 1000

print(f"PyTorch inference:  {pytorch_ms:.2f}ms per image")
print(f"ONNX inference:     {onnx_ms:.2f}ms per image")
print(f"\nONNX file saved to: runs/detect/runs/fod/weights/best.onnx")
print(f"Model size: {os.path.getsize('runs/detect/runs/fod/weights/best.onnx') / 1e6:.1f}MB")