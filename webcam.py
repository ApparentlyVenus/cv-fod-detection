import cv2
from ultralytics import YOLO
import time

model = YOLO('runs/detect/runs/fod/weights/best.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: could not open webcam")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Webcam running. Press Q to quit.")

fps_list = []

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: could not read frame")
        break

    start = time.perf_counter()

    results = model(frame, conf=0.25, iou=0.45, verbose=False)

    end = time.perf_counter()
    inference_ms = (end - start) * 1000
    fps = 1000 / inference_ms
    fps_list.append(fps)

    annotated = results[0].plot()

    num_detections = len(results[0].boxes)

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f"Detections: {num_detections}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f"Model: YOLOv8n FOD",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow('FOD Detection - Live', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
print(f"\nSession complete.")
print(f"Average FPS: {avg_fps:.1f}")
print(f"Average inference: {1000/avg_fps:.1f}ms" if avg_fps > 0 else "")

cap.release()
cv2.destroyAllWindows()