import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv4 model
model = YOLO("yolov8n.pt")
car_plate_model = YOLO("license_plate_detector.pt")

#LOAD VIDEO
cap = cv2.VideoCapture('traffic.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)[0]
    for detections in results.boxes.data.tolist():

        x1, y1, x2 ,y2, score, class_id = detections

        if (class_id) == 2:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    results2 = car_plate_model(frame)[0]
    for detections in results2.boxes.data.tolist():

        x1, y1, x2 ,y2, score, class_id = detections
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    # Draw bounding boxes and labels of detections

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()