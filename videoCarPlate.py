import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
from ultralytics import YOLO
import pytesseract

plate_model = YOLO(r"license_plate_detector.pt")

video = cv2.VideoCapture('Watch catastrophic parking fail as Jaguar driver hits two parked cars _ SWD Media.mp4')

# QUANDO FOR USAR NO WINDOWS
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


#frame_rate = video.get(cv2.CAP_PROP_FPS)
#interval = int(frame_rate)

kernel = np.ones((3,3), np.uint8)

while True:
    ret, frame = video.read()
    # if not ret:
    #     break

    # current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    # if current_frame % interval != 0:
    #     continue

    plate_detections = plate_model(frame)[0]

    for det in plate_detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = det
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        plate = frame[int(y1):int(y2), int(x1):int(x2)]
        plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        
        plate = cv2.resize(plate, (int(plate.shape[1] * 5), int(plate.shape[0] * 5)))
        plate = plate[55:-30, 45:-45]
        plate = cv2.medianBlur(plate, 3)

        _, binary_image = cv2.threshold(plate, 0 ,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        binary_image = cv2.dilate(binary_image, kernel, iterations=2)
        
        
        cv2.imshow('plate', binary_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        plate_text = pytesseract.image_to_string(binary_image, config='--psm 7')
        cv2.putText(frame, plate_text, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
