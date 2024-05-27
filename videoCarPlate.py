import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
from ultralytics import YOLO
import pytesseract

plate_model = YOLO(r"license_plate_detector.pt")

video = cv2.VideoCapture('2103099-hd_1920_1080_30fps.mp4')
#video = cv2.VideoCapture(0)


# QUANDO FOR USAR NO WINDOWS
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# frame_rate = video.get(cv2.CAP_PROP_FPS)
# interval = int(frame_rate)

kernel = np.ones((3,3), np.uint8)
plate_texts = []

while True:
    ret, frame = video.read()
    if not ret:
        break


    plate_detections = plate_model(frame)[0]

    for det in plate_detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = det
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        plate = frame[int(y1):int(y2), int(x1):int(x2)]
        plate = cv2.resize(plate, (int(plate.shape[1] * 5), int(plate.shape[0] * 5)))

        plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        
        
 
        #plate = plate[int(-plate.shape[1]*0.20):int(plate.shape[1]*0.250), int(plate.shape[0]*0.45):int(-plate.shape[0]*0.25)]
        


        # cv2.imshow('plate', plate)
        # cv2.waitKey(0)

        #plate = plate[55:-30, 45:-45]
        
        #plate = cv2.medianBlur(plate, 3)
        _, binary_image = cv2.threshold(plate, 0 ,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)   
        
        #cv2.imshow('plate', binary_image)
        #cv2.waitKey(0)

        binary_image = cv2.erode(binary_image, kernel, iterations = 2)
        #binary_image = cv2.dilate(binary_image, kernel, iterations = 2)
        
        # cv2.imshow('plate', binary_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        plate_text = pytesseract.image_to_string(binary_image, config='--psm 7')
        filtrado = ''.join(filter(lambda x: x.isalnum(), plate_text))
        
        #plate_texts.append(str(filtrado))
        #print(plate_texts)
        # if len(plate_texts) % 4 == 0:
        #     most_common_plate_text = max(set(plate_texts), key=plate_texts.count)
        #     print(most_common_plate_text)
        #     cv2.putText(frame, most_common_plate_text, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, filtrado, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
        
        cv2.imshow('plate', binary_image)
        cv2.waitKey(0)

        #cv2.putText(frame, filtrado, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
