import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
from ultralytics import YOLO
import pytesseract

plate_model = YOLO(r"license_plate_detector.pt")
img = cv2.imread('placa.png')
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


plate_detections = plate_model(img)[0]
texto2 = "oba"
for det in plate_detections.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = det
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    plate = img[int(y1):int(y2), int(x1):int(x2)]
    texto2 = pytesseract.image_to_string(plate)
    print(texto2)


    plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gaussian_blurred = cv2.GaussianBlur(plate, (3, 3), 0)

    cv2.imshow('plate2', gaussian_blurred)
    

    _, binary_image = cv2.threshold(gaussian_blurred, 135 ,255, cv2.THRESH_BINARY_INV)

    
    

    #cv2.putText(img, str(texto), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

print(texto2)
cv2.imshow('plate', img)
cv2.waitKey(0) 

