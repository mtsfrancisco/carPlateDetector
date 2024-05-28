import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
from ultralytics import YOLO
import pytesseract

plate_model = YOLO(r"license_plate_detector.pt")
img = cv2.imread(r'C:\Users\mathe\OneDrive\Documentos\Python\Ler_Placa\carPlateDetector\Fotos\placa.png')
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


plate_detections = plate_model(img)[0]

for det in plate_detections.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = det
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    plate = img[int(y1):int(y2), int(x1):int(x2)]

    cv2.imshow('plate1', plate)
    cv2.waitKey(0)

    plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    filtered = cv2.bilateralFilter(plate, 11, 17, 17)
    edged = cv2.Canny(filtered, 30, 200)

    cv2.imshow('plate2', edged)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    license_plate = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            license_plate = plate[y:y+h, x:x+w]
            break

    cv2.imshow('plate2', license_plate)
    
    gaussian_blurred = cv2.GaussianBlur(plate, (3, 3), 0)
    _, binary_image = cv2.threshold(gaussian_blurred, 135 ,255, cv2.THRESH_BINARY_INV)
    
    
    

    #cv2.putText(img, str(texto), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


cv2.imshow('plate', img)
cv2.waitKey(0) 

