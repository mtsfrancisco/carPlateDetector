import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
from ultralytics import YOLO
import pytesseract

plate_model = YOLO(r"license_plate_detector.pt")
img = cv2.imread("APENASAPLACA.png")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (int(img.shape[1] * 5), int(img.shape[0] * 5)))

img = cv2.GaussianBlur(img, (5, 5), 0)
_, img = cv2.threshold(img, 110 ,255, cv2.THRESH_BINARY)
cv2.imshow('plate2', img)
cv2.waitKey(0)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
texto2 = pytesseract.image_to_string(img)
print(texto2)
