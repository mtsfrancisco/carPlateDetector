import cv2
from matplotlib import pyplot as plt
import numpy as np
import pytesseract


img = cv2.imread("APENASAPLACA.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (int(img.shape[1] * 5), int(img.shape[0] * 5)))
img = img[35:-30, 30:-30]



img = cv2.medianBlur(img, 3)
#img = cv2.GaussianBlur(img, (5, 5), 0)
img = cv2.threshold(img, 0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


kernel = np.ones((3,3), np.uint8)
img = cv2.dilate(img, kernel, iterations=2)

texto = pytesseract.image_to_string(img, config='--psm 7')
print("aaaa")
print(texto)

cv2.imshow('plate2', img)
cv2.waitKey(0)

