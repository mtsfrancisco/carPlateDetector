import pytesseract
import cv2 as cv

img = cv.imread("carta.jpeg")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.medianBlur(img, 3)
img = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
texto = pytesseract.image_to_string(img, config='--psm 7')
print(texto)
