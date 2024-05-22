import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

video = cv2.VideoCapture('traffic.mp4')


def plateDetect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
    edged = cv2.Canny(bfilter, 30, 200) #Edge detection
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    global location
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    maskk = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(maskk, [location], 0,255, -1)
    new_image = cv2.bitwise_and(img, img, mask=maskk)

    (x,y) = np.where(maskk==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]
    return cropped_image

    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)

    if result == []:
        return img
    
    text = result[0][-2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
    res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
    return res


while True:
    ret, frame = video.read()
    new_frame = plateDetect(frame)
    cv2.imshow('Video', new_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()