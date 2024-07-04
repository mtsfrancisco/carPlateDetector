import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import easyocr

reader = easyocr.Reader(['en'])

def grabcut_algorithm(original_image, bounding_box):
    
    segment = np.zeros(original_image.shape[:2],np.uint8)
    
    x,y,width,height = bounding_box
    segment[y:y+height, x:x+width] = 1

    background_mdl = np.zeros((1,65), np.float64)
    foreground_mdl = np.zeros((1,65), np.float64)
    
    cv2.grabCut(original_image, segment, bounding_box, background_mdl, foreground_mdl, 5,
    cv2.GC_INIT_WITH_RECT)

    new_mask = np.where((segment==2)|(segment==0),0,1).astype('uint8')

    original_image = original_image*new_mask[:,:,np.newaxis]
    text = reader.readtext(original_image)
    print(text)
    cv2.imshow('Result', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image = cv2.imread("Fotos/carro.webp")
car_model = YOLO(r"license_plate_detector.pt")

car_detections = car_model(image)[0]
for det in car_detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = det
        grabcut_algorithm(image, (int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1)))