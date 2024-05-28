import cv2
from matplotlib import pyplot as plt
import numpy as np
from imutils.object_detection import non_max_suppression
import easyocr
from ultralytics import YOLO
import pytesseract
import time

plate_model = YOLO(r"license_plate_detector.pt")
easypcr = easyocr.Reader(['en'])


min_confidence = 0.8
text_model = "frozen_east_text_detection.pb"
net = cv2.dnn.readNet(text_model)

# QUANDO FOR USAR NO WINDOWS
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def normalizeImg(image, newSize = (320, 320)):
    (H, W) = image.shape[:2]
    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = newSize
    rW = W / float(newW)
    rH = H / float(newH)
    # resize the image and grab the new image dimensions
    image_norm = cv2.resize(image, (newW, newH))
    (H, W) = image_norm.shape[:2]

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    return image_norm, H,W, rW,rH

def bounding_boxes(scores,geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < min_confidence:
                continue
            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    return non_max_suppression(np.array(rects), probs=confidences), confidences

layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

kernel = np.ones((3,3), np.uint8)
video = cv2.VideoCapture('Video/VideoYoutube.mp4')
while True:
    ret, frame = video.read()
    if not ret:
        break


    plate_detections = plate_model(frame)[0]

    for det in plate_detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = det
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        plate = frame[int(y1):int(y2), int(x1):int(x2)]
        binary_image = cv2.resize(plate, (int(plate.shape[1] * 5), int(plate.shape[0] * 5)))
        #binary_image = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        # _, binary_image = cv2.threshold(plate, 0 ,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)   
        # binary_image = cv2.erode(binary_image, kernel, iterations = 2)
        # binary_image = cv2.dilate(binary_image, kernel, iterations = 1)


        image_norm, H,W, rW,rH = normalizeImg(binary_image)
        blob = cv2.dnn.blobFromImage(image_norm, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
        
        start = time.time()
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        end = time.time()
        # show timing information on text prediction
        print("[INFO] text detection took {:.6f} seconds".format(end - start))

        boxes,confidences  = bounding_boxes(scores, geometry)
        # boxes = np.array(rects)
        # remove the bounding boxes that are too narrow or too short
        boxes = [box for box in boxes if (box[2]-box[0]) > 50]

        # make sure the largetest bounding box is fully expended horizontally
        boxes = sorted(boxes, key=lambda box: (box[2]-box[0])*(box[3]-box[1]))
        startX, startY, endX, endY = boxes[-1]
        startX = 0 if startX > W/4 else startX
        endX = W if endX < 3*W/4 else endX
        boxes[-1] = (startX, startY, W, endY)
            
        orig = binary_image.copy()
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            # draw the bounding box on the image
            cntImage = cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
      

        #plate_text = pytesseract.image_to_string(binary_image, config='--psm 6')
        #filtrado = ''.join(filter(lambda x: x.isalnum(), plate_text))

        for (startX, startY, endX, endY) in boxes:
                # scale the bounding box coordinates based on the respective
                # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

        new_image = orig[startY-3:endY+3, startX:endX]
        plate_text = easypcr.readtext(new_image)
        plate = plate_text[0][1] if len(plate_text) > 0 else ""
        
        
        #plate_texts.append(str(filtrado))
        #print(plate_texts)
        # if len(plate_texts) % 4 == 0:
        #     most_common_plate_text = max(set(plate_texts), key=plate_texts.count)
        #     print(most_common_plate_text)
        #     cv2.putText(frame, most_common_plate_text, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if plate_text != "":
            cv2.putText(frame, plate, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
