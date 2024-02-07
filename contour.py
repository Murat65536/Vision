import cv2
import numpy as np

def nothing(n):
    pass

def biggestContourI(contours):
    maxVal = 0
    maxI = None
    for i in range(0, len(contours)):
        if len(contours[i]) > maxVal:
            cs = contours[i]
            maxVal = len(contours[i])
            maxI = i
    return maxI
            

iLowH = 0
iHighH = 10
iLowS = 116
iHighS = 226
iLowV = 182
iHighV = 255

cv2.namedWindow('Control')
cv2.createTrackbar("LowH", "Control", iLowH, 255, nothing)
cv2.createTrackbar("HighH", "Control", iHighH, 255, nothing)
cv2.createTrackbar("LowS", "Control", iLowS, 255, nothing)
cv2.createTrackbar("HighS", "Control", iHighS, 255, nothing)
cv2.createTrackbar("LowV", "Control", iLowV, 255, nothing)
cv2.createTrackbar("HighV", "Control", iHighV, 255, nothing)

cam = cv2.VideoCapture(0)

while True:
    ret_val, img = cam.read()

    lh = cv2.getTrackbarPos('LowH', 'Control')
    ls = cv2.getTrackbarPos('LowS', 'Control')
    lv = cv2.getTrackbarPos('LowV', 'Control')
    hh = cv2.getTrackbarPos('HighH', 'Control')
    hs = cv2.getTrackbarPos('HighS', 'Control')
    hv = cv2.getTrackbarPos('HighV', 'Control')

    lower = np.array([lh, ls, lv], dtype = "uint8")
    higher = np.array([hh, hs, hv], dtype = "uint8")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    flt = cv2.inRange(hsv, lower, higher)

    contours, hierarchy = cv2.findContours(flt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    bc = biggestContourI(contours)
    cv2.drawContours(img, contours, bc, (0,255,0), 3)

    if (len(contours) > 0):
        x1,y1,x2,y2 = cv2.boundingRect(contours[bc])
        cv2.rectangle(img, (x1, y1), (x1+x2, y1+y2), (0, 255, 0), 2)
        imageCenterX = round(x1+x2/2)
        imageCenterY = round(y1+y2/2)
        cv2.circle(img, (imageCenterX, imageCenterY), 5, (0, 255, 0), -1)

    cv2.circle(img, (round(cam.get(cv2.CAP_PROP_FRAME_WIDTH)/2), round(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)), 5, (255, 0, 0), -1)

    cv2.imshow('cam', img)
    cv2.imshow('hsv', hsv)
    cv2.imshow('flt', flt)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()