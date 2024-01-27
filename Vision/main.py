import cv2
from PIL import Image
import numpy

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimit = numpy.array([6, 110, 110], dtype=numpy.uint8)
    upperLimit = numpy.array([14, 240, 240], dtype=numpy.uint8)

    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    mask_ = Image.fromarray(mask)

    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        
        frame = cv2.putText(frame, ("X1: " + str(x1) + " Y1: " + str(y1) + " X2: " + str(x2) + " Y2: " + str(y2)), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()