import cv2
import numpy as np
from PIL import Image

from util import get_limits


colors = {
    'yellow':[0,255,255],
    'red':[0,0,255],
    'green':[0,255,0],
    'blue':[255,0,0]
}
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, brg in colors.items():

        lowerLimit, upperLimit = get_limits(color=brg)

        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

        kernel =  np.ones((5,5),np.uint8)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)

        mask_ = Image.fromarray(mask)

        bbox = mask_.getbbox()

        if bbox is not None:
            x1, y1, x2, y2 = bbox

            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()