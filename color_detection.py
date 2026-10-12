import cv2 as cv
import numpy as np 
from PIL import Image
from  color import Select_color

color_detect = [255,0,0]
web_cam = cv.VideoCapture(0)
while True:
    ret,frame = web_cam.read()
    hsv_image = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    lower_limit,upper_limit =  Select_color(color= color_detect)
    mask = cv.inRange(hsv_image,lower_limit,upper_limit)
    mask_ = Image.fromarray(mask)
    bbox= mask_.getbbox()
    if bbox is not None:
        x1,y1,x2,y2 = bbox
        cv.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
    cv.imshow("",frame)
    if cv.waitKey(40) & 0xFF == ord('e'):
        break
web_cam.release()
cv.destroyAllWindows()