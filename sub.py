import cv2 as cv
from os import path, listdir
from math import sqrt, pi, atan
import numpy as np

def MinValue(point, ts1, ts2,ts3):
    min = 99999
    for x,y in point:
        a = sqrt((ts1 - x)**2 + (ts2 - y)**2)
        if a<min and a > ts3:
            min = a
            A = (x,y)
    return A
def Detect_Two_point(point,center):
        A = MinValue(point,center[0],center[1],0)
        point.remove(A)
        B = MinValue(point, A[0], A[1],350) # 350
        return (A,B)
def Calculate_Angle(A,B):
    if B[1] == A[1]:
        if B[0] < A[0]:
            return 0
        if B[0] > A[0]:
            return 180
    if B[0] == A[0]:
        if B[1] < A[1]:
            return 90
        if B[1] > A[1]:
            return 270
    if (A[0] != B[0]) and A[1] != B[1]:
        angle = (atan(abs((A[1] - B[1]) / (A[0] - B[0]))) / pi) * 180

        if B[0] < A[0] and B[1] > A[1]:
            return 360-angle
        if B[0] > A[0] and B[1] > A[1]:
            return 180+angle
        if B[0] > A[0] and B[1] < A[1]:
            return 180 - angle
        if B[0] < A[0] and B[1] < A[1]:
            return angle

#function xoay ảnh theo góc, tâm và trả về một ảnh mới
def Rotate_Img(img, angle, center):
    rows = img.shape[0]
    cols = img.shape[1]
    T = cv.getRotationMatrix2D(center, angle, 1)
    dst = cv.warpAffine(img, T, (cols, rows))

    return dst

#Crop 4 vị trí ốc ta cần xác định trả về 4 vùng bao quanh vị trí đó và tọa độ tâm vị trí đó
def Crop_img(img, center):
    x,y = center
    x= int(x)
    y = int(y)

    A = (x+390, y+ 91)
    B = (x + 318, y + 242)
    C = (x -135, y + 534)
    D = (x+177,  y - 535)
    point = [A,B,C,D]

    Crop1 = img[( y+ 91 -40) :( y+91+40 ),(x+ 390 -40):(x+390 + 40)]
    Crop2 = img[ (y+242 -40): (y+242+40),(x + 318 -40):(x+318+40)]
    Crop3 = img[ (y+534 - 40): (y+534+ 40),(x- 135 - 40):(x - 135 + 40)]
    Crop4 = img[ (y -535 -40): (y-535+40),(x+177 -40): (x+177+40)]

    Crop = [Crop1, Crop2, Crop3, Crop4]
    return Crop ,point
def Check_Crop_img(img, name):

       gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

       blur = cv.GaussianBlur(gray, (5,5), 0)
       ret, thresh = cv.threshold(blur, 5, 255, cv.THRESH_BINARY)
       crop = thresh[30:50,30:50]

       x = np.where(crop != 255)
       for i in range(len(x)):
            if len(x[i]) >0:
                return False
       return True



def DetectCircle(img,mode,model):
    point = []

    imgcp = cv.resize(img, (1224,1024))
    graycp = cv.cvtColor(imgcp, cv.COLOR_BGR2GRAY)
    circles =  cv.HoughCircles(graycp, cv.HOUGH_GRADIENT,1, 100,
                             param1 = 100, param2 = 200, minRadius= 100,maxRadius= 500) #500
    detected = np.uint16(np.around(circles))
    for x,y,r in detected[0, :]:
        center = (x*2,y*2,r*2)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)
    ret, thresh = cv.threshold(blur, 62, 255, cv.THRESH_BINARY)

    contours, hiearchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, h, w = cv.boundingRect(cnt)
        if w * h < 5700 and w * h > 2000 and w<1.3*h and h<1.3*w:

            point.append((x + w / 2, y + h / 2))

    point = Detect_Two_point(point, center)
    angle = Calculate_Angle(point[0], point[1])
    img = Rotate_Img(img, angle, point[0])


    crop, point = Crop_img(img, point[0])

    return Draw_Rec(img, crop,point,mode,model)

from model import DL_Check_Missing_Screw

def Draw_Rec(img,crop,point,mode,model):
    if mode == 'opencv':
        for i in range(len(crop)):
            s = Check_Crop_img(crop[i], str(i))
            x,y = point[i]
            if s == True:
                cv.rectangle(img, (x - 40, y - 40), (x + 40, y + 40), (0, 255, 0), 3)
            if s == False:
                cv.rectangle(img, (x - 40, y - 40), (x + 40, y + 40), (0, 0, 255), 3)
    elif mode == 'Dl model':
        for i in range(len(crop)):
            s = DL_Check_Missing_Screw(crop[i], model)
            x,y = point[i]
            if s == 1:
                cv.rectangle(img, (x - 40, y - 40), (x + 40, y + 40), (0, 255, 0), 3)
            if s == 0:
                cv.rectangle(img, (x - 40, y - 40), (x + 40, y + 40), (0, 0, 255), 3)
    return img