import cv2
import numpy as np
import os
from cvzone.HandTrackingModule import HandDetector
import time

wCam,hCam = 640,480
cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)

pTime = 0

detector = HandDetector(minTrackCon=0.8,maxHands=2)

tipIds = [4,8,12,16,20]

while True:
    _,img = cap.read()
    img = cv2.flip(img,1)

    hands,img = detector.findHands(img,flipType=False)
    if hands:
        hand1 = hands[0]
        lmList1 = hand1['lmList']
        bbox1 = hand1['bbox']

        if len(lmList1) != 0:
            fingers = []

            #Left Thumb
            if lmList1[tipIds[0]][0] > lmList1[tipIds[0]-1][0]:
                fingers.append(1)
            else:
                fingers.append(0)

            # #Right Thumb
            # if lmList1[tipIds[0]][0] > lmList1[tipIds[0]-1][0]:
            #     fingers.append(1)
            # else:
            #     fingers.append(0)

            #4 fingers
            for id in range(1,5):
                if lmList1[tipIds[id]][1] < lmList1[tipIds[id]-2][1]: #if index finger tip y is lower than landmark 6 y(below index finger tip)
                    fingers.append(1)
                else:
                    fingers.append(0)
            
            totalFingers = fingers.count(1)
            print(totalFingers)

            cv2.rectangle(img,(20,225),(170,425),(0,255,0),cv2.FILLED)
            cv2.putText(img,str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),25)
                


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS {int(fps)}',(400,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow("Images",img)
    cv2.waitKey(1)