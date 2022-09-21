import cv2
import time
import os
from cvzone.HandTrackingModule import HandDetector

wCam, hCam = 640, 480
 
cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)

detector = HandDetector(minTrackCon=0.8,maxHands=2)
pTime = 0

while True:
    _, img = cap.read()
    hands , img = detector.findHands(img)

    if hands:
        hand1 = hands[0]
        lmList1 = hand1['lmList']

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
 
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
 
    cv2.imshow("Image", img)
    cv2.waitKey(1)