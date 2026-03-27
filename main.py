import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np 

width, height = 1280, 720
folderPath = "presentation"

if not os.path.exists(folderPath):
    print("presentation folder missing")
    exit()

pathImages = sorted(os.listdir(folderPath), key=len)
if len(pathImages) == 0:
    print("No slides found")
    exit()

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

imgNumber = 0
hs, ws = 180, 320

gestureThreshold = 300
buttonPressed = False
buttonCounter = 0
buttonDelay = 15

annotations = [[]]
annotationsNumber = 0
annotationsStart = False

detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hCam, wCam, _ = img.shape   # webcam size

    # Load slide
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    imgCurrent = cv2.resize(imgCurrent, (width, height))

    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0,255,0), 5)

    # ================= HAND DETECTED =================
    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmlist = hand['lmList']

        # ---------- COORDINATE MAPPING ----------
        x, y = lmlist[8][0], lmlist[8][1]
        xSlide = int(np.interp(x, [0, wCam], [0, width]))
        ySlide = int(np.interp(y, [0, hCam], [0, height]))
        indexFinger = (xSlide, ySlide)

        # =========================================================
        # ✋ ZONE 1 : SLIDE CONTROL (ONLY ABOVE GREEN LINE)
        # =========================================================
        if cy < gestureThreshold and not buttonPressed:

            # 👍 Previous slide
            if fingers == [1,0,0,0,0]:
                if imgNumber > 0:
                    imgNumber -= 1
                    buttonPressed = True
                    annotations = [[]]
                    annotationsNumber = 0

            # 🤙 Next slide
            elif fingers == [0,0,0,0,1]:
                if imgNumber < len(pathImages)-1:
                    imgNumber += 1
                    buttonPressed = True
                    annotations = [[]]
                    annotationsNumber = 0

        # =========================================================
        # ✏️ ZONE 2 : POINTER + DRAW (WORKS EVERYWHERE)
        # =========================================================

        # 👉 POINTER
        if fingers == [0,1,1,0,0]:
            cv2.circle(imgCurrent, indexFinger, 12, (255,0,255), cv2.FILLED)
            annotationsStart = False

        # ✍️ DRAW
        elif fingers == [0,1,0,0,0]:
            if annotationsStart is False:
                annotationsStart = True
                annotationsNumber += 1
                annotations.append([])
            cv2.circle(imgCurrent, indexFinger, 12, (255,0,255), cv2.FILLED)
            annotations[annotationsNumber].append(indexFinger)

        else:
            annotationsStart = False

        # 🧽 ERASE LAST DRAW
        if fingers == [0,1,1,1,0]:
            if len(annotations) > 1:
                annotations.pop(-1)
                annotationsNumber = max(annotationsNumber-1,0)
                buttonPressed = True

    # Delay between slide changes
    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False

    # Draw all annotations
    for i in range(len(annotations)):
        for j in range(1, len(annotations[i])):
            cv2.line(imgCurrent, annotations[i][j-1], annotations[i][j], (0,0,200), 12)

    # Webcam overlay
    imgSmall = cv2.resize(img, (ws, hs))
    imgCurrent[0:hs, width-ws:width] = imgSmall

    cv2.imshow("Webcam", img)
    cv2.imshow("Presentation", imgCurrent)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
