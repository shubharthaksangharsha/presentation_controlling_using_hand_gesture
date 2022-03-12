import cv2
from cv2 import flip 
import numpy as np 
from cvzone.HandTrackingModule import HandDetector
import os 
#VARIABLES
width, height = 640, 480
folderPath = 'Presentation'
imgNumber = 0
hs, ws = int(640//8 * 2), int(480//8 * 3)
gestureThreshold = 640//2 - 80
buttonPressed = False
buttonCounter = 0
buttonDelay = 10
annotations = [[]]
annotationNumber = 0
annotationStart = False
#Camera Setup
cap = cv2.VideoCapture(0)
#Get the list of presentation images 
pathImages = sorted(os.listdir(folderPath))

#Hand Detector 
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    
    hands, img =  detector.findHands(img)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)
    if hands and buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmList = hand['lmList']
        
        #Constraints values for easier drawing
        xVal = int(np.interp(lmList[8][0], [width // 2, w], [0, width + 1000]))
        yVal = int(np.interp(lmList[8][1], [170, height - 170], [0, height]))
        indexFinger = xVal, yVal
         
        
        
        if cy <= gestureThreshold: #if hand is at the height of the face 
            
            #Gesture 1- Left 
            if fingers == [1, 0, 0, 0, 0]:
                annotationStart = False
                print('Left')
                if imgNumber > 0:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = 0
                    
                    imgNumber -= 1
                
            #Gesture 2- Right
            if fingers == [0, 0, 0, 0, 1]:
                annotationStart = False
                print('Right')
                if imgNumber < len(pathImages) -1 :
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = 0
                
                    imgNumber += 1
            
        #Gesture 3 - Show Pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED )       
            

        #Gesture 4 - Draw Pointer
        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:                     
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED ) 
            annotations[annotationNumber].append(indexFinger)
        else:
            annotationStart = False
            
        #Gesture 5 - Erase
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                if annotationNumber >= 0:
                    annotations.pop(-1)
                    annotationNumber -= 1
                    buttonPressed = True
    else:
        annotationStart = False
        
    #Button Pressed Iterations 
    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False
    
    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j != 0:
                cv2.line(imgCurrent, annotations[i][j-1], annotations[i][j], (0, 0, 200), 12)
    #Adding webcam image on the slides
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w-ws: w] = imgSmall
    
    cv2.imshow('Presentation', img)
    cv2.imshow('Slides', imgCurrent)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
