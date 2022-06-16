import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector

# import my trained selfie detection model
selfie_model = tf.keras.models.load_model('./selfie_model.h5')

# use handDetector in cvzone
handDetector = HandDetector(detectionCon=0.8)
poseDetector = PoseDetector()

# use haarcascade in opencv to implement face detection
faceCascPath = "haarcascade_upperbody.xml" 
faceCascade = cv2.CascadeClassifier(faceCascPath)

# get camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access camera!")
    exit()

while True:
    # get a frame
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame!")
        break
    
    # detect hands
    hands, img = handDetector.findHands(frame)
    
    # detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1, #比例因子
        minNeighbors = 3,  #最小鄰距
        minSize=(30, 30) #視窗大小
    )

    #print("Found {0} faces!".format(len(faces)))
    
    # mark the face with bbox
    for (x, y, w, h) in faces:
        bbox_array = cv2.rectangle( frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(bbox_array, "Face", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 0), 2)
    
    
    cv2.imshow("Faces found",  frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()