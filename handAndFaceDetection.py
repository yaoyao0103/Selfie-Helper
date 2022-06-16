import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector
import keras.backend as K
from cvzone.SelfiSegmentationModule import SelfiSegmentation


# import my trained selfie detection and hand sign detection model
selfie_model = tf.keras.models.load_model('./selfie_model.h5')
hand_model = tf.keras.models.load_model('./hand_model.h5')

# use handDetector in cvzone
segmentor = SelfiSegmentation()
handDetector = HandDetector(detectionCon=0.8)
poseDetector = PoseDetector()

# use haarcascade in opencv to implement face detection
faceCascPath = "haarcascade_frontalface_default.xml" 
faceCascade = cv2.CascadeClassifier(faceCascPath)

# get camera
cap = cv2.VideoCapture(0)
predicted_action = ''
prediction = 0.0
body_rate = 0.0

if not cap.isOpened():
    print("Cannot access camera!")
    exit()

while True:
    # get a frame
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame!")
        break
    
    # detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1, #比例因子
        minNeighbors = 3,  #最小鄰距
        minSize=(30, 30) #視窗大小
    )
    
    for (x, y, w, h) in faces:
        bbox_array = cv2.rectangle( frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(bbox_array, predicted_action + ", acc: " + str(prediction), (x, y -10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 0), 2)
        cv2.putText(bbox_array, "body rate: " + str(body_rate), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 0), 2)
    
    # detect hands
    hands, img = handDetector.findHands(frame)
       
    # detect body
    img = poseDetector.findPose(frame)
    lmList, bboxInfo = poseDetector.findPosition(img, bboxWithHands=False)

    #print("Found {0} faces!".format(len(faces)))
    
    """
    if 'bbox' in bboxInfo:
        (body_x,body_y,body_w,body_h) = bboxInfo['bbox']
        body_rate = (480-body_x)*(640-body_y)/(480*640)
        if body_rate > 1:
            body_rate = 1.0
        if body_rate >= 0.7 and len(faces)!=0:
            new_body_w = int(body_w*0.5)
            new_body_h = int(body_w*0.5)
            (center_x,  center_y) = bboxInfo['center']
            center_y = body_y + int(new_body_h/2)

            if center_y-int(new_body_h/2) >= 0:
                center_y1 = center_y-int(new_body_h/2)
            else:
                center_y1 = 0
            if center_y+int(new_body_h/2) <= 480:
                center_y2 = center_y+int(new_body_h/2)
            else:
                center_y2 = 480

            
            
            body_array = frame[center_y1:center_y2,center_x-int(new_body_w/2):center_x+int(new_body_w/2),:]
            #cv2.imshow("Faces found",  body_array)

            
            new_body_array = np.resize(body_array, [256,256])
            new_body_array = np.stack((new_body_array,)*3, axis=-1)
            new_body_array = np.reshape(new_body_array, [1,256,256,3])
            input_arr = np.array(new_body_array).astype('float32') / 255
            predictions = selfie_model.predict(input_arr)
            prediction = predictions[0][0]
            
            if(prediction > 0.5):
                predicted_action='Selfie'
            else:
                predicted_action='NonSelfie'
            
            print(predicted_action)
            # mark the face with bbox
    """
    if len(hands)>=1 :
        [x,y,w,h] = hands[0]['bbox']
        
        if x-10 >= 0:
            x1 = x-10
        else:
            x1 = 0
        if x+w+10 <= 640:
            x2 = x+w+10
        else:
            x2 = x+w
        if y-10 >= 0:
            y1 = y-10
        else:
            y1 = 0
        if y+h+10 <= 480:
            y2 = y+h+10
        else:
            y2 = 480
        
        hand_array = frame[y1:y2,x1:x2,:]


        #new_hand_array = segmentor.removeBG(hand_array, (200,200,200), threshold=0.8)
        #cv2.imshow("Faces found",  new_hand_array)
        new_hand_array = segmentor.removeBG(hand_array, (255,255,255), threshold=0.0001)
        new_hand_array = cv2.cvtColor(new_hand_array, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Faces found",  new_hand_array)
        #new_hand_array = segmentor.removeBG(hand_array, (200,200,200), threshold=0.6)
        new_hand_array = cv2.resize(new_hand_array, (40, 40), interpolation=cv2.INTER_AREA)
        temp = np.ones((50,50), dtype=np.int)
        temp *= 255
        temp[5:45, 5:45] = new_hand_array
        new_hand_array = np.reshape(temp, (50,50,1))
        new_hand_array = new_hand_array/255.0
        X = np.array([new_hand_array])

        Y_pred = hand_model.predict(X)
        hand_prediction = K.argmax(Y_pred,axis=-1)
        hand_prediction = hand_prediction.numpy()[0].astype(str)
        print(hand_prediction)
        #cv2.imshow("Faces found",  hand_array)
    else:
        prediction = 0.0
        predicted_action='NonSelfie'
        cv2.imshow("Faces found",  frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
"""if len(hands)>=1 :
        [x,y,w,h] = hands[0]['bbox']
        
        if x-40 >= 0:
            x1 = x-40
        else:
            x1 = 0
        if x+w+40 <= 640:
            x2 = x+w+40
        else:
            x2 = x+w
        if y-40 >= 0:
            y1 = y-40
        else:
            y1 = 0
        if y+w+40 <= 480:
            y2 = y+w+40
        else:
            y2 = y+w
        hand_array = frame[y1:y2,x1:x2,:]
        cv2.imshow("Faces found",  hand_array)
        """