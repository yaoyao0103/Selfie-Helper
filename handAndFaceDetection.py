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
hand_model = tf.keras.models.load_model('./hand_model_new.h5')

# use handDetector in cvzone
segmentor = SelfiSegmentation()
handDetector = HandDetector(detectionCon=0.8)
poseDetector = PoseDetector()

# use haarcascade in opencv to implement face detection
faceCascPath = "haarcascade_frontalface_default.xml" 
faceCascade = cv2.CascadeClassifier(faceCascPath)



def handSignPrediction(frame, hands):
    hand_prediction = '0'
    [x,y,w,h] = hands[0]['bbox']
    
    # get a suitable part or hand
    if x-20 >= 0:
        x1 = x-20
    else:
        x1 = 0
    if x+w+20 <= 640:
        x2 = x+w+20
    else:
        x2 = x+w
    if y-20 >= 0:
        y1 = y-20
    else:
        y1 = 0
    if y+h+20 <= 480:
        y2 = y+h+20
    else:
        y2 = 480
    
    hand_array = frame[y:y+h,x:x+w,:]
    # step 1. remove back ground
    new_hand_array = segmentor.removeBG(hand_array, (220,220,220), threshold=0.000001)
    # step 2. transform to gray scale
    new_hand_array = cv2.cvtColor(new_hand_array, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Faces found",  new_hand_array)
    # step 3. resize to 40*40
    new_hand_array = cv2.resize(new_hand_array, (26, 42), interpolation=cv2.INTER_AREA)
    
    # step 4. put the hand array to the center 50*50 white array
    temp = np.ones((50,50), dtype=np.int)
    temp *= 220
    temp[4:46, 12:38] = new_hand_array
    
    # step 5. reshape to 50*50*1
    new_hand_array = np.reshape(temp, (50,50,1))
    # step 6. data normalization
    new_hand_array = new_hand_array/255.0

    # step 7. predict
    X = np.array([new_hand_array])
    Y_pred = hand_model.predict(X)
    hand_prediction = K.argmax(Y_pred,axis=-1)
    hand_prediction = hand_prediction.numpy()[0].astype(str)
    print("hand_prediction: {0}".format(hand_prediction))
    
    return hand_prediction

def selfiePrediction(frame, bboxInfo):
    selfie_prediction = 0.0
    predicted_action = 'NonSelfie'
    # get bbox of body object
    (body_x,body_y,body_w,body_h) = bboxInfo['bbox']

    # calculate the proportion
    body_proportion = (480-body_x)*(640-body_y)/(480*640)
    if body_proportion > 1:
        body_proportion = 1.0

    # condition of selfie detection: 1. body_proportion more than 70%   2. found face 
    if body_proportion >= 0.7 and len(faces)!=0:

        # get a suitable part or face
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

        # step 1. resize to 256*256
        new_body_array = np.resize(body_array, [256,256])
        # step 2. transform to 256*256*3 
        new_body_array = np.stack((new_body_array,)*3, axis=-1)
        # step 3. reshape to 1*256*256*3
        new_body_array = np.reshape(new_body_array, [1,256,256,3])
        # step 4. data normalization
        input_arr = np.array(new_body_array).astype('float32') / 255
        # step 5. predict
        predictions = selfie_model.predict(input_arr)
        selfie_prediction = predictions[0][0]
        
        if(selfie_prediction > 0.5):
            predicted_action='Selfie'
        else:
            predicted_action='NonSelfie'

    return selfie_prediction, predicted_action, body_proportion


# get camera
cap = cv2.VideoCapture(0)
selfie_prediction = 0.0
predicted_action = 'NonSelfie'
hand_prediction = '0'
body_proportion = 0.0

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
    
    # show border and information on face
    for (x, y, w, h) in faces:
        bbox_array = cv2.rectangle( frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(bbox_array, predicted_action + ", acc: " + str(selfie_prediction), (x, y -10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(bbox_array, "body rate: " + str(body_proportion), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # detect hands
    hands, img = handDetector.findHands(frame)

    # detect body
    img = poseDetector.findPose(frame)
    lmList, bboxInfo = poseDetector.findPosition(img, bboxWithHands=False)
    
    # found hand!
    if len(hands)>=1 :
        hand_prediction = handSignPrediction(frame, hands)
    # found body!
    if 'bbox' in bboxInfo:
        selfie_prediction, predicted_action, body_proportion = selfiePrediction(frame, bboxInfo)
    else:
        prediction = 0.0
        predicted_action='NonSelfie'
    cv2.imshow("Faces found",  frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()