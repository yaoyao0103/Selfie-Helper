# modules of makeup
import sys,os
import numpy as np
import cv2
import json
import time
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage,QIcon,QPixmap
from PyQt5.QtWidgets import QFileDialog,QMessageBox
from AIMakeup import Makeup,Face,Organ,NoFace

# modules of models
import tensorflow as tf
from queue import Queue
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector
import keras.backend as K
from cvzone.SelfiSegmentationModule import SelfiSegmentation



class SelfieHelper(object):
    def __init__(self, MainWindow):
        self.window=MainWindow
        # makeup values
        self.mode = []
        for num in range(1,6):
            path = "./data/mode{}.json".format(str(num))
            if os.path.exists(path):
                with open(path, 'r') as fp:
                    self.mode.append(json.load(fp))
        self.values = {'brightening': 0.0, 'sharpen': 0.0, 'whitening': 0.0, 'smooth': 0.0}

        self._setupUi()
        self.bg_edit=[]
        self.bg_op=[self.bt_reset, self.bt_saveImage]
        self.bg_result=[self.bt_save1,self.bt_save2,self.bt_save3,self.bt_save4,self.bt_save5]
        self.sls=[self.sl_brightening,self.sl_sharpen,self.sl_whitening,self.sl_smooth]
        
        # set label on image
        self.label=QtWidgets.QLabel(self.window)
        self.sa.setWidget(self.label)
        # set status
        self._set_statu(self.bg_edit,False)
        self._set_statu(self.bg_op,False)
        self._set_statu(self.bg_result,False)
        self._set_statu(self.sls,False)
        # import dlib model
        if os.path.exists("./data/shape_predictor_68_face_landmarks.dat"):
            self.path_predictor=os.path.abspath("./data/shape_predictor_68_face_landmarks.dat")
        else:
            QMessageBox.warning(self.centralWidget,'警告','默認的dlib模型文件路徑不存在')
            self.path_predictor,_=QFileDialog.getOpenFileName(self.centralWidget,'選擇dlib模型文件','./','Data Files(*.dat)')
        # make up
        self.mu = Makeup(self.path_predictor)
        self.path_img=''
        self._set_connect()

        # detectors in cvzone
        self.segmentor = SelfiSegmentation()
        self.handDetector = HandDetector(detectionCon=0.8)
        self.poseDetector = PoseDetector()

        # import my trained selfie detection and hand sign detection model
        self.selfie_model = tf.keras.models.load_model('./models/selfie_model.h5')
        self.hand_model = tf.keras.models.load_model('./models/hand_model.h5')

        # predictions
        self.number = 'unknown'
        self.selfie_prediction = 0.0
        self.predicted_action = 'NonSelfie'
        self.body_proportion = 0.0
        self.hand_predictions = Queue(maxsize=10)

        self.countdown = False
        self.remaining = 10.0
        self.autoStart = True
        self.imageCount = 0

    # set btn connection
    def _set_connect(self):
        self.bt_open.clicked.connect(self._open_img)
        for op in ['reset']:
            self.__getattribute__('bt_'+op).clicked.connect(self.__getattribute__('_'+op))
        self.bt_saveImage.clicked.connect(self._saveImage)
        self.bt_save1.clicked.connect(lambda: self._save('1'))
        self.bt_save2.clicked.connect(lambda: self._save('2'))
        self.bt_save3.clicked.connect(lambda: self._save('3'))
        self.bt_save4.clicked.connect(lambda: self._save('4'))
        self.bt_save5.clicked.connect(lambda: self._save('5'))
        self.bt_camera.clicked.connect(lambda: self._openCamera())


    # open an image
    def _open_img(self):
        self.path_img,_=QFileDialog.getOpenFileName(self.centralWidget,'打開圖片文件','./','Image Files(*.png *.jpg *.bmp)')
        if self.path_img and os.path.exists(self.path_img):
            print(self.path_img)
            self.im_bgr,self.temp_bgr,self.faces=self.mu.read_and_mark(self.path_img)
            self.im_ori,self.previous_bgr=self.im_bgr.copy(),self.im_bgr.copy()
            self._set_statu(self.bg_edit,True)
            self._set_statu(self.bg_op,True)
            self._set_statu(self.bg_result,True)
            self._set_statu(self.sls,True)
            self._set_img()
        else:
            QMessageBox.warning(self.centralWidget,'無效路徑','無效路徑，請重新選擇！')

    # opencv image to QImage    
    def _cv2qimg(self,cvImg):
        height, width, channel = cvImg.shape
        bytesPerLine = 3 * width
        return QImage(cv2.cvtColor(cvImg,cv2.COLOR_BGR2RGB).data, width, height, bytesPerLine, QImage.Format_RGB888)
        
    # set Image
    def _set_img(self):
        self.label.setPixmap(QPixmap.fromImage(self._cv2qimg(self.temp_bgr)))

    def _set_statu(self,group,value):
        [item.setEnabled(value) for item in group]

    # reset Image    
    def _reset(self):
        self.temp_bgr[:]=self.im_ori[:]
        self._set_img()
        
    # apply on all faces
    def _mapfaces(self,fun,value,isCamera=False):
        if isCamera == False:
            self.previous_bgr[:]=self.temp_bgr[:]
        for face in self.faces[self.path_img]:
            fun(face,value)
        if isCamera == False:
            self._set_img()

    # trigger when slider change
    def _changeValue(self):
        self.values['sharpen'] = min(1,max(self.sl_sharpen.value()/200,0))
        self.values['whitening'] = min(1,max(self.sl_whitening.value()/200,0))
        self.values['brightening'] = min(1,max(self.sl_brightening.value()/200,0))
        self.values['smooth'] = min(1,max(self.sl_smooth.value()/100,0))
        self._reset()
        self._sharpen()
        self._whitening()
        self._brightening()
        self._smooth()

    # apply makeup filter on camera frame
    def _applyMakeup(self, mode):
        self.im_bgr=self.frame
        im_hsv=cv2.cvtColor(self.im_bgr, cv2.COLOR_BGR2HSV)
        self.temp_bgr,temp_hsv=self.im_bgr.copy(),im_hsv.copy()
        try:
            self.faces = self.mu.get_faces(self.im_bgr,im_hsv,self.temp_bgr,temp_hsv,"camera")
            self.values['sharpen'] = mode['sharpen']
            self.values['whitening'] = mode['whitening']
            self.values['brightening'] = mode['brightening']
            self.values['smooth'] = mode['smooth']
            
            self._sharpen(True)
            self._whitening(True)
            self._brightening(True)
            self._smooth(True)
            cv2.imshow("Faces found",  self.temp_bgr)
            
        except:
            print("too many face")
        
    # makeup: sharpen
    def _sharpen(self, isCamera = False):
        value=self.values['sharpen']
        def fun(face,value):
            face.organs['left eye'].sharpen(value,confirm=False)
            face.organs['right eye'].sharpen(value,confirm=False)

        self._mapfaces(fun,value,isCamera)

    # makeup: whitening  
    def _whitening(self, isCamera = False):
        value=self.values['whitening']
        def fun(face,v):
            face.organs['left eye'].whitening(value,confirm=False)
            face.organs['right eye'].whitening(value,confirm=False)
            face.organs['left brow'].whitening(value,confirm=False)
            face.organs['right brow'].whitening(value,confirm=False)
            face.organs['nose'].whitening(value,confirm=False)
            face.organs['forehead'].whitening(value,confirm=False)
            face.organs['mouth'].whitening(value,confirm=False)
            face.whitening(value,confirm=False)

        self._mapfaces(fun,value,isCamera)

    # makeup: brightening  
    def _brightening(self, isCamera = False):
        value=self.values['brightening']
        def fun(face,value):
            face.organs['mouth'].brightening(value,confirm=False)

        self._mapfaces(fun,value,isCamera)
        
    # makeup: smooth
    def _smooth(self, isCamera = False):
        value=self.values['smooth']
        def fun(face,value):
            face.smooth(value,confirm=False)
            face.organs['left eye'].smooth(value*2/3,confirm=False)
            face.organs['right eye'].smooth(value*2/3,confirm=False)
            face.organs['left brow'].smooth(value*2/3,confirm=False)
            face.organs['right brow'].smooth(value*2/3,confirm=False)
            face.organs['nose'].smooth(value*2/3,confirm=False)
            face.organs['forehead'].smooth(value*3/2,confirm=False)
            face.organs['mouth'].smooth(value,confirm=False)
        
        self._mapfaces(fun,value, isCamera)
    
    # save mode to json file
    def _save(self, number='1'):
        values = self.values
        path = "./data/mode{}.json".format(number)
        with open(path, 'w+') as fp:
            json.dump(values, fp)
        with open(path, 'r') as fp:
            self.mode[int(number)-1] = json.load(fp)

    # save image
    def _saveImage(self):
        output_path,_=QFileDialog.getSaveFileName(self.centralWidget,'选择保存位置','./','Image Files(*.png *.jpg *.bmp)')
        if output_path:
            self.save(output_path,self.temp_bgr)
        else:
            QMessageBox.warning(self.centralWidget,'无效路径','无效路径，请重新选择！')

    def save(self,output_path,output_im):
        cv2.imencode('.jpg',output_im)[1].tofile(output_path)

    # hand sign prediction
    def _handSignPrediction(self, hands):
        [x,y,w,h] = hands[0]['bbox']
        
        if y < 0:
            y = 0
        if x < 0:
            x = 0
        hand_array = self.frame[y:y+h,x:x+w,:]

        # step 1. remove back ground
        new_hand_array = self.segmentor.removeBG(hand_array, (220,220,220), threshold=0.000001)
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
        Y_pred = self.hand_model.predict(X)
        tempPrediction = K.argmax(Y_pred,axis=-1)
        tempPrediction = str(tempPrediction.numpy()[0])
        if self.hand_predictions.full():
            self.hand_predictions.get()
        self.hand_predictions.put(tempPrediction)
        print("hand_prediction: {0}".format(tempPrediction))

    # selfie prediction
    def _selfiePrediction(self, bboxInfo):
        
        (body_x,body_y,body_w,body_h) = bboxInfo['bbox']
        # condition of selfie detection: body_proportion more than 70%
        if self.body_proportion >= 0.7:

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
            
            self.body_array = self.frame[center_y1:center_y2,center_x-int(new_body_w/2):center_x+int(new_body_w/2),:]

            # step 1. resize to 256*256
            new_body_array = np.resize(self.body_array, [256,256])
            # step 2. transform to 256*256*3 
            new_body_array = np.stack((new_body_array,)*3, axis=-1)
            # step 3. reshape to 1*256*256*3
            new_body_array = np.reshape(new_body_array, [1,256,256,3])
            # step 4. data normalization
            input_arr = np.array(new_body_array).astype('float32') / 255
            # step 5. predict
            predictions = self.selfie_model.predict(input_arr)
            self.selfie_prediction = predictions[0][0]
            if(self.selfie_prediction > 0.5):
                self.predicted_action='Selfie'
                self._countdownStart()
            else:
                self.predicted_action='NonSelfie'

        else:
            self.selfie_prediction = 0.0
            self.predicted_action='NonSelfie'

    # check hand sign in nearly ten predictions
    def _checkSign(self):
        hand_signs = list(self.hand_predictions.queue)
        statistics = {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'9':0,'10':0}
        for i in hand_signs:
            statistics[i] += 1
        if statistics['9'] >= 1:
            return '9'
        elif statistics['1'] >= 3:
            return '1'
        elif statistics['0'] >= 3:
            return '0'
        elif statistics['2'] >= 3:
            return '2'
        elif statistics['5'] >= 4:
            return '5'
        elif statistics['3'] >= 5:
            return '3'
        elif statistics['4'] >= 5:
            return '4'
        else:
            return 'unknown'

    # start countdown
    def _countdownStart(self):
            self.countdown = True
            now = time.time()
            self.end = now + 10
            self.remaining = np.round(self.end - now, 1)

    # stop countdown
    def _countdownStop(self):
            self.countdown = False
            self.autoStart = False
            self.remaining = 10

    # open Camera
    def _openCamera(self):
        # get camera
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Cannot access camera!")
            exit()

        self.path_img = 'camera'
        while True:
            # get a frame
            ret, frame = cap.read()
            if not ret:
                print("Cannot receive frame!")
                break

            self.frame = frame
            # detect hands
            hands, img = self.handDetector.findHands(self.frame)
            # detect body
            if self.countdown == False:
                img = self.poseDetector.findPose(self.frame, draw=False)
                lmList, bboxInfo = self.poseDetector.findPosition(img, bboxWithHands=False)

            
            ############################################ found hand! ###########################################
            if len(hands)>=1 :
                self._handSignPrediction(hands)
                cv2.putText
                for hand in hands:
                    [x,y,w,h] = hand['bbox']
                    self.number = self._checkSign()
                    #bbox_array = cv2.rectangle( frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
                    cv2.putText(self.frame, "Sign: " + self.number , (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            ############################################ found body! ###########################################
            if 'bbox' in bboxInfo and self.predicted_action == 'NonSelfie' and self.autoStart == True:
                # get bbox of body object
                (body_x,body_y,body_w,body_h) = bboxInfo['bbox']
                # calculate the proportion
                self.body_proportion = (480-body_x)*(640-body_y)/(480*640)
                if self.body_proportion > 1:
                    self.body_proportion = 1.0
                #if self.predicted_action == 'NonSelfie':
                self._selfiePrediction(bboxInfo)
                cv2.putText(self.frame, self.predicted_action + ", acc: " + str(self.selfie_prediction), (body_x, body_y -10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(self.frame, "body rate: " + str(self.body_proportion), (body_x, body_y +15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                self.body_proportion = 0.0
                self.selfie_prediction = 0.0
                #self.predicted_action='NonSelfie'
            
            ########################################### countdown timer ###########################################

            if self.countdown == True:
                now = time.time()
                self.remaining = self.end - now
                if self.remaining < 0:
                    self._countdownStop()
                    ret, frame = cap.read()
                    self.frame = frame
                    if self.number != 'unknown' :
                        number = int(self.number)
                        if number <= 5 and number >= 1:
                            mode = self.mode[number-1]
                            self._applyMakeup(mode)
                            cv2.imwrite('./output/{0}.jpg'.format(self.imageCount), self.temp_bgr)
                        else:
                            cv2.imwrite('./output/{0}.jpg'.format(self.imageCount), self.frame)
                    else:
                        cv2.imwrite('./output/{0}.jpg'.format(self.imageCount), self.frame)
                    self.imageCount += 1
            
            cv2.putText(self.frame, self.predicted_action , (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(self.frame, "Timer: " + str(self.remaining), (0,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(self.frame, "Auto Start: " + str(self.autoStart), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(self.frame, "Makeup Mode: " + str(self.number), (0,65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            ############################################ hand sign operation ###########################################
            if self.number != 'unknown':
                number = int(self.number)
                if number <= 5 and number >= 1:
                    mode = self.mode[number-1]
                    self._applyMakeup(mode)
                # stop taking selfie
                elif number == 0:
                    self._countdownStop()
                    cv2.imshow("Faces found",  self.frame)
                # start taking selfie
                elif number == 9 and self.countdown == False:
                    self._countdownStart()
                    cv2.imshow("Faces found",  self.frame)
                else:
                    cv2.imshow("Faces found",  self.frame)
                
            else:
                cv2.imshow("Faces found",  self.frame)

            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # setup UI
    def _setupUi(self):
        self.window.setObjectName("MainWindow")
        self.window.resize(837, 838)
        self.centralWidget = QtWidgets.QWidget(self.window)
        self.centralWidget.setObjectName("centralWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralWidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.sa = QtWidgets.QScrollArea(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sa.sizePolicy().hasHeightForWidth())
        self.sa.setSizePolicy(sizePolicy)
        self.sa.setWidgetResizable(True)
        self.sa.setObjectName("sa")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 813, 532))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.sa.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout.addWidget(self.sa)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")

        self.label_whitening = QtWidgets.QLabel(self.centralWidget)
        self.label_whitening.setText("美白")
        self.gridLayout.addWidget(self.label_whitening, 0, 0, 1, 1)
        self.sl_whitening = QtWidgets.QSlider(self.centralWidget)
        self.sl_whitening.setOrientation(QtCore.Qt.Horizontal)
        self.sl_whitening.setObjectName("sl_whitening")
        self.sl_whitening.valueChanged.connect(self._changeValue)
        self.gridLayout.addWidget(self.sl_whitening, 0, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 2, 1, 1)

        self.label_smooth = QtWidgets.QLabel(self.centralWidget)
        self.label_smooth.setText("磨皮")
        self.gridLayout.addWidget(self.label_smooth, 1, 0, 1, 1)
        self.sl_smooth = QtWidgets.QSlider(self.centralWidget)
        self.sl_smooth.setOrientation(QtCore.Qt.Horizontal)
        self.sl_smooth.setObjectName("sl_smooth")
        self.sl_smooth.valueChanged.connect(self._changeValue)
        self.gridLayout.addWidget(self.sl_smooth, 1, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 1, 2, 1, 1)

        self.label_sharpen = QtWidgets.QLabel(self.centralWidget)
        self.label_sharpen.setText("亮眼")
        self.gridLayout.addWidget(self.label_sharpen, 2, 0, 1, 1)
        self.sl_sharpen = QtWidgets.QSlider(self.centralWidget)
        self.sl_sharpen.setOrientation(QtCore.Qt.Horizontal)
        self.sl_sharpen.setObjectName("sl_sharpen")
        self.sl_sharpen.valueChanged.connect(self._changeValue)
        self.gridLayout.addWidget(self.sl_sharpen, 2, 1, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 2, 2, 1, 1)

        self.label_brightening = QtWidgets.QLabel(self.centralWidget)
        self.label_brightening.setText("紅脣")
        self.gridLayout.addWidget(self.label_brightening, 3, 0, 1, 1)
        self.sl_brightening = QtWidgets.QSlider(self.centralWidget)
        self.sl_brightening.setOrientation(QtCore.Qt.Horizontal)
        self.sl_brightening.setObjectName("sl_brightening")
        self.sl_brightening.valueChanged.connect(self._changeValue)
        self.gridLayout.addWidget(self.sl_brightening, 3, 1, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem3, 3, 2, 1, 1)

        # op
        self.bt_open = QtWidgets.QPushButton(self.centralWidget)
        self.bt_open.setObjectName("bt_open")
        self.gridLayout.addWidget(self.bt_open, 4, 0, 1, 1)

        self.bt_reset = QtWidgets.QPushButton(self.centralWidget)
        self.bt_reset.setObjectName("bt_reset")
        self.gridLayout.addWidget(self.bt_reset, 4, 1, 1, 1)

        self.bt_saveImage = QtWidgets.QPushButton(self.centralWidget)
        self.bt_saveImage.setObjectName("bt_saveImage")
        self.gridLayout.addWidget(self.bt_saveImage, 4, 2, 1, 1)

        self.bt_save1 = QtWidgets.QPushButton(self.centralWidget)
        self.bt_save1.setObjectName("bt_save1")
        self.gridLayout.addWidget(self.bt_save1, 6, 0, 1, 1)

        self.bt_save2 = QtWidgets.QPushButton(self.centralWidget)
        self.bt_save2.setObjectName("bt_save2")
        self.gridLayout.addWidget(self.bt_save2, 6, 1, 1, 1)

        self.bt_save3 = QtWidgets.QPushButton(self.centralWidget)
        self.bt_save3.setObjectName("bt_save3")
        self.gridLayout.addWidget(self.bt_save3, 6, 2, 1, 1)

        self.bt_save4 = QtWidgets.QPushButton(self.centralWidget)
        self.bt_save4.setObjectName("bt_save4")
        self.gridLayout.addWidget(self.bt_save4, 6, 3, 1, 1)

        self.bt_save5 = QtWidgets.QPushButton(self.centralWidget)
        self.bt_save5.setObjectName("bt_save5")
        self.gridLayout.addWidget(self.bt_save5, 6, 4, 1, 1)

        self.bt_camera = QtWidgets.QPushButton(self.centralWidget)
        self.bt_camera.setObjectName("bt_camera")
        self.gridLayout.addWidget(self.bt_camera, 0, 4, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout)
        self.window.setCentralWidget(self.centralWidget)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self.window)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.window.setWindowTitle(_translate("MainWindow", "AI美顏"))
        self.bt_open.setText(_translate("MainWindow", "打開文件"))
        self.bt_reset.setText(_translate("MainWindow", "還原"))
        self.bt_saveImage.setText(_translate("MainWindow", "保存照片"))
        self.bt_camera.setText(_translate("MainWindow", "開啟鏡頭"))
        self.bt_save1.setText(_translate("MainWindow", "保存至mode1"))
        self.bt_save2.setText(_translate("MainWindow", "保存至mode2"))
        self.bt_save3.setText(_translate("MainWindow", "保存至mode3"))
        self.bt_save4.setText(_translate("MainWindow", "保存至mode4"))
        self.bt_save5.setText(_translate("MainWindow", "保存至mode5"))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = SelfieHelper(MainWindow)
    ui.window.show()
    sys.exit(app.exec_())

