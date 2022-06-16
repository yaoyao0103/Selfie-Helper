import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import keras.backend as K
import random

hand_model = tf.keras.models.load_model('./hand_model.h5')
dict_labels = {
    '0':1,
    '1':2,
    '2':3,
    '3':4,
    '4':5,
    '5':6,
    '6':7,
    '7':8,
    '8':9,
    '9':10 
}
x, y = [], []
number = str(random.randint(0, 5))
img_path = './hands/'+number+'.jpg'
image = cv2.imread(img_path,0)

image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
img = np.reshape(image, (50,50,1))

plt.figure()
plt.imshow(image)

img = img/255.0
x.append(img)
X = np.array(x)


Y_pred = hand_model.predict(X)
predict = K.argmax(Y_pred,axis=-1)
predict = predict.numpy()[0].astype(str)
print(predict)
cv2.waitKey(0)
cv2.destroyAllWindows()

actual_action = number

plt.title('Predicted: ' + predict[0] + '\n' +
        'Actual: ' + actual_action)

plt.show()