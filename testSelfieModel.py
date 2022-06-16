import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

selfie_model = tf.keras.models.load_model('./selfie_model.h5')

image = tf.keras.preprocessing.image.load_img('stranger3.jpg', target_size=(256,256))
input_arr = np.array([tf.keras.preprocessing.image.img_to_array(image)]).astype('float32') / 255

predictions = selfie_model.predict(input_arr)
predicted_action = ''
print(predictions[0][0])
if(predictions[0][0]>0.5):
  predicted_action='Selfie'
else:
  predicted_action='NonSelfie'

plt.figure()
plt.imshow(image)

actual_action = 'Selfie'

plt.title('Predicted: ' + predicted_action + '\n' +
        'Actual: ' + actual_action)

plt.show()