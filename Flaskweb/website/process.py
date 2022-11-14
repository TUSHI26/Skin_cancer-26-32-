import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
from tensorflow import keras
#import views.homeimage_location


model = tf.keras.models.Sequential()
# Convolutional & Max Pooling layers
model.add(tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(128,128,4)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# Flatten & Dense layers
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
# performing binary classification
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss = tfa.losses.SigmoidFocalCrossEntropy(),
              optimizer = tf.keras.optimizers.Adam(),
              metrics = ['binary_accuracy',
                       tf.keras.metrics.FalsePositives(),
                       tf.keras.metrics.FalseNegatives(),
                       tf.keras.metrics.TruePositives(),
                       tf.keras.metrics.TrueNegatives()
                      ]
             )








test = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(path = image_location, color_mode = "rgba", target_size = (128,128)), dtype="float32")
print(image_location)
test = test / 255
test=np.reshape(test,(1,128,128,4))
test = np.array(test)
model.load_weights('C:/Users/User/Downloads/weights.hdf5')
print(model.predict(test)[0][0])