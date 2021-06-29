import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tflite_runtime.interpreter as tflite
import tensorflow_model_optimization as tfmot
import os
import sys
import numpy as np
import platform
import matplotlib.pyplot as plt
import time

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

batch_size = 1
epochs = 1
IMG_HEIGHT = 224
IMG_WIDTH = 224

data_path = "\\\\freenas.local\\fast_storage\\python\\gh3\\data\\cats_and_dogs_filtered"
train_dir = os.path.join(data_path, 'train')

# apply some data-aug to artificially inflate the dataset in a pragmatic way. Trade-off is more compute.
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )

# The generator interacts with the filesystem to load and format images one at a time for model input. Abstracts away a lot of functionality well.
train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')

interpreter = tflite.Interpreter(
      model_path="models/google_automl_flower_classifier.tflite",
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB, {})
      ])
interpreter.allocate_tensors()

for i in range(20):
    # Pick a random sample
    # x, y = train_data_gen.next()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Pre-processing: add batch dimension and convert to unsigned 8-bit int data type
    test_image = np.expand_dims(train_data_gen[0][0][0], axis=0).astype(np.uint8)
    interpreter.set_tensor(input_index, test_image)
    start = time.time()

    # Run the model / inference.
    interpreter.invoke()
    output = interpreter.tensor(output_index)

    end = time.time()
    print(output)
    print('%d, %0.3f' % (i, (end-start)))