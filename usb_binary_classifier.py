# -*- coding: utf-8 -*-
# This script trains a multi-class classifier using the Keras TensorFlow API

import tensorflow as tf # >= 2.3.0
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
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

batch_size = 5
epochs = 1
img_width  = 390 # 390
img_height = 267 # 267
save_file = sys.argv[1]

def build_keras_model():
    # load the VGG16 network without the final few fully-connected layers
    base_model = VGG16(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(img_height, img_width, 3)))

    # Freeze all the lower layers. I.e. make it so their weights don't change w/ training
    # for layer in base_model.layers:
    #     layer.trainable = False

    # this will be the new final few layers.
    head_model = base_model.output
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(512, activation="relu")(head_model)
    head_model = Dense(1)(head_model)

    # put the head on the base model.
    model = Model(inputs=base_model.input, outputs=head_model)
    return model

data_path = "\\\\freenas.local\\fast_storage\\python\\gh3\\data\\cats_and_dogs_filtered"
#data_path = "/home/rbain/links/fast_storage/python/gh3/data/cats_and_dogs_filtered"
train_dir = os.path.join(data_path, 'train')
validation_dir = os.path.join(data_path, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))
num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))
total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)
print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("Total training images:", total_train)
print("Total validation images:", total_val)

# apply some data-aug to artificially inflate the dataset in a pragmatic way. Trade-off is more compute.
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )

# Generator for our validation data. Do not apply data-aug. We want a sense of how well it does on realistic data/scenarios.
validation_image_generator = ImageDataGenerator(rescale=1./255) 

# The generator interacts with the filesystem to load and format images one at a time for model input. Abstracts away a lot of functionality well.
train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(img_height, img_width),
                                                     class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(img_height, img_width),
                                                              class_mode='binary')

# cuDNN crashes without these 3 lines
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.InteractiveSession(config=config)

if len(sys.argv) <= 2:
    # if save_file doesn't exist w/ os module
    if not os.path.isdir(save_file) and not save_file.endswith(".tflite"):
        print("initializing model")
        model = build_keras_model()
        model.summary()

        print("compiling model...")
        model.compile(optimizer='adam',
                        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                        metrics=['accuracy'])

        model.fit_generator(train_data_gen,
            steps_per_epoch=total_train // batch_size // 8,
            epochs=epochs,
            validation_data=val_data_gen,
            validation_steps=total_val // batch_size,
            max_queue_size = 20,
            workers = 4
        )
        
        model.save(save_file)
        print('model saved!')

    def representative_dataset_gen():
        for _i in range(20):
            yield [np.array([train_data_gen.next()[0][0]], dtype=np.float32)]
    
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(save_file)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    model = converter.convert()
    save_file += ".tflite"
    open(save_file, "wb").write(model)
else:
    interpreter = tflite.Interpreter(
        model_path=save_file,
        experimental_delegates=[
            tflite.load_delegate(EDGETPU_SHARED_LIB, {})
        ])
    interpreter.allocate_tensors()

    for i in range(10):
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        # Pre-processing: add batch dimension and convert to float32 to match with the model's input data format.
        test_image = np.expand_dims(train_data_gen[0][0][0], axis=0).astype(np.uint8)
        interpreter.set_tensor(input_index, test_image)
        start = time.time()
        interpreter.invoke() # Run inference.

        # Post-processing: remove batch dimension and find the digit with highest probability.
        output = interpreter.tensor(output_index)

        end = time.time()
        print(output)
        print('%d, %0.3f' % (i, (end-start)))
