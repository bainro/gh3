# -*- coding: utf-8 -*-
# Original file: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/classification.ipynb
# This tutorial shows how to classify cats or dogs from images.

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_model_optimization as tfmot
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

print(tf.__version__) # 2.X.X

"""The dataset has the following directory structure:

cats_and_dogs_filtered</b>
|__ train
    |______ cats: [cat.0.jpg, cat.1.jpg, cat.2.jpg ....]
    |______ dogs: [dog.0.jpg, dog.1.jpg, dog.2.jpg ...]
|__ validation
    |______ cats: [cat.2000.jpg, cat.2001.jpg, cat.2002.jpg ....]
    |______ dogs: [dog.2000.jpg, dog.2001.jpg, dog.2002.jpg ...]
"""

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

# Let's look at how many cats and dogs images are in the training and validation directory:

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

# will want to change this to 5 after training, for prediction on GH3
batch_size = 5
epochs = 80
IMG_HEIGHT = 267 # 150
IMG_WIDTH = 390 # 150

save_dir_exists = os.path.isdir("models")
assert save_dir_exists and len(sys.argv) >= 2
save_file = os.path.join("models", sys.argv[1])

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
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

# show me some data-aug nurd
#augmented_images = [train_data_gen[0][0][0] for i in range(5)]
#plotImages(augmented_images)
#_x, labels = train_data_gen.next()
#print(labels)

# cuDNN crashes without these 3 lines
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# if save_file doesn't exist w/ os module
if os.path.isdir(save_file):
    if len(sys.argv) == 2:
        print("loading model...")
        model = load_model(save_file)
    else:
        print("loading and quantizing the specified saved model...")
        converter = tf.lite.TFLiteConverter.from_saved_model(save_file)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # tflite quantized model
        model = converter.convert()
        #model.save(save_file)
else:
    print("initializing model")

    # Create the model as a deep artificial neural network 
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', 
            input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1)
    ])

    # print("quantizing the entire model...")
    # model = tfmot.quantization.keras.quantize_model(model)

    # add pruning
    #model = tfmot.sparsity.keras.prune_low_magnitude(model)

    print("compiling model...")
    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])

    # print all the layers of the network
    model.summary()

    #callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

    # Choo-choo, we training.
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_val // batch_size,
        workers=4,
        #callbacks=callbacks,
    )

    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # tflite quantized model
    # model = converter.convert()

    model.save(save_file)
    print('model saved!')

    # Visualize the new model after training
    acc = history.history['accuracy']
    loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    #plt.show()

# do prediction on an input of all 0s
#predictions = model(np.zeros(shape=train_data_gen[0][0].shape, dtype=np.float32))
# predictions = model(np.array([train_data_gen[0][0][0]], dtype=np.float32))
# print(predictions)
# print(train_data_gen[0][0].shape)

# interpreter = tf.lite.Interpreter(model_content=model)
# interpreter.allocate_tensors()

# used to see how fast inference of ~400x300px inputs w/ batch size 5 took. Best on pro is 10 FPS.
for i in range(200):
    # Pick a random sample
    x, y = train_data_gen.next()
    
    # print(interpreter.get_input_details())

    # input_index = interpreter.get_input_details()[0]["index"]
    # output_index = interpreter.get_output_details()[0]["index"]

    # # Pre-processing: add batch dimension and convert to float32 to match with the model's input data format.
    # test_image = np.expand_dims(train_data_gen[0][0][0], axis=0).astype(np.float32)
    # interpreter.set_tensor(input_index, test_image)
    start = time.time()
    # # Run inference.
    # interpreter.invoke()

    # # Post-processing: remove batch dimension and find the digit with highest probability.
    # output = interpreter.tensor(output_index)

    y_pred = model(x)
    end = time.time()
    #print(output)
    print('%d, %0.3f' % (i, (end-start)))