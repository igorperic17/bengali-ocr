import pandas as pd
import pyarrow.parquet as pq
import cv2
import numpy as np
import keras
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization, Concatenate, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import applications

from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sprinkles import sprinkles

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

HEIGHT = 137
WIDTH = 236
INPUT_SIZE = 128

LABEL_ROOT_CLASSES = 168
LABEL_VOWEL_CLASSES = 11
LABEL_CONSONANT_CLASSES = 7

BATCH_SIZE = 100
EPOCHS = 3

# build network
input_layer = Input(shape=(INPUT_SIZE, INPUT_SIZE, 1))
# model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu', input_shape=(INPUT_SIZE, INPUT_SIZE, 1))(input_layer)
# model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
# model = BatchNormalization(momentum=0.15)(model)
# model = MaxPool2D(pool_size=(2, 2))(model)
# model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
# model = MaxPool2D(pool_size=(3, 3))(model)

# model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
# model = BatchNormalization(momentum=0.15)(model)
# model = MaxPool2D(pool_size=(2, 2))(model)
# model = Dropout(rate=0.3)(model)
# model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
# model = MaxPool2D(pool_size=(2, 2))(model)
# model = Dropout(rate=0.3)(model)

# flattened_model = Flatten()(model)
# flattened_model = Dense(1024, activation='relu')(flattened_model)
# flattened_model = Dropout(rate=0.3)(flattened_model)
# flattened_model = Dense(512, activation='relu')(flattened_model)

# # construct new branch for vowel, feeding the input of the root into the vowel branch

# vowel_branch = Dense(LABEL_VOWEL_CLASSES, activation='softmax', name='vowel_out')(flattened_model)
# consonant_branch = Dense(LABEL_CONSONANT_CLASSES, activation='softmax', name='consonant_out')(flattened_model)
# root_branch = Dense(LABEL_ROOT_CLASSES, activation='softmax', name='root_out')(flattened_model)
# # one leaf node ends here


base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (INPUT_SIZE,INPUT_SIZE,1))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
vowel_branch = Dense(LABEL_VOWEL_CLASSES, activation='softmax', name='vowel_out')(x)
consonant_branch = Dense(LABEL_CONSONANT_CLASSES, activation='softmax', name='consonant_out')(x)
root_branch = Dense(LABEL_ROOT_CLASSES, activation='softmax', name='root_out')(x)

model = Model(inputs = base_model.input, outputs = [root_branch, vowel_branch, consonant_branch])
lossWeights = {"root_out": 1.0, "consonant_out": 0.5, "vowel_out": 0.5}
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], loss_weights=lossWeights)
model.summary()

def center_letter(image):
    mask = image.copy()
    mask = mask.astype(np.uint8)
    mask[mask >= 50] = 255
    mask[mask < 50] = 0

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    minX, minY, maxX, maxY = INPUT_SIZE, INPUT_SIZE, 0, 0
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        minX = min(minX, x)
        minY = min(minY, y)
        maxX = max(maxX, x + w)
        maxY = max(maxY, y + h)
    isolated = image[minY:maxY, minX:maxX].copy()

    max_dim = max(maxY - minY, maxX - minX)
    centered = np.zeros((max_dim, max_dim), dtype=np.uint8)
    cx, cy = (max_dim - isolated.shape[1])//2, (max_dim - isolated.shape[0]) // 2
    centered[cy:(isolated.shape[0]+cy), cx:(cx+isolated.shape[1])] = isolated
    centered = cv2.resize(centered, (INPUT_SIZE, INPUT_SIZE))

    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(5, 3))
    # axes[0].imshow(image,'gray')
    # axes[1].imshow(centered,'gray', vmin=0, vmax=255)
    # axes[2].imshow(isolated, 'gray', vmin=0, vmax=255)
    # plt.show()
    # plt.clf()
    # plt.pause(0.0001)

    # normalize the image
    centered = centered.astype(np.float32) / 255.0
    centered /= centered.max()

    return centered

## construct the training data

train_metadata = pd.read_csv('data/train.csv')
for i in range(100):
    feather_file_idx = i % 4
    print('Iteration {0}'.format(i))
    print('Reading the data from batch {0}'.format(feather_file_idx))
    train_df = pd.read_feather('data/train_image_data_{0}.feather'.format(feather_file_idx))
    # train_images = train_df.iloc[:, 1:].values.reshape([-1, HEIGHT, WIDTH])
    train_df = pd.merge(train_df, train_metadata, on = 'image_id').drop(['image_id'], axis = 1)

    X_train_orig = train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'grapheme'], axis = 1).values.astype(np.uint8)
    X_train_orig = X_train_orig.reshape(-1, HEIGHT, WIDTH)
    # resize
    X_train = []
    for image in X_train_orig:
        image_resized = center_letter(255 - image)
        X_train.append(image_resized)

    X_train = np.reshape(X_train, (-1, INPUT_SIZE, INPUT_SIZE, 1))

    def one_hot(target_label, num_of_classes):
        Y_train_orig = train_df[target_label].values
        Y_train = []
        for label in Y_train_orig:
            label_resized = np.zeros((num_of_classes), dtype=np.float32)
            label_resized[label] = 1
            Y_train.append(label_resized)
        Y_train = np.reshape(Y_train, (-1, num_of_classes))
        return Y_train

    Y_train_root = one_hot('grapheme_root', LABEL_ROOT_CLASSES)
    Y_train_vowel = one_hot('vowel_diacritic', LABEL_VOWEL_CLASSES)
    Y_train_consonant = one_hot('consonant_diacritic', LABEL_CONSONANT_CLASSES)

    print(Y_train_root.shape)
    print(Y_train_vowel.shape)
    print(Y_train_consonant.shape)

    x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, y_train_consonant, y_test_consonant = \
        train_test_split(X_train, Y_train_root, Y_train_vowel, Y_train_consonant, test_size=0.15)
        # train_test_split(X_train, Y_train_root, Y_train_vowel, Y_train_consonant, test_size=0.15, random_state=42)


    # from keras.callbacks import ModelCheckpoint
    # checkpoint = ModelCheckpoint(filepath='best_model.h5',save_best_only=True, monitor='val_aucroc', mode='max', period=5, verbose=1) 
    # checkpoint
    filepath="model/weights-improvement-{epoch:02d}-{val_root_out_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_root_out_accuracy', verbose=1, save_best_only=False, mode='max')
    callbacks_list = [checkpoint]

    history = model.fit(x_train, { 'root_out' : y_train_root, 'vowel_out': y_train_vowel, 'consonant_out': y_train_consonant }, \
        callbacks=callbacks_list, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, validation_data=(x_test, { 'root_out' : y_test_root, 'vowel_out': y_test_vowel, 'consonant_out': y_test_consonant }))

    print(history)

    del train_df
    del x_train
    del x_test
    del y_train_root
    del y_test_root
    del y_train_vowel
    del y_test_vowel
    del y_train_consonant
    del y_test_consonant
    del X_train_orig
    del X_train
    del Y_train_root
    del Y_train_vowel
    del Y_train_consonant




