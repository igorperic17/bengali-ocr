import pandas as pd
import pyarrow.parquet as pq
import cv2
import numpy as np
import keras
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.models import Model

from sklearn.model_selection import train_test_split

HEIGHT = 137
WIDTH = 236
INPUT_SIZE = 64

LABEL_ROOT_CLASSES = 168
LABEL_VOWEL_CLASSES = 11
LABEL_CONSONANT_CLASSES = 7

BATCH_SIZE = 90
EPOCHS = 30

print('Reading the data...')
train_metadata = pd.read_csv('data/train.csv')
train_df = pd.read_feather('data/train_image_data_0.feather')
# train_images = train_df.iloc[:, 1:].values.reshape([-1, HEIGHT, WIDTH])
train_df = pd.merge(train_df, train_metadata, on = 'image_id').drop(['image_id'], axis = 1)

print(train_df.head())


### TODO: data cleanup, centering

# build network
input_layer = Input(shape=(INPUT_SIZE, INPUT_SIZE, 1))
model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu', input_shape=(INPUT_SIZE, INPUT_SIZE, 1))(input_layer)
model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
model = MaxPool2D(pool_size=(3, 3))(model)

# model = Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
# model = BatchNormalization(momentum=0.15)(model)
# model = MaxPool2D(pool_size=(2, 2))(model)
# model = Dropout(rate=0.3)(model)
# model = Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
# model = MaxPool2D(pool_size=(2, 2))(model)

model = Flatten()(model)
model = Dense(1024, activation='relu')(model)
model = Dense(512, activation='relu')(model)

last_layer = Dense(LABEL_ROOT_CLASSES, activation='softmax')(model)
last_layer_vowel = Dense(LABEL_VOWEL_CLASSES, activation='softmax')(model)
last_layer_consonant = Dense(LABEL_CONSONANT_CLASSES, activation='softmax')(model)

model = Model(inputs = input_layer, outputs = [last_layer, last_layer_vowel, last_layer_consonant])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# inputs = Input(shape = (INPUT_SIZE, INPUT_SIZE, 1))

# model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(INPUT_SIZE, INPUT_SIZE, 1))(inputs)
# model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
# # model = BatchNormalization(momentum=0.15)(model)
# model = MaxPool2D(pool_size=(2, 2))(model)
# model = Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
# model = Dropout(rate=0.3)(model)

# model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
# model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
# # model = BatchNormalization(momentum=0.15)(model)
# model = MaxPool2D(pool_size=(2, 2))(model)
# model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
# model = BatchNormalization(momentum=0.15)(model)
# model = Dropout(rate=0.3)(model)
# #Added More Layers
# model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
# model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
# model = BatchNormalization(momentum=0.15)(model)
# model = MaxPool2D(pool_size=(2, 2))(model)
# model = Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
# model = BatchNormalization(momentum=0.15)(model)
# model = Dropout(rate=0.3)(model)

# model = Flatten()(model)
# model = Dense(512, activation = "relu")(model)
# model = Dropout(rate=0.3)(model)
# dense = Dense(512, activation = "relu")(model)

# head_root = Dense(168, activation = 'softmax')(dense)
# # head_vowel = Dense(11, activation = 'softmax')(dense)
# # head_consonant = Dense(7, activation = 'softmax')(dense)

# model = Model(inputs=inputs, outputs=head_root)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


## construct the training data

X_train_orig = train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'grapheme'], axis = 1).values.astype(np.float32)
X_train_orig = X_train_orig.reshape(-1, HEIGHT, WIDTH) / 255.0
# resize
X_train = []
for image in X_train_orig:
    image_resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
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
    train_test_split(X_train, Y_train_root, Y_train_vowel, Y_train_consonant, test_size=0.1, random_state=666)

history = model.fit(x_train, { 'dense_3' : y_train_root, 'dense_4': y_train_vowel, 'dense_5': y_train_consonant }, \
    batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, validation_data=(x_test, { 'dense_3' : y_test_root, 'dense_4': y_test_vowel, 'dense_5': y_test_consonant }))

print(history)






