# !ls
# !git clone https://github.com/druzsan/test.git
# import os
# os.chdir('test')
# !ls
# !git pull
# !ls

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import os
import numpy as np
from keras import backend, models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator
backend.tensorflow_backend.set_session(session)

from models.vgg16 import VGG16

# Setting the main parameters
train_dir = "data/small_dataset/cropped/train"
validation_dir = "data/small_dataset/cropped/validation"
classes = len([dir for dir in os.listdir(train_dir)
               if os.path.isdir(os.path.join(train_dir, dir))])
train_batchsize = 64
val_batchsize = 16

# Building the model
vgg_no_fc = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

for layer in vgg_no_fc.layers:
    layer.trainable = False

# for layer in vgg_no_fc.layers:
#     print(layer, layer.trainable)

model = models.Sequential()

model.add(vgg_no_fc)

model.add(layers.Flatten(name='flatten'))
model.add(layers.Dense(4096, activation='relu', name='fc1'))
model.add(layers.Dense(4096, activation='relu', name='fc2'))
model.add(layers.Dense(classes, activation='softmax', name='predictions'))

# vgg_no_fc.summary()
# model.summary()

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=train_batchsize,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False
)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    verbose=1
)

model.save('weights/classification1.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

print("train accuracy:", acc)
print("validation accuracy:", val_acc)
print("train loss:", loss)
print("validation loss:", val_loss)

model.predict_generator(
    validation_generator,
    steps=validation_generator.samples / validation_generator.batch_size,
    verbose=1
)

ground_truth = validation_generator.classes

predictions = model.predict_generator(
    validation_generator,
    steps=validation_generator.samples / validation_generator.batch_size,
    verbose=1
)
predicted_classes = np.argmax(predictions, axis=1)

errors = len([1 for (a, b) in zip(ground_truth, predicted_classes) if a != b])
print("No of errors = {}/{}".format(errors, validation_generator.samples))