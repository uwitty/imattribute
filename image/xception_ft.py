import os

from keras.applications.xception import Xception
from keras.models import Model, Sequential
from keras.layers import Input, Flatten
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.pooling import GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import keras.callbacks

from mylib.image import utils
import datetime

ROWS = 299
COLS = 299

def Xception_ft(classes, weights=None):
    input_tensor = Input(shape=(ROWS, COLS, 3))
    xception = Xception(include_top=False, weights='imagenet', input_tensor=input_tensor)
    for layer in xception.layers:
        layer.trainable = False
    #print('len(xception.layers)', len(xception.layers))

    x = xception.output
    x = GlobalAveragePooling2D()(x)

    if isinstance(classes, list):
        predictions = [Dense(cls, activation='softmax', name='output{}'.format(idx))(x)
                       for idx, cls in enumerate(classes)]
        losses = {'output{}'.format(idx): 'categorical_crossentropy'
                  for idx in range(len(classes))}
    else:
        predictions = Dense(classes, activation='softmax', name='output')(x)
        losses = 'categorical_crossentropy'

    #predictions = Dense(classes, activation='softmax', name='output1')(x)
    #predictions2 = Dense(classes, activation='softmax', name='output2')(x)
    #predictions3 = Dense(classes, activation='softmax', name='output3')(x)

    #model = Model(xception.input, [predictions, predictions2, predictions3])
    model = Model(xception.input, predictions)
    #top_model = Sequential()
    #top_model.add(Flatten(input_shape=xception.output_shape[1:]))
    #top_model.add(Dense(256, activation='relu'))
    #top_model.add(Dropout(0.5))
    #top_model.add(Dense(classes, activation='softmax'))
    #model = Model(inputs=xception.input, outputs=top_model(xception.output))

    #print(losses)
    if weights is not None:
        model.load_weights(weights)

    print('len(model.layers)', len(model.layers))
    return model


def train(train_generator, validation_data,
          classes, losses,
          epochs=100,
          default_weights=None,
          model_name=None,
          output_dir='output',
          steps_per_epoch=None,
          patience=10,
          epochs_fc=None
          ):
    if epochs_fc is None:
        epochs_fc = epochs // 5

    m = Xception_ft(weights=default_weights, classes=classes)
    #m.summary()

    # callbacks
    weight_fname_format = 'weights{epoch:02d}-loss{loss:.2f}-vloss{val_loss:.2f}.hdf5'
    if model_name is not None:
        weight_fname_format = model_name + '-' + weight_fname_format
    cp_cb = keras.callbacks.ModelCheckpoint(filepath = os.path.join(output_dir, weight_fname_format), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
    
    es_cb = keras.callbacks.EarlyStopping(patience=patience)
    
    histories = []
    if default_weights is None:
        for layer in m.layers[:132]:
            layer.trainable = False

        m.compile(optimizer='nadam',
                      #loss='categorical_crossentropy',
                      loss=losses,
                      metrics=['accuracy'])

        histories += [m.fit_generator(
            train_generator,
            steps_per_epoch,
            epochs=epochs_fc,
            validation_data=validation_data,
            verbose=2,
            callbacks=[cp_cb, es_cb])]

    for layer in m.layers[:126]:
        layer.trainable = False
    for layer in m.layers[126:]:
        layer.trainable = True

    m.compile(optimizer='nadam',
                  #loss='categorical_crossentropy',
                  loss=losses,
                  metrics=['accuracy'])

    histories += [m.fit_generator(
        train_generator,
        steps_per_epoch,
        epochs=epochs,
        initial_epoch=epochs_fc,
        validation_data=validation_data,
        verbose=2,
        callbacks=[cp_cb, es_cb])]
    
    m.save(output_dir + '/final.h5')
    utils.plot_history(histories, output_dir + '/plot.png')
    return histories

