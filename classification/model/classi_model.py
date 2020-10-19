import numpy as np

from tensorflow.keras.layers import Input
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.models import Model


def classic_model(input_shape=(503, 136, 1)):
    x_input = Input(input_shape)

    x = Conv2D(32, (3,3), name='conv1', activation='relu')(x_input)
    x = MaxPooling2D(pool_size=(2,2), name='max_pool1')(x)
    x = Conv2D(64, (3,3), name ='conv2', activation = 'relu')(x)
    x = MaxPooling2D(pool_size=(2,2), name='max_pool2')(x)
    x = Dropout(rate = 0.25, name='dropout1')(x)

    x = Conv2D(64, (3,3), name='conv3', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2), name = 'max_pool3')(x)
    x = Dropout(rate=0.5, name='dropout2')(x)


    x = Flatten(name ='flatten')(x)

    x = Dense(2048, activation='relu', name='dense1')(x)
    x = Dropout(rate=0.5, name='dropout3')(x)
    x = Dense(1024, activation='relu', name='dense2')(x)


    #final output
    left_arm = Dense(2, activation='softmax', name='left_arm')(x)
    right_arm = Dense(2, activation='softmax', name = 'right_arm')(x)

    head = Dense(2, activation='softmax', name='head')(x)
    leg = Dense(2, activation='softmax', name='leg')(x)

    model = Model(inputs = x_input, outputs = [head, leg, right_arm, left_arm], name='classic_model')
    return model 