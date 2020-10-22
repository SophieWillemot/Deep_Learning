""" U-net model with keras """

import tensorflow as tf


class CustomUNet2D(object):
    """
    Implements UNet architecture https://arxiv.org/abs/1606.06650
    """

    def __init__(self,
                 image_shape,
                 filters=(8, 16, 32),
                 kernel=(3, 3),
                 activation=tf.keras.layers.LeakyReLU(),
                 padding='same',
                 pooling=(2, 2)):

        #self.num_classes = number_class
        self.image_shape = image_shape

        self.filters = filters
        self.kernel = kernel
        self.activation = activation
        self.padding = padding  # "same" or "valid"
        self.pooling = pooling

    def double_convolution(self, input_, num_filters):
        layer1 = tf.keras.layers.Conv2D(filters=num_filters,
                                        kernel_size=self.kernel,
                                        padding=self.padding
                                        )(input_)
        layer2 = self.activation(layer1)
        layer3 = tf.keras.layers.Conv2D(filters=num_filters,
                                        kernel_size=self.kernel,
                                        padding=self.padding
                                        )(layer2)
        layer4 = self.activation(layer3)
        layer5 = tf.keras.layers.SpatialDropout2D(0.2)(layer4)  # adapatation in progress
        return layer5

    def maxpooling(self, intput_):
        layer = tf.keras.layers.MaxPool2D(pool_size=self.pooling,
                                          padding=self.padding
                                          )(intput_)
        return layer

    def upsampling(self, input_):
        layer = tf.keras.layers.UpSampling2D(size=self.pooling
                                             )(input_)
        return layer

    @staticmethod
    def concatenate(upconv_input, forward_input):
        layer = tf.keras.layers.Concatenate()([upconv_input, forward_input])
        return layer

    def final_convolution(self, input_):
        layer = tf.keras.layers.Conv2D(filters=self.num_classes,
                                       kernel_size=(1, 1),
                                       padding=self.padding
                                       )(input_)
        return layer

    def compression_block(self, input_, num_filters):
        layer1 = self.double_convolution(input_, num_filters)
        layer2 = self.maxpooling(layer1)
        layer3 = tf.keras.layers.BatchNormalization()(layer2)
        return layer3

    def bottleneck(self, input_, num_filters):
        layer1 = self.double_convolution(input_, num_filters)
        layer2 = tf.keras.layers.BatchNormalization()(layer1)
        return layer2

    def expansion_block(self, upconv_input, forward_input, num_filters):
        upconv_input = self.upsampling(upconv_input)
        layer1 = self.concatenate(upconv_input, forward_input)
        layer2 = self.double_convolution(layer1, num_filters)
        layer3 = tf.keras.layers.BatchNormalization()(layer2)
        return layer3

    def create_model(self):
        input_ = tf.keras.layers.Input(shape=self.image_shape, dtype=tf.float32, name="input")

        x = input_

        # compression/encoder
        for i in range(len(self.filters) - 1):
            x = self.compression_block(x, num_filters=self.filters[i])

        # bottleneck
        #x = self.bottleneck(x, num_filters=self.filters[-1])

        # expansion/decoder
        #for i in reversed(range(len(self.filters) - 1)):
        #    x = self.expansion_block(x, forwards[i], num_filters=self.filters[i])

        # final layer
        #logits = self.final_convolution(x)
        #x = self.final_convolution(x)
        #output_ = tf.keras.layers.Softmax(name='output')(logits)

        x = tf.keras.layers.Flatten()(x)

        #final output
        left_arm = tf.keras.layers.Dense(2, activation='softmax', name='left_arm')(x)
        right_arm = tf.keras.layers.Dense(2, activation='softmax', name = 'right_arm')(x)

        head = tf.keras.layers.Dense(2, activation='softmax', name='head')(x)
        leg = tf.keras.layers.Dense(2, activation='softmax', name='leg')(x)

        model = tf.keras.models.Model(inputs = input_, outputs = [head, leg, right_arm, left_arm], name='UNet')

        return model