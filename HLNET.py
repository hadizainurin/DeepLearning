import tensorflow as tf
import math
from keras.utils.vis_utils import model_to_dot
import keras as k
from IPython.display import SVG

from tensorflow.python.keras.engine import training
NUM_CLASSES = 1000

#def swish(x):
#    x= x * tf.nn.sigmoid(x)
#    return x

def round_filters(filters, multiplier):
    depth_divisor = 8
    min_depth = None
    min_depth = min_depth or depth_divisor
    filters = filters * multiplier
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, multiplier):
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, ratio=0.25):
        super(SEBlock, self).__init__()
        self.num_reduced_filters = max(1, int(input_channels * ratio))
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.reduce_conv = tf.keras.layers.Conv2D(filters=self.num_reduced_filters,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same")
        self.expand_conv = tf.keras.layers.Conv2D(filters=input_channels,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same")

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = self.reduce_conv(branch)
        branch = tf.keras.activations.swish(branch)
        branch = self.expand_conv(branch)
        branch = tf.nn.sigmoid(branch)
        se_output = inputs * branch
        return se_output


class MBConv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate):
        super(MBConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.conv1 = tf.keras.layers.Conv2D(filters=in_channels * expansion_factor,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k),
                                                      strides=stride,
                                                      padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.se = SEBlock(input_channels=in_channels * expansion_factor)
        self.conv2 = tf.keras.layers.Conv2D(filters=out_channels,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(rate=drop_connect_rate)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.keras.activations.swish(x)
        x = self.dwconv(x)
        x = self.bn2(x, training=training)
        x = self.se(x)
        x = tf.keras.activations.swish(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        if self.stride == 1 and self.in_channels == self.out_channels:
          if self.drop_connect_rate:
            x = self.dropout(x, training=training)
          #x = tf.keras.layers.add([x, inputs])
          x = tf.keras.layers.concatenate([x, inputs])
        
        return x

def build_mbconv_block(in_channels, out_channels, layers, stride, expansion_factor, k, drop_connect_rate):
    for i in range(layers):
        if i == 0:
            x = MBConv(in_channels=in_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=stride,
                             k=k,
                             drop_connect_rate=drop_connect_rate)
        else:
            x.add = MBConv(in_channels=out_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=1,
                             k=k,
                             drop_connect_rate=drop_connect_rate)
    return x


def HLNet (img_shape, width_coefficient, depth_coefficient, dropout_rate, drop_connect_rate=0.2):
  block1 = build_mbconv_block(in_channels=round_filters(32, width_coefficient),
                                         out_channels=round_filters(16, width_coefficient),
                                         layers=round_repeats(1, depth_coefficient),
                                         stride=1,
                                         expansion_factor=1, k=3, drop_connect_rate=drop_connect_rate)
  block2 = build_mbconv_block(in_channels=round_filters(16, width_coefficient),
                                         out_channels=round_filters(24, width_coefficient),
                                         layers=round_repeats(2, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)
  block3 = build_mbconv_block(in_channels=round_filters(24, width_coefficient),
                                         out_channels=round_filters(40, width_coefficient),
                                         layers=round_repeats(2, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)
  block4 = build_mbconv_block(in_channels=round_filters(40, width_coefficient),
                                         out_channels=round_filters(80, width_coefficient),
                                         layers=round_repeats(3, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)
  block5 = build_mbconv_block(in_channels=round_filters(80, width_coefficient),
                                         out_channels=round_filters(112, width_coefficient),
                                         layers=round_repeats(3, depth_coefficient),
                                         stride=1,
                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)
  block6 = build_mbconv_block(in_channels=round_filters(112, width_coefficient),
                                         out_channels=round_filters(192, width_coefficient),
                                         layers=round_repeats(4, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)
  block7 = build_mbconv_block(in_channels=round_filters(192, width_coefficient),
                                         out_channels=round_filters(320, width_coefficient),
                                         layers=round_repeats(1, depth_coefficient),
                                         stride=1,
                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)
  #repetition = 1,2,2,3,4,1
  #layers = 1,2,3,4
  input = tf.keras.Input(shape=img_shape)
  x = tf.keras.layers.Conv2D(filters=round_filters(32, width_coefficient),
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")(input)
  x = tf.keras.layers.BatchNormalization()(x, training=None)
  x = tf.keras.activations.swish(x)
  x = block1(x)
  x = block2(x)
  x = block3(x)
  x = block4(x)
  x = block5(x)
  x = block6(x)
  x = block7(x)
  x = tf.keras.layers.Conv2D(filters=round_filters(1280, width_coefficient),
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")(x)
  x = tf.keras.layers.BatchNormalization()(x, training=None)
  x = tf.keras.activations.swish(x)
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dropout(rate=dropout_rate)(x, training=None)
  output =tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

  model = tf.keras.Model(input, output)
  return model

tf.keras.backend.clear_session()
image_shape = 224, 224, 3
model = HLNet(image_shape, 1.0, 1.0, 0.2)
model.summary()

tf.keras.utils.plot_model(model, to_file="my_model.png", show_shapes=True, show_layer_names=True)

SVG(model_to_dot(model).create(prog='dot', format='svg'))

def MBConv2(input2, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate, training=None, **kwargs):
       
      in_channels = in_channels
      out_channels = out_channels
      stride = stride
      drop_connect_rate = drop_connect_rate
      conv1 = tf.keras.layers.Conv2D(filters=in_channels * expansion_factor,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
      bn1 = tf.keras.layers.BatchNormalization()
      dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k),
                                                      strides=stride,
                                                      padding="same")
      bn2 = tf.keras.layers.BatchNormalization()
      se = SEBlock(input_channels=in_channels * expansion_factor)
      conv2 = tf.keras.layers.Conv2D(filters=out_channels,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
      bn3 = tf.keras.layers.BatchNormalization()
      dropout = tf.keras.layers.Dropout(rate=drop_connect_rate)

      x = conv1(input2)
      x = bn1(x, training=training)
      x = tf.keras.activations.swish(x)
      x = dwconv(x)
      x = bn2(x, training=training)
      x = se(x)
      x = tf.keras.activations.swish(x)
      x = conv2(x)
      x = bn3(x, training=training)
      if stride == 1 and in_channels == out_channels:
        if drop_connect_rate:
          x = dropout(x, training=training)
        x = tf.keras.layers.add([x, input2])
        x = tf.keras.layers.concatenate([x, input2])
      return x
        
def HLNet2 (img_shape, width_coefficient, dropout_rate, drop_connect_rate=0.2):
  layers =1
  input1 = tf.keras.Input(shape=img_shape)
  x = tf.keras.layers.Conv2D(filters=round_filters(32, width_coefficient),
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")(input1)
  x = tf.keras.layers.BatchNormalization()(x, training=None)
  x = tf.keras.activations.swish(x)
  x = MBConv2(x, in_channels=round_filters(32, width_coefficient),
                                         out_channels=round_filters(16, width_coefficient),
                                         stride=1,
                                         expansion_factor=1, k=3, drop_connect_rate=drop_connect_rate)
  x = MBConv2(x, in_channels=round_filters(16, width_coefficient),
                                         out_channels=round_filters(16, width_coefficient),
                                         stride=1,
                                         expansion_factor=1, k=3, drop_connect_rate=drop_connect_rate)
  x = MBConv2(x, in_channels=round_filters(16, width_coefficient),
                                         out_channels=round_filters(16, width_coefficient),
                                         stride=1,
                                         expansion_factor=1, k=3, drop_connect_rate=drop_connect_rate)
  x = tf.keras.layers.Conv2D(filters=round_filters(1280, width_coefficient),
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")(x)
  x = tf.keras.layers.BatchNormalization()(x, training=None)
  x = tf.keras.activations.swish(x)
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dropout(rate=dropout_rate)(x, training=None)
  output1 =tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

  model2 = tf.keras.Model(input1, output1)
  return model2

tf.keras.backend.clear_session()
image_shape = 224, 224, 3
model2 = HLNet2(image_shape, 1.0, 1.0, 0.2)
model2.summary()
tf.keras.utils.plot_model(model2, to_file="my_model2.png", show_shapes=True, show_layer_names=True)