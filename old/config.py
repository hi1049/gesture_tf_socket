import tensorflow as tf

# model config
tf.app.flags.DEFINE_integer("batch_size", 10, "batch size")
tf.app.flags.DEFINE_integer("num_units", 256, "number of LSTM/GRU units") # 256
tf.app.flags.DEFINE_integer("num_layers", 1, "number of layers of RNN network")
tf.app.flags.DEFINE_integer("height", 112, "height of a frame")
tf.app.flags.DEFINE_integer("width", 112, "width of a frame")
tf.app.flags.DEFINE_integer("FRAMES_PER_CLIP", 16, "frames per a single video clip")
tf.app.flags.DEFINE_integer("STRIDE", 8, "stride of time window")

# path
tf.app.flags.DEFINE_string("vgg_dir", './vgg16_weights_keras/*', "vgg16 weights dir for glob")
tf.app.flags.DEFINE_string("weights_path", 'resnet152_weights_tf.h5', "resnet152 weights for tensorflow backend")
tf.app.flags.DEFINE_string("frame_path", '../dataset', "frame path")

tf.app.flags.DEFINE_string("which", None, "which dataset to use")
tf.app.flags.DEFINE_boolean("debug", False, "activate tfdebug or not")

FLAGS = tf.app.flags.FLAGS