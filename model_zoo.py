import tensorflow as tf
import i3d
import re
from stn import spatial_transformer_network as transformer


def preprocess(inps, batch_size, is_training):
    _shape = tf.shape(inps)
    l, h, w = tf.unstack(_shape[1:-1])

    # first, crop it randomly
    crop_inputs = tf.cond(is_training,
                          lambda: tf.random_crop(inps, tf.unstack(_shape[:2]) + [224, 224, 3]),
                          lambda: inps[:, :, (h - 224) / 2:(h + 224) / 2, (w - 224) / 2:(w + 224) / 2])
    crop_inputs = tf.reshape(crop_inputs, (batch_size, -1, 224, 224, 3))  # for channel dimension restore

    processed_inputs_trans = []

    for _in in tf.unstack(crop_inputs):
        scale_x = tf.cond(is_training,
                          lambda: tf.random_uniform([], 1.0, 5.0),
                          lambda: tf.constant(1.0))
        scale_y = tf.cond(is_training,
                          lambda: tf.random_uniform([], 1.0, scale_x),
                          lambda: tf.constant(1.0))

        def body(t, seq_img_trans):
            img = tf.image.per_image_standardization(_in[t])  # standardization
            img_trans = transformer(tf.expand_dims(img, 0), tf.stack([[scale_x, 0., 0., 0., scale_y, 0.]]))

            seq_img_trans = seq_img_trans.write(t, img_trans[0])

            return t + 1, seq_img_trans

        t = tf.constant(0)
        seq_img_trans = tf.TensorArray(dtype=tf.float32, size=l)

        _, seq_img_trans = tf.while_loop(cond=lambda t, *_: t < l,
                                         body=body, loop_vars=(t, seq_img_trans))

        processed_inputs_trans.append(seq_img_trans.stack())

    return processed_inputs_trans

class I3DNet:
    def __init__(self, inps, n_class, batch_size, pretrained_model_path, final_end_point, dropout_keep_prob, is_training, scope='v/SenseTime_I3D'):

        self.final_end_point = final_end_point
        self.n_class = n_class
        self.batch_size = batch_size
        self.dropout_keep_prob = dropout_keep_prob
        self.is_training = is_training
        self.scope = scope

        # build entire pretrained networks (dummy operation!)
        i3d.I3D(preprocess(inps,batch_size,is_training), num_classes=n_class,
            final_endpoint=final_end_point, scope=scope,
            dropout_keep_prob=dropout_keep_prob, is_training=is_training)

        var_dict = { re.sub(r':\d*','',v.name):v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope) }
        self.assign_ops = []
        if pretrained_model_path:
            for var_name, var_shape in tf.contrib.framework.list_variables(pretrained_model_path):
                if var_name.startswith('v/SenseTime_I3D/Logits'):
                    continue
                # load variable
                var = tf.contrib.framework.load_variable(pretrained_model_path, var_name)
                assign_op = var_dict[var_name].assign(var)
                self.assign_ops.append(assign_op)

    def __call__(self, inps):
        proc = preprocess(inps,self.batch_size,self.is_training)
        tf.summary.image('crop', proc[0])
        merge_op = tf.summary.merge_all()
        out, _ = i3d.I3D(proc, num_classes=self.n_class,
                        final_endpoint=self.final_end_point, scope=self.scope,
                        dropout_keep_prob=self.dropout_keep_prob, is_training=self.is_training, reuse=True)

        return out, merge_op
