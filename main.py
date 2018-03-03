import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import sys
import scipy.misc


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    tags = [vgg_tag]
    tensor_names = [
      'image_input:0',
      'keep_prob:0',
      'layer3_out:0',
      'layer4_out:0',
      'layer7_out:0'
    ]

    tf.saved_model.loader.load(sess, tags, vgg_path)
    graph = tf.get_default_graph()

    loader = graph.get_tensor_by_name
    output = tuple([loader(tensor_name) for tensor_name in tensor_names])
    
    return output
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # prevent weights from getting too large and overfitting
    regularization_const = 1e-3
    regularizer = tf.contrib.layers.l2_regularizer(regularization_const)
    #adding relu broke the model, probably because it does not match the VGG
    activation = None # tf.nn.relu

    # the encoder layer is already mostly written, but is ending with a convolution layer
    # we need a 1x1 convolution to act like the dense layer in original vgg
    # but preserves spatial information
    encoder_1x1 = tf.layers.conv2d(
      vgg_layer7_out,                 # the last tensor in the VGG model
      1024,                           # number of channels
      1,                              # 1x1 kernel
      strides=(1, 1),                 # 1x1 stride
      padding='same',                 # don't want to change the shape
      activation=activation,          # nonlinearity
      kernel_regularizer=regularizer, # penalize big weights to prevent gradient explosion
      name='encoder_1x1'
    )
    
    decoder_conv1 = tf.layers.conv2d_transpose(
      encoder_1x1,
      512,
      4,
      strides=(2, 2),
      padding='same',
      activation=activation,
      kernel_regularizer=regularizer,
      name='decoder_conv1'
    )

    decoder_1x1_1 = tf.layers.conv2d(
      vgg_layer4_out,
      512,
      1,
      strides=(1, 1),
      padding='same',
      activation=activation,
      kernel_regularizer=regularizer,
      name='decoder_1x1_1'
    )

    skip1 = tf.add(decoder_conv1, decoder_1x1_1)

    decoder_conv2 = tf.layers.conv2d_transpose(
      skip1,
      256,
      4,
      strides=(2, 2),
      padding='same',
      activation=activation,
      kernel_regularizer=regularizer,
      name='decoder_conv2'
    )

    decoder_1x1_2 = tf.layers.conv2d(
      vgg_layer3_out,
      256,
      1,
      strides=(1, 1),
      padding='same',
      activation=activation,
      kernel_regularizer=regularizer,
      name='decoder_1x1_2'
    )

    skip2 = tf.add(decoder_conv2, decoder_1x1_2)

    output = tf.layers.conv2d_transpose(
      skip2,
      num_classes,
      16,                   # bigger stride here to match vgg
      strides=(8, 8),       # likewise, bigger kernel
      padding='same',
      activation=None,
      kernel_regularizer=regularizer,
      name='decoder_conv3'
    )

    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=correct_label))
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, accuracy, saver):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param accuracy: TF op to measure training accuracy
    :param saver: TF saver to save model for later use
    """
    for epoch in range(epochs):
      batch_count = 0
      for images, labels in get_batches_fn(batch_size):
        batch_count += 1
        _, loss, acc = sess.run([train_op, cross_entropy_loss, accuracy], feed_dict={input_image: images, correct_label: labels, keep_prob: 0.5, learning_rate: 1e-3})
        print("Batch {}, loss is {}, accuracy is {}".format(batch_count, loss, acc))

      saver.save(sess, "./model/model.ckpt")

# the test worked, but I changed the method signature
#tests.test_train_nn(train_nn)


def run(load_existing=False):
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(data_dir, image_shape)

        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        output_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        epochs = 64
        batch_size = 16

        labels = tf.placeholder(tf.float32, shape=((None,) + image_shape + (num_classes,)), name='labels_placeholder')
        learning_rate = tf.placeholder(tf.float32)

        logits, train_op, cross_entropy_loss = optimize(output_layer, labels, learning_rate, num_classes)

        correct_pred = tf.equal(tf.argmax(output_layer, 3), tf.argmax(labels, 3))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver()

        if load_existing:
          saver.restore(sess, "./model/model.ckpt")

          print("Model loaded. Generating images...")
          image_outputs = helper.gen_test_output(
            sess, logits, keep_prob, image_input, os.path.join(data_dir, 'video_frames'), image_shape)
          print("Images generated!")
          for name, image in image_outputs:
            scipy.misc.imsave(os.path.join('./segmented_video', name), image)
          
          helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        else:
          train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,
            labels, keep_prob, learning_rate, accuracy, saver)

          helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)



if __name__ == '__main__':
    load_existing = False
    if len(sys.argv) > 1:
      load_existing = True
    run(load_existing)
