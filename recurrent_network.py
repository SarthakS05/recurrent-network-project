from __future__ import absolute_import, division, print_function

# Import TensorFlow v2.
import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np

# MNIST dataset parameters.
num_classes = 10
num_features = 784

# Training Parameters
learning_rate = 0.001
training_steps = 1000
batch_size = 32
display_step = 100

# Network Parameters
# MNIST images are processed as 28 sequences of 28 timesteps.
num_input = 28
timesteps = 28
num_units = 32

# Prepare MNIST data.
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train = x_train.reshape([-1, 28, 28])
x_test = x_test.reshape([-1, num_features])
x_train, x_test = x_train / 255.0, x_test / 255.0

# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)


class LSTM(Model):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm_layer = layers.LSTM(units=num_units)
        self.out = layers.Dense(num_classes)

    def call(self, x, is_training=False):
        x = self.lstm_layer(x)
        x = self.out(x)
        if not is_training:
            x = tf.nn.softmax(x)
        return x


lstm_net = LSTM()


def cross_entropy_loss(x, y):
    y = tf.cast(y, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    return tf.reduce_mean(loss)


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


optimizer = tf.optimizers.Adam(learning_rate)


def run_optimization(x, y):
    with tf.GradientTape() as gradient_tape:
        predictions = lstm_net(x, is_training=True)
        loss = cross_entropy_loss(predictions, y)

    trainable_variables = lstm_net.trainable_variables
    gradients = gradient_tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))


for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    run_optimization(batch_x, batch_y)

    if step % display_step == 0:
        predictions = lstm_net(batch_x, is_training=True)
        loss = cross_entropy_loss(predictions, batch_y)
        batch_accuracy = accuracy(predictions, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, batch_accuracy))