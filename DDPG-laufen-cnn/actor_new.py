


import keras.backend as keras_backend
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.initializers import normal
from keras.layers import Dense, Input, concatenate, AveragePooling2D, Conv2D, Flatten
from keras.models import Model






class ActorNetwork:
    def __init__(self, tf_session, state_size,hidden_units, action_size, width_cnn,height_cnn,n_channel, tau=0.001, lr=0.0001):
        self.tf_session = tf_session
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.lr = lr
        self.hidden_units = hidden_units
        self.width = width_cnn
        self.height = height_cnn
        self.n_channel = n_channel

        keras_backend.set_session(tf_session)

        self.action_gradient = tf.placeholder(tf.float32,[None, self.action_size])