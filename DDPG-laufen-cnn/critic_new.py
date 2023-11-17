import keras.backend as keras_backend
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.layers import Dense, Input, add, Conv2D, Concatenate, Flatten, AveragePooling2D, MaxPooling2D, concatenate
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

image_network = "NETWORKS/"

class CriticNetwork:
    def __init__(self, tf_session,action_size,hidden_units,width_cnn,height_cnn,n_channel, tau, lr):
        self.tf_session = tf_session
        self.tau = tau
        self.lr = lr
        self.action_size = action_size
        self.hidden_units = hidden_units
        self.width = width_cnn
        self.height = height_cnn
        self.n_channel = n_channel
        keras_backend.set_session(tf_session)

        self.model, self.state_input, self.action_input = self.generate_model()
        self.target_model, _, _ = self.generate_model()

        self.critic_gradients = tf.gradients(self.model.output, self.action_input)
        self.tf_session.run(tf.global_variables_initializer())

    def get_gradients(self, states_im, actions):
        return self.tf_session.run(
            self.critic_gradients,
            feed_dict={self.state_input: states_im, self.action_input: actions},
        )[0]

    def train_target_model(self):
        main_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        target_weights = [
            self.tau * main_weight + (1 - self.tau) * target_weight
            for main_weight, target_weight in zip(main_weights, target_weights)
        ]
        self.target_model.set_weights(target_weights)

    def generate_model(self):
        state_input = Input(shape=(self.height, self.width, self.n_channel), name='Input_image')

        cnn_1 = Conv2D(32, (5, 5), padding='same', activation='relu')(state_input)
        cnn_1ap = MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(cnn_1)

        cnn_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(cnn_1ap)
        cnn_2ap = MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(cnn_2)

        cnn_3 = Conv2D(128, (3, 3), padding='same', activation='relu')(cnn_2ap)
        cnn_3ap = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(cnn_3)


        flat = Flatten()(cnn_3ap)

        state_h1 = Dense(self.hidden_units[0], activation="relu")(flat)
        state_h2 = Dense(self.hidden_units[1], activation="linear")(state_h1)

        action_input = Input(shape=[2])
        action_h1 = Dense(self.hidden_units[1], activation="linear")(action_input)

        #merged = concatenate([state_h2, action_h1])
        merged = concatenate([state_h2, action_h1])
        merged_h1 = Dense(self.hidden_units[1], activation="relu")(merged)

        output_layer = Dense(1, activation="linear")(merged_h1)
        model = Model(inputs=[state_input, action_input], outputs=output_layer)

        model.compile(loss="mse", optimizer=Adam(lr=self.lr))
        tf.keras.utils.plot_model(model, to_file=image_network + 'critic_model_CNN.png',
                                  show_shapes=True, show_layer_names=True, rankdir='TB')


        return model, state_input, action_input