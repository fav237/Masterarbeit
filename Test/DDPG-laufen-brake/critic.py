

import keras.backend as keras_backend
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.layers import Dense, Input, add, concatenate
from keras.models import Model
from keras.optimizers import Adam

image_network = "NETWORKS/"



class CriticNetwork:
    def __init__(self, tf_session, state_size, action_size=3, tau=0.001, lr=0.001, hidden_units = (128, 256, 550)):
        self.tf_session = tf_session
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.lr = lr
        self.hidden_units = hidden_units

        keras_backend.set_session(tf_session)

        self.model, self.state_input, self.action_input = self.generate_model()

        self.target_model, _, _ = self.generate_model()

        self.critic_gradients = tf.gradients(self.model.output, self.action_input)
        self.tf_session.run(tf.global_variables_initializer())

    def get_gradients(self, states, actions):
        return self.tf_session.run(
            self.critic_gradients,
            feed_dict={self.state_input: states, self.action_input: actions},
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
        state_input = Input(shape=[self.state_size])
        action_input = Input(shape=[self.action_size])

        state_h1 = Dense(self.hidden_units[0], activation="relu")(state_input)
        state_h2 = Dense(self.hidden_units[1], activation="linear")(state_h1)

       
        action_h1 = Dense(self.hidden_units[1], activation="linear")(action_input)

        merged = concatenate([state_h2, action_h1])
        merged_h1 = Dense(self.hidden_units[2], activation="relu")(merged)

        output_layer = Dense(1, activation="linear")(merged_h1)
        model = Model(inputs=[state_input, action_input], outputs=output_layer)




        model.compile(loss="mse", optimizer=Adam(lr=self.lr))
        return model, state_input, action_input