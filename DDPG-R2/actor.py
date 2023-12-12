

import keras.backend as keras_backend
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.initializers import normal
from keras.layers import Dense, Input, concatenate
from keras.models import Model


class ActorNetwork:
    def __init__(self, tf_session, state_size, action_size, hidden_units, tau=0.001, lr=0.0001 ):
        self.tf_session = tf_session
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.lr = lr
        self.hidden_units = hidden_units

        keras_backend.set_session(tf_session)

        self.model, self.model_states = self.generate_model()
        model_weights = self.model.trainable_weights
        self.target_model, _ = self.generate_model()

        # Generate tensors to hold the gradients for Policy Gradient update
        self.action_gradients = tf.placeholder(tf.float32, [None, action_size])
        self.parameter_gradients = tf.gradients(self.model.output, model_weights, -self.action_gradients)
        self.gradients = zip(self.parameter_gradients, model_weights)
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(self.gradients)
        self.tf_session.run(tf.global_variables_initializer())
        

    def train(self, states, action_gradients):
        self.tf_session.run(
            self.optimize,
            feed_dict={
                self.model_states: states,
                self.action_gradients: action_gradients,
            },
        )

    
    def train_target_model(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        actor_target_weights = [
            self.tau * actor_weights + (1 - self.tau) * actor_target_weights
            for actor_weights, actor_target_weights in zip(actor_weights, actor_target_weights)
        ]
        self.target_model.set_weights(actor_target_weights)

    
    def generate_model(self):
        input_layer = Input(shape=[self.state_size])
        h0 = Dense(self.hidden_units[0], activation="relu")(input_layer)
        h1 = Dense(self.hidden_units[1], activation="relu")(h0)
        steering = Dense(1, activation="tanh")(h1)
        acceleration = Dense(1, activation="sigmoid")(h1)
        output_layer = concatenate([steering, acceleration])
        model = Model(inputs=input_layer, outputs=output_layer)

        return model, input_layer
