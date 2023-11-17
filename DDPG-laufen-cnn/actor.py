

import keras.backend as keras_backend
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.initializers import normal
from keras.layers import Dense, Input, concatenate, AveragePooling2D, Conv2D, Flatten
from keras.models import Model

image_network = "NETWORKS/"

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

        self.model, self.model_image, self.model_input = self.generate_model()
        model_weights = self.model.trainable_weights

        self.target_model, _ , _= self.generate_model()


        # Generate tensors to hold the gradients for Policy Gradient update
        self.action_gradients = tf.placeholder(tf.float32, [None, action_size])
        self.parameter_gradients = tf.gradients(self.model.output, model_weights, -self.action_gradients)
        self.gradients = zip(self.parameter_gradients, model_weights)

        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(self.gradients)
        self.tf_session.run(tf.global_variables_initializer())

    def train(self, image,input, action_gradients):
        self.tf_session.run(
            self.optimize,
            feed_dict={
                self.model_image: image ,
                self.model_input: input,
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
        # network for image processing
        image_input = Input(shape=(self.height, self.width, self.n_channel))

        cnn_1 = Conv2D(32, (5, 5), padding='same', activation='relu')(image_input)
        cnn_1ap = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(cnn_1)

        cnn_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(cnn_1ap)
        cnn_2ap = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(cnn_2)

        cnn_3 = Conv2D(128, (3, 3), padding='same', activation='relu')(cnn_2ap)
        cnn_3ap = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(cnn_3)

        # cnn_4 = Conv2D(256, (3, 3), padding='same', activation='relu')(cnn_3ap)
        # cnn_4ap = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(cnn_4)

        image_output = Flatten()(cnn_3ap)
        image_layer = Dense(self.hidden_units[1], activation="relu")(image_output)

        # network for additional inputs
        state_input = Input(shape=[self.state_size], name="state_input")
        state_layer = Dense(self.hidden_units[2], activation="relu")(state_input)

        # concatenate both networks
        input_layer = [image_input,state_input]
        model_concat = concatenate([image_layer,state_layer])
        
        output_layer = Dense(self.action_size, activation="tanh")(model_concat)
        model = Model(inputs=[image_input,state_input], outputs=output_layer)

        return model, image_input,state_input