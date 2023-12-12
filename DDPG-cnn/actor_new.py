import keras.backend as keras_backend
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.initializers import normal
from keras.layers import Dense, Input, Conv2D,concatenate, AveragePooling2D, Activation, Flatten, Concatenate, MaxPooling2D
from keras.models import Model

image_network = "NETWORKS/"
class ActorNetwork:
    def __init__(self, tf_session,action_size,hidden_units,width_cnn,n_channel,height_cnn, tau, lr):
        self.tf_session = tf_session
        self.action_size = action_size
        self.tau = tau
        self.lr = lr
        self.hidden_units = hidden_units
        self.width = width_cnn
        self.height = height_cnn
        self.n_channel = n_channel


        keras_backend.set_session(tf_session)

        self.model, self.model_states = self.generate_model()
        model_weights = self.model.trainable_weights

        self.target_model, _ = self.generate_model()

        # Generate tensors to hold the gradients for Policy Gradient update
        self.action_gradients = tf.placeholder(tf.float32, [None, self.action_size])
        self.parameter_gradients = tf.gradients(self.model.output, model_weights, -self.action_gradients)
        self.gradients = zip(self.parameter_gradients, model_weights)

        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(self.gradients)
        self.tf_session.run(tf.global_variables_initializer())

    def train(self, states_im, action_gradients):
        self.tf_session.run(
            self.optimize,
            feed_dict={
                self.model_states: states_im,
                self.action_gradients: action_gradients,
            },
        )

    def train_target_model(self):
        main_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        target_weights = [
            self.tau * main_weight + (1 - self.tau) * target_weight
            for main_weight, target_weight in zip(main_weights, target_weights)
        ]
        self.target_model.set_weights(target_weights)

    def generate_model(self):
        input_layer = Input(shape=(self.height, self.width, self.n_channel))

        cnn_1 = Conv2D(32, (5, 5), padding='same', activation='relu')(input_layer)
        cnn_1ap = MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(cnn_1)

        cnn_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(cnn_1ap)
        cnn_2ap = MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(cnn_2)

        cnn_3 = Conv2D(128, (3, 3), padding='same', activation='relu')(cnn_2ap)
        cnn_3ap = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(cnn_3)

        flat = Flatten()(cnn_3ap)

        h0 = Dense(self.hidden_units[1], activation="relu", kernel_initializer='he_uniform')(flat)
        h1 = Dense(self.hidden_units[2], activation="relu", kernel_initializer='he_uniform')(h0)
        steer = Dense(1, activation="tanh")(h1)
        acc = Dense(1, activation="sigmoid")(h1)
        output_layer = concatenate([steer,acc])
        model = Model(inputs=input_layer, outputs=output_layer)
        tf.keras.utils.plot_model(model,
                                  to_file=image_network + 'actor_model_CNN.png',
                                  show_shapes=True,
                                  show_layer_names=True, rankdir='TB')



        # state_input = Input(shape=(IM_WIDTH_CNN, IM_HEIGHT_CNN, IM_LAYERS))
        #
        # conv0 = Conv2D(32, (3, 3), padding='same', activation='relu', init='uniform')(state_input)
        # av0 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv0)
        #
        # conv1 = Conv2D(64, (3, 3), padding='same', activation='relu', init='uniform')(av0)
        # av1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
        #
        # conv2 = Conv2D(128, (3, 3), padding='same', activation='relu', init='uniform')(av1)
        # av2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        #
        # flat = Flatten()(av2)
        #
        # merged_h1 = Dense(128, activation="relu")(flat)
        #
        # output_layer = Dense(self.action_size, activation="tanh")(merged_h1)
        # model = Model(input=state_input, output=output_layer)
        #
        # tf.keras.utils.plot_model(model,
        #                           to_file='NETWORKS/actor_model_image.png',
        #                           show_shapes=True,
        #                           show_layer_names=True, rankdir='TB')



        return model, input_layer
