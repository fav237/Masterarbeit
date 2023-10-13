


import argparse
import time
import keras.backend as keras_backend
import numpy as np
import tensorflow as tf
from carla_env import Carla_Env
from actor import ActorNetwork
from critic import CriticNetwork
from keras.callbacks import TensorBoard
import cv2
from keras.models import load_model
from replay_buffer import ReplayBuffer

time_buff = []

AGGREGATE_STATS_EVERY = 10
actor_weights_file  = "data/data_actor.h5"
critic_weights_file = "data/data_critic.h5"
save_weights_path = "data/data_" + "/"
N_save_stats = 50

#TRAIN PARAMETERS
tau = 0.001  # Target Network HyperParameter
lra = 0.0001  # Learning rate for Actor
lrc = 0.001  # Learning rate for Critic
state_dim = 30
max_steps = 100000
episodes_num = 8000
buffer_size = 100000
batch_size = 32
gamma = 0.99  # discount factor
hidden_units = (100, 400, 600)


class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)



def play(train_indicator):

    # ou_sigma = 0.3
    tensorboard = ModifiedTensorBoard(log_dir=f"logs/logs_carlaWaypoint-{int(time.time())}")
    step = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf_session = tf.Session(config=config)

    keras_backend.set_session(tf_session)

    actor = ActorNetwork(tf_session=tf_session, state_size= state_dim, action_size=9,
                             tau=tau, lr=lra)
    critic = CriticNetwork(tf_session=tf_session, state_size=state_dim, action_size=9,
                               tau=tau, lr=lrc)
    

    env =Carla_Env()

    buffer = ReplayBuffer(buffer_size)

    try:
        # print(critic_weights_file)
        actor.model.load_weights(actor_weights_file)
        critic.model.load_weights(critic_weights_file)
        actor.target_model.load_weights(actor_weights_file)
        critic.target_model.load_weights(critic_weights_file)
        print("Weights loaded successfully")
    except:
        print("Cannot load weights")

    ep_rewards = []


    for i in range(episodes_num):
        tensorboard.step = i
        print("Episode : %s Replay buffer %s" % (i, len(buffer)))

        _, state = env.reset()


        total_reward = 0.0
        for j in range(max_steps):
            tm1 = time.time()

            loss = 0
        
            action_predicted = actor.model.predict(state.reshape(1, state.shape[0]))  # + ou()  # predict and add noise
            [new_image, new_state], reward, done, info = env.step(action_predicted[0])
            buffer.add((state, action_predicted[0], reward, new_state, done))  # add replay buffer
            print(new_state)


            # batch update
            batch = buffer.get_batch(batch_size)

            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.zeros((len(batch), 1))
            #try:
            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + gamma * target_q_values[k]

            if train_indicator:
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.get_gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.train_target_model()
                critic.train_target_model()

        
            total_reward += reward

            state = new_state

            # Print statistics each step
            print("Episode %s - Step %s - Action %s - Reward %s" % (i, step, action_predicted[0], reward))

            step += 1
            if done:
                print(env.summary)
                # Imprimir estadisticas cada episode
                print("Episode %s - Step %s - Action %s - Reward %s" % (i, step, action_predicted[0], reward))
                break

        # Save data to tensorboard
        ep_rewards.append(total_reward)
        if (i > 0) and ((i % AGGREGATE_STATS_EVERY == 0) or (i ==1)):
            average_reward = np.mean(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            average_dist = np.mean(env.distance_acum[-AGGREGATE_STATS_EVERY:])
            tensorboard.update_stats(average_reward=average_reward, min_reward=min_reward, max_reward=max_reward,
                                     distance=average_dist, loss=loss)
            
        
        #Save training data to files
        if i % 3 == 0 and train_indicator:
        
            actor.model.save_weights(save_weights_path +  "_actor.h5", overwrite=True)
            critic.model.save_weights(save_weights_path +  "_critic.h5", overwrite=True)
        if i % N_save_stats == 0 and train_indicator:
        
            actor.model.save_weights(save_weights_path +  "_" + str(i) + "_actor.h5", overwrite=True)
            critic.model.save_weights(save_weights_path +  "_" + str(i) + "_critic.h5", overwrite=True)
        if (i > 10) and (total_reward > np.max(ep_rewards[:-1])):
            actor.model.save_weights(save_weights_path + "_best_reward_actor.h5", overwrite=True)
            critic.model.save_weights(save_weights_path +  "_best_reward_critic.h5", overwrite=True)


        time_buff.append((time.time() - tm1))
        tm = time.strftime("%Y-%m-%d %H:%M:%S")
        episode_stat = "%s -th Episode. %s total steps. Total reward: %s. Time %s" % (i, step, total_reward, tm)
        dif_time = "Step time %s" % (time.time() - tm1)
        print(episode_stat)
        
        
        for actor_world in env.actor_list:
            actor_world.destroy()


    actor.model.save_weights(save_weights_path + "_actor.h5", overwrite=True)
    critic.model.save_weights(save_weights_path +  "_critic.h5", overwrite=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", type=int, help="train indicator", default=1)
    args = parser.parse_args()
    play(args.train)
