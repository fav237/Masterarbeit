import argparse
import time
import keras.backend as keras_backend
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from carla_env import CarEnv
from actor_new import ActorNetwork
from critic_new import CriticNetwork

from keras.callbacks import TensorBoard
# from util.noise import OrnsteinUhlenbeckActionNoise

import cv2
from replay_buffer import ReplayBuffer
from ou import OU
from keras.models import load_model

time_buff = []

AGGREGATE_STATS_EVERY = 10

class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self,log_dir):
        self.step = 1
        self.log_dir = log_dir
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
    # More or less the same writer as in Keras' Tensorboard callback
    # Physically writes to the log files
    def _write_logs(self, logs, index):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, index)
        self.writer.flush()



def play(train_indicator):

    # ou_sigma = 0.3
    tensorboard = ModifiedTensorBoard(log_dir=f"logs/logs_semantic_CNN/straight-{int(time.time())}")
    step = 0
    time_buff = []
    AGGREGATE_STATS_EVERY = 10
    actor_weights_file  = "data/data_actor_semantic_cnn.h5"
    critic_weights_file = "data/data_critic_semantic_cnn.h5"
    save_weights_path = "data/" 
    N_save_stats = 50


    #TRAIN PARAMETERS
    tau = 0.001        # Target Network HyperParameter
    lra = 0.0001       # Learning rate for Actor
    lrc = 0.001        # Learning rate for Critic
    gamma = 0.99       # discount factor
    BATCH_SIZE = 32
    BUFFER_SIZE = 100000
    #state_dim = 31
    state_dim = 12
    action_dim = 2
    max_steps = 3000
    episodes_num = 20000
    explore = 95000.
    epsilon = 1.0
    epsilon_min = 0.1
    step = 0
    hidden_units =[128, 256, 512]
    im_width = 160             #480  640  obs_range/lidar_bin 
    im_height =  60          #270  640  obs_range/lidar_bin
    n_channel = 3

    ou= OU() 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf_session = tf.Session(config=config)

    keras_backend.set_session(tf_session)


    
    actor = ActorNetwork(tf_session=tf_session,action_size=action_dim, hidden_units=hidden_units, 
                             width_cnn=im_width, height_cnn=im_height, n_channel=n_channel,tau=tau, lr=lra)
    critic = CriticNetwork(tf_session=tf_session, hidden_units=hidden_units, action_size=action_dim,
                             width_cnn=im_width, height_cnn=im_height, n_channel=n_channel,tau=tau, lr=lrc)

    buffer = ReplayBuffer(BUFFER_SIZE)

    env = CarEnv()

    # noise function for exploration
    # ou = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim), sigma=ou_sigma * np.ones(action_dim))

    # Torcs environment - throttle and gear change controlled by client


    try:
        
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
        
        current_state,_ = env.reset()

        total_reward = 0.0
        for j in range(max_steps):
            tm1 = time.time()

            if epsilon > epsilon_min:
                epsilon = epsilon - 1.0 / explore
            loss = 0
              # + ou()  # predict and add noise
            # initialize numpy matrices to hold action values with OU noise
            action_noise = np.zeros([1,action_dim])
            noise = np.zeros([1,action_dim])
            
            action = actor.model.predict(np.array(current_state).reshape(-1, *current_state.shape))  # + ou()  # predict and add noise

            noise[0][0] =  max(epsilon, 0) * ou.function(action[0][0],  0.0 , 0.20, 0.05)
            #if action[0][1] >=0 :
            noise[0][1] =  max(epsilon, 0) * ou.function(action[0][1],  0.30 , 0.70, 0.10)
            #else:
            #    noise[0][1] =  max(epsilon, 0) * ou.function(action[0][1],  -0.1 , 1.00, 0.05)
            
            action_noise[0][0] = action[0][0] + noise[0][0]
            action_noise[0][1] = action[0][1] + noise[0][1]
            
            new_current_state,st, reward, done, info = env.step(action_noise[0])
                
            buffer.add((current_state, action_noise[0], reward, new_current_state, done))  # add replay buffer


            # batch update
            batch = buffer.get_batch(BATCH_SIZE)

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

            current_state = new_current_state

            # Imprimir estadísticas cada step
            print("Episode %s - Step %s - Action %s - Reward %s" % (i, step,action_noise[0], reward))

            step += 1
            if done:
                
                # Imprimir estadisticas cada episode
                print("Episode %s - Step %s - Action %s - Reward %s" % (i, step, action_noise[0], reward))
                break


        #Guardar datos en tensorboard
        ep_rewards.append(total_reward)
        if (i > 0) and ((i % AGGREGATE_STATS_EVERY == 0) or (i ==1)):
            average_reward = np.mean(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            average_dist = np.mean(env.distance_travel[-AGGREGATE_STATS_EVERY:])
            tensorboard.update_stats(average_reward=average_reward, min_reward=min_reward, max_reward=max_reward,
                                     distance=average_dist, loss=loss)

        #Guardar datos del entrenamiento en ficheros
        if i % 10 == 0 and (total_reward > 10) :
            
            actor.model.save_weights(save_weights_path +  "data_actor.h5", overwrite=True)
            critic.model.save_weights(save_weights_path +  "data_critic.h5", overwrite=True)

        if (i > 10) and (total_reward > 100):
            actor.model.save_weights(save_weights_path + "data_best_reward_actor.h5", overwrite=True)
            critic.model.save_weights(save_weights_path +  "data_best_reward_critic.h5", overwrite=True)


        time_buff.append((time.time() - tm1))
        #print(np.mean(np.array(time_buff)))
        tm = time.strftime("%Y-%m-%d %H:%M:%S")
        episode_stat = "%s -th Episode. %s total steps. Total reward: %s. Time %s" % (i, step, total_reward, tm)
        dif_time = "Step time %s" % (time.time() - tm1)
        print(episode_stat)

        # Guardar estadísticas de cada episode
        # with open(train_stat_file, "a") as outfile:
        #     outfile.write(episode_stat + "\n")
        for actor_world in env.actor_list:
            actor_world.destroy()

    

if __name__ == "__main__":
    
    play(1)