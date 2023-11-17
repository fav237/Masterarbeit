


import argparse
import time
import keras.backend as keras_backend
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from carla_env import CarEnv
from actor import ActorNetwork
from critic import CriticNetwork
from keras.callbacks import TensorBoard
import cv2
from keras.models import load_model
from replay_buffer import ReplayBuffer
from ou import OU
OU = OU()       #Ornstein-Uhlenbeck Process



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
    time_buff = []

    AGGREGATE_STATS_EVERY = 10
    actor_weights_file  = "data/data_actor.h5"
    critic_weights_file = "data/data_critic.h5"
    save_weights_path = "data/" 
    N_save_stats = 50

    
    #TRAIN PARAMETERS
    tau = 0.001  # Target Network HyperParameter
    lra = 0.0001  # Learning rate for Actor
    lrc = 0.001  # Learning rate for Critic
    gamma = 0.99  # discount factor
    BATCH_SIZE = 32
    BUFFER_SIZE = 100000
    #state_dim = 31
    state_dim = 13
    action_dim = 2

    np.random.seed(1337)

    max_steps = 100000
    episodes_num = 2000
    hidden_units = (100, 400, 600)
    EXPLORE = 100000.
    epsilon = 1
    tensorboard = ModifiedTensorBoard(log_dir=f"logs/logs_carlaWaypoint-{int(time.time())}")
    step = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf_session = tf.Session(config=config)

    keras_backend.set_session(tf_session)

    actor = ActorNetwork(tf_session=tf_session, state_size= state_dim, action_size=action_dim,
                              tau=tau, lr=lra)
    critic = CriticNetwork(tf_session=tf_session, state_size=state_dim, action_size=action_dim,
                                tau=tau, lr=lrc)
    
    #actor = ActorNetwork(tf_session, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    #critic = CriticNetwork(tf_session, state_dim, action_dim, BATCH_SIZE, TAU, LRC)

    env = CarEnv()

    buffer = ReplayBuffer(BUFFER_SIZE)

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

        print(f'state_ddpg: {state}')
        total_reward = 0.0
        for j in range(max_steps):
            tm1 = time.time()
            loss = 0
            
            epsilon = epsilon - 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])

            action = actor.model.predict(state.reshape(1, state.shape[0]))  # + ou()  # predict and add noise
            #print(f'action: {action}')
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(action[0][0],  0.0 , 0.20, 0.05)
            if action[0][1] >=0 :
                 noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(action[0][1],  0.5 , 0.80, 0.10)
            else:
                noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(action[0][1],  -0.1 , 1.00, 0.05)
            

            a_t[0][0] = action[0][0] + noise_t[0][0]
            a_t[0][1] = action[0][1] + noise_t[0][1]
            


            new_image, new_state, reward, done, info = env.step(a_t[0])

            buffer.add((state, a_t[0], reward, new_state, done))  # add replay buffer
            print(new_state)


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

            state = new_state

            # Print statistics each step
            print("Episode %s - Step %s - Action %s - Reward %s" % (i, step, action[0], reward))

            step += 1
            if done:
                print(env.summary)
                # Imprimir estadisticas cada episode
                print("Episode %s - Step %s - Action %s - Reward %s" % (i, step, action[0], reward))
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
        
            actor.model.save_weights(save_weights_path +  "data_actor.h5", overwrite=True)
            critic.model.save_weights(save_weights_path +  "data_critic.h5", overwrite=True)
        if i % N_save_stats == 0 and train_indicator:
        
            actor.model.save_weights(save_weights_path +  "data_" + str(i) + "_actor.h5", overwrite=True)
            critic.model.save_weights(save_weights_path +  "data_" + str(i) + "_critic.h5", overwrite=True)
        if (i > 10) and (total_reward > np.max(ep_rewards[:-1])):
            actor.model.save_weights(save_weights_path + "data_best_reward_actor.h5", overwrite=True)
            critic.model.save_weights(save_weights_path +  "data_best_reward_critic.h5", overwrite=True)


        time_buff.append((time.time() - tm1))
        tm = time.strftime("%Y-%m-%d %H:%M:%S")
        episode_stat = "%s -th Episode. %s total steps. Total reward: %s. Time %s" % (i, step, total_reward, tm)
        dif_time = "Step time %s" % (time.time() - tm1)
        print(episode_stat)
        
        
        for actor_world in env.actor_list:
            actor_world.destroy()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", type=int, help="train indicator", default=1)
    args = parser.parse_args()
    play(args.train)
