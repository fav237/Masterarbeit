


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
import matplotlib.pyplot as plt
      #Ornstein-Uhlenbeck Process
import sys


class DDPGAgent:


    def __init__(self):
        self.time_buff = []
        self.AGGREGATE_STATS_EVERY = 10
        self.actor_weights_file  = "data/data_actor.h5"
        self.critic_weights_file = "data/data_critic.h5"
        self.save_weights_path = "data/" 
        self.N_save_stats = 50

    
        #TRAIN PARAMETERS
        self.tau = 0.001        # Target Network HyperParameter
        self.lra = 0.0001       # Learning rate for Actor
        self.lrc = 0.001        # Learning rate for Critic
        self.gamma = 0.99       # discount factor
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 100000
        #self.state_dim = 31
        self.state_dim = 12
        self.action_dim = 2
        self.max_steps = 3000
        self.episodes_num = 500
        self.explore = 95000.
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.step = 0
        self.hidden_units =[100, 400, 500]
        self.im_width = 320             #480  640  obs_range/lidar_bin 
        self.im_height =  240           #270  640  obs_range/lidar_bin
        self.n_channel = 3

        self.ou= OU() 

        

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.tf_session = tf.Session(config=config)
        keras_backend.set_session(self.tf_session)

        self.actor = ActorNetwork(tf_session=self.tf_session, hidden_units=self.hidden_units, state_size= self.state_dim, action_size=self.action_dim,
                                tau=self.tau, lr=self.lra, width_cnn=self.im_width, height_cnn=self.im_height, n_channel=self.n_channel,)
        self.critic = CriticNetwork(tf_session=self.tf_session, hidden_units=self.hidden_units, state_size=self.state_dim, action_size=self.action_dim,
                                    tau=self.tau, lr=self.lrc, width_cnn=self.im_width, height_cnn=self.im_height, n_channel=self.n_channel,)

    def trainAgent(self):

        self.all_rewards = []
        self.all_dist_raced = []
        self.all_dist_percentage = []
        self.all_avg_speed = []
        buffer = ReplayBuffer(self.BUFFER_SIZE)
        env = CarEnv()

        ep_rewards = []
        epsilon =self.epsilon

        for i in range(self.episodes_num):
            
            print("Episode : %s Replay buffer %s" % (i, len(buffer)))

            if np.mod(i, 15) == 0:
                state = env.reset(relaunch=True)
                
            else:
                state = env.reset()
                
            states_input = np.array(state[-1:]).reshape(1, -1)
            #image_in = np.expand_dims(np.array(state[0]),-1)/255 
            states_image = np.array(state[0:-1])
            
            #print(f'state_ddpg: {state}')
            total_reward = 0.0

            for j in range(self.max_steps):
                tm1 = time.time()
                loss = 0
                
                if epsilon > self.epsilon_min:
                    epsilon = epsilon - 1.0 / self.explore

                # initialize numpy matrices to hold action values with OU noise
                action_noise = np.zeros([1,self.action_dim])
                noise = np.zeros([1,self.action_dim])

                # get action from actor
                action = self.actor.model.predict(states_image,states_input)  # + ou()  # predict and add noise
                #print(f'action: {action}')

                noise[0][0] =  max(epsilon, 0) * self.ou.function(action[0][0],  0.0 , 0.20, 0.05)
                #if action[0][1] >=0 :
                noise[0][1] =  max(epsilon, 0) * self.ou.function(action[0][1],  0.30 , 0.70, 0.10)
                #else:
                #    noise[0][1] =  max(epsilon, 0) * self.ou.function(action[0][1],  -0.1 , 1.00, 0.05)
                

                action_noise[0][0] = action[0][0] + noise[0][0]
                action_noise[0][1] = action[0][1] + noise[0][1]
                


                new_state, reward, done, info = env.step(action_noise[0])

                ### save to experience replay memory for batch selection
                buffer.add((state, action_noise[0], reward, new_state, done))  # add replay buffer
                print(new_state)


                # batch update
                batch = buffer.get_batch(self.BATCH_SIZE)

                # build arrays for models from mini batch
                states_image = np.asarray([e[0][0:-1] for e in batch])
                states_input = np.asarray([e[0][-1:] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_states_image = np.asarray([e[3][0:-1] for e in batch])
                new_states_input = np.asarray([e[3][-1:] for e in batch])
                
                dones = np.asarray([e[4] for e in batch])
                target = np.asarray([e[2] for e in batch])
                #try:
                ### get q values from target critic model
                target_action = self.actor.target_model.predict(new_states_image,new_states_input)
                target_q_values = self.critic.target_model.predict([new_states_image,new_states_input,target_action ])

                for k in range(len(batch)):
                    if dones[k]:
                        target[k] = rewards[k]
                    else:
                        target[k] = rewards[k] + self.gamma * target_q_values[k]

                ## train networks
                
                loss += self.critic.model.train_on_batch([states_image,states_input, actions], target)
                action_for_grad = self.actor.model.predict(states_image,states_input)
                gradients = self.critic.get_gradients(states_image,states_input, action_for_grad)
                self.actor.train(states_image,states_input, gradients)
                ## soft update
                self.actor.train_target_model()
                self.critic.train_target_model()

            
                total_reward += reward

                state = new_state

                # Print statistics each step
                print("Episode %s - Step %s - Action %s - Reward %s" % (i, self.step, action_noise[0], reward))

                self.step += 1
                if done:
                    print(env.summary)
                    # Imprimir estadisticas cada episode
                    print("Episode %s - Step %s - Action %s - Reward %s" % (i, self.step, action_noise[0], reward))
                    
                    self.all_rewards.append(total_reward)
                    self.step = 0
                    
                    break

            # Save data to tensorboard
            ep_rewards.append(total_reward)
            if (i > 0) and ((i % self.AGGREGATE_STATS_EVERY == 0) or (i ==1)):
                average_reward = np.mean(ep_rewards[-self.AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-self.AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-self.AGGREGATE_STATS_EVERY:])
                average_dist = np.mean(env.distance_acum[-self.AGGREGATE_STATS_EVERY:])

            
            
            #Save training data to files
            if i % 10 == 0 :
            
                self.actor.model.save_weights(self.save_weights_path +  "data_actor.h5", overwrite=True)
                self.critic.model.save_weights(self.save_weights_path +  "data_critic.h5", overwrite=True)

            if (i > 10) and (total_reward > np.max(ep_rewards[:-1])):
                self.actor.model.save_weights(self.save_weights_path + "data_best_reward_actor.h5", overwrite=True)
                self.critic.model.save_weights(self.save_weights_path +  "data_best_reward_critic.h5", overwrite=True)


            self.time_buff.append((time.time() - tm1))
            tm = time.strftime("%Y-%m-%d %H:%M:%S")
            episode_stat = "%s -th Episode. %s total steps. Total reward: %s. Time %s" % (i, self.step, total_reward, tm)
            dif_time = "Step time %s" % (time.time() - tm1)
            print(episode_stat)
            
            
            for actor_world in env.actor_list:
                actor_world.destroy()
        # plot
        self.plot()


    def testAgent(self):
        ### 
        self.epsilon = self.epsilon_min

        try:
            # print(critic_weights_file)
            self.actor.target_model.load_weights(self.actor_weights_file)
            self.critic.target_model.load_weights(self.critic_weights_file)
            print("Weights loaded successfully")
        except:
            print("Cannot load weights")
        
        self.trainAgent()


    def plot(self):
        print("Plotting rewards!")
        plt.plot(self.all_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()
        print("Plotting distances!")
        # plt.plot(self.all_dist_raced)
        # plt.xlabel("Episode")
        # plt.ylabel("Distanz von Startlinie [m]")
        # plt.show()

if __name__ == "__main__":

    

    if len(sys.argv) < 1:
        print("syntax: python ddpg.py <train | test> ")
        sys.exit("Syntax error")

    if sys.argv[1] == "train":
        agent = DDPGAgent()
        agent.trainAgent()
    elif sys.argv[1] == "test":
        agent = DDPGAgent()
        agent.testAgent()
    else:
        print("syntax: python ddpg.py <train | test> ")
        sys.exit("Syntax error")
    