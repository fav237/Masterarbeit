


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
        self.state_dim = 13
        self.action_dim = 2
        self.max_steps = 100000
        self.episodes_num = 2000
        self.explore = 100000.
        self.epsilon = 1.0
        self.step = 0


        self.ou= OU() 

        

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.tf_session = tf.Session(config=config)
        keras_backend.set_session(self.tf_session)

        self.actor = ActorNetwork(tf_session=self.tf_session, state_size= self.state_dim, action_size=self.action_dim,
                                tau=self.tau, lr=self.lra)
        self.critic = CriticNetwork(tf_session=self.tf_session, state_size=self.state_dim, action_size=self.action_dim,
                                    tau=self.tau, lr=self.lrc)

    def trainAgent(self):

        self.all_rewards = []
        self.all_dist_raced = []
        self.all_dist_percentage = []
        self.all_avg_speed = []
        
        env = CarEnv()

        buffer = ReplayBuffer(self.BUFFER_SIZE)

        ep_rewards = []


        for i in range(self.episodes_num):
            
            print("Episode : %s Replay buffer %s" % (i, len(buffer)))

            image, state = env.reset()

            print(f'state_ddpg: {state}')
            total_reward = 0.0

            for j in range(self.max_steps):
                tm1 = time.time()
                loss = 0
                
                epsilon = self.epsilon - 1.0 / self.explore

                # initialize numpy matrices to hold action values with OU noise
                action_noise = np.zeros([1,self.action_dim])
                noise = np.zeros([1,self.action_dim])

                # get action from actor
                action = self.actor.model.predict(state.reshape(1, state.shape[0]))  # + ou()  # predict and add noise
                #print(f'action: {action}')

                noise[0][0] =  max(epsilon, 0) * OU.function(action[0][0],  0.0 , 0.20, 0.05)
                if action[0][1] >=0 :
                    noise[0][1] =  max(epsilon, 0) * OU.function(action[0][1],  0.5 , 0.80, 0.10)
                else:
                    noise[0][1] =  max(epsilon, 0) * OU.function(action[0][1],  -0.1 , 1.00, 0.05)
                

                action_noise[0][0] = action[0][0] + noise[0][0]
                action_noise[0][1] = action[0][1] + noise[0][1]
                


                new_image, new_state, reward, done, info = env.step(action_noise[0])

                ### save to experience replay memory for batch selection
                buffer.add((state, action_noise[0], reward, new_state, done))  # add replay buffer
                print(new_state)


                # batch update
                batch = buffer.get_batch(self.BATCH_SIZE)

                # build arrays for models from mini batch
                states = np.asarray([e[0] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_states = np.asarray([e[3] for e in batch])
                dones = np.asarray([e[4] for e in batch])
                target = np.zeros((len(batch), 1))
                #try:
                ### get q values from target critic model
                target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])

                for k in range(len(batch)):
                    if dones[k]:
                        target[k] = rewards[k]
                    else:
                        target[k] = rewards[k] + self.gamma * target_q_values[k]

                ## train networks
                if train_indicator:
                    loss += self.critic.model.train_on_batch([states, actions], target)
                    action_for_grad = self.actor.model.predict(states)
                    gradients = self.critic.get_gradients(states, action_for_grad)
                    self.actor.train(states, gradients)
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

                    
                    break

            # plot
            self.plot()
            
            #Save training data to files
            if i % 10 == 0 :
            
                self.actor.model.save_weights(self.save_weights_path +  "data_actor.h5", overwrite=True)
                self.critic.model.save_weights(self.save_weights_path +  "data_critic.h5", overwrite=True)

            if (i > 10) and (total_reward > np.max(ep_rewards[:-1])):
                self.actor.model.save_weights(self.save_weights_path + "data_best_reward_actor.h5", overwrite=True)
                self.critic.model.save_weights(self.save_weights_path +  "data_best_reward_critic.h5", overwrite=True)


            self.time_buff.append((time.time() - tm1))
            tm = time.strftime("%Y-%m-%d %H:%M:%S")
            episode_stat = "%s -th Episode. %s total steps. Total reward: %s. Time %s" % (i, step, total_reward, tm)
            dif_time = "Step time %s" % (time.time() - tm1)
            print(episode_stat)
            
            
            for actor_world in env.actor_list:
                actor_world.destroy()


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
        plt.plot(self.all_dist_raced)
        plt.xlabel("Episode")
        plt.ylabel("Distanz von Startlinie [m]")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", type=int, help="train indicator", default=1)
    args = parser.parse_args()
    play(args.train)
