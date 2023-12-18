import glob
import os
import sys


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

sys.path.append('../carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner


import random
import carla
from carla import ColorConverter
import time as time
import math
import numpy as np
import cv2

import gym
from gym import spaces
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras.backend as keras_backend
from keras.models import load_model


red = carla.Color(255, 0, 0)
green = carla.Color(0, 255, 0)
blue = carla.Color(47, 210, 231)
cyan = carla.Color(0, 255, 255)
yellow = carla.Color(255, 255, 0)
orange = carla.Color(255, 162, 0)
white = carla.Color(255, 255, 255)


class CarEnv(gym.Env):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (30, 30)      # this line will be used to show current speed
    org1 = (30, 50)     # this line will be used to show number of episode
    org2 = (30, 70)      # this line will be used to show number of Target reach
    episode = 0
    fontScale = 0.5
    thickness = 1
    im_width = 640
    im_height = 640
    front_camera = None
    bev_camera = None
    angle_rw = 0
    trackpos_rw = 0
    cmd_vel = 0
    summary = {'Target': 0, 'Steps': 0}
    distance_acum = []
    speed = 0
    line_time = 5
    line_widht = 0.1
    cam_x = 1.4            #-5
    cam_z = 1.3             #3
    cam_pitch = -40
    SHOW_CAM = True
    state_dim = 16
    n_channel = 3
    train_mode = 'right'
    train =['right', 'straight', 'random', 'Left']
    #
    config2 = tf.ConfigProto()
    config2.gpu_options.allow_growth = True
    tf_session2 = tf.Session(config=config2)

    keras_backend.set_session(tf_session2)

    def __init__(self):
        # using continous actions
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        # using image as input normlised to 0 1
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,0 ))

        self.client = carla.Client("localhost", 2000)
        #self.client.set_timeout(4.0)
        self.world = self.client.get_world()
        
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = 0.2
        #elf.settings.no_rendering_mode = True
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)
    
        self.blueprint_library = self.world.get_blueprint_library()
        self.a2 = self.blueprint_library.filter("a2")[0]
        self.target = 0
        self.data = {}
        self.pos_a = 0
        self.pos_b = 0


    def reset(self, relaunch=False):
        
        if relaunch == True:
            print('kill carla')
            time.sleep(1)
            print('restart carla')
            time.sleep(0.5)
        self.tm = time.time()
        self.nospeed_times = 0
        self.toofast_times = 0
        global acum
        global x_prev
        global y_prev
        acum = 0
        self.direction = 1
        self.episode +=1
        self.actor_list = []
        self.collision_hist = []
        self.laneInv_hist = []
        self.waypoints_route = []
        self.dif_angle_routes = 0
        self.total_dist = 1
        self.map = self.world.get_map()
        self.route_planner = GlobalRoutePlanner(self.map, 1)
        #############################
        self.spawn_points = self.map.get_spawn_points()
        if self.train_mode == self.train[0]:
            self.pos_a = carla.Transform(carla.Location(x=392.5, y=290.5, z=0.0),carla.Rotation(pitch=0.0, yaw=90., roll=0.0))
            self.pos_b = carla.Transform(carla.Location(x=350.50, y=326.50, z=0.0),carla.Rotation(pitch=0.0, yaw=180., roll=0.0))
            a = self.pos_a.location
            b = self.pos_b.location
            self.current_route = self.route_planner.trace_route(a,b)
            self.total_dist = self.total_distance(self.current_route)
            self.transform = self.pos_a
            
        if self.train_mode == self.train[1]:
            self.pos_a = carla.Transform(carla.Location(x=392.25, y=20.5, z=0.0),carla.Rotation(pitch=0.0, yaw=89.9, roll=0.0))
            self.pos_b = carla.Location(x=392.25, y=300.50, z=0.0)
            a = self.pos_a.location
            self.current_route = self.route_planner.trace_route(a,self.pos_b)
            self.total_dist = self.total_distance(self.current_route)
            self.transform = self.pos_a
            
        if self.train_mode == self.train[3]:
            self.pos_a = carla.Transform(carla.Location(x=40.75, y=302.5, z=0.6),carla.Rotation(pitch=0.0, yaw=-180.0, roll=0.0))
            self.pos_b = carla.Transform(carla.Location(x=-3.680, y=280.50, z=0.6),carla.Rotation(pitch=0.0, yaw=-89.9, roll=0.0))
            a = self.pos_a.location
            b = self.pos_b.location
            self.current_route = self.route_planner.trace_route(a,b)
            self.total_dist = self.total_distance(self.current_route)
            self.transform = self.pos_a
            
        if self.train_mode == self.train[2]:
            while self.total_dist < 500 :
                self.pos_a = random.choice(self.spawn_points)
                self.pos_b = random.choice(self.spawn_points)
                a = self.pos_a.location
                b = self.pos_b.location
                self.current_route = self.route_planner.trace_route(a, b)
                self.total_dist = self.total_distance(self.current_route)
                self.transform = self.pos_a

        print(self.transform)

        for i in range(len(self.current_route)):
            w1 = self.current_route[i][0]
            self.waypoints_route.append([w1.transform.location.x, w1.transform.location.y,
                w1.transform.rotation.yaw])
        self.target = w1.transform.location

        self.draw_path( self.current_route, tl=self.line_time)

        self.vehicle = None
        while self.vehicle is None:
            try:
                self.vehicle = self.world.spawn_actor(self.a2, self.transform)
                time.sleep(0.01)
            except:
                pass
        
        # add the car tothe list of actor
        self.actor_list.append(self.vehicle)

        # get the blueprint for camera
        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')

        # change the dimensions of the image
        self.camera_bp.set_attribute('image_size_x', f'{self.im_width}')
        self.camera_bp.set_attribute('image_size_y', f'{self.im_height}')
        self.camera_bp.set_attribute('fov', '110')
        self.camera_bp.set_attribute('sensor_tick', '0.02')

        # Adjust sensor relative to vehicle
        sensor_transform = carla.Transform(carla.Location(z=self.cam_z,x=self.cam_x))

        # spawn the sensor and attach to vehicle.
        self.camera = self.world.spawn_actor(self.camera_bp,sensor_transform,attach_to=self.vehicle)

        # add sensor to list of actors
        self.actor_list.append(self.camera)

        # display what the camera see
        self.camera.listen(lambda data: self.process_img(data))

        # control the the car
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))

        # And here's another workaround - it takes a bit over 3 seconds for Carla simulator to start accepting any
        # control commands. Above time adds to this one, but by those 2 tricks we start driving right when we start an episode
        # But that adds about 5 seconds of a pause time between episodes.
        time.sleep(5)

        # get the blueprint for the collision sensor
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        #https://carla.readthedocs.io/en/latest/ts_traffic_simulation_overview/
        sensor_transform = carla.Transform(carla.Location(z=self.cam_z,x=self.cam_x))
        # spawn the sensor and attach to vehicle.
        self.collision = self.world.spawn_actor(collision_bp, sensor_transform, attach_to=self.vehicle)

        # add sensor to list of actors
        self.actor_list.append(self.collision)

        # retrievve data
        self.collision.listen(lambda event: self.collision_data(event))

        # # get the blueprint for the lane invasion sensor
        # lane_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        # self.lane = self.world.spawn_actor(lane_bp, sensor_transform, attach_to=self.vehicle)
        # self.actor_list.append(self.lane)
        # self.lane.listen(lambda event1: self.lane_data(event1))
        
        # to be sure tha when the car falls into the simulator
        # it is not record as a error
        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        location_reset = self.vehicle.get_transform()
        x_prev = location_reset.location.x
        y_prev = location_reset.location.y

        self.get_obs(self.front_camera)

        return  self.front_camera, self.state()


    def total_distance(self, current_plan):
        sum = 0
        for i in range(len(current_plan) - 1):
            sum = sum + self.distance_wp(current_plan[i + 1][0], current_plan[i][0])
        return sum


    def distance_wp(self, target, current):
        dx = target.transform.location.x - current.transform.location.x
        dy = target.transform.location.y - current.transform.location.y
        return math.sqrt(dx * dx + dy * dy)


    def distance_target(self, target, current):
        dx = target.x - current.x
        dy = target.y - current.y
        return math.sqrt(dx * dx + dy * dy)


    def draw_path(self,  current_plan, tl):
        for i in range(len(current_plan) - 1):
            w1 = current_plan[i][0]
            w2 = current_plan[i + 1][0]
            self.world.debug.draw_line(w1.transform.location, w2.transform.location, thickness=self.line_widht,
                                       color=green, life_time=tl)


    # def lane_data(self, event1):  
    #     self.laneInv_hist.append(1)


    def collision_data(self, event):
        self.collision_hist.append(1)


    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        #i3 = i2[:, :, :3]
        
        self.front_camera = i2
        

    def step(self, action):
        global x_prev
        global y_prev
        global acum
        global acum_prev
        global d_i_prev
        done = False
        reward = 0
        v_min = 20
        v_target = 30

        # Action is applied for steering and throttle 
        steer = float(np.clip(action[0], -0.5, 0.5))
        throttle = float(np.clip(action[1], 0, 1))
        #brake = float(np.abs(np.clip(action[1], -0.3, 0)))
        
        print(f'acc: {throttle}, steer: {steer}')
        self.vehicle.apply_control(carla.VehicleControl(throttle =throttle , steer=steer))
        # print(f'acc: {throttle}, steer: {steer}, brake: {brake}')
        # self.vehicle.apply_control(carla.VehicleControl(throttle =throttle , steer=steer, brake=brake))
        
        location_rv = self.vehicle.get_transform()

        d_i = math.sqrt((x_prev - location_rv.location.x) ** 2 + (y_prev - location_rv.location.y) ** 2)

        acum += d_i
        x_prev = location_rv.location.x
        y_prev = location_rv.location.y
        
        x_prev = location_rv.location.x
        y_prev = location_rv.location.y
        d_i_prev = d_i

        v = self.vehicle.get_velocity()
        speed = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        # print(f'speed: {speed}')
        
        location = self.vehicle.get_location()
        d2target = self.distance_target(self.target, location)

        image = cv2.resize(self.front_camera, (160, 60))
        self.get_obs(image)
        # time.sleep(0.01)

        # reward for steering
        angle =self.data['delta_yaw_t']
        trackpos = self.data['lateral_dist_t']
        #print(f'angle:{angle}')
        reward += np.abs(v.x)*((np.cos(angle)) -np.abs(np.sin(angle))*4 - np.abs(trackpos)*4)

        # reward for acceleration
        if speed > v_min and speed <= (v_target+v_min):
            reward -= ((speed - v_target)/2)* (speed/v_target)
        elif speed <1:
            reward -= 10
            self.nospeed_times +=1
            if self.nospeed_times > 100:
                done = True 
                print('no speed for long time...' )
        elif speed > (v_target+v_min):
            self.toofast_times +=1
            reward -= 10
            if self.toofast_times > 100:
                done = True
                print('speed too fast...')
        else:
            reward +=2

        # punish for collision
        if len(self.collision_hist) != 0:  
            reward -= 20
            done = True
            print('Collision occurred...')

        # punish for lane invasion
        if abs(self.data['lateral_dist_t']) > 1.2:
            reward -= 20
            done = True
            print('Lane Invasion occurred...')
            self.summary['Steps'] += 1

        # reward for reaching the targe
        if self.distance_target(self.target, location) < 2:
            done = True
            reward += 10
            print('The target has been reached')
            self.summary['Steps'] += 1
            self.summary['Target'] += 1

        # when timer runs out
        if self.episode_start + 30 < time.time():
            print('End of timer, distance to target: ', d2target)
            done = True
            self.summary['Steps'] += 1
            reward -= 10

        img = cv2.putText(self.front_camera, 'Speed: '+str(int(speed))+' kmh', self.org, 
                           self.font, self.fontScale, (255,255,255), self.thickness, cv2.LINE_AA)
        img = cv2.putText(img, 'Episode: '+str(int(self.episode)), self.org1, 
                           self.font, self.fontScale, (255,255,255), self.thickness, cv2.LINE_AA)
        img = cv2.putText(img, 'Goal reached: '+str(int(self.summary['Target'])), self.org2, 
                           self.font, self.fontScale, (255,255,255), self.thickness, cv2.LINE_AA)

        if self.SHOW_CAM :
            cv2.namedWindow('Real', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Real', img)
            cv2.waitKey(1)

        if done == True:
            self.distance_acum.append(acum)

        return image, self.state(), reward, done, self.data


    def get_obs(self, image):
        self.distances = [ 1., 5., 10.]   
        angles =[]
        # waypoint information
        # actual position of the car
        actual_position = self.vehicle.get_transform()
        pos_x = actual_position.location.x
        pos_y = actual_position.location.y
        velocity = self.vehicle.get_velocity()
        acceleration = self.vehicle.get_acceleration()
        velocity_t = np.array([velocity.x, velocity.y])
        acceleration_t = np.array([acceleration.x, acceleration.y])
        self.speed = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2)
        
        # actual waypoint
        actual_waypoint = self.map.get_waypoint(location= self.vehicle.get_location())
        current_waypoint = np.array((actual_waypoint.transform.location.x, actual_waypoint.transform.location.y,
                                        actual_waypoint.transform.rotation.yaw))
        
        pos_vec = np.array([pos_x - current_waypoint[0], pos_y - current_waypoint[1]])
        lv = np.linalg.norm(np.array(pos_vec))
       
        # calculate the deta_yaw between the car and current waypoint
        waypoint_yaw = current_waypoint[2] % 360
        vehicle_yaw = self.vehicle.get_transform().rotation.yaw % 360
        delta_yaw = waypoint_yaw - vehicle_yaw
        if 180 <= delta_yaw and delta_yaw <= 300:
            delta_yaw -= 360
        elif -300 <= delta_yaw and delta_yaw <= -180:
            delta_yaw += 360

        route_heading = np.array(
            [np.cos(waypoint_yaw/180 * np.pi),
             np.sin(waypoint_yaw/180 * np.pi)])

        lateral_distance = lv * np.sign(pos_vec[0] * route_heading[1] -
                                         pos_vec[1] * route_heading[0])
        
        # get next waypoint in distance
        for d in self.distances:  
            waypoint_heading = actual_waypoint.next(d)[0].transform.rotation.yaw
            
            delta_heading = (vehicle_yaw) -(waypoint_heading % 360)
            if 180 <= delta_heading and delta_heading <= 360:
                delta_heading -= 360
            elif -360 <= delta_heading and delta_heading <= -180:
                delta_heading += 360
            angles.append(delta_heading)
        future_angles = np.array(angles, dtype=np.float32)

        self.draw_path( self.current_route, tl=self.line_time+1)

        self.data['velocity_t'] = velocity_t
        self.data['acc_t'] = acceleration_t
        self.data['delta_yaw_t'] = delta_yaw
        self.data['dyaw_dt_t'] = dyaw_dt
        self.data['lateral_dist_t'] = lateral_distance
        self.data['speed'] = self.speed
        self.data['angles_t'] = future_angles
        

    def state(self):
        velocity_t = (self.data['velocity_t'])/12.0
        accel_t = (self.data['acc_t'])/4.0
        speed =  ((np.array(self.data['speed']).reshape(
            (1, )))-40.0) /40.0

        delta_yaw_t = np.array(self.data['delta_yaw_t']).reshape(
            (1, )) /45.0

        lateral_dist_t = (self.data['lateral_dist_t'].reshape(
            (1, )))/1.2 

        future_angles = self.data['angles_t'] /45.0
         
        info_vec = np.concatenate([  delta_yaw_t, future_angles,
                                     lateral_dist_t, velocity_t
        ],
                                  axis=0)
        info_vec = info_vec.squeeze()

        return np.float32(info_vec)
