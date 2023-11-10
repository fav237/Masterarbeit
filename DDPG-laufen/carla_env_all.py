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
#import sympy as sym
import matplotlib.pyplot as plt

import gym
from gym import spaces
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras.backend as keras_backend
from keras.models import load_model

from misc import _vec_decompose


red = carla.Color(255, 0, 0)
green = carla.Color(0, 255, 0)
blue = carla.Color(47, 210, 231)
cyan = carla.Color(0, 255, 255)
yellow = carla.Color(255, 255, 0)
orange = carla.Color(255, 162, 0)
white = carla.Color(255, 255, 255)


class CarEnv:
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (30, 30) # this line will be used to show current speed
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
    line_time = 5
    line_widht = 0.1
    cam_x = -5
    cam_z = 3
    cam_pitch = -40
    SHOW_CAM = True
    state_dim = 16
    #state_dim = 31
    n_channel = 3
    train_mode = 'straight'
    train =['straight', 'random']
    #
    config2 = tf.ConfigProto()
    config2.gpu_options.allow_growth = True
    tf_session2 = tf.Session(config=config2)

    keras_backend.set_session(tf_session2)

    def __init__(self):
        # using continous actions
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        # using image as input normlised to 0 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(160, 60,self.n_channel ))

        self.client = carla.Client("localhost", 2000)
        #self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.a2 = self.blueprint_library.filter("a2")[0]
        self.prev_d2goal = 10000
        self.target = 0
        self.numero_tramo = 0
        self.error_lateral = []
        self.position_array = []
        self.prev_next = 0
        self.waypoints_txt = []
        self.data = {}
        self.pos_a = 0
        self.pos_b = 0

        src = np.float32([[0, self.im_height], [1200, self.im_height], [0, 0], [self.im_width, 0]])
        dst = np.float32([[569, self.im_height], [711, self.im_height], [0, 0], [self.im_width, 0]])
        self.M = cv2.getPerspectiveTransform(src, dst)



    def reset(self):
        self.tm = time.time()
        self.dif_tm = 0
        global acum
        global x_prev
        global y_prev
        acum = 0
        self.actor_list = []
        self.collision_hist = []
        self.laneInv_hist = []
        self.coeficientes = np.zeros((51-1, 8))
        self.pos_array_wp = 0
        self.waypoints_current_route = []
        self.dif_angle_routes = 0
        #############################NUEVO
        self.total_dist = 1
        self.map = self.world.get_map()
        self.route_planner = GlobalRoutePlanner(self.map, 1)
        #############################
        
        self.spawn_points = self.map.get_spawn_points()
        self.waypoints_current_route = []

        if self.train_mode == self.train[0]:
            self.pos_a = carla.Transform(carla.Location(x=-90, y=24.5, z=0.6),carla.Rotation(pitch=0.0, yaw=0.16, roll=0.0))
            self.pos_b = carla.Location(x=80, y=24.50, z=0.6)
            a = self.pos_a.location
            self.current_route = self.route_planner.trace_route(a,self.pos_b)
            self.total_dist = self.total_distance(self.current_route)
            self.transform = self.pos_a

        # while self.total_dist > 200 or self.total_dist < 180:
        if self.train_mode == self.train[1]:
            while self.total_dist < 2000 and self.dif_angle_routes == 0:
                self.pos_a = random.choice(self.spawn_points)
                self.pos_b = random.choice(self.spawn_points)
                angles_dif = abs(abs(self.pos_a.rotation.yaw) - abs(self.pos_b.rotation.yaw))
                if angles_dif > 80 and angles_dif < 100:
                    self.dif_angle_routes = 1

                a = self.pos_a.location
                b = self.pos_b.location
                self.current_route = self.route_planner.trace_route(a, b)
                self.total_dist = self.total_distance(self.current_route)


                self.transform = self.pos_a

        print(self.transform)

        for i in range(len(self.current_route)):
            w1 = self.current_route[i][0]
            self.waypoints_current_route.append([w1.transform.location.x, w1.transform.location.y, w1.transform.location.z,
                 w1.transform.rotation.pitch, w1.transform.rotation.yaw, w1.transform.rotation.roll])
        self.waypoints_current_route.append([0, 0, 0, 0, 0, 0])
        self.target = w1.transform.location
        ##################

        self.draw_path( self.current_route, tl=self.line_time)
        #self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.a2, self.transform)

        
        # add the car tothe list of actor
        self.actor_list.append(self.vehicle)

        # get the blueprint for camera
        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')

        # change the dimensions of the image
        self.camera_bp.set_attribute('image_size_x', f'{self.im_width}')
        self.camera_bp.set_attribute('image_size_y', f'{self.im_height}')
        self.camera_bp.set_attribute('fov', '110')

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

        # sleep for 5 seconds
        time.sleep(5)

        # get the blueprint for the collision sensor
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        #https://carla.readthedocs.io/en/latest/ts_traffic_simulation_overview/
        # spawn the sensor and attach to vehicle.
        self.collision = self.world.spawn_actor(collision_bp, sensor_transform, attach_to=self.vehicle)

        # add sensor to list of actors
        self.actor_list.append(self.collision)

        # retrievve data
        self.collision.listen(lambda event: self.collision_data(event))

        # get the blueprint for the lane invasion sensor
        lane_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        self.lane = self.world.spawn_actor(lane_bp, sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.lane)
        self.lane.listen(lambda event1: self.lane_data(event1))

        # get the blueprint for gnss sensor
        gnss_bp = self.blueprint_library.find('sensor.other.gnss')
        self.gnss = self.world.spawn_actor(gnss_bp, sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.gnss)
        self.gnss.listen(lambda event3: self.gnss_data(event3))

        
        # to be sure tha when the car falls into the simulator
        # it is not record as a error
        while self.front_camera is None:
            time.sleep(0.01)
            # reset action
        #self.last_action = np.array([0.0, 0.0])
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        location_reset = self.vehicle.get_transform()
        x_prev = location_reset.location.x
        y_prev = location_reset.location.y

        image = cv2.resize(self.front_camera, (160, 60))

        self.observation(image)

        return  image, self.state()


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
    

    def draw_waypoint_info(self, world, w, lt=(700+ 5.0)):
        w_loc = w.transform.location
        world.debug.draw_point(w_loc, 0.5, red, lt)


    def lane_data(self, event1):  
        self.laneInv_hist.append(1)

    def gnss_data(self, event3):
        global latitude
        global longitude

        latitude = event3.latitude
        longitude = event3.longitude


    def collision_data(self, event):
        self.collision_hist.append(1)


    def process_img(self, image):
        
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        
        self.front_camera = i3
        

    def step(self, action):
        global x_prev
        global y_prev
        global acum
        global acum_prev
        global d_i_prev
        #self.steering_lock == False
        #

        #Action is applied like steerin while throttle is cte
        throttle = float(np.clip(action[1]/3, 0, 0.7))
        brake = float(np.abs(np.clip(action[1]/8, -1, 0)))
        steer = float(np.clip(action[0], -0.5, 0.5))
        #brake = np.abs(action[2])
        print(f'acc: {throttle}, steer: {steer}, brake: {brake}')
        self.vehicle.apply_control(carla.VehicleControl(throttle =throttle , steer=steer, brake=brake))
        #print(f'acc: {throttle}, steer: {steer}')
        
        #print("acumulado: ", acum)
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
        print(f'speed: {speed}')
        #reward for acceleration
        if speed < 5:
            done = False
            reward = -200
        elif speed < 15:
            done = False
            reward = -100
        elif speed > 65:
            done = True
            reward = -200
            print('speed too fast, distance to target: ', d2target)
            self.summary['Steps'] += 1
        else:
            done = False
            reward = 2

        location = self.vehicle.get_location()
        # print(self.angle_rw)
        # progress = np.cos(self.angle_rw) - abs(np.sin(self.angle_rw)) - abs(self.trackpos_rw)
        if self.train_mode == 'straight':
            if steer < -0.1 and steer > 0.1:
                reward = -100
                done = True
                print('steer too big, distance to target: ', d2target)
                self.summary['Steps'] += 1

        d2target = self.distance_target(self.target, location)

        
        # punish for collision
        if len(self.collision_hist) != 0:  
            done = True
            reward = -400
            print('Collision occurred, distance to target: ', d2target)
            self.summary['Steps'] += 1

        #punish IF THERE IS LANE DEPARTURE
        if abs(self.data['lateral_dist_t']) > 1.2:
            done = True
            reward = -400
            print('Lane Invasion occurred, distance to target: ', d2target)
            self.summary['Steps'] += 1


        # IF YOU HAVE REACHED THE TARGET, THE REWARD IS CHANGED AND YOU LEAVE
        if self.distance_target(self.target, location) < 15:
            done = True
            reward = 100
            self.summary['Steps'] += 1
            self.summary['Target'] += 1
            print('The target has been reached')

        # IF THE TIMER HAS ELAPSED, THE REWARD IS CHANGED AND YOU LEAVE
        if self.episode_start + (10*70) < time.time():
            print('End of timer, distance to target: ', d2target)
            done = True
            self.summary['Steps'] += 1
        if done :
            dis_trav = self.total_dist - d2target

            print(f"distance_travelled 1: {dis_trav} ")
            print(f"distance_travelled acum: {acum} ")
            if acum <= 50:
                reward = -200
            elif (acum > 50) and (acum < 160):
                reward = -100
            else:
                reward = 100

        self.cmd_vel = speed / 120  # normalizo la velocidad

        if self.SHOW_CAM :
            cv2.namedWindow('Real', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Real', self.front_camera)
            cv2.waitKey(1)
           
       
        image = cv2.resize(self.front_camera, (160, 60))
        self.observation(image)
        image = cv2.putText(image, 'Speed: '+str(int(speed)), self.org, 
                            self.font, self.fontScale, white, self.thickness, cv2.LINE_AA)
        


        if done == True:
            self.distance_acum.append(acum)

        return image, self.state(), reward, done, self.data


    def observation(self, image):
        
        # waypoint information
        # actual position of the car
        actual_position = self.vehicle.get_transform()
        pos_x = actual_position.location.x
        pos_y = actual_position.location.y
        vehicle_yaw = actual_position.rotation.yaw % 360
        velocity = self.vehicle.get_velocity()
        acceleration = self.vehicle.get_acceleration()
        dyaw_dt = self.vehicle.get_angular_velocity().z
        speed = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2)
        waypoints = self.waypoints_current_route
        vehicle_vector = actual_position.get_forward_vector()
        #actual waypoint
        actual_waypoint = self.map.get_waypoint(location= self.vehicle.get_location())
        
        current_waypoint = np.array((actual_waypoint.transform.location.x, actual_waypoint.transform.location.y,
                                        actual_waypoint.transform.rotation.yaw))
        d = 1000
        dis_next = np.sqrt((current_waypoint[0]-waypoints[2][0])**2 + (current_waypoint[1]-waypoints[2][1])**2)
        
        if dis_next < d: 
            waypoints = waypoints[2:]  
            
        waypoint = waypoints[2]
        pos_vec = np.array([pos_x - waypoint[0], pos_y - waypoint[1]])
        lv = np.linalg.norm(np.array(pos_vec))
        w = np.array([np.cos(waypoint[2]/180*np.pi), np.sin(waypoint[2]/180*np.pi)])
        unit_vec = pos_vec/lv
        cross = np.cross(w, unit_vec)
        dis = - lv * cross
        
        # calculate the deta_yaw between the car and current waypoint
        waypoint_yaw = waypoint.transform.rotation.yaw % 360
        vehicle_yaw = self.vehicle.get_transform().rotation.yaw % 360
        delta_yaw = math.degrees(np.arctan2(unit_vec[1], unit_vec[0]) 
                                 - np.arctan2(vehicle_vector.y, vehicle_vector.x))
        if 180 <= delta_yaw and delta_yaw <= 360:
            delta_yaw -= 360
        elif -360 <= delta_yaw and delta_yaw <= -180:
            delta_yaw += 360

        route_heading = np.array(
            [np.cos(waypoint_yaw/180 * np.pi),
             np.sin(waypoint_yaw/180 * np.pi)])
        vehicle_heading = np.float32(vehicle_yaw / 180.0 * np.pi)
        vehicle_heading_vec =  np.array((np.cos(vehicle_heading),
                                          np.sin(vehicle_heading)))

        self.draw_path( self.current_route, tl=self.line_time+1)
        #get next waypoint in distance
        
        velocity_t_absolute = np.array([velocity.x, velocity.y])
        acceleration_absolute = np.array([acceleration.x, acceleration.y])

        
        print(f' vehicle_heading_vec: {vehicle_heading_vec}\n vector: {velocity_t_absolute}')
        # decompose v and a to tangential and normal in vehicle coordinates
        velocity_t = _vec_decompose(vec_to_be_decomposed=velocity_t_absolute, direction=vehicle_heading_vec)
        acceleration_t = _vec_decompose(vec_to_be_decomposed=acceleration_absolute, direction=vehicle_heading_vec)
        
        
        lateral_distance = lv * np.sign(pos_vec[0] * route_heading[1] -
                                         pos_vec[1] * route_heading[0])
        self.data['velocity_t'] = velocity_t
        self.data['acc_t'] = acceleration_t
        self.data['delta_yaw_t'] = delta_yaw
        self.data['dyaw_dt_t'] = dyaw_dt
        self.data['lateral_dist_t'] = lateral_distance
        
        #print(f"location{self.data['location']}\n lateral_dist: {self.data['lateral_dist_t']}")
        #self.state['action_t_1'] = self.last_action
        # print(f"\npoint: {self.points}")


    def vec_decompose(vector, direction):
        assert vector.shape[0] == 2, direction.shape[0] == 2
        scalar_longitudinal = np.inner(vector, direction)
        vec_lateral = vector - scalar_longitudinal * direction
        scalar_lateral = np.linalg.norm(vec_lateral) * np.sign(vec_lateral[0] * direction[1] -
                                                                vec_lateral[1] * direction[0])
        return np.array([scalar_longitudinal, scalar_lateral], dtype=np.float32)

    def state(self):
        '''
        params: dict of ego state(velocity_t, accelearation_t, dist, command, delta_yaw_t, dyaw_dt_t)
        type: np.array
        return: array of size[9,], torch.Tensor (v_x, v_y, a_x, a_y
                                                 delta_yaw, dyaw, d_lateral, action_last,
                                                 future_angles)
        '''
        velocity_t = self.data['velocity_t']
        accel_t = self.data['acc_t']
        #speed =  self.data['speed'] /30

        delta_yaw_t = np.array(self.data['delta_yaw_t']).reshape(
            (1, )) / 2.0

        lateral_dist_t = self.data['lateral_dist_t'].reshape(
            (1, )) * 10.0
         
        info_vec = np.concatenate([ delta_yaw_t,  
                                     lateral_dist_t,  velocity_t, accel_t
        ],
                                  axis=0)
        info_vec = info_vec.squeeze()


        return np.float32(info_vec)
    
    def get_lane_dis(waypoints, x, y):
        """
        Calculate distance from (x, y) to waypoints.
        :param waypoints: a list of list storing waypoints like [[x0, y0], [x1, y1], ...]
        :param x: x position of vehicle
        :param y: y position of vehicle
        :return: a tuple of the distance and the closest waypoint orientation
        """
        dis_min = 1000
        waypt = waypoints[0]
        for pt in waypoints:
            d = np.sqrt((x-pt[0])**2 + (y-pt[1])**2)
            if d < dis_min:
                dis_min = d
                waypt=pt
        vec = np.array([x - waypt[0], y - waypt[1]])
        lv = np.linalg.norm(np.array(vec))
        w = np.array([np.cos(waypt[2]/180*np.pi), np.sin(waypt[2]/180*np.pi)])
        cross = np.cross(w, vec/lv)
        dis = - lv * cross
        return dis, w
