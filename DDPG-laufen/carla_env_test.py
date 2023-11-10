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
    
    config2 = tf.ConfigProto()
    config2.gpu_options.allow_growth = True
    tf_session2 = tf.Session(config=config2)

    keras_backend.set_session(tf_session2)
    proximity_threshold = 15

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
        self.waypoints_route = []
        self.dif_angle_routes = 0
        #############################NUEVO
        self.total_dist = 1
        self.map = self.world.get_map()
        self.route_planner = GlobalRoutePlanner(self.map, 1)
        #############################
        
        self.spawn_points = self.map.get_spawn_points()
        self.waypoints_route = []

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
            self.waypoints_route.append([w1.transform.location.x, w1.transform.location.y, 
                  w1.transform.rotation.yaw])
        self.waypoints_route.append([0, 0, 0])
        self.target = w1.transform.location
        #print(f"waypoins: {self.waypoints_route}")
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

        state = self.observation(image)

        return  image, state


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
        self.position_array.append([x_prev, y_prev, location_rv.location.z, location_rv.rotation.pitch, location_rv.rotation.yaw, location_rv.rotation.roll])
        # print(reward)
        x_prev = location_rv.location.x
        y_prev = location_rv.location.y
        d_i_prev = d_i

        v = self.vehicle.get_velocity()
        speed = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        print(f'speed: {speed}')
        #reward for acceleration
        if speed < 5:
            done = False
            reward = -50
        elif speed < 10:
            done = False
            reward = -10
        elif speed > 40:
            done = False
            reward = -1
        elif speed > 50:
            done = False
            reward = -10
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
                self.summary['Steps'] += 1

        d2target = self.distance_target(self.target, location)

        lock_time = 0
        # if  self.steering_lock == False:
        #     self.steering_lock == True
        
        # punish for collision
        if len(self.collision_hist) != 0:  
            done = True
            reward = -200
            print('Collision occurred, distance to target: ', d2target)
            self.summary['Steps'] += 1

        #punish IF THERE IS LANE DEPARTURE
        if len(self.laneInv_hist) != 0:
            done = True
            reward = -100
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
            if acum <= 50:
                reward = -200
            elif (acum > 50) and (acum < 160):
                reward = -100
            else:
                reward = 100

        self.cmd_vel = speed / 120  

        # if out of lane
        dis, _ =self.get_lane_dis(self.waypoints_route,x_prev, y_prev)
        if abs(dis) > 1.5 :
            done = True
            print('out of lane, distance to target: ', d2target)
            self.summary['Steps'] += 1

        if self.SHOW_CAM :
            cv2.namedWindow('Real', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Real', self.front_camera)
            cv2.waitKey(1)
           
       
        image = cv2.resize(self.front_camera, (160, 60))
        state = self.observation(image)
            


        if done == True:
            self.distance_acum.append(acum)

        return image, state, reward, done, None


    def observation(self, image):
        self.distances = [1., 5., 10.]
        angles =[]
        
        # waypoint information
        # actual position of the car
        actual_position = self.vehicle.get_transform()
        ego_x = actual_position.location.x
        ego_y = actual_position.location.y
        ego_yaw = actual_position.rotation.yaw / 180 *np.pi

        # lateral distance of ego car from lane line, angle to lane line
        lateral_dis, w = self.get_preview_lane_dis(self.waypoints_route, ego_x, ego_y)

        # ego car and lane line heading
        delta_yaw = np.arcsin(np.cross(w, 
                                np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))

        #                        
        # calculate the speed of the car                        
        v = self.ego.get_velocity()
        
        self.vehicle_front, self.walker_front = self.get_hazard()
        self.draw_path( self.current_route, tl=self.line_time+1)
        self.data['speed'] = np.sqrt(v.x**2 + v.y**2)
        self.data['delta_yaw'] = delta_yaw
        self.data['lateral_dis'] = lateral_dis
        self.data['vehicle_front'] = self.vehicle_front
        speed = np.array(self.data['speed']).reshape((1, )) 
        delta_yaw = np.array(self.data['delta_yaw']).reshape((1, )) 
        lateral_dis = np.array(self.data['lateral_dis']).reshape((1, )) 
        vehicle_front = np.array(self.data['vehicle_front']).reshape((1, )) 
        
        print(f"lateral: {lateral_dis}\n delta: {delta_yaw}\n speed:{speed}\n vehicle:{vehicle_front}")
        state = np.array([lateral_dis, - delta_yaw, speed, self.vehicle_front])


        return state

    def get_preview_lane_dis(waypoints, x, y, idx=2):
        """
        Calculate distance from (x, y) to a certain waypoint
        :param waypoints: a list of list storing waypoints like [[x0, y0], [x1, y1], ...]
        :param x: x position of vehicle
        :param y: y position of vehicle
        :param idx: index of the waypoint to which the distance is calculated
        :return: a tuple of the distance and the waypoint orientation
        """
        waypt = waypoints[2]
        vec = np.array([x - waypt[0], y - waypt[1]])
        lv = np.linalg.norm(np.array(vec))
        w = np.array([np.cos(waypt[2]/180*np.pi), np.sin(waypt[2]/180*np.pi)])
        cross = np.cross(w, vec/lv)
        dis = - lv * cross
        return dis, w

    def distance_vehicle(waypoint, vehicle_transform):
        loc = vehicle_transform.location
        dx = waypoint.transform.location.x - loc.x
        dy = waypoint.transform.location.y - loc.y

        return math.sqrt(dx * dx + dy * dy)


    def get_hazard(self):
        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        walker_list = actor_list.filter("*walker*")

        # check possible obstacles
        vehicle_state = self._is_vehicle_hazard(vehicle_list)

        walker_state = self._is_vehicle_hazard(walker_list)

        return  vehicle_state, walker_state

    def _is_vehicle_hazard(self, vehicle_list):
        """
        Check if a given vehicle is an obstacle in our way. To this end we take
        into account the road and lane the target vehicle is on and run a
        geometry test to check if the target vehicle is under a certain distance
        in front of our ego vehicle.

        WARNING: This method is an approximation that could fail for very large
        vehicles, which center is actually on a different lane but their
        extension falls within the ego vehicle lane.

        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
            - bool_flag is True if there is a vehicle ahead blocking us
            and False otherwise
            - vehicle is the blocker object itself
        """

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self.map.get_waypoint(ego_vehicle_location)

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self.map.get_waypoint(target_vehicle.get_location())
            if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            loc = target_vehicle.get_location()
            if self.is_within_distance_ahead(loc, ego_vehicle_location,
                            self.vehicle.get_transform().rotation.yaw,
                            self.proximity_threshold):
                return True

        return False

    def is_within_distance_ahead(target_location, current_location, orientation, max_distance):
        """
        Check if a target object is within a certain distance in front of a reference object.

        :param target_location: location of the target object
        :param current_location: location of the reference object
        :param orientation: orientation of the reference object
        :param max_distance: maximum allowed distance
        :return: True if target object is within max_distance ahead of the reference object
        """
        target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
        norm_target = np.linalg.norm(target_vector)
        if norm_target > max_distance:
            return False

        forward_vector = np.array(
            [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
        d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

        return d_angle < 90.0

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