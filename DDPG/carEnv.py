

import glob
import os
from re import I
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

import numpy as np
import carla
import random
import time
import cv2
import math


SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
cam_x = 1.5
cam_z = 0.8


class CarEnv:
    #  SHOW_CAM is whether or not we want to show a preview.
    SHOW_CAM = SHOW_PREVIEW
    # STEER_AMT is how much we want to apply to steering
    STEER_AMT = 1.0

    im_width = IMG_WIDTH
    im_height = IMG_HEIGHT
    camera_x = cam_x
    camera_z = cam_z

    # The collision_hist is going to be used because the collision sensor reports a history of incidents
    collision_hist = []
	

    def __init__(self):
        # Connect to the client and retrieve the world object
        self.client = carla.Client('localhost', 2000)
        self.world = self.client.get_world()

        # Get the blueprint library
        self.blueprint_library = self.world.get_blueprint_library()

        #filter for the first vehicle blueprints
        self.a2 = self.blueprint_library.filter('A2')[0]

        ### position for route_planning
        self.point_a = 0
        self.point_b = 0


    def process_img(self, image):
        i = np.array(image.raw_data)
        #np.save("iout.npy", i)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("",i3)
            cv2.waitKey(1)
        self.front_camera = i3
          

    def collision_data(self, event):
        self.collision_hist.append(event)

    def laneInv_data(self, event1):
        self.laneInv_hist.append(event1)

    def obstacle_data(self, event2):
        self.obstacle_hist.append(event2)

    def gnss_data(self, event3):
        global latitude
        global longitude

        latitude = event3.latitude
        longitude = event3.longitude


    def total_distance(self, current_route):
        sum = 0
        for i in range(len(current_route) - 1):
            sum = sum + self.distance_wp(current_route[i + 1][0], current_route[i][0])
        return sum

    def distance_wp(self, target, current):
        dx = target.transform.location.x - current.transform.location.x
        dy = target.transform.location.y - current.transform.location.y
        return math.sqrt(dx * dx + dy * dy)

    def distance_target(self, target, current):
        dx = target.x - current.x
        dy = target.y - current.y
        return math.sqrt(dx * dx + dy * dy)
    

    def draw_line(self, world, current_route):
        for waypoint in range(len(current_route) - 1):
            waypoint_1 = current_route[i][0]
            waypoint_2 = current_route[i + 1][0]
            self.world.debug.draw_line(waypoint_1.transform.location, waypoint_2.transform.location, thickness=2,
                                       color=carla.Color(0, 255, 0), life_time=20)
            
    
    def draw_point(self, world, current_route):
        for  waypoint in current_route:
            
            self.world.debug.draw_string(waypoint[0].transform.location, '°', draw_shadow=False,
                color=carla.Color(r=0, g=0, b=255), life_time=20,
                persistent_lines=True)

    def reset(self):
        self.actor_list = []
        self.collision_hist = []
        self.laneInv_hist = []
        self.obstacle_hist = []
        self.total_distance = 0
        self.route_planner = GlobalRoutePlanner(self.world.get_map(), 1)


        spawn_points = self.map.get_spawn_points()

        ### planing randomly a route ###
        while self.total_distance < 2500:
            self.point_a = random.choice(spawn_points)
            self.point_b = random.choice(spawn_points)

            a = self.pos_a.location
            b = self.pos_b.location
            self.current_route = self.grp.trace_route(a, b)
            self.total_distance = self.total_distance(self.current_route)

        self.total_distance = self.total_distance(self.current_route)

        self.transform = self.point_a
        # Get the map's spawn point
        self.spawn_point = random.choice(self.a2, self.spawn_point)

        # spawn the car
        self.vehicle = self.world.spawn_actor(self.a2, self.spawn_point)

        # add the car tothe list of actor
        self.actor_list.append(self.vehicle)

        # get the blueprint for camera
        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')

        # change the dimensions of the image
        self.camera_bp.set_attribute('image_size_x', f'{self.im_width}')
        self.camera_bp.set_attribute('image_size_y', f'{self.im_height}')
        self.camera_bp.set_attribute('fov', '110')

        # Adjust sensor relative to vehicle
        sensor_transform = carla.Transform(carla.Location(z=self.camera_z,x=self.camera_x))

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

        # get the blueprint for Obstacle sensor
        obstacle_bp = self.blueprint_library.find('sensor.other.obstacle')
        self.obstacle = self.world.spawn_actor(obstacle_bp, sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.obstacle)
        self.obstacle.listen(lambda event2: self.lane_data(event2))

        # get the blueprint for gnss sensor
        gnss_bp = self.blueprint_library.find('sensor.other.gnss')
        self.gnss = self.world.spawn_actor(gnss_bp, sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.obstacle)
        self.gnss.listen(lambda event3: self.gnss_data(event3))

        # to be sure tha when the car falls into the simulator
        # it is not record as a error
        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))

        return self.front_camera
    

    def step(self, action):

        '''
        define all action here


        '''

        steer = action[0]
        throttle = action[1]
        brake = action[3]

        # steering action
        if steer == 0:
            steer = -0.5
        elif steer == 1:
            steer = -0.25  
        elif steer == 2:
            steer = -0.1
        elif steer == 3:
            steer = 0.05
        elif steer == 4:
            steer = 0.0
        elif steer == 5:
            steer = 0.05
        elif steer == 6:
            steer = 0.1
        elif steer == 7:
            steer = 0.25  
        elif steer == 8:
            steer =  0.5
        
        # Brake action
        if brake ==0 :
            brake = 0
        elif brake ==1:
            brake = 0.5
        elif brake == 2:
            brake = 1

        if throttle == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=steer, brake=brake))
        elif throttle == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.25, steer=steer, brake=brake))
        elif throttle == 2:
             self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=steer, brake=brake))
        else:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.75, steer=steer, brake=brake))


        # get the velocity of the car, calcul his speed and apply the predefined speed
        v = self.vehicle.get_velocity()
        speed = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        ### start defining reward from each step ###


        # punish for collision
        if len(self.collision_hist) != 0 :
            done = True
            reward = -200

        # punish for lane invasion
        if len(self.laneInv_hist) != 0 :
            done = True
            reward = -100

        #reward for acceleration
        if speed < 10:
            done = False
            reward = -1
        elif speed > 30:
            done = False
            reward = -1
        elif speed > 40:
            done = False
            reward = -5
        else:
            done = False
            reward = 1


        # check for episode duration
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
        	done = True


        return self.front_camera, reward, done, None
    
    

