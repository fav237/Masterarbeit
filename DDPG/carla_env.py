
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
#import gym
#from gym import spaces

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
cam_x = 1.5
cam_z = 0.8

## https://carla.readthedocs.io/en/latest/tuto_G_retrieve_data/
## https://carla.readthedocs.io/en/latest/core_sensors/#cameras


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

class Carla_Env:
    #  SHOW_CAM is whether or not we want to show speed preview.
    SHOW_CAM = SHOW_PREVIEW
    # STEER_AMT is how much we want to apply to steering
    STEER_AMT = 1.0

    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    camera_x = cam_x
    camera_z = cam_z
    actor_list = []
    summary = {"goal": 0, "done": 0}
    distance_acum = []

    front_camera = None
    # The collision_hist is going to be used because the collision sensor reports speed history of incidents
    collision_hist = []

    def __init__(self):
        
        # Connect to the client and retrieve the world object
        self.client = carla.Client('localhost', 2000)
        self.world = self.client.get_world()

        # Get the blueprint library
        self.blueprint_library = self.world.get_blueprint_library()

        #filter for the first vehicle blueprints
        self.a2 = self.blueprint_library.filter('*a2*')[0]

    
    def reset(self):
        self.actor_list = []
        self.collision_hist = []
        self.laneInv_hist = []
        self.radar_hist = []
        self.distance_to_goal = 0
        self.route_planner = GlobalRoutePlanner(self.world.get_map(), 1)
        self.Target =0
        global acum
        global x_prev
        global y_prev
        acum = 0

        spawn_points = self.world.get_map().get_spawn_points()

        ### planing randomly a route ###
        while self.distance_to_goal < 2500:
            self.point_a = random.choice(spawn_points)
            self.point_b = random.choice(spawn_points)

            a = self.point_a.location
            b = self.point_b.location
            
            self.current_route = self.route_planner.trace_route(a, b)
            self.distance_to_goal = self.total_distance(self.current_route)

        self.distance_to_goal = self.total_distance(self.current_route)

        self.Target = b
        self.transform = self.point_a
        # Get the map's spawn point
        self.spawn_point = random.choice(self.a2, self.spawn_point)


        self.draw_line(self.world, self.current_route)
        #self.draw_point(self.world, self.current_route)

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


        # get the blueprint for gnss sensor
        gnss_bp = self.blueprint_library.find('sensor.other.gnss')
        self.gnss = self.world.spawn_actor(gnss_bp, sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.radar)
        self.gnss.listen(lambda event2: self.gnss_data(event2))

        # to be sure tha whenthe car falls into the simulator
        # it is not record as speed error
        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))
        location_reset = self.vehicle.get_transform()
        x_prev = location_reset.location.x
        y_prev = location_reset.location.y
        angle, distance_to_route = self.get_closest_wp_forward()
        self.last_distance_to_route = distance_to_route

        return {'image': self.front_camera, 'float_input': angle}
    


    def collision_data(self, event):
        self.collision_hist.append(event)

    def laneInv_data(self, event1):
        self.laneInv_hist.append(event1)


    def gnss_data(self, event2):
        global latitude
        global longitude

        latitude = event2.latitude
        longitude = event2.longitude



    def process_img(self, image):
        i = np.array(image.raw_data)
        #np.save("iout.npy", i)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        
        self.front_camera = i3


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
    

    def draw_line(self, current_route):
        for i in range(len(current_route) - 1):
            waypoint_1 = current_route[i][0]
            waypoint_2 = current_route[i + 1][0]
            self.world.debug.draw_line(waypoint_1.transform.location, waypoint_2.transform.location, thickness=2,
                                       color=carla.Color(0, 255, 0), life_time=20)
            
    
    def draw_point(self, current_route):
        for  waypoint in current_route:
            
            self.world.debug.draw_string(waypoint[0].transform.location, '°', draw_shadow=False,
                color=carla.Color(r=0, g=0, b=255), life_time=20,
                persistent_lines=True)
            
    def get_wp_forward(self):
        '''
		this function is to find the closest point looking forward
		if there in no points behind, then we get first available
		'''
        # first we create a list of angles and distances to each waypoint
		# yeah - maybe a bit wastefull
        points_ahead = []
        points_behind = []
        for i, wp in enumerate(self.current_route):
            ## get angle
            vehicle_pos = self.vehicle.get_transform()
            wp_transform = wp[0].transform
            distance = math.sqrt((wp_transform.location.y - vehicle_pos.location.y)**2 + (wp_transform.location.x - vehicle_pos.location.x)**2)
            angle = math.degrees(math.atan2(wp_transform.location.y - vehicle_pos.location.y,
                                            wp_transform.location.x - vehicle_pos.location.x)) - vehicle_pos.rotation.yaw
            
            if angle > 360:
                angle -= 360
            elif angle < -360:
                angle += 360
            elif angle > 180:
                angle = -360 + angle
            elif angle < 180:
                angle = 360 - angle
            
            if abs(angle)<= 90:
                points_ahead.append([i, distance, angle])
            else:
                points_behind.append([i, distance, angle])

        # now we pick a point we need to get angle to
        if len(points_ahead)==0:
            closest = min(points_behind, key=lambda x: x[1])
            if closest[2]>0:
                closest = [closest[0],closest[1],90]
            else:
                closest = [closest[0],closest[1],-90] 
        else:
            closest = min(points_ahead, key=lambda x: x[1])
            # move forward if too close
            for i, point in enumerate(points_ahead):
                if point[1]>=10 and point[1]<20:
                    closest = point
                    break
            return closest[2]/90.0, closest[1] # we convert angle to [-1 to +1] and also return distance



    def step(self, action):
        '''
        explain the step function

        '''
        global x_prev
        global y_prev
        global acum
        global acum_prev
        global d_i_prev


        action_control = {
            0: [-0.5, 0.0, 0.3],       # left_turn 90° angle
            1: [-0.25, 0.0, 0.3],      # left_turn curve or 45° angle
            2: [-0.05, 0.0, 0.3],      # left steer to stay in the route
            3: [0.0, 0.0, 0.5],        # straight
            4: [-0.05, 0.0, 0.3],      # right steer to stay in the route
            5: [-0.25, 0.0, 0.3],      # rifght_turn curve or 45° angle
            6: [-0.05, 0.0, 0.3],      # right_turn 90° angle
            7: [0.0, 0.5, 0.0],        # medium stop
            8: [0.0, 1.0, 0.0]         # stark stop
        }
        steer = action_control[action][0]
        brake = action_control[action][1]
        throttle = action_control[action][2]

        self.vehicle.apply_control(carla.VehicleControl(throttle =throttle , steer=steer, brake=brake))

        location_new = self.vehicle.get_transform()
        
        location = self.vehicle.get_location()

        distance_new = math.sqrt((x_prev - location_new.location.x) ** 2 + (y_prev - location_new.location.y) ** 2)

        acum += distance_new
        x_prev = location_new.location.x
        y_prev = location_new.location.y


        
        distance_prev = distance_new

        # acceleration
        v = self.vehicle.get_velocity()
        speed = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if self.SHOW_CAM:
            cv2.namedWindow('Real', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Real', self.front_camera)
            cv2.waitKey(1)


        # get angle and distance to the navigation route
        angle, distance = None, None
        while angle is None:
            try:
                angle, distance = self.get_closest_wp_forward()
            except:
                pass

        #print('angle ',angle,' distance',distance)

        # punish for collision
        if len(self.collision_hist) != 0 :
            done = True
            reward = -200
            self.summary['done'] += 1

        # punish for lane invasion
        if len(self.laneInv_hist) != 0 :
            done = True
            reward = -100
            self.summary['done'] += 1

        #reward for acceleration
        if speed < 10:
            done = False
            reward = -1
        elif speed > 40:
            done = False
            reward = -1
        elif speed > 50:
            done = False
            reward = -5
        else:
            done = False
            reward = 1

        # punish for deviating from the route
        route_loss =  distance - self.last_distance_to_route 
        if route_loss > 0.5:
            reward = -1
        if distance > 20:
            reward= -1


        # reward if the vehicle get to the goal location
        if self.distance_target(self.Target, location) < 15:
                done = True
                reward = 100
                self.summary['done'] += 1
                self.summary['goal'] += 1

        # reward if the distance travelled by the vehicle is more that 160
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
            self.summary['done'] += 1
            self.distance_acum.append(acum)
            if acum <= 50:
                    reward = -200
            elif (acum > 50) and (acum < 160):
                reward = -100
            else:
                reward = 100

        return {'image': self.front_camera, 'float_input': angle}, reward, done, None
    

   