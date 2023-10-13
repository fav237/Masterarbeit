

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
# Wrangler Rubicon
actor_list = []
IM_WIDTH = 640
IM_HEIGHT = 480
CAMERA_POS_Z = 3 
CAMERA_POS_X = 3
# define speed contstants
PREFERRED_SPEED = 30 # what it says
SPEED_THRESHOLD = 2 #defines when we get close to desired speed so we drop the

# Max steering angle
MAX_STEER_DEGREES = 20

def angle_between(v1, v2):
    return math.degrees(np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0]))


# maintain speed function
def maintain_speed(current_speed):
    ''' 
    this is a very simple function to maintan desired speed
    
    '''
    if current_speed >= PREFERRED_SPEED:
        return 0
    elif current_speed < PREFERRED_SPEED - SPEED_THRESHOLD:
        return 0.9 # think of it as % of "full gas"
    else:
        return 0.4 # tweak this if the car is way over or under preferred speed 

# function to get angle between the car and target waypoint
def get_angle(vehicle,wp):
    '''
    this function to find direction to selected waypoint
    '''
    vehicle_pos = vehicle.get_transform()
    vehicle_x = vehicle_pos.location.x
    vehicle_y = vehicle_pos.location.y
    wp_x = wp.transform.location.x
    wp_y = wp.transform.location.y
    
    # vector to waypoint
    x = (wp_x - vehicle_x)/((wp_y - vehicle_y)**2 + (wp_x - vehicle_x)**2)**0.5
    y = (wp_y - vehicle_y)/((wp_y - vehicle_y)**2 + (wp_x - vehicle_x)**2)**0.5
    
    #vehicle vector
    vehicle_vector = vehicle_pos.get_forward_vector()
    degrees = angle_between((x,y),(vehicle_vector.x,vehicle_vector.y))

    return degrees

try: 

    # Connect to the client and retrieve the world object
    client = carla.Client('localhost', 2000)
    world = client.get_world()

    # Get the blueprint library
    blueprint_library = world.get_blueprint_library()

    # Get the map's spawn points randomly
    spawn_point = random.choice(world.get_map().get_spawn_points())

    #filter for the vehicle blueprints
    vehicle_bp = blueprint_library.filter('*wrangler_rubicon*')[0]

    # Set up the vehicle transform
    vehicle_loc = carla.Location(x=-10, y=40.0, z=1.8)
    vehicle_rot = carla.Rotation(pitch=0.0, yaw=92.0, roll=0.0)
    vehicle_trans = carla.Transform(vehicle_loc,vehicle_rot)

    # spawn the car
    vehicle = world.spawn_actor(vehicle_bp, vehicle_trans)

    # send vehicle with autovisuel
    #vehicle.set_autopilot(True)

    # add the car tothe list of actor
    actor_list.append(vehicle)

    ''' 
    work with the RGB camera
    
    https://carla.readthedocs.io/en/latest/core_sensors/#cameras

    '''
    # get the blueprint for this sensor yet for camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')

    # change the dimensions of the image
    camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
    camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
    camera_bp.set_attribute('fov', '110')

    # Adjust sensor relative to vehicle
    camera_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))

    # spawn the sensor and attach to vehicle.
    camera = world.spawn_actor(camera_bp,camera_trans,attach_to=vehicle)

    # add sensor to list of actors
    actor_list.append(camera)
    def process_img(image):
        i = np.array(image.raw_data)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        return i3
    
    # this actually opens a live stream from the camera
    camera.listen(lambda image: process_img(image))
   
    ### Route planing

    point_a = carla.Location(x=-10, y=45.0, z=1.8)
    point_b = carla.Location(x=-10, y=180.0, z=1.8)

    route_planner= GlobalRoutePlanner(world.get_map(), 1)
    route = route_planner.trace_route(point_a, point_b)

    # draw the route in sim window -
    # Note it does not get into the camera of the car
    for waypoint in route:
        world.debug.draw_string(waypoint[0].transform.location, '°', draw_shadow=False,
        color=carla.Color(r=0, g=255, b=0), life_time=40.0,
        persistent_lines=True)
        
    # the cheating loop of moving the car along the route
    # we will be tracking waypoints in the route and switch to next one wen we get close to current one
    current_wp = 5 
    while current_wp<len(route)-1:
        cv2.imshow('Camera',process_img('image'))
        cv2.waitKey(1)
        while current_wp<len(route) and vehicle.get_transform().location.distance(route[current_wp][0].transform.location)<5:
            current_wp +=1 #move to next wp if we are too close
        
        predicted_angle = get_angle(vehicle,route[current_wp][0])
        v = vehicle.get_velocity()
        speed = round(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2),0)
        estimated_throttle = maintain_speed(speed)
        # extra checks on predicted angle when values close to 360 degrees are returned
        if predicted_angle<-300:
            predicted_angle = predicted_angle+360
        elif predicted_angle > 300:
            predicted_angle = predicted_angle -360
        steer_input = predicted_angle
        # limit steering to max angel, say 40 degrees
        if predicted_angle<-MAX_STEER_DEGREES:
            steer_input = -MAX_STEER_DEGREES
        elif predicted_angle>MAX_STEER_DEGREES:
            steer_input = MAX_STEER_DEGREES
        # conversion from degrees to -1 to +1 input for apply control function
        steer_input = steer_input/75
        vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer_input))
    # sleep for 20 seconds, the finish
        

finally:
    print('detroying actors')
    for actor in actor_list:
        actor.destroy()
    for actor in world.get_actors().filter('*vehicle*'):
        actor.destroy()
    for actor in world.get_actors().filter('*walker*'):
        actor.destroy()
    for sensor in world.get_actors().filter('*sensor*'):
        sensor.destroy()
    print('done.')


