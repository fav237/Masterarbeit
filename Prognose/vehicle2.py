

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

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
world = client.get_world()

# Get the blueprint library
blueprint_library = world.get_blueprint_library()

# Get the map's spawn points randomly
spawn_point = random.choice(world.get_map().get_spawn_points())

#filter for the vehicle blueprints
vehicle_bp = blueprint_library.filter('*a2*')[0]
if vehicle_bp.has_attribute('is_invincible'):
    vehicle_bp.set_attribute('is_invincible', 'false')

# Set up the vehicle transform
vehicle_loc = carla.Location(x=-35, y=135.0, z=1.8)           
vehicle_rot = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
vehicle_trans = carla.Transform(vehicle_loc,vehicle_rot)

# wait for 15s
time.sleep(9.5)

# spawn the car
vehicle = world.spawn_actor(vehicle_bp, vehicle_trans)

### Route planing ###
# from A to B
point_a = carla.Location(x=-35, y=135.0, z=1.8)
point_b = carla.Location(x=1.8, y=105.0, z=1.8)
route_planner= GlobalRoutePlanner(world.get_map(), 1)
route = route_planner.trace_route(point_a, point_b)

#draw the route in sim window -
for waypoint in route:
    world.debug.draw_string(waypoint[0].transform.location, '°', draw_shadow=False,
    color=carla.Color(r=0, g=0, b=255), life_time=30.0,
    persistent_lines=True)

for waypoint in route:
    vehicle.set_transform(waypoint[0].transform)

    time.sleep(0.12)