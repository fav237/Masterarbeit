


import glob
import os
import sys
import queue
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import numpy as np
import carla
import random
import time
import queue
import cv2
import math


IM_WIDTH = 640
IM_HEIGHT = 480
CAMERA_POS_Z = 3 
CAMERA_POS_X = 3


# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
world = client.get_world()

# Get the blueprint library
blueprint_library = world.get_blueprint_library()

# We will aslo set up the spectator so we can see what we do
spectator = world.get_spectator()

# Get the map's spawn points
spawn_points = world.get_map().get_spawn_points()
start_point =spawn_points[1]
# spawn a walker the link https://carla.readthedocs.io/en/latest/tuto_G_pedestrian_bones/
walker_bp = blueprint_library.filter('*walker.pedestrian*')[7]
if walker_bp.has_attribute('is_invincible'):
    walker_bp.set_attribute('is_invincible', 'false')
transform = carla.Transform(carla.Location(x=-18,y=123.3,z=1.8))

walker_loc = carla.Location(x=-10, y=120.0, z=1.8)
walker_rot = carla.Rotation(pitch=0.0, yaw=92.0, roll=0.0)
walker_trans = carla.Transform(walker_loc,walker_rot)

walker = world.try_spawn_actor(walker_bp, transform)

# wait for 10s
time.sleep(6.5)

### manual control of the walker ###
control_walker = carla.WalkerControl()
control_walker.speed = 0.9
control_walker.direction.y = 0
control_walker.direction.x = 1
control_walker.direction.z = 0
walker.apply_control(control_walker)

# wait for 5s
time.sleep(40)

for actor in world.get_actors().filter('*walker*'):
    actor.destroy()

### Set up the AI controller for the pedestrian ###
#walker_controller_bp = blueprint_library.find('controller.ai.walker')
#walker_controller = world.spawn_actor(walker_controller_bp, walker.get_transform(), walker)
# startthe control and give him a destination
#walker_controller.start()
#walker_controller.go_to_location(world.get_random_location_from_navigation()) # try to find a correctlocation for the walker next-time 





    