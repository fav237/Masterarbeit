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
import carla #the sim library itself
import time # to set a delay after each photo
import cv2 #to work with images from cameras
import numpy as np #in this example to change image representation - re-shaping
import traceback


# Connect to the client and retrieve the world object
try:

    client = carla.Client('localhost', 2000)
    #world = client.load_world('Town01')
    world = client.get_world()
    weather = carla.WeatherParameters(
        cloudiness = 0.0,
        precipitation=0.0,
        sun_altitude_angle=10.0,
        sun_azimuth_angle=70.0,
        precipitation_deposits=0.0,
        wind_intensity=0.0,
        wetness=0.0,
    )
    world.set_weather(weather)
except Exception as e:
    print(traceback.format_exc())

for actor in world.get_actors().filter('*vehicle*'):
    actor.destroy()
for sensor in world.get_actors().filter('*sensor*'):
    sensor.destroy()

# Get the blueprint library
blueprint_library = world.get_blueprint_library()

# Get the map's spawn points
spawn_points = world.get_map().get_spawn_points()
start_point = spawn_points[0]

# Set up the vehicle transform
vehicle_loc = carla.Location(x=334.75, y=10.0, z=0.3)
vehicle_rot = carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0)
vehicle_trans = carla.Transform(vehicle_loc,vehicle_rot)

# Filter for the Vehicle
vehicle_bp = blueprint_library.filter('*a2*')

# Add Vehicle
#vehicle = world.try_spawn_actor(vehicle_bp[0], start_point)
vehicle = world.try_spawn_actor(vehicle_bp[0], vehicle_trans)
time.sleep(1)
print(f'waypoint location : {vehicle.get_transform()}')

# send vehicle off
#vehicle.set_autopilot(True)

### camera mount offset on the car - you can tweak these to have the car in view or not
# CAMERA_POS_Z = 3 
# CAMERA_POS_X = -5 
# CAMERA_POS_Z = 5 
# CAMERA_POS_X = 1

# camera_bp = blueprint_library.find('sensor.camera.rgb')
# camera_bp.set_attribute('image_size_x', '640')
# camera_bp.set_attribute('image_size_x', '480')

# camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))
# #this creates the camera in the sim
# camera = world.spawn_actor(camera_bp,camera_init_trans,attach_to=vehicle)
# def camera_callback(image,data_dict):
#     data_dict['image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))

# image_w = camera_bp.get_attribute('image_size_x').as_int()
# image_h = camera_bp.get_attribute('image_size_y').as_int()

# camera_data = {'image': np.zeros((image_h,image_w,4))}
# # this actually opens a live stream from the camera
# camera.listen(lambda image: camera_callback(image,camera_data))


# ### route planning bit
# point_a = start_point.location #we start at where the car is


# now look at the map

# a_loc = carla.Location(x=-10, y=45.0, z=1.8)
# b_loc = carla.Location(x=-10, y=180.0, z=1.8)

# map = world.get_map()

# sampling_resolution = 1
# grp = GlobalRoutePlanner(map, 1)

# route = grp.trace_route(a_loc, b_loc)
# now let' pick the longest possible route
# distance = 0
# for loc in spawn_points: # we start trying all spawn points 
#                             #but we just exclude first at zero index
#     cur_route = grp.trace_route(point_a, loc.location)
#     if len(cur_route)>distance:
#         distance = len(cur_route)
#         route = cur_route
#draw the route in sim window - Note it does not get into the camera of the car
# for waypoint in route:
#     world.debug.draw_string(waypoint[0].transform.location, '^', draw_shadow=False,
#         color=carla.Color(r=0, g=0, b=255), life_time=60.0,
#         persistent_lines=True)

# time.sleep(5)
# # the cheating loop of moving the car along the route
# for waypoint in route:

#     # move the car to current waypoint
#     vehicle.set_transform(waypoint[0].transform)
#     # Dispaly with imshow
#     cv2.imshow('Fake self-driving',camera_data['image'])
#     cv2.waitKey(50)
    
time.sleep(1)
#cv2.destroyAllWindows()
#camera.stop() # this is the opposite of camera.listen
for actor in world.get_actors().filter('*vehicle*'):
    actor.destroy()
for sensor in world.get_actors().filter('*sensor*'):
    sensor.destroy()
    




