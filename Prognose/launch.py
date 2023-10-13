import os

import time

# spawn the walker
print('### spawn the walker ###')
os.system('python walker.py ')

# wait for 5s
time.sleep(5)

# spawn the vehicle commimg from the right side
print('####### spawn the vehicle 2 #######')
os.system('python3 vehicle2.py')

# wait for 5s
time.sleep(5)

# spawn our ego-vehicle
print('####### spawn our ego-vehicle  #######')
os.system('python3 vehicle-control.py')