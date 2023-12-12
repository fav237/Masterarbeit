

import random
from collections import deque


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.max_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)


    def __len__(self):
        return len(self.buffer) 

    def add(self, experience):
        self.buffer.append(experience)

    def get_batch(self, size):
        if len(self.buffer) >= size: 
            return random.sample(self.buffer, size)
        else : 
            return random.sample(self.buffer, len(self.buffer))
        
        

    def clear(self):
        self.buffer.clear()
    