

import random
from collections import deque


class ReplayBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.current_size = 0
        self.buffer = deque()


    def __len__(self):
        return self.current_size

    def add(self, experience):
        if self.current_size < self.max_size:
            self.buffer.append(experience)
            self.current_size += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def get_batch(self, size):
        sample_size = size if size <= self.current_size else self.current_size
        return random.sample(self.buffer, sample_size)

    def clear(self):
        self.buffer.clear()
        self.current_size = 0
    