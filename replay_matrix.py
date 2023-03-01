import random
from collections import deque
import numpy as np


class ReplayMatrix:
    def __init__(self, buffer_size, eligibility_decay, window_size=10, human_delay=1):
        self.buffer_size = buffer_size
        self.window_size = window_size
        self.eligibility_decay = eligibility_decay
        self.count = 0
        self.buffer = deque()
        self.curr_window = []

        self.sap_container = []
        self.human_delay = human_delay

    def append_sap(self, sap_list):
        if len(self.sap_container) == self.human_delay:
            self.pop_sap(0)

        self.sap_container.append(sap_list)
    
    def pop_sap(self, index):
        if len(self.sap_container) >0:
            return self.sap_container.pop(index)

    def get_first_sap(self):
        return self.sap_container[0]

    def add(self, s, a, f):
        experience = (s, a, f)
        self.curr_window.append(experience)
        # If we experience non-zero feedback, we have a complete window
        if f != 0.0:
            # Add to replay matrix
            self.buffer.append(self.curr_window)
            # Reset for next window
            self.curr_window = []
            # Resolve overall buffer size
            if self.count < self.buffer_size:
                self.count += 1
            else:
                self.buffer.popleft()

        if len(self.curr_window) == self.window_size:
            self.curr_window.pop(0)

    def add2(self, s, a, p, f):
        experience = (s, a, p, f)
        self.curr_window.append(experience)
        # If we experience non-zero feedback, we have a complete window
        if f != 0.0:
            # Add to replay matrix
            self.buffer.append(self.curr_window)
            # Reset for next window
            self.curr_window = []
            # Resolve overall buffer size
            if self.count < self.buffer_size:
                self.count += 1
            else:
                self.buffer.popleft()

        if len(self.curr_window) == self.window_size:
            self.curr_window.pop(0)

    def size(self):
        return self.count

    def softmax(self, x, beta=0.0):
        x = np.array(x)
        return np.exp(beta * x) / np.sum(np.exp(beta * x))

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = self.buffer
        else:
            # batch = random.sample(self.buffer, batch_size)
            #sample_distro = self.softmax(range(len(self.buffer)), beta=0.1)
            length = self.count
            idx = random.sample(range(0,length),batch_size)

            #batch = np.random.choice(self.buffer, size=batch_size, replace=False, p=sample_distro)

            batch = []
            for each_idx in idx:
                batch.append(self.buffer[each_idx])

        return batch

    def clear(self):
        self.buffer.clear()
        self.curr_window = []
        self.count = 0
