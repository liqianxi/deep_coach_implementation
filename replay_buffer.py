""" 
Data structure for implementing experience replay

Author: Patrick Emami
"""
from __future__ import division
from collections import deque
import random
import numpy as np

class ReplayBuffer(object):

    def __init__(self, buffer_limit, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        # The maximum number of *completed* experience tuples to store in buffer
        self.buffer_limit = buffer_limit
        # Buffer for incompleted experience tuples of state, action
        self.buffer = deque()
        # Buffer for completed experience tuples of state, action, feedback
        self.f_buffer = deque()
        random.seed(random_seed)

    def append(self, inc_experience):
        self.buffer.append(inc_experience)

    def apply_feedback(self, f):
        # Outside of this function call, check to make sure human delay timesteps have passed
        state, action = self.buffer.popleft()
        # Only if feedback was non-zero does the experience move to the f_buffer
        if f != 0.0:
            self.f_buffer.append((state, action, f))

        # Remove oldest experience tuples once limit is hit
        while len(self.f_buffer) > self.buffer_limit:
            self.f_buffer.popleft()

        return state, action, f

    def size(self):
        return len(self.buffer)

    def f_size(self):
        return len(self.f_buffer)

    def add(self, s, a, f):
        experience = (s, a, f)
        self.f_buffer.append(experience)

    def sample_batch(self, batch_size):
        if len(self.f_buffer) < batch_size:
            batch = self.f_buffer
        else:
            batch = random.sample(self.f_buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        f_batch = np.array([_[2] for _ in batch])

        return s_batch, a_batch, f_batch

    def clear(self):
        self.buffer.clear()
        self.f_buffer.clear()

