
import os
import gymnasium as gym
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from replay_matrix import ReplayMatrix 



class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.linear1 = nn.Linear(self.state_size, 128)
        #self.linear2 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(128, self.action_size)
        self.all_weights = list(self.linear1.parameters()) + list(self.linear2.parameters())
        self.lr = 1e-3
        self.optimizer = optim.Adam(list(self.linear1.parameters()) + list(self.linear2.parameters()), lr=self.lr)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = self.linear2(output)
        tmp = F.softmax(output, dim=-1)
        distribution = Categorical(tmp)
        return distribution, torch.argmax(tmp)

    def get_log_prob(self, state,action):
        output = F.relu(self.linear1(state))
        output = self.linear2(output)
        
        return Categorical(F.softmax(output, dim=-1)).log_prob(action)

    def get_prob(self, state, action):
        output = F.relu(self.linear1(state))
        output = self.linear2(output)
        
        return Categorical(F.softmax(output, dim=-1)).probs[action]

