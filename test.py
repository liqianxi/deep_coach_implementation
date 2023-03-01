from graphviz import Digraph
import torch
from torch.autograd import Variable
import torch
from actor import Actor
from torchviz import make_dot


#actor_model = Actor(3,3)


x1 = torch.tensor([[1.0,2.0]],requires_grad=True)
x2 = torch.tensor([[2.0,2.0]],requires_grad=True)

y = torch.sum(x1 * torch.log(x1))
# with torch.no_grad():
#     action = torch.tensor([[0]])
# forward = actor_model(input)[0].log_prob(action)

# forward.backward()
y.backward()
print(y)
make_dot(y).view()


# class new2():
#     def __init__(self):
#         self.layer1 = [1,2]
#         self.layer2 = [3,4]
#         self.list = [self.layer1,self.layer2]

#     def play(self):
#         print(self.list)
#         self.layer1  = [99,1]
#         print(self.list)


# # new2_obj = new2()
# # new2_obj.play()

# import torch
# import torch.nn as nn
# x = torch.tensor([1., 3., 1., 3.])
# c = torch.distributions.categorical.Categorical(x)
# p = c.probs
# print(p)
# log_p = torch.log(p)
# print(log_p)
# print(torch.sum(p*log_p))

# import random
# from collections import deque
# import numpy as np


# from human_thread import FeedbackCollector


# feedback_thread = FeedbackCollector()
# feedback_thread.start()
# while True:
#     tmp = feedback_thread.poll()
#     if tmp:
#         print(tmp)