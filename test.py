
import torch
from actor import Actor

actor_model = Actor(3,3)

input = torch.randn(1, 3)
input = torch.tensor([[1.0,1.0,1.0]])
action = torch.tensor([[0]])
forward = actor_model(input)[0].log_prob(action)

forward.backward()
print(actor_model.linear1.weight.grad.shape)


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