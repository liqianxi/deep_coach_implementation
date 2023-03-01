import os
import gymnasium as gym
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from replay_matrix import ReplayMatrix 
from human_thread import FeedbackCollector
from collections import defaultdict
from functools import reduce # only in Python 3
import sys
import numpy as np


from graphviz import Digraph
import torch
from torch.autograd import Variable


# make_dot was moved to https://github.com/szagoruyko/pytorchviz
from torchviz import make_dot


def replay_matrix_train(actor, replay, batch_size,beta, state_observation):

    # If we have enough samples for a batch update:
    if replay.size() >= batch_size:
        size_list = [actor.linear1.weight.shape, actor.linear2.weight.shape]

        # Sample one minibatch of windows from the buffer.
        minibatch_of_windows = replay.sample_batch(batch_size)

        # Store the average eligibility traces. (same size as the weights in out policy net)
        avg_eligibility_trace_list = [torch.zeros(each_size) for each_size in size_list]

        # i.e. for each window in the mini-batch
        for w in minibatch_of_windows:
            # Initralize the average eligibility traces. (same size as the weights in out policy net)
            eligibility_trace_lamda = [torch.zeros(each_size) for each_size in size_list]

            # Retrieve final feedback signal F from window w.
            final_feedback_signal = w[-1][-1]

            # For s, a, p, f in w do:
            for state, action, log_prob, feed in w:

                # Clear the accumulated gradients.
                actor.optimizer.zero_grad()

                # Calculate the loss.
                log_p = actor.get_log_prob(state,action)

                # Backprop to get gradients.
                log_p.backward()

                # The important sampling calculation requires p value, not log(p) value.
                pi_theta_t = torch.exp(log_p)

                # Retrieve the gradients of weights.
                layer1_grad = actor.linear1.weight.grad
                layer2_grad = actor.linear2.weight.grad

                # Update eligibility_trace.
                eligibility_trace_lamda[0] = replay.eligibility_decay * eligibility_trace_lamda[0] + pi_theta_t / log_prob * layer1_grad
                eligibility_trace_lamda[1] = replay.eligibility_decay * eligibility_trace_lamda[1] + pi_theta_t / log_prob * layer2_grad


            # For all policy net layers, add them to the average eligibility_trace and weighted by final_feedback_signal.
            for i in range(2):
                avg_eligibility_trace_list[i] += final_feedback_signal * eligibility_trace_lamda[i]
        
        # Clear gradients.
        actor.optimizer.zero_grad()

        # Below several lines calculate the entropy gradient.
        cur_distribution = actor(state_observation)[0].probs
        entropy = - torch.sum(cur_distribution * torch.log(cur_distribution))
        entropy.backward()

        #make_dot(entropy).view()

        # Update avg eligibility trace.
        avg_eligibility_trace_list[0] = avg_eligibility_trace_list[0] / batch_size + beta * actor.linear1.weight.grad
        avg_eligibility_trace_list[1] = avg_eligibility_trace_list[1] / batch_size + beta * actor.linear2.weight.grad

        # Manually update weights in the net.
        actor.linear1.weight = nn.Parameter(actor.linear1.weight + actor.lr * avg_eligibility_trace_list[0])
        actor.linear2.weight = nn.Parameter(actor.linear2.weight + actor.lr * avg_eligibility_trace_list[1])


        
            



def trainIters(env, actor, experiment, n_iters, device):
    # if experiment.feedback_mode == 1:  # if using human feedback, begin collector thread
    #     feedback_thread = FeedbackCollector()
    #     feedback_thread.setDaemon(True)
    #     feedback_thread.start()
    
    experience = experiment.experience
    
    for iter in range(n_iters):
        state_observation,_ = env.reset()

        a = None
        ep_steps = 0
        for _ in range(experiment.max_steps):
            env.render()


            with torch.no_grad():
                # Observe current state s_t
                state_observation = torch.tensor(state_observation)

                # Collect stochastic policy given current state, get the argmax action!
                d, best_action = actor(state_observation)

                # Record probability
                prob = d.probs[best_action]

                # We need a queue to track experienced [s,a,p],since we have human delay, we need to append
                # previous [s,a,p] pairs later to the window.
                pair = [state_observation, best_action, prob]
                experience.append_sap(pair)

            # Execute action.
            next_state_observation, env_reward, terminal, truncated, info = env.step(best_action.item())

            if terminal:
                break

            # Here get human feedback.
            f = 1.0#np.random.choice([1.0])
            # if experiment.feedback_mode == 0:
            #     f += env_reward
            # else:
            #     f = feedback_thread.poll()
            # if f != 0.0:
            #     #feedback_histogram[f] += 1
            #     print(f"receive human feedback {f}")
            # print(best_action.item())
            # f = fake_feedback(best_action.item())
            # print("f",f)
            f = torch.tensor(f)
            
            sap = experience.get_first_sap()
            s = sap[0] # s_(t-d)
            a = sap[1] # a_(t-d)
            p = sap[2] # p_(t-d)

            # Here, if f==0, it will still add to the window.
            # Whenever the f !=0, the trajectory in a window end, append 
            # the window with L most recent entries to
            # the replay buffer.

            experience.add2(s,a,p,f)

            if ep_steps >= experiment.human_delay:
                if experiment.train_mode == 2:
                    replay_matrix_train(actor, experience, experiment.batch_size,experiment.beta, state_observation)
                else:
                    print ('Unknown training procedure specification!')
                    sys.exit(1)

            ep_steps += 1
            print ('ep_steps',ep_steps)
            state_observation = next_state_observation
