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



def replay_batch_train(actor, replay, f, human_delay, batch_size):
    # Make sure that we have enough entries to being applying human feedback
    if replay.size() >= human_delay:
        # For now, we only keep transitions with non-zero feedback
        state, action, feedback = replay.apply_feedback(f)
        if len(state) <= 10 and len(np.array(state).shape) == 2:
            print ('Training state: {0} \t Training action: {1} \t Feedback: {2} \t Buffer size: {3}'.format(state, action, feedback, replay.f_size()))
        else:
            print ('Training action: {0} \t Feedback: {1} \t Buffer size: {2}'.format(action, feedback, replay.f_size()))

    if replay.f_size() >= batch_size:
        states, actions, feeds = replay.sample_batch(batch_size)
        batch_grad_updates = []
        # print states, actions, feeds
        for s, a, f in zip(states, actions, feeds):
            a_one_hot = np.zeros((1, actor.a_dim))
            a_one_hot[:, a] = 1.
            curr_sa_prob, actor_gradient = actor.predict_with_gradients(s,a_one_hot)
            curr_sa_prob = curr_sa_prob[0][a]
            if not np.isnan(sum(map(lambda x: np.sum(x), actor_gradient))):
                grad = map(lambda x: f * x, actor_gradient)
                batch_grad_updates.append(grad)
        if len(batch_grad_updates) > 0:
            grad_update = reduce(lambda x, y: map(lambda a_one_hot: a_one_hot[0] + a_one_hot[1], zip(x, y)), batch_grad_updates)
            grad_update = list(map(lambda x: x / len(batch_grad_updates), grad_update))
            print ('RM: Sum of mean gradient update: {0}'.format(reduce(lambda x, y: np.mean(x) + np.mean(y), grad_update)))
            actor.train(grad_update)
        else:
            print('Skipping training step')

def replay_matrix_train(actor, replay, f, human_delay, batch_size,betta, state_observation):
    # If we have enough samples for a batch update:

    #print("weights",actor.all_weights)
    if replay.size() >= batch_size:
        size_list = [actor.linear1.weight.shape, actor.linear2.weight.shape]
        # Sample one minibatch of window from the buffer.
        minibatch_of_windows = replay.sample_batch(batch_size)
        

        avg_eligibility_trace_list = [torch.zeros(each_size) for each_size in size_list]
        
        #avg_eligibility_trace = np.zeros((1, actor.a_dim))
        

        # i.e. for each window in the mini-batch
        for w in minibatch_of_windows:
            
            eligibility_trace_lamda = [torch.zeros(each_size) for each_size in size_list]

            # Retrieve final feedback signal F from window w.
            final_feedback_signal = w[-1][-1]
            for state, action, log_prob, feed in w:
                actor.optimizer.zero_grad()
                log_distribution = actor.get_log_prob(state,action)
                pi_theta_t = torch.exp(log_distribution)
                log_distribution.backward()

                layer1_grad = actor.linear1.weight.grad
                layer2_grad = actor.linear2.weight.grad

                eligibility_trace_lamda[0] = replay.eligibility_decay * eligibility_trace_lamda[0] + pi_theta_t / log_prob * layer1_grad
                eligibility_trace_lamda[1] = replay.eligibility_decay * eligibility_trace_lamda[1] + pi_theta_t / log_prob * layer2_grad

            
            for i in range(2):
                avg_eligibility_trace_list[i] += final_feedback_signal * eligibility_trace_lamda[i]
            

            
        
        actor.optimizer.zero_grad()

        cur_distribution = actor(state_observation)[0].probs
        entropy = - torch.sum(cur_distribution * torch.log(cur_distribution))
        entropy.backward()

        avg_eligibility_trace_list[0] = avg_eligibility_trace_list[0] / batch_size + betta * actor.linear1.weight.grad
        avg_eligibility_trace_list[1] = avg_eligibility_trace_list[1] / batch_size + betta * actor.linear2.weight.grad


        #print(actor.linear1.weight)
        actor.linear1.weight = nn.Parameter(actor.linear1.weight+ actor.lr * avg_eligibility_trace_list[0])
        actor.linear2.weight = nn.Parameter(actor.linear2.weight+ actor.lr * avg_eligibility_trace_list[1])

        
            
            
                





def trainIters(env, actor, experiment, n_iters, device):
    # From the deep COACH paper
    lr = 0.00025 
    human_delay = 1
    eligibility_decay = 0.35
    window_size = 10
    mini_batch_size = 16
    # RMSProp optimizer
    replay_limit = 10000
    window = []
    eligibility_replay_buffer = []

    if experiment.feedback_mode == 1:  # if using human feedback, begin collector thread
        feedback_thread = FeedbackCollector()
        feedback_thread.start()
    
    feedback_histogram = defaultdict(int)


    experience = experiment.experience
    #print(experience)
    #optimizerA = optim.Adam(actor.parameters())

    
    for iter in range(n_iters):
        state_observation,_ = env.reset()

        a = None
        ep_steps = 0
        for _ in range(experiment.max_steps):
            env.render()


            with torch.no_grad():
                # Observe current state s_t
                state_observation = torch.tensor(state_observation)

                # Collect stochastic policy given current state
                d, best_action = actor(state_observation)
                # print 'Logits: {0}'.format(l)

                # Randomly sample the stochastic policy
                #TODO:why is this random sample? why not argmax?
                #a = np.random.choice(experiment.num_actions, replace=False, p=d[0])
                prob = d.probs[best_action]

                pair = [state_observation, best_action, prob]
                #print(log_prob)
                experience.append_sap(pair)



            
            # Execute action.
            #for _ in range(experiment.action_repeat):

            next_state_observation, env_reward, terminal, truncated, info = env.step(best_action.item())

            if terminal:
                break



            f = 0.0 #np.random.choice([-1.0,0,1.0])
            if experiment.feedback_mode == 0:
                f += env_reward
            else:
                f += feedback_thread.poll()
            if f != 0.0:
                #feedback_histogram[f] += 1
                print(f"receive human feedback {f}")
            
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
                    replay_matrix_train(actor, experience, f, experiment.human_delay, experiment.batch_size,experiment.betta, state_observation)
                else:
                    print ('Unknown training procedure specification!')
                    sys.exit(1)

            ep_steps += 1
            print ('ep_steps',ep_steps)
            state_observation = next_state_observation
