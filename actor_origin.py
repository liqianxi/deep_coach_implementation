import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from replay_matrix import ReplayMatrix 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("MountainCar-v0").unwrapped

state_size = env.observation_space.shape[0]
action_size = env.action_space.n


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns



def wait_human_feedback():
    # Async function
    return 0


def trainIters(actor, critic, n_iters):
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


    experience = ReplayMatrix(replay_limit, eligibility_decay, window_size=10)

    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    for iter in range(n_iters):

        state, _ = env.reset()
        
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        env.reset()

        # Save the trajectory (s_(t-d), a_(t-d), p_(t-d))
        cache_queue = []
       

        for i in count():
            # i is timestamp

            env.render()
            #print("state",state) #(array([-0.52061,  0.     ], dtype=float32), {})

            # Observe current state s_t.
            state = torch.tensor(state).to(device)

            # Select an action that is argmax (dist)
            dist, best_action = actor(state) 

            # Record log_prob p_t = pi(a_t|s_t)
            log_prob = dist.log_prob(best_action).unsqueeze(0)

            # Collect human feedback f_t here.
            #value = critic(state) # TODO: remove all critics.
            human_feedback = wait_human_feedback()

            #print(next_state)

            # Update pairs history.
            if i >= human_delay:
                cache_queue.pop(0)
                
            cache_queue.append([state, best_action, log_prob])

            # Execute the action, get the next state.
            next_state, reward, done, _, _ = env.step(best_action.cpu().numpy())

            if human_feedback != 0:
                # If we received feedback at this timestamp, that means the trainer gave feedback
                # at timestamp t = (current - human_delay), we append the pair 
                # at history[current - human_delay] to the window
                window.append(cache_queue[0] + [human_feedback])

                # Truncate window to the window_length most recent entries 
                # and store in eligibility replay buffer
                truncated_window = window[len(window)-window_size:]
                eligibility_replay_buffer.append(truncated_window)

                # Reset window.
                window = []


            









            
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            state = next_state

            if done:
                print('Iteration: {}, Score: {}'.format(iter, i))
                break


        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
    torch.save(actor, 'model/actor.pkl')
    torch.save(critic, 'model/critic.pkl')
    env.close()


if __name__ == '__main__':

    actor = Actor(state_size, action_size).to(device)

    critic = Critic(state_size, action_size).to(device)
    trainIters(actor, critic, n_iters=100)