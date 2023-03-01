import gymnasium as gym
from experiment import Experiment
from actor import Actor
from deep_coach import trainIters
import torch

def main():
    #env = gym.make("MountainCar-v0",render_mode="human").env
    env = gym.make("MountainCar-v0").env
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("state,action",state_size,action_size) 2,3
    policy_lr = 0.00025 
    human_delay = 1
    eligibility_decay = 0.35
    window_size = 10 # Set as default in replay_matrix or replay_buffer

    # RMSProp optimizer
    replay_limit = 10000
    '''
        :param state_dim: Dimensionality of the state space
        :param num_actions: Dimensionality of the action space
        :param train_mode: Training procedure (0 for eligibility trace; 1 for replay memory)
        :param feedback_mode: Feedback collection procedure (0 for environment; 1 for keyboard thread)
        :param action_map: Mapping from network action space to environment action space
        :param human_delay: Number of timesteps between feedback collection and application to transition
        :param max_episodes: Maximum number of learning episodes to run
        :param max_steps: Maximum number of agent steps per learning episode
        :param learning_rate: Learning rate used for training actor network
        :param hidden_size: Number of hidden units used in each hidden layer of actor network
        :param Lambda: Exponential weight decay for eligibility trace
        :param alpha: Entropy regularization coefficient
        :param tau: Exponential weight decay for target network updates
        :param batch_size: Number of samples per minibatch of actor network training (only for replay memory training)
        :param action_select: Number of timesteps between actor receiving new observations to act on
        :param action_repeat: Number of times to repeat each action selected by the actor
        :param sleep_time: Number of seconds to sleep between each step
        :param replay_limit: Maximum number of transition to be stored in replay memory
        :param gamma: Threshold of hinge entropy regularization loss
        :param temperature: Temperature to use for Boltzmann softmax
        :param grad_norm_clip: Clipping threshold for gradient norms
        :param replacing: Boolean for using a replacing or accumulating eligibility trace
        :param plot_kld: Flag for plotting KLD between the current policy and the uniform distribution
        :param plot_feed: Flag for plotting feedback received by agent over time
        :param meta_file: Path to .meta file for reloading a pre-trained convolutional encoder
        :param malmo_mission: Path to Malmo mission XML to use for experiment
    '''
    exp = Experiment(state_dim=state_size, num_actions=action_size, 
                      learning_rate=policy_lr, Lambda=eligibility_decay, alpha=0.0,
                      train_mode=2, hidden_size=30, human_delay=human_delay, 
                      gamma=0.0, grad_norm_clip=None, temperature=None, 
                      replacing=False, replay_limit=replay_limit,feedback_mode=1)

    actor = Actor(state_size, action_size).to(device)

    trainIters(env, actor, exp, n_iters=200, device=device)


if __name__ == '__main__':
    main()