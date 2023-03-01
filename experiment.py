import sys

from replay_buffer import ReplayBuffer
from replay_matrix import ReplayMatrix
#from utils.utils import IdentityDict

class IdentityDict(dict):
    def __missing__(self, key):
        return key
class Experiment:

    def __init__(self, state_dim, num_actions, train_mode=1, feedback_mode=0, action_map=IdentityDict(), human_delay=0,
                 max_episodes=100, max_steps=10000, learning_rate=0.01, hidden_size=100, Lambda=0.95, alpha=0.01,betta=1.5,
                 tau=0.001, batch_size=32, action_select=1, action_repeat=1, sleep_time=0.01, replay_limit=10000,
                 gamma=0.0, temperature=None, grad_norm_clip=None, replacing=True, plot_kld=True, plot_feed=True,
                 meta_file=None, malmo_mission=None, name=''):
        '''
        :param betta: entropy regularization coefficient.
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
        self.betta = betta
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.action_map = action_map
        self.train_mode = train_mode
        self.feedback_mode = feedback_mode
        self.human_delay = human_delay
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.Lambda = Lambda
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size
        self.action_select = action_select
        self.action_repeat = action_repeat
        self.sleep_time = sleep_time
        self.replay_limit = replay_limit
        self.gamma = gamma
        self.temperature = temperature
        self.grad_norm_clip = grad_norm_clip
        self.replacing = replacing
        self.plot_kld = plot_kld
        self.plot_feed = plot_feed
        self.meta_file = meta_file
        self.malmo_mission = malmo_mission
        self.experience = None
        self.name = name

        self.init_experience()

    def init_experience(self):
        if self.train_mode == 0:
            print ('Using list for caching experiences')
            self.experience = []
        elif self.train_mode == 1:
            print ('Using replay memory buffer for caching experiences')
            self.experience = ReplayBuffer(self.replay_limit, self.human_delay)
        elif self.train_mode == 2:
            print ('Using replay matrix for caching experiences')
            self.experience = ReplayMatrix(self.replay_limit, self.Lambda)
        else:
            print ('Unknown training procedure {0} specified!....Exiting'.format(self.train_mode))
            sys.exit(1)

    def __str__(self):
        if len(self.name) == 0:
            fields = [self.state_dim, self.num_actions, self.train_mode, self.feedback_mode,
                      self.human_delay, self.max_episodes, self.max_steps, self.learning_rate, self.hidden_size,
                      self.Lambda, self.alpha, self.betta,self.tau, self.batch_size, self.action_select, self.action_repeat,
                      self.sleep_time, self.replay_limit, self.gamma, self.temperature, self.grad_norm_clip, self.replacing,
                      self.malmo_mission.split('/')[-1]]
            fields = map(str, fields)
            return '_'.join(fields)
        else:
            return self.name