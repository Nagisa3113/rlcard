import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple
import random

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])


class ActorCriticAgent(nn.Module):
    ''' A2C网络模型，包含一个Actor和Critic
    '''

    def __init__(self, input_dim, output_dim, hidden_dim,
                 replay_memory_size=20000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=1000,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=20000,
                 batch_size=32,
                 num_actions=2,
                 state_shape=None,
                 train_every=1,
                 mlp_layers=None,
                 learning_rate=0.00005,
                 device=None,
                 save_path=None,
                 save_every=float('inf'),
                 ):
        super(ActorCriticAgent, self).__init__()


        self.use_raw = False
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.train_every = train_every

        # Torch device
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        # The epsilon decay scheduler
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        # Create estimators
        # self.q_estimator = Estimator(num_actions=num_actions, learning_rate=learning_rate, state_shape=state_shape, \
        #                              mlp_layers=mlp_layers, device=self.device)
        # self.target_estimator = Estimator(num_actions=num_actions, learning_rate=learning_rate, state_shape=state_shape, \
        #                                   mlp_layers=mlp_layers, device=self.device)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 0)
        )

        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=0),
        )

        # Create replay memory
        self.memory = Memory(replay_memory_size, batch_size)

        # Checkpoint saving parameters
        self.save_path = save_path
        self.save_every = save_every

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value

    def feed(self, ts):
        (state, action, reward, next_state, done) = tuple(ts)
        self.feed_memory(state['obs'], action, reward, next_state['obs'], list(next_state['legal_actions'].keys()), done)
        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size
        if tmp>=0 and tmp%self.train_every == 0:
            self.train()



class Memory(object):
    ''' Memory for saving transitions
    '''

    def __init__(self, memory_size, batch_size):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
        '''
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, legal_actions, done):
        ''' Save transition into memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, done, legal_actions)
        self.memory.append(transition)

    def sample(self):
        ''' Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        '''
        samples = random.sample(self.memory, self.batch_size)
        samples = tuple(zip(*samples))
        return tuple(map(np.array, samples[:-1])) + (samples[-1],)

    def checkpoint_attributes(self):
        ''' Returns the attributes that need to be checkpointed
        '''

        return {
            'memory_size': self.memory_size,
            'batch_size': self.batch_size,
            'memory': self.memory
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        '''
        Restores the attributes from the checkpoint

        Args:
            checkpoint (dict): the checkpoint dictionary

        Returns:
            instance (Memory): the restored instance
        '''

        instance = cls(checkpoint['memory_size'], checkpoint['batch_size'])
        instance.memory = checkpoint['memory']
        return instance
