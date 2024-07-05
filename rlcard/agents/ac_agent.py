import os

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple
import random

from torch.nn.utils import clip_grad_norm_

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])


class ACAgent(object):
    '''
    Approximate clone of rlcard.agents.dqn_agent.DQNAgent
    that depends on PyTorch instead of Tensorflow
    '''

    def __init__(self,
                 replay_memory_size=20000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=1000,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay_steps=20000,
                 batch_size=128,
                 num_actions=None,
                 train_every=10,
                 state_shape=None,
                 mlp_layers=None,
                 learning_rate=0.0005,
                 device=None,
                 save_path=None,
                 hidden_layers_sizes=[64, 64],
                 q_mlp_layers=[64, 64],
                 save_every=float('inf'), ):

        '''
        Q-Learning algorithm for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
            replay_memory_size (int): Size of the replay memory
            replay_memory_init_size (int): Number of random experiences to sample when initializing
              the reply memory.
            update_target_estimator_every (int): Copy parameters from the Q estimator to the
              target estimator every N steps
            discount_factor (float): Gamma discount factor
            epsilon_start (float): Chance to sample a random action when taking an action.
              Epsilon is decayed over time and this is the start value
            epsilon_end (float): The final minimum value of epsilon after decaying is done
            epsilon_decay_steps (int): Number of steps to decay epsilon over
            batch_size (int): Size of batches to sample from the replay memory
            evaluate_every (int): Evaluate every N steps
            num_actions (int): The number of the actions
            state_space (list): The space of the state vector
            train_every (int): Train the network every X steps.
            mlp_layers (list): The layer number and the dimension of each layer in MLP
            learning_rate (float): The learning rate of the DQN agent.
            device (torch.device): whether to use the cpu or gpu
            save_path (str): The path to save the model checkpoints
            save_every (int): Save the model every X training steps
        '''

        self.use_raw = True
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

        # Create replay memory
        self.memory = Memory(replay_memory_size, batch_size)

        # Checkpoint saving parameters
        self.save_path = save_path
        self.save_every = save_every

        self.obs_dim = state_shape[0]
        self.act_dim = num_actions
        self.device = device
        self.a_lr = 0.0001
        self.c_lr = 0.0001
        self.batch_size = 64
        self.gamma = 0.95
        self.tau = 0.001
        self.model_episode = 0
        self.eps = 0.5
        self.decay_speed = 0.99998
        self.output_activation = 'softmax'

        self.actor = Actor(self.obs_dim, 128, self.act_dim).to(self.device)
        self.critic = Critic(self.obs_dim, hidden_dim=128).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)
        self.memory = Memory(memory_size=1e5, batch_size=64)
        self.c_loss = 0
        self.a_loss = 0

    def feed(self, ts):
        ''' Store data in to replay buffer and train the agent. There are two stages.
            In stage 1, populate the memory without training
            In stage 2, train the agent every several timesteps

        Args:
            ts (list): a list of 5 elements that represent the transition
        '''
        (state, action, reward, next_state, done) = tuple(ts)
        self.feed_memory(state['obs'], action, reward, next_state['obs'], list(next_state['legal_actions'].keys()),
                         done)
        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size
        if tmp >= 0 and tmp % self.train_every == 0:
            self.train()

    def step(self, state):
        ''' Predict the action for genrating training data but
            have the predictions disconnected from the computation graph

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
        '''
        q_values = self.predict(state)
        # epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]
        legal_actions = list(state['legal_actions'].keys())
        for i in range(q_values.size):
            if i in legal_actions:
                q_values[i] = q_values[i]
            else:
                q_values[i] = 0

        return q_values

    def eval_step(self, state):
        ''' Predict the action for evaluation purpose.

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
            info (dict): A dictionary containing information
        '''
        q_values = self.predict(state)
        best_action = np.argmax(q_values)

        info = {}
        info['values'] = {state['raw_legal_actions'][i]: float(q_values[list(state['legal_actions'].keys())[i]]) for i
                          in range(len(state['legal_actions']))}

        return best_action, info

    def predict(self, state):
        ''' Predict the masked Q-values

        Args:
            state (numpy.array): current state

        Returns:
            q_values (numpy.array): a 1-d array where each entry represents a Q value
        '''

        obs = np.expand_dims(state['obs'], 0)
        obs = torch.Tensor(obs).to(self.device)
        q_values = self.actor(obs).cpu().detach().numpy()[0]
        masked_q_values = -np.inf * np.ones(self.num_actions, dtype=float)
        legal_actions = list(state['legal_actions'].keys())
        masked_q_values[legal_actions] = q_values[legal_actions]

        return masked_q_values

    def train(self):
        ''' Train the network

        Returns:
            loss (float): The loss of the current batch.
        '''
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, legal_actions_batch = self.memory.sample()

        state_batch = torch.Tensor(state_batch).to(self.device)
        action_batch = torch.Tensor(action_batch).to(self.device)
        action_batch = torch.argmax(action_batch, dim=1).unsqueeze(1)
        reward_batch = torch.Tensor(reward_batch).reshape(-1, 1).to(self.device)
        next_state_batch = torch.Tensor(next_state_batch).to(self.device)
        done_batch = torch.Tensor(done_batch).reshape(-1, 1).to(self.device)

        td_target = reward_batch + self.gamma * self.critic(next_state_batch) * (1 - done_batch)
        td_delta = td_target - self.critic(state_batch)
        actions = self.actor(state_batch).gather(1, action_batch)
        log_probs = torch.log(actions)
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss = torch.mean(
            F.mse_loss(self.critic(state_batch), td_target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), 1)
        clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数

        self.train_t += 1

    def feed_memory(self, state, action, reward, next_state, legal_actions, done):
        ''' Feed transition to memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        self.memory.save(state, action, reward, next_state, legal_actions, done)

    def set_device(self, device):
        self.device = device

    def checkpoint_attributes(self):
        '''
        Return the current checkpoint attributes (dict)
        Checkpoint attributes are used to save and restore the model in the middle of training
        Saves the model state dict, optimizer state dict, and all other instance variables
        '''

        return {
            'agent_type': 'DDPGAgent',
            # 'q_estimator': self.q_estimator.checkpoint_attributes(),
            'memory': self.memory.checkpoint_attributes(),
            'total_t': self.total_t,
            'train_t': self.train_t,
            'replay_memory_init_size': self.replay_memory_init_size,
            'update_target_estimator_every': self.update_target_estimator_every,
            'discount_factor': self.discount_factor,
            'epsilon_start': self.epsilons.min(),
            'epsilon_end': self.epsilons.max(),
            'epsilon_decay_steps': self.epsilon_decay_steps,
            'batch_size': self.batch_size,
            'num_actions': self.num_actions,
            'train_every': self.train_every,
            'device': self.device,
            'save_path': self.save_path,
            'save_every': self.save_every
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        '''
        Restore the model from a checkpoint

        Args:
            checkpoint (dict): the checkpoint attributes generated by checkpoint_attributes()
        '''

        print("\nINFO - Restoring model from checkpoint...")
        agent_instance = cls(
            replay_memory_size=checkpoint['memory']['memory_size'],
            replay_memory_init_size=checkpoint['replay_memory_init_size'],
            update_target_estimator_every=checkpoint['update_target_estimator_every'],
            discount_factor=checkpoint['discount_factor'],
            epsilon_start=checkpoint['epsilon_start'],
            epsilon_end=checkpoint['epsilon_end'],
            epsilon_decay_steps=checkpoint['epsilon_decay_steps'],
            batch_size=checkpoint['batch_size'],
            num_actions=checkpoint['num_actions'],
            state_shape=checkpoint['q_estimator']['state_shape'],
            train_every=checkpoint['train_every'],
            mlp_layers=checkpoint['q_estimator']['mlp_layers'],
            learning_rate=checkpoint['q_estimator']['learning_rate'],
            device=checkpoint['device'],
            save_path=checkpoint['save_path'],
            save_every=checkpoint['save_every'],
        )

        agent_instance.total_t = checkpoint['total_t']
        agent_instance.train_t = checkpoint['train_t']

        agent_instance.memory = Memory.from_checkpoint(checkpoint['memory'])

        return agent_instance

    def save_checkpoint(self, path, filename='checkpoint_dqn.pt'):
        ''' Save the model checkpoint (all attributes)

        Args:
            path (str): the path to save the model
            filename(str): the file name of checkpoint
        '''
        torch.save(self.checkpoint_attributes(), os.path.join(path, filename))


class Critic(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Actor(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


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
