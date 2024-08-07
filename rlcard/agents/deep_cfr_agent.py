# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements Deep CFR Algorithm.

See https://arxiv.org/abs/1811.00164.

The algorithm defines an `advantage` and `strategy` networks that compute
advantages used to do regret matching across information sets and to approximate
the strategy profiles of the game.  To train these networks a fixed ring buffer
(other data structures may be used) memory is used to accumulate samples to
train the networks.
"""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import collections
import math
import random
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlcard.utils.utils import remove_illegal

AdvantageMemory = collections.namedtuple(
    "AdvantageMemory", "info_state iteration advantage action")

StrategyMemory = collections.namedtuple(
    "StrategyMemory", "info_state iteration strategy_action_probs")


class DeepCFRAgent():
    """Implements a solver for the Deep CFR Algorithm with PyTorch.

    See https://arxiv.org/abs/1811.00164.

    Define all networks and sampling buffers/memories.  Derive losses & learning
    steps. Initialize the game state and algorithmic variables.

    Note: batch sizes default to `None` implying that training over the full
          dataset in memory is done by default.  To sample from the memories you
          may set these values to something less than the full capacity of the
          memory.
    """

    def __init__(self,
                 env,
                 policy_network_layers=(256, 256),
                 advantage_network_layers=(128, 128),
                 num_iterations: int = 100,
                 num_traversals: int = 20,
                 learning_rate: float = 1e-4,
                 batch_size_advantage=None,
                 batch_size_strategy=None,
                 memory_capacity: int = int(1e6),
                 policy_network_train_steps: int = 10,
                 advantage_network_train_steps: int = 10,
                 reinitialize_advantage_networks: bool = True):
        """Initialize the Deep CFR algorithm.

        Args:
          game: Open Spiel game.
          policy_network_layers: (list[int]) Layer sizes of strategy net MLP.
          advantage_network_layers: (list[int]) Layer sizes of advantage net MLP.
          num_iterations: (int) Number of training iterations.
          num_traversals: (int) Number of traversals per iteration.
          learning_rate: (float) Learning rate.
          batch_size_advantage: (int or None) Batch size to sample from advantage
            memories.
          batch_size_strategy: (int or None) Batch size to sample from strategy
            memories.
          memory_capacity: Number af samples that can be stored in memory.
          policy_network_train_steps: Number of policy network training steps (per
            iteration).
          advantage_network_train_steps: Number of advantage network training steps
            (per iteration).
          reinitialize_advantage_networks: Whether to re-initialize the advantage
            network before training on each iteration.
        """
        self._env = env
        self._batch_size_advantage = batch_size_advantage
        self._batch_size_strategy = batch_size_strategy
        self._policy_network_train_steps = policy_network_train_steps
        self._advantage_network_train_steps = advantage_network_train_steps
        self._num_players = env.num_players
        self._root_node = self._env.reset()
        self._embedding_size = 68
        self._num_iterations = num_iterations
        self._num_traversals = num_traversals
        self._reinitialize_advantage_networks = reinitialize_advantage_networks
        self._num_actions = self._env.num_actions
        self._iteration = 1
        self.use_raw = False
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Define strategy network, loss & memory.
        self._strategy_memories = ReservoirBuffer(memory_capacity)
        self._policy_network = MLP(self._embedding_size,
                                   list(policy_network_layers),
                                   self._num_actions).to(self.device)
        # Illegal actions are handled in the traversal code where expected payoff
        # and sampled regret is computed from the advantage networks.
        self._policy_sm = nn.Softmax(dim=-1).to(self.device)
        self._loss_policy = nn.MSELoss()
        self._optimizer_policy = torch.optim.Adam(
            self._policy_network.parameters(), lr=learning_rate)

        # Define advantage network, loss & memory. (One per player)
        self._advantage_memories = [
            ReservoirBuffer(memory_capacity) for _ in range(self._num_players)
        ]
        self._advantage_networks = [
            MLP(self._embedding_size, list(advantage_network_layers),
                self._num_actions).to(self.device) for _ in range(self._num_players)
        ]
        self._loss_advantages = nn.MSELoss(reduction="mean")
        self._optimizer_advantages = []
        for p in range(self._num_players):
            self._optimizer_advantages.append(
                torch.optim.Adam(
                    self._advantage_networks[p].parameters(), lr=learning_rate))
        self._learning_rate = learning_rate

    @property
    def advantage_buffers(self):
        return self._advantage_memories

    @property
    def strategy_buffer(self):
        return self._strategy_memories

    def clear_advantage_buffers(self):
        for p in range(self._num_players):
            self._advantage_memories[p].clear()

    def reinitialize_advantage_network(self, player):
        self._advantage_networks[player].reset()
        self._optimizer_advantages[player] = torch.optim.Adam(
            self._advantage_networks[player].parameters(), lr=self._learning_rate)

    def reinitialize_advantage_networks(self):
        for p in range(self._num_players):
            self.reinitialize_advantage_network(p)

    def solve(self):
        """Solution logic for Deep CFR.

        Traverses the game tree, while storing the transitions for training
        advantage and policy networks.

        Returns:
          1. (nn.Module) Instance of the trained policy network for inference.
          2. (list of floats) Advantage network losses for
            each player during each iteration.
          3. (float) Policy loss.
        """
        advantage_losses = collections.defaultdict(list)

        for _ in range(self._num_iterations):
            for p in range(self._num_players):
                for _ in range(self._num_traversals):
                    init_state, init_player = self._env.reset()
                    self._root_node = init_state
                    self._traverse_game_tree(self._root_node, p)
                # if self._reinitialize_advantage_networks:
                # Re-initialize advantage network for player and train from scratch.
                # self.reinitialize_advantage_network(p)
                # Re-initialize advantage networks and train from scratch.
                advantage_losses[p].append(self._learn_advantage_network(p))
            self._iteration += 1

        # Train policy network.
        policy_loss = self._learn_strategy_network()
        return self._policy_network, advantage_losses, policy_loss

    def eval_step(self, state):
        ''' Predict the action given state for evaluation

        args:
            state (dict): current state

        returns:
            action (int): an action id
        '''
        obs = state['obs']
        legal_actions = list(state['legal_actions'])
        action_prob = self.action_probabilities(obs)
        action_prob = remove_illegal(action_prob, legal_actions)
        action_prob /= action_prob.sum()
        action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
        return action, action_prob

    def _traverse_game_tree(self, state, player):
        """Performs a traversal of the game tree.

        Over a traversal the advantage and strategy memories are populated with
        computed advantage values and matched regrets respectively.

        Args:
          state: Current OpenSpiel game state.
          player: (int) Player index for this traversal.

        Returns:
          (float) Recursively returns expected payoffs for each action.
        """
        expected_payoff = collections.defaultdict(float)
        current_player = self._env.get_player_id()
        actions = state['legal_actions']

        if self._env.is_over():
            # Terminal state get returns.
            payoffs = self._env.get_payoffs()
            return payoffs[player]

        if current_player == player:
            sampled_regret = collections.defaultdict(float)
            # Update the policy over the info set & actions via regret matching.
            _, strategy = self._sample_action_from_advantage(state, player)
            cur_step = self._env.timestep
            for action in actions:
                while self._env.timestep != cur_step:
                    self._env.step_back()
                child_state, child = self._env.step(action)
                expected_payoff[action] = self._traverse_game_tree(child_state, player)
                self._env.step_back()

            cfv = 0
            for a_ in actions:
                cfv += strategy[a_] * expected_payoff[a_]

            for action in actions:
                sampled_regret[action] = expected_payoff[action]
                sampled_regret[action] -= cfv
            sampled_regret_arr = [0] * self._num_actions

            for action in sampled_regret:
                sampled_regret_arr[action] = sampled_regret[action]

            adv_mem = AdvantageMemory(state['obs'].flatten(), self._iteration,
                                      sampled_regret_arr, action)
            self._advantage_memories[player].add(adv_mem)
            return cfv
        else:
            other_player = current_player
            _, strategy = self._sample_action_from_advantage(state, other_player)
            # Recompute distribution for numerical errors.
            probs = np.array(strategy)
            probs /= probs.sum()
            sampled_action = np.random.choice(range(self._num_actions), p=probs)
            self._strategy_memories.add(
                StrategyMemory(
                    state['obs'].flatten(), self._iteration, strategy))

            child_state, _ = self._env.step(sampled_action)
            return self._traverse_game_tree(child_state, player)

    def _sample_action_from_advantage(self, state, player):
        """Returns an info state policy by applying regret-matching.

        Args:
          state: Current OpenSpiel game state.
          player: (int) Player index over which to compute regrets.

        Returns:
          1. (list) Advantage values for info state actions indexed by action.
          2. (list) Matched regrets, prob for actions indexed by action.
        """
        info_state = state['obs'].flatten()
        legal_actions = state['legal_actions']
        with torch.no_grad():
            state_tensor = torch.FloatTensor(np.expand_dims(info_state, axis=0)).to(self.device)
            raw_advantages = self._advantage_networks[player](state_tensor).cpu().detach().numpy()[0]
        advantages = [max(0., advantage) for advantage in raw_advantages]
        cumulative_regret = np.sum([advantages[action] for action in legal_actions])
        matched_regrets = np.array([0.] * self._num_actions)
        if cumulative_regret > 0.:
            for action in legal_actions:
                matched_regrets[action] = advantages[action] / cumulative_regret
        else:
            matched_regrets[max(legal_actions, key=lambda a: raw_advantages[a])] = 1
        return advantages, matched_regrets

    @staticmethod
    def _flatten_state(state):
        ''' Flatten the given state represenatation

        Args:
            state (tf.placeholder): current state

        Returns:
            flattened (tensor): flattened state
        '''
        shape = state.get_shape().as_list()
        dim = np.prod(shape[1:])
        flattened = torch.reshape(state, [-1, dim])
        return flattened

    def action_probabilities(self, state):
        """Computes action probabilities for the current player in state.

        Args:
          state: (pyspiel.State) The state to compute probabilities for.

        Returns:
          (dict) action probabilities for a single batch.
        """
        info_state_vector = state
        # legal_actions = state['legal_actions']
        # if len(info_state_vector.shape) == 1:
        #     info_state_vector = np.expand_dims(info_state_vector, axis=0)
        with torch.no_grad():
            logits = self._policy_network(torch.FloatTensor(info_state_vector).to(self.device))
            probs = self._policy_sm(logits).cpu().numpy()
        return probs
        # {action: probs[0][action] for action in legal_actions}

    def _learn_advantage_network(self, player):
        """Compute the loss on sampled transitions and perform a Q-network update.

        If there are not enough elements in the buffer, no loss is computed and
        `None` is returned instead.

        Args:
          player: (int) player index.

        Returns:
          (float) The average loss over the advantage network.
        """
        for _ in range(self._advantage_network_train_steps):

            if self._batch_size_advantage:
                if self._batch_size_advantage > len(self._advantage_memories[player]):
                    ## Skip if there aren't enough samples
                    return None
                samples = self._advantage_memories[player].sample(
                    self._batch_size_advantage)
            else:
                samples = self._advantage_memories[player]
            info_states = []
            advantages = []
            iterations = []
            for s in samples:
                info_states.append(s.info_state)
                advantages.append(s.advantage)
                iterations.append([s.iteration])
            # Ensure some samples have been gathered.
            if not info_states:
                return None
            self._optimizer_advantages[player].zero_grad()
            advantages = torch.FloatTensor(np.array(advantages)).to(self.device)
            iters = torch.FloatTensor(np.sqrt(np.array(iterations))).to(self.device)
            infos = torch.FloatTensor(np.array(info_states)).to(self.device)
            outputs = self._advantage_networks[player](infos)
            loss_advantages = self._loss_advantages(iters * outputs,
                                                    iters * advantages)
            loss_advantages.backward()
            self._optimizer_advantages[player].step()

        return loss_advantages.cpu().detach().numpy()

    def _learn_strategy_network(self):
        """Compute the loss over the strategy network.

        Returns:
          (float) The average loss obtained on this batch of transitions or `None`.
        """
        for _ in range(self._policy_network_train_steps):
            if self._batch_size_strategy:
                if self._batch_size_strategy > len(self._strategy_memories):
                    ## Skip if there aren't enough samples
                    return None
                samples = self._strategy_memories.sample(self._batch_size_strategy)
            else:
                samples = self._strategy_memories
            info_states = []
            action_probs = []
            iterations = []
            for s in samples:
                info_states.append(s.info_state)
                action_probs.append(s.strategy_action_probs)
                iterations.append([s.iteration])

            self._optimizer_policy.zero_grad()
            iters = torch.FloatTensor(np.sqrt(np.array(iterations))).to(self.device)
            ac_probs = torch.FloatTensor(np.array(np.squeeze(action_probs))).to(self.device)
            logits = self._policy_network(torch.FloatTensor(np.array(info_states)).to(self.device))
            outputs = self._policy_sm(logits)
            loss_strategy = self._loss_policy(iters * outputs, iters * ac_probs)
            loss_strategy.backward()
            self._optimizer_policy.step()

        return loss_strategy.cpu().detach().numpy()


class SonnetLinear(nn.Module):
    """A Sonnet linear module.

    Always includes biases and only supports ReLU activations.
    """

    def __init__(self, in_size, out_size, activate_relu=True):
        """Creates a Sonnet linear layer.

        Args:
          in_size: (int) number of inputs
          out_size: (int) number of outputs
          activate_relu: (bool) whether to include a ReLU activation layer
        """
        super(SonnetLinear, self).__init__()
        self._activate_relu = activate_relu
        self._in_size = in_size
        self._out_size = out_size
        # stddev = 1.0 / math.sqrt(self._in_size)
        # mean = 0
        # lower = (-2 * stddev - mean) / stddev
        # upper = (2 * stddev - mean) / stddev
        # # Weight initialization inspired by Sonnet's Linear layer,
        # # which cites https://arxiv.org/abs/1502.03167v3
        # # pytorch default: initialized from
        # # uniform(-sqrt(1/in_features), sqrt(1/in_features))
        self._weight = None
        self._bias = None
        self.reset()

    def forward(self, tensor):
        y = F.linear(tensor, self._weight, self._bias)
        return F.relu(y) if self._activate_relu else y

    def reset(self):
        stddev = 1.0 / math.sqrt(self._in_size)
        mean = 0
        lower = (-2 * stddev - mean) / stddev
        upper = (2 * stddev - mean) / stddev
        # Weight initialization inspired by Sonnet's Linear layer,
        # which cites https://arxiv.org/abs/1502.03167v3
        # pytorch default: initialized from
        # uniform(-sqrt(1/in_features), sqrt(1/in_features))
        self._weight = nn.Parameter(
            torch.Tensor(
                stats.truncnorm.rvs(
                    lower,
                    upper,
                    loc=mean,
                    scale=stddev,
                    size=[self._out_size, self._in_size])))
        self._bias = nn.Parameter(torch.zeros([self._out_size]))


class MLP(nn.Module):
    """A simple network built from nn.linear layers."""

    def __init__(self,
                 input_size,
                 hidden_sizes,
                 output_size,
                 activate_final=False):
        """Create the MLP.

        Args:
          input_size: (int) number of inputs
          hidden_sizes: (list) sizes (number of units) of each hidden layer
          output_size: (int) number of outputs
          activate_final: (bool) should final layer should include a ReLU
        """

        super(MLP, self).__init__()
        self._layers = []
        # Hidden layers
        for size in hidden_sizes:
            self._layers.append(SonnetLinear(in_size=input_size, out_size=size))
            input_size = size
        # Output layer
        self._layers.append(
            SonnetLinear(
                in_size=input_size,
                out_size=output_size,
                activate_relu=activate_final))

        self.model = nn.ModuleList(self._layers)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

    def reset(self):
        for layer in self._layers:
            layer.reset()


class ReservoirBuffer(object):
    """Allows uniform sampling over a stream of data.

    This class supports the storage of arbitrary elements, such as observation
    tensors, integer actions, etc.
    See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
    """

    def __init__(self, reservoir_buffer_capacity):
        self._reservoir_buffer_capacity = reservoir_buffer_capacity
        self._data = []
        self._add_calls = 0

    def add(self, element):
        """Potentially adds `element` to the reservoir buffer.

        Args:
          element: data to be added to the reservoir buffer.
        """
        if len(self._data) < self._reservoir_buffer_capacity:
            self._data.append(element)
        else:
            idx = np.random.randint(0, self._add_calls + 1)
            if idx < self._reservoir_buffer_capacity:
                self._data[idx] = element
        self._add_calls += 1

    def sample(self, num_samples):
        """Returns `num_samples` uniformly sampled from the buffer.

        Args:
          num_samples: `int`, number of samples to draw.

        Returns:
          An iterable over `num_samples` random elements of the buffer.
        Raises:
          ValueError: If there are less than `num_samples` elements in the buffer
        """
        if len(self._data) < num_samples:
            raise ValueError("{} elements could not be sampled from size {}".format(
                num_samples, len(self._data)))
        return random.sample(self._data, num_samples)

    def clear(self):
        self._data = []
        self._add_calls = 0

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)
