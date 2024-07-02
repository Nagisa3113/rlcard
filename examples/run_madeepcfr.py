''' An example of solve Leduc Hold'em with CFR (chance sampling)
'''
import os
import argparse

from tensorboardX import SummaryWriter

import rlcard
from rlcard.agents import (
    RandomAgent,
)
from rlcard.agents.ma_deep_cfr_agent import MADeepCFRAgent
from rlcard.utils import (
    set_seed,
    tournament,
    Logger,
    plot_curve,
)
from utils import make_logpath, save_config


def train(args):
    # Make environments, CFR only supports Leduc Holdem
    env = rlcard.make(
        'multi-leduc-holdem',
        config={
            'seed': 0,
            'allow_step_back': True,
        }
    )
    eval_env = rlcard.make(
        'multi-leduc-holdem',
        config={
            'seed': 0,
        }
    )

    _, args.log_dir = make_logpath(args.env, args.algorithm)
    writer = SummaryWriter(args.log_dir)
    save_config(args, args.log_dir)
    # Seed numpy, torch, random
    set_seed(args.seed)

    agent = MADeepCFRAgent(
        env,
        policy_network_layers=(64, 64),
        advantage_network_layers=(64, 64),
        num_iterations=200,
        num_traversals=100,
        learning_rate=1e-3,
        batch_size_advantage=None,
        batch_size_strategy=None,
        memory_capacity=1e7)

    # agent.load()  # If we have saved model, we first load the model

    # Evaluate CFR against random
    eval_env.set_agents([
        agent,
        RandomAgent(num_actions=env.num_actions),
        RandomAgent(num_actions=env.num_actions),
        RandomAgent(num_actions=env.num_actions),
    ])
    eval_reward = 0
    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            # agent.train()
            agent.solve()
            print('\rIteration {}'.format(episode), end='')
            # Evaluate the performance. Play with Random agents.
            if episode % args.evaluate_every == 0:
                # agent.save()  # Save model
                rewards = tournament(eval_env, args.num_eval_games)
                eval_reward = rewards[0]
                writer.add_scalar('eval_reward', eval_reward, global_step=episode)
        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path
    # Plot the learning curve
    plot_curve(csv_path, fig_path, 'cfr')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CFR")
    parser.add_argument(
        '--env',
        type=str,
        default='multi-leduc-holdem',
        choices=[
            'leduc-holdem',
            'limit-holdem',
            'no-limit-holdem',
        ],
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='deepcfr',
        # default='dqn',
        # default='nfsp',
        choices=[
            'dqn',
            'nfsp',
            'ddpg',
        ],
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=200,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/leduc_holdem_deepcfr_result/',
    )

    args = parser.parse_args()

    train(args)
