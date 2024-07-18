''' An example of solve Leduc Hold'em with CFR (chance sampling)
'''
import os
import argparse

import torch
from tensorboardX import SummaryWriter

import rlcard
from rlcard.agents import (
    RandomAgent,
)
from rlcard.agents.deep_cfr_agent import DeepCFRAgent
from rlcard.utils import (
    set_seed,
    tournament,
    Logger,
    plot_curve,
)
from utils.utils import make_logpath, save_config


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

    agent = DeepCFRAgent(
        env,
        policy_network_layers=(256, 256),
        advantage_network_layers=(256, 256),
        num_iterations=1,
        num_traversals=1,
        learning_rate=0.0001,
        batch_size_advantage=256,
        batch_size_strategy=256,
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
    for episode in range(args.num_episodes):
        # agent.train()
        agent.solve()
        print('\rIteration {}'.format(episode), end='')
        # Evaluate the performance. Play with Random agents.
        if episode % args.evaluate_every == 0:
            # agent.save()  # Save model
            rewards = tournament(eval_env, args.num_eval_games)
            eval_reward = rewards[0]
            writer.add_scalar('eval_reward', eval_reward, global_step=episode * 2)

    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)


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
        default=20000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=1000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/result/',
    )

    args = parser.parse_args()

    train(args)
