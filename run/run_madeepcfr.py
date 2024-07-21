import os
import argparse

import torch
from tensorboardX import SummaryWriter

import rlcard
from rlcard.agents import (
    RandomAgent,
)
from rlcard.agents.ma_deep_cfr_agent import MADeepCFRAgent
from rlcard.utils import (
    tournament,
)
from utils.utils import make_logpath, save_config


def train(args):
    # Make environments, CFR only supports Leduc Holdem
    env = rlcard.make(
        'multi-leduc-holdem',
        config={
            # 'seed': 0,
            'allow_step_back': True,
        }
    )
    eval_env = rlcard.make(
        'multi-leduc-holdem',
        config={
            # 'seed': 0,
        }
    )

    _, args.log_dir = make_logpath(args.env, args.algorithm)
    writer = SummaryWriter(args.log_dir)
    save_config(args, args.log_dir)
    # Seed numpy, torch, random
    # set_seed(args.seed)

    agent = MADeepCFRAgent(
        env,
        policy_network_layers=(args.policy_network_layers, args.policy_network_layers),
        advantage_network_layers=(args.advantage_network_layers, args.advantage_network_layers),
        num_iterations=1,
        num_traversals=1,
        learning_rate=args.learning_rate,
        batch_size_advantage=args.batch_size_advantage,
        batch_size_strategy=args.batch_size_strategy,
        policy_network_train_steps=1,
        advantage_network_train_steps=1,
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
        evey = args.evaluate_every
        if episode % args.evaluate_every == 0:
            # agent.save()  # Save model
            rewards = tournament(eval_env, args.num_eval_games)
            eval_reward = rewards[0]
            writer.add_scalar('eval_reward', eval_reward, global_step=episode * 4)

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
        default='madeepcfr',
        # default='dqn',
        # default='nfsp',
        choices=[
            'dqn',
            'nfsp',
            'ddpg',
        ],
    )
    # parser.add_argument(
    #     '--seed',
    #     type=int,
    #     default=42,
    # )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=5000,
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
    parser.add_argument(
        '--policy_network_layers',
        type=int,
        default=256,
    )
    parser.add_argument(
        '--advantage_network_layers',
        type=int,
        default=256,
    )
    parser.add_argument(
        '--batch_size_advantage',
        type=int,
        default=256,
    )
    parser.add_argument(
        '--batch_size_strategy',
        type=int,
        default=256,
    )
    parser.add_argument(
        '--learning_rate',
        type=int,
        default=0.0003,
    )

    args = parser.parse_args()

    train(args)
