''' An example of training a reinforcement learning agent on the environments in RLCard
'''
import os
import argparse
from datetime import datetime

import torch
from tensorboardX import SummaryWriter

import rlcard
from rlcard.agents import RandomAgent
from rlcard.agents.maddpg_agent import MADDPGAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)
from utils.utils import make_logpath, save_config


def train(args):
    # Check whether gpu is available
    device = get_device()
    _, args.log_dir = make_logpath(args.env, args.algorithm)
    writer = SummaryWriter(args.log_dir)
    save_config(args, args.log_dir)
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
        }
    )

    if args.load_checkpoint_path != "":
        agent = MADDPGAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path))
    else:
        agent = MADDPGAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            device=device,
            save_path=args.log_dir,
            save_every=args.save_every
        )

    agents = [agent,
              agent,
              agent,
              agent,
              ]
    env.set_agents(agents)

    eval_env = rlcard.make(
        'multi-leduc-holdem',
        config={
            'seed': 0,
        }
    )
    eval_env.set_agents([
        agent,
        RandomAgent(num_actions=env.num_actions),
        RandomAgent(num_actions=env.num_actions),
        RandomAgent(num_actions=env.num_actions),
    ])

    eval_reward = 0

    for episode in range(args.num_episodes):
        trajectories, payoffs = env.run(is_training=True)
        trajectories = reorganize(trajectories, payoffs)

        for ts in trajectories:
            for s in ts:
                agent.feed(s)
            # agent.feed(ts)

        # Evaluate the performance. Play with random agents.
        if episode > 0 and episode % args.evaluate_every == 0:
            rewards = tournament(eval_env, args.num_eval_games)
            eval_reward = rewards[0]
            writer.add_scalar('eval_reward', eval_reward, global_step=episode)

    # Save model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
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
        default='maddpg',
        choices=[
            'dqn',
            'nfsp',
            'ddpg',
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
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
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=datetime.now().strftime('%Y%m%d_%H%M%S'),
    )

    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        default="",
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=-1)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)
