import os
import argparse

from tensorboardX import SummaryWriter

import rlcard
from rlcard.agents import (
    CFRAgent,
    RandomAgent,
)
from rlcard.utils import (
    set_seed,
    tournament,
    Logger,
    plot_curve,
)
from utils.utils import make_logpath


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

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Initilize CFR Agent
    agent = CFRAgent(
        env,
        os.path.join(
            args.log_dir,
            'cfr_model',
        ),
    )
    agent.load()  # If we have saved model, we first load the model

    # Evaluate CFR against random
    eval_env.set_agents([
        agent,
        RandomAgent(num_actions=env.num_actions),
        RandomAgent(num_actions=env.num_actions),
        RandomAgent(num_actions=env.num_actions),
    ])


    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            agent.train()
            print('\rIteration {}'.format(episode), end='')
            # Evaluate the performance. Play with Random agents.
            rewards = tournament(eval_env, args.num_eval_games)
            eval_reward = rewards[0]
            writer.add_scalar('eval_reward', eval_reward, global_step=episode * 4)

            if episode % args.evaluate_every == 0:
                agent.save()  # Save model
                logger.log_performance(
                    episode,
                    tournament(
                        eval_env,
                        args.num_eval_games
                    )[0]
                )

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path
    # Plot the learning curve
    plot_curve(csv_path, fig_path, 'cfr')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CFR example in RLCard")
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
        default='cfr',
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
        default=1000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=20,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/leduc_holdem_cfr_result/',
    )

    args = parser.parse_args()

    train(args)
