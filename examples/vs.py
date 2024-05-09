import os
import csv
import matplotlib.pyplot as plt
import argparse
import torch

import rlcard
from rlcard.agents import (
    CFRAgent,
    RandomAgent,
)
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    Logger,
    plot_curve,
)
from rlcard.agents import DQNAgent
from rlcard.agents import NFSPAgent


def train(args):
    env = rlcard.make(
        'leduc-holdem',
        config={
            'seed': 0,
            'allow_step_back': True,
        }
    )

    eval_env = rlcard.make(
        'leduc-holdem',
        config={
            'seed': 0,
        }
    )
    # Initilize CFR Agent
    agent = CFRAgent(
        env,
        os.path.join(
            args.log_dir,
            'cfr_model',
        ),
    )
    agent.model_path = r"D:\Documents\PycharmProjects\rlcard\examples\experiments\leduc_holdem_cfr_result\cfr_model"
    agent.load()  # If we have saved model, we first load the model

    DQNagent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[64, 64],
        device=get_device(),
        save_path=args.log_dir,
        save_every=args.save_every
    )

    NFSPagent = NFSPAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers_sizes=[64, 64],
        q_mlp_layers=[64, 64],
        device=get_device(),
        save_path=args.log_dir,
        save_every=args.save_every
    )

    nsave_path = r"D:\Documents\PycharmProjects\rlcard\examples\experiments\leduc_holdem_nfsp_result\model.pth"
    NFSPagent = torch.load(nsave_path)

    dsave_path = r"D:\Documents\PycharmProjects\rlcard\examples\experiments\leduc_holdem_dqn_result\model.pth"
    DQNagent = torch.load(dsave_path)

    # Evaluate CFR against random
    eval_env.set_agents([
        agent,
        DQNagent,
        DQNagent,
        DQNagent,
        # RandomAgent(num_actions=env.num_actions),
        # RandomAgent(num_actions=env.num_actions),
        # RandomAgent(num_actions=env.num_actions),
        # RandomAgent(num_actions=env.num_actions),
    ])

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            # agent.train()
            # print('\rIteration {}'.format(episode), end='')
            # Evaluate the performance. Play with Random agents.
            if episode % args.evaluate_every == 0:
                # agent.save()  # Save model
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
    parser = argparse.ArgumentParser("VS")
    parser.add_argument(
        '--env',
        type=str,
        default='leduc-holdem',
        choices=[
            'blackjack',
        ],
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='nfsp',
        choices=[
            'dqn',
            'nfsp',
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='0',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        # default='experiments/leduc_holdem_dqn_result/',
        default='experiments/werwerw/',
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
