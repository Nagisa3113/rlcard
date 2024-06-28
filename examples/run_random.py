''' An example of playing randomly in RLCard
'''
import argparse
import pprint
from datetime import datetime

from tensorboardX import SummaryWriter

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed, tournament


def run(args):
    # Make environment
    env = rlcard.make(
        args.env,
        config={
            'seed': 42,
        }
    )

    writer = SummaryWriter(args.log_dir)
    # Seed numpy, torch, random
    set_seed(42)

    # Set agents
    agent = RandomAgent(num_actions=env.num_actions)
    env.set_agents([agent for _ in range(env.num_players)])

    for episode in range(2000):
        # Generate data from the environment
        trajectories, player_wins = env.run(is_training=False)

        # Evaluate the performance. Play with random agents.
        rewards = tournament(env, 200)
        eval_reward = rewards[0]
        writer.add_scalar('eval_reward', eval_reward, global_step=episode)
    #     logger.log_performance(


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Random example in RLCard")
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
        '--log_dir',
        type=str,
        default=datetime.now().strftime('%Y%m%d_%H%M%S'),
    )

    args = parser.parse_args()

    run(args)
