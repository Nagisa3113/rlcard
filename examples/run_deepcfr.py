''' An example of solve Leduc Hold'em with CFR (chance sampling)
'''
import os
import argparse

import rlcard
from rlcard.agents import (
    CFRAgent,
    RandomAgent,
)
from rlcard.agents.deep_cfr_agent import DeepCFRAgent
from rlcard.utils import (
    set_seed,
    tournament,
    Logger,
    plot_curve,
)
import tensorflow as tf

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

    # Seed numpy, torch, random
    set_seed(args.seed)
    sess = tf.compat.v1.InteractiveSession()

    agent = DeepCFRAgent(session=sess,
                    scope='deepcfr',
                    env=env,
                    policy_network_layers=(4, 4),
                    advantage_network_layers=(4, 4),
                    num_traversals=1,
                    num_step=1,
                    learning_rate=1e-4,
                    batch_size_advantage=10,
                    batch_size_strategy=10,
                    memory_capacity=int(1e7))



    # agent.load()  # If we have saved model, we first load the model

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

    sess.close()
    tf.reset_default_graph()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CFR example in RLCard")
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
        default=20,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/leduc_holdem_cfr_result/',
    )

    args = parser.parse_args()

    train(args)
