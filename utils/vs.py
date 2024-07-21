import os
import argparse
import torch

import rlcard
from rlcard.agents import (
    CFRAgent,
    RandomAgent,
    ACAgent,
    MADeepCFRAgent,
    DQNAgent,
    NFSPAgent)

from rlcard.utils import (
    get_device,
    tournament_wp,
)


def evaluate(args):
    env = rlcard.make(
        'multi-leduc-holdem',
        config={
            'seed': 0,
        }
    )

    device = get_device()

    cfr_agent = CFRAgent(
        env,
        os.path.join(
            args.log_dir,
            'cfr_model',
        ),
    )

    ac_agent = ACAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        device=device,
        save_path=args.log_dir,
        save_every=args.save_every
    )

    dqn_agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[64, 64],
        device=get_device(),
        save_path=args.log_dir,
        save_every=args.save_every
    )

    nfsp_agent = NFSPAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers_sizes=[64, 64],
        q_mlp_layers=[64, 64],
        device=get_device(),
        save_path=args.log_dir,
        save_every=args.save_every
    )

    madeepcfr_agent = MADeepCFRAgent(
        env,
        policy_network_layers=(args.policy_network_layers, args.policy_network_layers),
        advantage_network_layers=(args.advantage_network_layers, args.advantage_network_layers),
        num_iterations=1,
        num_traversals=1,
        learning_rate=0.0003,
        batch_size_advantage=args.batch_size_advantage,
        batch_size_strategy=args.batch_size_strategy,
        policy_network_train_steps=1,
        advantage_network_train_steps=1,
        memory_capacity=1e7)

    nsave_path = r"../experiments\cfr\1000.pth"
    cfr_agent = torch.load(nsave_path)

    nsave_path = r"../experiments\multi-leduc-holdem\ac\model.pth"
    ac_agent = torch.load(nsave_path)

    nsave_path = r"../experiments\multi-leduc-holdem\dqn\model.pth"
    dqn_agent = torch.load(nsave_path)

    nsave_path = r"../experiments\multi-leduc-holdem\nfsp\model.pth"
    nfsp_agent = torch.load(nsave_path)

    nsave_path = r"../experiments\multi-leduc-holdem\madeepcfr\model.pth"
    madeepcfr_agent = torch.load(nsave_path)

    random_agent = RandomAgent(4)

    # Evaluate CFR against random
    env.set_agents([
        madeepcfr_agent,
        cfr_agent,
        madeepcfr_agent,
        cfr_agent,
    ])

    a = tournament_wp(env, args.num_eval_games)
    print(a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VS")
    parser.add_argument(
        '--env',
        type=str,
        default='multi-leduc-holdem',
    )

    parser.add_argument(
        '--cuda',
        type=str,
        default='0',
    )
    parser.add_argument(
        '--seed',
        type=int,
        # default=42,
    )

    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=1000,
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/vs/',
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

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    evaluate(args)
