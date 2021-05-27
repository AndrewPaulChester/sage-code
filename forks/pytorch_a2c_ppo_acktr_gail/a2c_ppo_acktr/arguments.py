import argparse

import torch


def get_args(args):
    # print(arglist)
    # args = parser.parse_args(arglist)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ["a2c", "ppo", "acktr"]
    if args.recurrent_policy:
        assert args.algo in [
            "a2c",
            "ppo",
        ], "Recurrent policy is not implemented for ACKTR"

    return args
