"""
Run DQN on grid world.
"""
import sys
import argparse
import torch.multiprocessing as mp


# from forks.rlkit.rlkit.launchers.dqn import experiment


def main(arglist):
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        "--env-name", default="box-taxi-v1", help="Select the environment to run"
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument(
        "--anneal-rate",
        type=float,
        default=0.99998,
        help="exploration anneal rate per step",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.00025, help="learning rate"
    )
    parser.add_argument(
        "--replay-buffer-size",
        type=int,
        default=10000,
        help="max size of replay buffer",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=128,
        help="number of forward steps in A2C (default: 5)",
    )

    parser.add_argument(
        "--eval-steps",
        type=int,
        default=2000,
        help="number of eval steps taken per epoch",
    )

    parser.add_argument(
        "--train-loops",
        type=int,
        default=10,
        help="number of training loops done between evals",
    )

    parser.add_argument(
        "--algo", default="ppo", help="algorithm to use: a2c | ppo | acktr"
    )
    parser.add_argument(
        "--gail",
        action="store_true",
        default=False,
        help="do imitation learning with gail",
    )
    parser.add_argument(
        "--gail-experts-dir",
        default="./gail_experts",
        help="directory that contains expert demonstrations for gail",
    )
    parser.add_argument(
        "--gail-batch-size",
        type=int,
        default=128,
        help="gail batch size (default: 128)",
    )
    parser.add_argument(
        "--gail-epoch", type=int, default=5, help="gail epochs (default: 5)"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.99,
        help="RMSprop optimizer apha (default: 0.99)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--use-gae",
        action="store_true",
        default=True,
        help="use generalized advantage estimation",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="gae lambda parameter (default: 0.95)",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=0.5,
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="max norm of gradients (default: 0.5)",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--cuda-deterministic",
        action="store_true",
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="how many training CPU processes to use (default: 16)",
    )
    parser.add_argument(
        "--ppo-epoch", type=int, default=4, help="number of ppo epochs (default: 4)"
    )
    parser.add_argument(
        "--num-mini-batch",
        type=int,
        default=4,
        help="number of batches for ppo (default: 32)",
    )
    parser.add_argument(
        "--clip-param",
        type=float,
        default=0.1,
        help="ppo clip parameter (default: 0.2)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="log interval, one log per n updates (default: 10)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="save interval, one save per n updates (default: 100)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="eval interval, one eval per n updates (default: None)",
    )
    parser.add_argument(
        "--num-env-steps",
        type=int,
        default=10000000,
        help="number of environment steps to train (default: 10e6)",
    )

    parser.add_argument(
        "--log-dir",
        default="./logs/",
        help="directory to save agent logs (default: ./logs/)",
    )
    parser.add_argument(
        "--save-dir",
        default="./trained_models/",
        help="directory to save agent logs (default: ./trained_models/)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--use-proper-time-limits",
        action="store_true",
        default=True,
        help="compute returns taking into account time limits",
    )
    parser.add_argument(
        "--recurrent-policy",
        action="store_true",
        default=False,
        help="use a recurrent policy",
    )
    parser.add_argument(
        "--use-linear-lr-decay",
        action="store_true",
        default=False,
        help="use a linear schedule on the learning rate",
    )

    parser.add_argument(
        "--deep-fc",
        action="store_true",
        default=False,
        help="add an extra hidden layer in the FC part of the network",
    )
    parser.add_argument(
        "--symbolic",
        default="none",
        help="symbolic action representation used",
        choices=[
            "none",
            "discrete",
            "factored",
            "ordinal",
            "three-tier",
            "three-tier-shared",
            "multihead",
            "multihead-flat",
            "three-tier-pretrained",
        ],
    )

    parser.add_argument(
        "--action-space",
        default="full",
        help="symbolic action space representation",
        choices=["full", "move-only", "move-continuous", "move-uniform", "rooms"],
    )
    parser.add_argument(
        "--no-plan-penalty",
        action="store_true",
        default=False,
        help="remove negative reward for failed plan",
    )
    parser.add_argument(
        "--naive-discounting",
        action="store_true",
        default=False,
        help="discount one step per plan, all reward in plan undiscounted",
    )

    parser.add_argument(
        "--render", action="store_true", default=False, help="show image of environment"
    )

    parser.add_argument(
        "--meta-num-steps",
        type=int,
        default=128,
        help="number of forward steps in A2C for the meta-controller (default: 128)",
    )
    args = parser.parse_args(arglist)

    def change_defaults():
        if args.log_interval == 100:
            args.log_interval = 1
        if args.save_interval == 100:
            args.save_interval = 1

    if args.symbolic == "none":
        from forks.rlkit.rlkit.launchers.ppo import experiment
    elif args.symbolic == "discrete":
        from forks.rlkit.rlkit.launchers.a2c_goal_gen import experiment

        change_defaults()
    elif args.symbolic == "factored":
        from forks.rlkit.rlkit.launchers.ppo_factored_goal_gen import experiment

        change_defaults()
    elif args.symbolic == "ordinal":
        from forks.rlkit.rlkit.launchers.ppo_ordinal_goal_gen import experiment

        change_defaults()
    elif args.symbolic == "three-tier":
        from forks.rlkit.rlkit.launchers.ppo_three_tier_goal_gen import experiment

        change_defaults()
    elif args.symbolic == "three-tier-shared":
        from forks.rlkit.rlkit.launchers.ppo_three_tier_shared_goal_gen import experiment

        change_defaults()
    elif args.symbolic == "multihead":
        from forks.rlkit.rlkit.launchers.ppo_multihead_goal_gen import experiment

        change_defaults()

    elif args.symbolic == "multihead-flat":
        from forks.rlkit.rlkit.launchers.ppo_multihead_flat import experiment

        change_defaults()

    elif args.symbolic == "three-tier-pretrained":
        from forks.rlkit.rlkit.launchers.ppo_pre_trained_goal_gen import experiment

        change_defaults()

    else:
        raise ValueError(
            f"Baseline argument expected one of 'none','discrete','factored','ordinal','three-tier','three-tier-shared','multihead','three-tier-pretrained' got: '{args.symbolic}'"
        )

    if args.action_space == "full":
        ancillary_goal_size = 5
    elif args.action_space == "move-only":
        ancillary_goal_size = 2
    elif args.action_space == "move-continuous":
        ancillary_goal_size = 3
    elif args.action_space == "move-uniform":
        ancillary_goal_size = 3
    elif args.action_space == "rooms":
        ancillary_goal_size = 5

    variant = dict(
        algorithm="PPO",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(args.replay_buffer_size),
        env_name=args.env_name,
        anneal_rate=args.anneal_rate,
        seed=args.seed,
        num_processes=args.num_processes,
        gamma=args.gamma,
        log_dir=args.log_dir,
        recurrent_policy=args.recurrent_policy,
        deep_fc=args.deep_fc,
        num_steps=args.num_steps,
        meta_num_steps=args.meta_num_steps,
        render=args.render,
        no_plan_penalty=args.no_plan_penalty,
        naive_discounting=args.naive_discounting,
        action_space=args.action_space,
        ancillary_goal_size=ancillary_goal_size,
        algorithm_kwargs=dict(
            num_epochs=args.epochs,  # number of epochs, so total number of training steps is this*num_trains_per_train_loop*num_train_loops_per_epoch
            num_eval_steps_per_epoch=args.eval_steps,
            num_expl_steps_per_train_loop=args.num_steps,  # number of environment steps to train
            num_trains_per_train_loop=args.num_steps,
            num_train_loops_per_epoch=args.train_loops,
            min_num_steps_before_training=0,  # number of steps to take before training starts
            max_path_length=1000,  # max number of steps per episode
            batch_size=32,
            use_linear_lr_decay=args.use_linear_lr_decay,
        ),
        trainer_kwargs=dict(
            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef,
            lr=args.learning_rate,
            eps=args.eps,
            clip_param=args.clip_param,
            ppo_epoch=args.ppo_epoch,
            num_mini_batch=args.num_mini_batch,
            max_grad_norm=args.max_grad_norm,
            gamma=args.gamma,
            use_gae=args.use_gae,
            gae_lambda=args.gae_lambda,
            use_proper_time_limits=args.use_proper_time_limits,
        ),
    )
    # optionally set the GPU (default=False)
    experiment(variant)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main(sys.argv[1:])


# guild run merged:train-ppo env-name="rooms-craft-v2" train-loops=10 entropy-coef=0.01 symbolic="three-tier-pretrained" action-space=rooms eval-steps=0 learning-rate=0.001 seed=2
