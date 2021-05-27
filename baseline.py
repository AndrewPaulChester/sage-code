"""
Run DQN on grid world.
"""
import sys
import argparse
import domains.gym_taxi


def main(arglist):
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        "--env-name", default="large-taxi-v1", help="Select the environment to run"
    )
    parser.add_argument(
        "--baseline",
        default="scripted",
        choices=["random-atomic", "random-symbolic", "scripted"],
        help="Select the baseline algorithm to run",
    )
    parser.add_argument(
        "--render", action="store_true", default=False, help="show image of environment"
    )
    parser.add_argument(
        "--always-return",
        action="store_true",
        default=False,
        help="always return in same action as deliver",
    )

    args = parser.parse_args(arglist)

    if args.baseline == "scripted":
        from forks.rlkit.rlkit.launchers.scripted_goal_gen import experiment
    elif args.baseline == "random-symbolic":
        from forks.rlkit.rlkit.launchers.random_goal_gen import experiment
    elif args.baseline == "random-atomic":
        from forks.rlkit.rlkit.launchers.dqn import experiment
    else:
        raise ValueError(
            f"Baseline argument expected one of 'random-atomic','random-symbolic','scripted', got: '{args.baseline}'"
        )

    variant = dict(
        algorithm="DQN",
        version="normal",
        layer_size=256,
        replay_buffer_size=1000,
        env_name=args.env_name,
        recurrent_policy=False,
        anneal_rate=1,
        seed=1,
        num_processes=1,
        gamma=0.99,
        render=args.render,
        log_dir="./logs/",
        num_steps=1000,
        always_return=args.always_return,
        algorithm_kwargs=dict(
            num_epochs=3,  # number of epochs, so total number of training steps is this*num_trains_per_train_loop, and total training batches is this*num_trains_per_train_loop
            num_eval_steps_per_epoch=20000,
            num_trains_per_train_loop=1,  # number of batches to train each loop.
            num_expl_steps_per_train_loop=1,  # number of environment steps to train
            min_num_steps_before_training=0,  # number of steps to take before training starts
            max_path_length=2000,  # max number of steps per episode
            batch_size=32,
            use_linear_lr_decay=False,
            num_train_loops_per_epoch=1,
        ),
        trainer_kwargs=dict(
            value_loss_coef=0,
            entropy_coef=0,
            lr=0.0000000000001,
            eps=0.000001,
            clip_param=0,
            ppo_epoch=32,
            num_mini_batch=4,
            max_grad_norm=0,
            gamma=0.99,
            use_gae=False,
            gae_lambda=1,
            use_proper_time_limits=False,
        ),
    )
    # optionally set the GPU (default=False)
    experiment(variant)


if __name__ == "__main__":
    main(sys.argv[1:])
