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
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument(
        "--anneal-rate",
        type=float,
        default=0.99998,
        help="exploration anneal rate per step",
    )
    parser.add_argument(
        "--anneal-schedule",
        type=float,
        default=200,
        help="number of epochs to linearly anneal over",
    )
    parser.add_argument(
        "--discount-rate", type=float, default=0.99, help="discount rate per timestep"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.0003, help="learning rate"
    )
    parser.add_argument(
        "--adam-eps", type=float, default=0.00001, help="adam epsilon value"
    )
    parser.add_argument(
        "--b-init-value", type=float, default=0.1, help="bias value for hidden layers"
    )
    parser.add_argument(
        "--init-w", type=float, default=0.003, help="magnitude of weights on all layers"
    )
    parser.add_argument(
        "--replay-buffer-size",
        type=int,
        default=100000,
        help="max size of replay buffer",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2000,
        help="number of steps per epoch (and eval, training)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="number of training examples per batch",
    )
    parser.add_argument(
        "--target-update-period",
        type=int,
        default=1,
        help="number of training steps before target network update",
    )
    parser.add_argument(
        "--soft-target-tau",
        type=float,
        default=0.001,
        help="update weight for network transfer",
    )
    parser.add_argument(
        "--double-dqn",
        action="store_true",
        default=False,
        help="use double dqn for training",
    )
    parser.add_argument(
        "--naive-discounting",
        action="store_true",
        default=False,
        help="discount plan steps by gamma (not gamma^length)",
    )
    parser.add_argument(
        "--huber-loss",
        action="store_true",
        default=False,
        help="clip loss function in trainer",
    )
    parser.add_argument(
        "--softmax",
        action="store_true",
        default=False,
        help="Use softmax for policy instead of argmax",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="temperature parameter for softmax",
    )
    parser.add_argument(
        "--experience-interval",
        type=int,
        default=1,
        help="number of controller steps between intermediate experience frames",
    ),
    parser.add_argument(
        "--train-loops", type=int, default=1, help="number of training loops each epoch"
    ), parser.add_argument(
        "--eval-steps", type=int, default=0, help="number of eval steps each epoch"
    )
    parser.add_argument(
        "--render", action="store_true", default=False, help="show image of environment"
    )
    parser.add_argument(
        "--action-space",
        default="planner",
        help="symbolic action space representation",
        choices=["planner", "skills"],
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")

    args = parser.parse_args(arglist)

    if args.env_name.startswith("craft-lottery") or args.env_name.startswith(
        "craft-mole"
    ):
        from forks.rlkit.rlkit.launchers.dqn import experiment
    else:
        from forks.rlkit.rlkit.launchers.dqn_pre_trained_goal_gen import experiment

    variant = dict(
        algorithm="DQN",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(args.replay_buffer_size),
        env_name=args.env_name,
        seed=args.seed,
        anneal_rate=args.anneal_rate,
        anneal_schedule=args.anneal_schedule,
        render=args.render,
        double_dqn=args.double_dqn,
        init_w=args.init_w,
        b_init_value=args.b_init_value,
        softmax=args.softmax,
        temperature=args.temperature,
        experience_interval=args.experience_interval,
        action_space=args.action_space,
        algorithm_kwargs=dict(
            num_epochs=args.epochs,  # number of epochs, so total number of training steps is this*num_trains_per_train_loop, and total training batches is this*num_trains_per_train_loop
            num_eval_steps_per_epoch=args.eval_steps,
            num_trains_per_train_loop=args.num_steps,  # number of batches to train each loop.
            num_expl_steps_per_train_loop=args.num_steps,  # number of environment steps to train
            min_num_steps_before_training=1000,  # number of steps to take before training starts
            max_path_length=1000,  # max number of steps per episode
            batch_size=args.batch_size,
            num_train_loops_per_epoch=args.train_loops,
        ),
        trainer_kwargs=dict(
            discount=args.discount_rate,
            learning_rate=args.learning_rate,
            target_update_period=args.target_update_period,
            soft_target_tau=args.soft_target_tau,
            adam_eps=args.adam_eps,
            naive_discounting=args.naive_discounting,
            huber_loss=args.huber_loss,
        ),
    )
    # optionally set the GPU (default=False)
    experiment(variant)


if __name__ == "__main__":
    main(sys.argv[1:])
