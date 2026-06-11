import os
import ogbench
import numpy as np
import torch
import argparse
from fql import FQL
from replay_buffer import ReplayBuffer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="cube-single-play-singletask-v0")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--max_timesteps", default=1_000_000, type=int)
    parser.add_argument("--eval_freq", default=5000, type=int)
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--alpha", default=10.0, type=float)
    parser.add_argument("--flow_steps", default=10, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    args = parser.parse_args()

    print(f"Env: {args.env}, Seed: {args.seed}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create env and load dataset
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(args.env)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    policy = FQL(obs_dim, action_dim, max_action, discount=args.discount, tau=args.tau, alpha=args.alpha, flow_steps=args.flow_steps)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    replay_buffer = ReplayBuffer(device)
    replay_buffer.load_ogbench(train_dataset)

    # Training loop
    for t in range(args.max_timesteps):
        policy.train(replay_buffer, args.batch_size)

        if (t + 1) % 1000 == 0:
            print(f"Step: {t + 1}/{args.max_timesteps}")

        if (t + 1) % 100_000 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            policy.save(f"checkpoints/fql_{args.env}_{t+1}.pt")
            print(f"  Saved checkpoint at step {t+1}")

        if (t + 1) % args.eval_freq == 0:
            print(f"Step: {t + 1}")
            successes = []
            for task_id in range(1, 6):
                for ep in range(args.eval_episodes):
                    obs, info = env.reset(options=dict(task_id=task_id))
                    done = False
                    while not done:
                        action = policy.select_action(np.array(obs))
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                    successes.append(info.get('success', 0))

            avg_success = np.mean(successes) * 100
            print(f"  Avg success rate: {avg_success:.1f}%")
            print("---------------------------------------")