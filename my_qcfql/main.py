import os
import ogbench
import numpy as np
import torch
import argparse
from qcfql import FQL
from replay_buffer import ReplayBuffer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="cube-single-play-singletask-v0")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--max_timesteps", default=1_000_000, type=int)
    parser.add_argument("--eval_freq", default=5000, type=int)
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--alpha", default=100.0, type=float)
    parser.add_argument("--flow_steps", default=10, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--chunk_size", default=5, type=int)
    parser.add_argument("--online_steps", default=1_000_000, type=int)
    args = parser.parse_args()

    print(f"Env: {args.env}, Seed: {args.seed}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create env and load dataset
    print("Loading dataset...")
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(args.env)
    print(f"Dataset loaded. Observations: {train_dataset['observations'].shape}")

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print("Creating policy...")
    policy = FQL(obs_dim, action_dim, max_action, chunk_size=args.chunk_size, discount=args.discount, tau=args.tau, alpha=args.alpha, flow_steps=args.flow_steps)
    print("Policy created.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Loading replay buffer...")
    replay_buffer = ReplayBuffer(device, chunk_size=args.chunk_size, discount=args.discount)
    replay_buffer.load_ogbench(train_dataset)

    print("Starting training...")
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
                        action_chunk = policy.select_action(np.array(obs))
                        for a in action_chunk:
                            obs, reward, terminated, truncated, info = env.step(a)
                            done = terminated or truncated
                            if done:
                                break
                    successes.append(info.get('success', 0))

            avg_success = np.mean(successes) * 100
            print(f"  Avg success rate: {avg_success:.1f}%")
            print("---------------------------------------")
        
    # Online fine-tuning
    if args.online_steps > 0:
        print("Starting online fine-tuning...")

        obs, info = env.reset()
        done = True
        online_step = 0

        for t in range(args.online_steps):
            # Collect a chunk
            if done:
                obs, info = env.reset()

            start_obs = np.array(obs)
            action_chunk = policy.select_action(np.array(obs))
            chunk_flat = action_chunk.flatten()
            rewards = []
            done = False

            for a in action_chunk:
                obs, reward, terminated, truncated, info = env.step(a)
                rewards.append(reward)
                done = terminated or truncated
                if done:
                    break

            mask = 0.0 if terminated else 1.0
            replay_buffer.add_chunk(start_obs, chunk_flat, np.array(obs), rewards, mask)

            # Train
            policy.train(replay_buffer, args.batch_size)

            if (t + 1) % 100_000 == 0:
                os.makedirs("checkpoints", exist_ok=True)
                policy.save(f"checkpoints/fql_{args.env}_online_{t+1}.pt")
                print(f"  Saved online checkpoint at step {t+1}")

            if (t + 1) % 1000 == 0:
                print(f"Online step: {t + 1}/{args.online_steps}")

            if (t + 1) % args.eval_freq == 0:
                print(f"Online step: {t + 1}")
                successes = []
                for task_id in range(1, 6):
                    for ep in range(args.eval_episodes):
                        eval_obs, eval_info = env.reset(options=dict(task_id=task_id))
                        eval_done = False
                        while not eval_done:
                            ac = policy.select_action(np.array(eval_obs))
                            for a in ac:
                                eval_obs, _, eval_term, eval_trunc, eval_info = env.step(a)
                                eval_done = eval_term or eval_trunc
                                if eval_done:
                                    break
                        successes.append(eval_info.get('success', 0))
                avg_success = np.mean(successes) * 100
                print(f"  Online avg success rate: {avg_success:.1f}%")
                print("---------------------------------------")