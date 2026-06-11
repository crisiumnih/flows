import ogbench
import numpy as np
import torch
import argparse
import imageio
from fql import FQL


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="cube-single-play-singletask-v0")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_episodes", default=5, type=int)
    parser.add_argument("--save_video", default=None, help="Path to save video (e.g. eval.mp4)")
    args = parser.parse_args()

    env, _, _ = ogbench.make_env_and_datasets(args.env)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = FQL(obs_dim, action_dim, max_action)
    policy.load(args.checkpoint)

    frames = []
    successes = []
    video_recorded = False
    for task_id in range(1, 6):
        for ep in range(args.eval_episodes):
            obs, info = env.reset(options=dict(task_id=task_id))
            done = False
            step = 0
            while not done and step < 300:
                action = policy.select_action(np.array(obs))
                obs, reward, terminated, truncated, info = env.step(action)
                if args.save_video and not video_recorded:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                done = terminated or truncated
                step += 1
            successes.append(info.get('success', 0))
            if args.save_video and not video_recorded:
                video_recorded = True

    avg_success = np.mean(successes) * 100
    print(f"Avg success rate: {avg_success:.1f}% ({sum(successes)}/{len(successes)} episodes)")

    if args.save_video and frames:
        imageio.mimsave(args.save_video, frames, fps=30)
        print(f"Saved video to {args.save_video}")
