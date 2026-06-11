"""Run a few episodes and save a video to see what the environment looks like."""
import ogbench
import imageio
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="cube-single-play-singletask-v0")
    parser.add_argument("--episodes", default=2, type=int)
    parser.add_argument("--out", default="env_preview.mp4")
    args = parser.parse_args()

    env, _, _ = ogbench.make_env_and_datasets(args.env)

    frames = []
    for task_id in range(1, args.episodes + 1):
        obs, info = env.reset(options=dict(task_id=task_id, render_goal=True))
        done = False
        step = 0
        while not done and step < 300:
            # Random actions to see the env
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            done = terminated or truncated
            step += 1

    if frames:
        imageio.mimsave(args.out, frames, fps=30)
        print(f"Saved {len(frames)} frames to {args.out}")
    else:
        print("No frames captured. Environment may not support rgb_array rendering.")


if __name__ == "__main__":
    main()
