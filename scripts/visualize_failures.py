"""
Visualize failure episodes: record RGB videos and save diagnostic plots.

Usage:
    python scripts/visualize_failures.py --model models/expert_trained_20.npz --normalizers models/expert_trained_20_normalizers.npz --episodes 5
"""
import argparse
from pathlib import Path
import json
import numpy as np

from src.envs.panda_env import PandaEnv
from src.models.bc_agent import BCAgent
from src.utils.preprocessing import Normalizer, NormalizationStats
from src.utils.visualization import CameraRenderer, VideoRecorder, TrajectoryVisualizer, create_comparison_plot


def load_normalizers(path: Path) -> tuple[Normalizer, Normalizer, str]:
    data = np.load(path)
    s = Normalizer(); s.stats = NormalizationStats(data['state_mean'], data['state_std'], data['state_mean'], data['state_mean'])
    a = Normalizer(); a.stats = NormalizationStats(data['action_mean'], data['action_std'], data['action_mean'], data['action_mean'])
    action_type = str(data['action_type']) if 'action_type' in data else 'qpos'
    return s, a, action_type


def run_and_record(agent: BCAgent, state_norm: Normalizer, action_norm: Normalizer, output_dir: Path, num_episodes: int = 5, max_steps: int = 500, action_type: str = 'qpos'):
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    plots_dir = output_dir / "plots"
    videos_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    env = PandaEnv(render_mode='rgb_array')
    traj_viz = TrajectoryVisualizer()

    summary = []

    for ep in range(num_episodes):
        rec = VideoRecorder(videos_dir / f"episode_{ep+1}.mp4", fps=20)

        obs, info = env.reset()
        obs_flat = np.concatenate([obs['joint_pos'], obs['joint_vel'], obs['ee_pos'], obs['target_pos']])

        ee_positions = []
        joint_positions = []
        actions = []
        dists = []

        for t in range(max_steps):
            # Normalize state and predict
            obs_n = state_norm.transform(obs_flat)
            a_pred = agent.predict(obs_n)
            # Inverse-transform action
            a_un = action_norm.inverse_transform(a_pred)

            # Interpret action according to saved action type
            if action_type == 'delta':
                current_q = env.data.qpos[:env.model.nv].copy()
                applied_action = current_q[:7] + a_un
            else:
                applied_action = a_un

            # Step env
            next_obs, info = env.step(applied_action, n_steps=5)

            # Render frame
            cam = CameraRenderer(env.model, env.data, width=320, height=240)
            frame = cam.render_rgb()
            rec.add_frame(frame)
            cam.close()

            # Record
            ee_positions.append(next_obs['ee_pos'].copy())
            joint_positions.append(next_obs['joint_pos'].copy())
            actions.append(a.copy())
            dists.append(info['ee_to_target_dist'])

            obs = next_obs
            obs_flat = np.concatenate([obs['joint_pos'], obs['joint_vel'], obs['ee_pos'], obs['target_pos']])

            if info['ee_to_target_dist'] < 0.05:
                break

        # Save video
        rec.save()

        # Save plots
        ee_positions = np.array(ee_positions)
        joint_positions = np.array(joint_positions)
        actions = np.array(actions)
        dists = np.array(dists)

        traj_viz.plot_3d_trajectory(ee_positions, title=f"Episode {ep+1} EE Trajectory", save_path=plots_dir / f"ep_{ep+1}_ee.png")
        traj_viz.plot_joint_trajectories(joint_positions, save_path=plots_dir / f"ep_{ep+1}_joints.png")

        create_comparison_plot({
            'distance_to_target': dists,
            'action_magnitude': np.linalg.norm(actions, axis=1),
        }, title=f"Episode {ep+1} distance & action", ylabel="Value", save_path=plots_dir / f"ep_{ep+1}_dist_action.png")

        # Save metrics
        ep_summary = {
            'episode': ep + 1,
            'steps': len(dists),
            'final_distance': float(dists[-1]) if len(dists)>0 else None,
            'mean_action_magnitude': float(np.mean(np.linalg.norm(actions, axis=1))) if len(actions)>0 else None,
            'video': str(videos_dir / f"episode_{ep+1}.mp4"),
            'plots': {
                'ee': str(plots_dir / f"ep_{ep+1}_ee.png"),
                'joints': str(plots_dir / f"ep_{ep+1}_joints.png"),
                'dist_action': str(plots_dir / f"ep_{ep+1}_dist_action.png"),
            }
        }
        summary.append(ep_summary)

        # Clear recorder
        rec.clear()

    # Save summary
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump({'episodes': summary}, f, indent=2)

    return output_dir / 'summary.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--normalizers', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--output-dir', type=str, default='analysis/failure_videos')

    args = parser.parse_args()

    agent = BCAgent(obs_dim=20, act_dim=7)
    agent.load(Path(args.model))

    state_norm, action_norm, action_type = load_normalizers(Path(args.normalizers))

    summary_path = run_and_record(agent, state_norm, action_norm, Path(args.output_dir), num_episodes=args.episodes, max_steps=args.max_steps, action_type=action_type)

    print('Saved summary to', summary_path)
