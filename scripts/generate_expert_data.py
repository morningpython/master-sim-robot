"""
Generate expert trajectories using IK-based Cartesian controller.

Usage:
    python scripts/generate_expert_data.py --num-trajectories 200 --steps 200 --output data/expert_traj.h5
"""

import argparse
from pathlib import Path
import numpy as np

from src.envs.panda_env import PandaEnv
from src.controllers.ik_controller import IKController
from src.utils.preprocessing import save_trajectories_to_hdf5


def generate_expert_trajectory(env: PandaEnv, steps: int = 200, action_format: str = "qpos") -> tuple:
    """Generate a single trajectory using IK-based position control.

    Args:
        action_format: 'qpos' for absolute joint targets, 'delta' for qpos deltas (recommended)
    """
    traj_states = []
    traj_actions = []

    # Reset env and IK solver
    obs, info = env.reset()
    ik = IKController(env.model, env.data)

    for _ in range(steps):
        obs = env._get_obs()
        current_qpos = env.data.qpos[:env.model.nv].copy()
        target_pos = obs["target_pos"].copy()

        # One-step IK position control
        qpos_target = ik.solve_position_control(target_pos, current_qpos, dt=0.01)
        if qpos_target is None:
            # Fallback: use small random perturbation around current qpos
            qpos_target = current_qpos + 0.01 * np.random.randn(*current_qpos.shape)

        # Compute action according to requested format
        if action_format == "qpos":
            action = qpos_target[:7]
        elif action_format == "delta":
            action = qpos_target[:7] - current_qpos[:7]
        else:
            raise ValueError(f"Unknown action_format: {action_format}")

        # Save state/action
        # Flatten observation to match environment observation structure (20 dims)
        state_flat = np.concatenate([
            obs["joint_pos"],
            obs["joint_vel"],
            obs["ee_pos"],
            obs["target_pos"],
        ])
        traj_states.append(state_flat)
        traj_actions.append(action)

        # Step env using absolute qpos target to keep simulation consistent
        env.step(qpos_target[:7], n_steps=5)

    return np.array(traj_states), np.array(traj_actions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trajectories", type=int, default=200)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", type=str, default="data/expert_traj.h5")
    parser.add_argument("--action-format", type=str, choices=["qpos","delta"], default="delta", help="Action labels to save: 'qpos' or 'delta' (recommended 'delta')")

    args = parser.parse_args()

    env = PandaEnv()
    trajectories = []

    print(f"Generating {args.num_trajectories} expert trajectories ({args.steps} steps each) with action_format={args.action_format}")
    for i in range(args.num_trajectories):
        traj = generate_expert_trajectory(env, steps=args.steps, action_format=args.action_format)
        trajectories.append(traj)

    p = Path(args.output)
    save_trajectories_to_hdf5(trajectories, p)
    print(f"Saved {len(trajectories)} trajectories to {p}")


if __name__ == "__main__":
    main()
