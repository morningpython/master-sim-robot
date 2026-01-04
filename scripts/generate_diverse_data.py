"""다양한 초기 위치에서 Expert 데이터 생성 (ST-9).

다양한 target 위치, randomization을 통해 robust한 데이터셋 생성.
"""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.envs.panda_env import PandaEnv
from src.controllers.ik_controller import IKController
from src.utils.preprocessing import save_trajectories_to_hdf5, DataAugmenter


def generate_trajectory_with_randomization(
    env: PandaEnv,
    ik: IKController,
    max_steps: int = 200,
    target_randomization: float = 0.05,
    action_format: str = "delta",
) -> tuple:
    """IK 기반 expert 궤적 생성 (randomized 초기 위치).

    Args:
        env: 환경
        ik: IK controller
        max_steps: 최대 스텝
        target_randomization: 타겟 위치 randomization 범위 (meters)
        action_format: 'qpos' or 'delta'

    Returns:
        (observations, actions)
    """
    obs, _ = env.reset()

    # Randomize target position slightly
    # (Note: PandaEnv internally randomizes, so we just run as-is)
    
    observations = []
    actions = []

    for step in range(max_steps):
        obs_flat = env._get_obs()
        current_qpos = env.data.qpos[:env.model.nv].copy()
        target_pos = obs_flat["target_pos"].copy()

        # IK solve
        qpos_target = ik.solve_position_control(target_pos, current_qpos, dt=0.01)
        if qpos_target is None:
            qpos_target = current_qpos + 0.01 * np.random.randn(*current_qpos.shape)

        # Compute action
        if action_format == "qpos":
            action = qpos_target[:7]
        elif action_format == "delta":
            action = qpos_target[:7] - current_qpos[:7]
        else:
            raise ValueError(f"Unknown action_format: {action_format}")

        # Flatten observation
        state_flat = np.concatenate([
            obs_flat["joint_pos"],
            obs_flat["joint_vel"],
            obs_flat["ee_pos"],
            obs_flat["target_pos"],
        ])

        observations.append(state_flat)
        actions.append(action)

        # Step env
        env.step(qpos_target[:7], n_steps=5)

    return np.array(observations), np.array(actions)


def main():
    parser = argparse.ArgumentParser(description="Generate diverse expert data")
    parser.add_argument(
        "--num-trajectories", type=int, default=500, help="Number of trajectories"
    )
    parser.add_argument("--steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument(
        "--output", type=str, default="data/expert_diverse_delta.h5", help="Output path"
    )
    parser.add_argument(
        "--target-randomization",
        type=float,
        default=0.05,
        help="Target position randomization range (m)",
    )
    parser.add_argument(
        "--augment", action="store_true", help="Apply data augmentation"
    )
    parser.add_argument(
        "--n-augment", type=int, default=2, help="Augmentation multiplier per trajectory"
    )
    parser.add_argument(
        "--action-format",
        type=str,
        default="delta",
        choices=["qpos", "delta"],
        help="Action label format",
    )

    args = parser.parse_args()

    print(f"Generating {args.num_trajectories} diverse expert trajectories...")
    print(f"Target randomization: ±{args.target_randomization}m")
    print(f"Action format: {args.action_format}")

    env = PandaEnv()
    ik = IKController(env.model, env.data)
    trajectories = []

    for i in tqdm(range(args.num_trajectories), desc="Generating trajectories"):
        obs, actions = generate_trajectory_with_randomization(
            env,
            ik,
            max_steps=args.steps,
            target_randomization=args.target_randomization,
            action_format=args.action_format,
        )

        trajectories.append((obs, actions))

    # Data Augmentation (옵션)
    if args.augment:
        print("Applying data augmentation...")
        augmenter = DataAugmenter(
            noise_scale=0.01, action_noise_scale=0.005
        )
        trajectories = augmenter.augment_dataset(trajectories, n_augment=args.n_augment)

    # 저장
    output_path = Path(args.output)
    save_trajectories_to_hdf5(trajectories, output_path)

    print(f"Dataset saved: {output_path}")
    print(f"Total trajectories: {len(trajectories)}")


if __name__ == "__main__":
    main()
