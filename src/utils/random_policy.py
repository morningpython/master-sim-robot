"""
Random Manipulation Simulation

랜덤 정책으로 로봇을 조작하여 초기 학습 데이터를 생성합니다.
"""
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import time
from tqdm import tqdm

from src.envs.panda_env import PandaEnv
from src.utils.data_collection import Trajectory, DatasetBuilder


class RandomPolicy:
    """
    랜덤 액션 정책.
    
    관절 공간에서 랜덤하게 목표 위치를 샘플링합니다.
    
    Args:
        action_dim: 액션 차원 (기본: 7)
        action_scale: 액션 스케일 (기본: 0.1)
    
    Example:
        >>> policy = RandomPolicy(action_dim=7)
        >>> state = np.zeros(21)
        >>> action = policy.predict(state)
    """
    
    def __init__(self, action_dim: int = 7, action_scale: float = 0.1):
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.prev_action = np.zeros(action_dim)
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        상태를 받아 랜덤 액션 반환.
        
        Args:
            state: 현재 상태 (21,)
        
        Returns:
            action: 관절 목표 위치 (7,)
        """
        # 현재 관절 위치 추출
        current_qpos = state[:7]
        
        # 랜덤 변화량
        delta = np.random.randn(self.action_dim) * self.action_scale
        
        # Smooth transition (momentum)
        delta = 0.7 * delta + 0.3 * (self.prev_action - current_qpos)
        
        # 새로운 목표 위치
        action = current_qpos + delta
        self.prev_action = action
        
        return action
    
    def reset(self) -> None:
        """정책 리셋."""
        self.prev_action = np.zeros(self.action_dim)


class SinusoidalPolicy:
    """
    사인파 기반 정책 (더 smooth한 움직임).
    
    Args:
        action_dim: 액션 차원
        frequency: 주파수 배열 (각 관절마다)
        amplitude: 진폭 배열
    """
    
    def __init__(
        self, 
        action_dim: int = 7,
        frequency: Optional[np.ndarray] = None,
        amplitude: Optional[np.ndarray] = None,
    ):
        self.action_dim = action_dim
        self.frequency = frequency if frequency is not None else np.random.uniform(0.5, 2.0, action_dim)
        self.amplitude = amplitude if amplitude is not None else np.random.uniform(0.2, 0.5, action_dim)
        self.time = 0.0
        self.phase = np.random.uniform(0, 2*np.pi, action_dim)
    
    def predict(self, state: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        사인파 패턴 액션 생성.
        
        Args:
            state: 현재 상태
            dt: 시간 간격
        
        Returns:
            action: 관절 목표 위치
        """
        self.time += dt
        
        # 사인파 생성
        action = self.amplitude * np.sin(2 * np.pi * self.frequency * self.time + self.phase)
        
        return action
    
    def reset(self) -> None:
        """정책 리셋."""
        self.time = 0.0
        self.phase = np.random.uniform(0, 2*np.pi, self.action_dim)


class DataGenerator:
    """
    랜덤 시뮬레이션을 통한 데이터 생성기.
    
    Example:
        >>> env = PandaEnv()
        >>> generator = DataGenerator(env, save_dir="data/random")
        >>> generator.generate(num_trajectories=100, steps_per_traj=200)
    """
    
    def __init__(
        self, 
        env: PandaEnv,
        save_dir: str = "data/random",
        policy_type: str = "random",
    ):
        """
        Args:
            env: PandaEnv 인스턴스
            save_dir: 저장 디렉토리
            policy_type: "random" 또는 "sinusoidal"
        """
        self.env = env
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Policy 생성
        if policy_type == "random":
            self.policy = RandomPolicy(action_dim=7)
        elif policy_type == "sinusoidal":
            self.policy = SinusoidalPolicy(action_dim=7)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
        
        self.trajectories = []
    
    def _get_state(self) -> np.ndarray:
        """현재 상태 벡터 생성."""
        obs = self.env._get_obs()
        state = np.concatenate([
            obs["joint_pos"],
            obs["joint_vel"],
            obs["ee_pos"],
            obs["target_pos"],
            np.array([0.0])  # gripper placeholder
        ])
        return state
    
    def generate_trajectory(
        self, 
        steps: int = 200,
        render: bool = False,
    ) -> Trajectory:
        """
        단일 trajectory 생성.
        
        Args:
            steps: 스텝 수
            render: 렌더링 여부
        
        Returns:
            Trajectory 객체
        """
        traj = Trajectory()
        traj.metadata["start_time"] = time.time()
        
        # Reset environment and policy
        self.env.reset()
        self.policy.reset()
        
        for step in range(steps):
            # Get state
            state = self._get_state()
            
            # Get action from policy
            action = self.policy.predict(state)
            
            # Clip to joint limits (simplified, assumes [-π, π])
            action = np.clip(action, -np.pi, np.pi)
            
            # Save state-action pair
            traj.states.append(state)
            traj.actions.append(action)
            
            # Step environment
            self.env.step(action, n_steps=5)
            
            # Render
            if render and step % 10 == 0:
                self.env.render()
        
        traj.metadata["end_time"] = time.time()
        traj.metadata["num_steps"] = steps
        
        return traj
    
    def generate(
        self,
        num_trajectories: int = 100,
        steps_per_traj: int = 200,
        save: bool = True,
    ) -> list[Trajectory]:
        """
        여러 trajectory 생성.
        
        Args:
            num_trajectories: 생성할 trajectory 수
            steps_per_traj: trajectory당 스텝 수
            save: 저장 여부
        
        Returns:
            Trajectory 리스트
        """
        print(f"Generating {num_trajectories} trajectories...")
        
        for i in tqdm(range(num_trajectories)):
            traj = self.generate_trajectory(steps=steps_per_traj)
            
            if save:
                filepath = self.save_dir / f"traj_{i:04d}.pkl"
                traj.save(str(filepath))
            
            self.trajectories.append(traj)
        
        print(f"✅ Generated {num_trajectories} trajectories ({num_trajectories * steps_per_traj} steps)")
        
        return self.trajectories
    
    def build_and_save_dataset(self, output_path: str = "data/random_dataset.pkl") -> Tuple[np.ndarray, np.ndarray]:
        """
        전체 데이터셋 빌드 및 저장.
        
        Args:
            output_path: 출력 파일 경로
        
        Returns:
            (states, actions)
        """
        if not self.trajectories:
            # Load from disk
            self.trajectories = DatasetBuilder.load_trajectories(str(self.save_dir))
        
        states, actions = DatasetBuilder.build_dataset(self.trajectories)
        DatasetBuilder.save_dataset(states, actions, output_path)
        
        return states, actions
    
    def get_statistics(self) -> dict:
        """데이터 통계 반환."""
        if not self.trajectories:
            return {"num_trajectories": 0}
        
        lengths = [len(traj) for traj in self.trajectories]
        
        # State/action 통계
        all_states = []
        all_actions = []
        for traj in self.trajectories:
            all_states.extend(traj.states)
            all_actions.extend(traj.actions)
        
        states = np.array(all_states)
        actions = np.array(all_actions)
        
        return {
            "num_trajectories": len(self.trajectories),
            "total_steps": sum(lengths),
            "avg_length": np.mean(lengths),
            "state_mean": states.mean(axis=0),
            "state_std": states.std(axis=0),
            "action_mean": actions.mean(axis=0),
            "action_std": actions.std(axis=0),
        }


def main():
    """메인 함수 - 랜덤 데이터 생성."""
    print("Random Manipulation Data Generation")
    print("=" * 60)
    
    # Environment 생성
    env = PandaEnv()
    
    # Generator 생성
    generator = DataGenerator(
        env, 
        save_dir="data/random",
        policy_type="random"  # or "sinusoidal"
    )
    
    # 데이터 생성
    generator.generate(
        num_trajectories=50,
        steps_per_traj=200,
        save=True,
    )
    
    # 데이터셋 빌드
    states, actions = generator.build_and_save_dataset("data/random_dataset.pkl")
    
    # 통계 출력
    stats = generator.get_statistics()
    print("\n" + "=" * 60)
    print("Statistics:")
    print(f"  Trajectories: {stats['num_trajectories']}")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Avg length: {stats['avg_length']:.1f}")
    print(f"  State shape: {states.shape}")
    print(f"  Action shape: {actions.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
