"""
Teleoperation Data Collection Tool

키보드 입력으로 로봇을 조작하고 demonstration data를 수집합니다.
"""
import mujoco
import mujoco.viewer
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import time


@dataclass
class Trajectory:
    """
    단일 시연(trajectory) 데이터.
    
    Attributes:
        states: 상태 리스트 [(joint_pos, joint_vel, ee_pos), ...]
        actions: 액션 리스트 [joint_target_pos, ...]
        metadata: 추가 정보 (시간, 성공 여부 등)
    """
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.states)
    
    def save(self, filepath: str) -> None:
        """trajectory를 pickle 파일로 저장."""
        data = {
            "states": np.array(self.states),
            "actions": np.array(self.actions),
            "metadata": self.metadata,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load(filepath: str) -> "Trajectory":
        """pickle 파일에서 trajectory 로드."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        traj = Trajectory()
        traj.states = list(data["states"])
        traj.actions = list(data["actions"])
        traj.metadata = data["metadata"]
        return traj


class TeleoperationCollector:
    """
    텔레오퍼레이션 데이터 수집기.
    
    키보드 입력으로 로봇을 조작하고 state-action pair를 저장합니다.
    
    Controls:
        - Q/A: Joint 1 (+/-)
        - W/S: Joint 2 (+/-)
        - E/D: Joint 3 (+/-)
        - R/F: Joint 4 (+/-)
        - T/G: Joint 5 (+/-)
        - Y/H: Joint 6 (+/-)
        - U/J: Joint 7 (+/-)
        - Space: Start/Stop recording
        - Enter: Save trajectory
        - ESC: Exit
    
    Example:
        >>> from src.envs.panda_env import PandaEnv
        >>> env = PandaEnv()
        >>> collector = TeleoperationCollector(env, save_dir="data/demos")
        >>> collector.run()
    """
    
    def __init__(self, env, save_dir: str = "data/demos", delta: float = 0.05):
        """
        Args:
            env: PandaEnv 인스턴스
            save_dir: 데이터 저장 디렉토리
            delta: 키 입력당 관절 이동량 (radians)
        """
        self.env = env
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.delta = delta
        
        # Current trajectory
        self.current_traj = Trajectory()
        self.is_recording = False
        
        # All collected trajectories
        self.trajectories: List[Trajectory] = []
        
        # Current target position
        self.target_qpos = self.env.home_qpos.copy()
        
        # Key mappings
        self.key_to_joint = {
            ord('Q'): (0, +1), ord('A'): (0, -1),
            ord('W'): (1, +1), ord('S'): (1, -1),
            ord('E'): (2, +1), ord('D'): (2, -1),
            ord('R'): (3, +1), ord('F'): (3, -1),
            ord('T'): (4, +1), ord('G'): (4, -1),
            ord('Y'): (5, +1), ord('H'): (5, -1),
            ord('U'): (6, +1), ord('J'): (6, -1),
        }
    
    def _get_state(self) -> np.ndarray:
        """
        현재 상태 벡터 생성.
        
        Returns:
            state: (21,) [joint_pos(7), joint_vel(7), ee_pos(3), target_pos(3), gripper(1)]
        """
        obs = self.env._get_obs()
        state = np.concatenate([
            obs["joint_pos"],      # 7
            obs["joint_vel"],      # 7
            obs["ee_pos"],         # 3
            obs["target_pos"],     # 3
            np.array([0.0])        # 1 (gripper, placeholder)
        ])
        return state
    
    def _handle_keyboard(self, key: int) -> bool:
        """
        키보드 입력 처리.
        
        Args:
            key: 키 코드
        
        Returns:
            계속 실행 여부
        """
        # Joint control
        if key in self.key_to_joint:
            joint_idx, direction = self.key_to_joint[key]
            self.target_qpos[joint_idx] += direction * self.delta
            
            # Clamp to joint limits
            joint_id = mujoco.mj_name2id(
                self.env.model, 
                mujoco.mjtObj.mjOBJ_JOINT, 
                self.env.joint_names[joint_idx]
            )
            limits = self.env.model.jnt_range[joint_id]
            self.target_qpos[joint_idx] = np.clip(
                self.target_qpos[joint_idx], limits[0], limits[1]
            )
        
        # Recording control
        elif key == ord(' '):  # Space - toggle recording
            self.is_recording = not self.is_recording
            if self.is_recording:
                self.current_traj = Trajectory()
                self.current_traj.metadata["start_time"] = time.time()
                print("🔴 Recording started")
            else:
                self.current_traj.metadata["end_time"] = time.time()
                duration = (self.current_traj.metadata["end_time"] - 
                           self.current_traj.metadata["start_time"])
                print(f"⏸️  Recording stopped ({len(self.current_traj)} steps, {duration:.1f}s)")
        
        elif key == ord('\r'):  # Enter - save trajectory
            if len(self.current_traj) > 0:
                self._save_current_trajectory()
            else:
                print("⚠️  No data to save")
        
        elif key == 27:  # ESC
            return False
        
        return True
    
    def _save_current_trajectory(self) -> None:
        """현재 trajectory 저장."""
        traj_id = len(self.trajectories)
        filepath = self.save_dir / f"traj_{traj_id:04d}.pkl"
        
        self.current_traj.metadata["trajectory_id"] = traj_id
        self.current_traj.save(str(filepath))
        
        self.trajectories.append(self.current_traj)
        
        print(f"✅ Saved trajectory {traj_id} ({len(self.current_traj)} steps) to {filepath}")
        
        # Reset
        self.current_traj = Trajectory()
        self.is_recording = False
    
    def run(self, headless: bool = False) -> None:
        """
        텔레오퍼레이션 수집 실행.
        
        Args:
            headless: True면 viewer 없이 실행 (테스트용)
        """
        print("\n" + "="*60)
        print("Teleoperation Data Collection")
        print("="*60)
        print("Controls:")
        print("  Q/A, W/S, E/D, R/F, T/G, Y/H, U/J: Control joints 1-7")
        print("  Space: Start/Stop recording")
        print("  Enter: Save current trajectory")
        print("  ESC: Exit")
        print("="*60 + "\n")
        
        if headless:
            # Headless mode (for testing)
            self.env.reset()
            for _ in range(100):
                state = self._get_state()
                action = self.target_qpos.copy()
                
                if self.is_recording:
                    self.current_traj.states.append(state)
                    self.current_traj.actions.append(action)
                
                self.env.step(action)
            return
        
        # Interactive mode
        self.env.reset()
        
        with mujoco.viewer.launch_passive(self.env.model, self.env.data) as viewer:
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 135
            
            while viewer.is_running():
                # Get current state
                state = self._get_state()
                action = self.target_qpos.copy()
                
                # Record if active
                if self.is_recording:
                    self.current_traj.states.append(state)
                    self.current_traj.actions.append(action)
                
                # Step environment
                self.env.step(action, n_steps=5)
                
                # Sync viewer
                viewer.sync()
                
                # Check for exit (this is simplified, real key handling needs viewer integration)
                # In practice, you'd use viewer's key callback
    
    def get_statistics(self) -> Dict[str, any]:
        """수집된 데이터 통계 반환."""
        if not self.trajectories:
            return {"num_trajectories": 0}
        
        lengths = [len(traj) for traj in self.trajectories]
        
        return {
            "num_trajectories": len(self.trajectories),
            "total_steps": sum(lengths),
            "avg_length": np.mean(lengths),
            "min_length": np.min(lengths),
            "max_length": np.max(lengths),
        }


class DatasetBuilder:
    """수집된 trajectory들을 학습용 데이터셋으로 변환."""
    
    @staticmethod
    def load_trajectories(data_dir: str) -> List[Trajectory]:
        """디렉토리에서 모든 trajectory 로드."""
        data_dir = Path(data_dir)
        trajectories = []
        
        for pkl_file in sorted(data_dir.glob("traj_*.pkl")):
            traj = Trajectory.load(str(pkl_file))
            trajectories.append(traj)
        
        return trajectories
    
    @staticmethod
    def build_dataset(
        trajectories: List[Trajectory]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trajectory 리스트를 (states, actions) 배열로 변환.
        
        Args:
            trajectories: Trajectory 리스트
        
        Returns:
            states: (N, state_dim)
            actions: (N, action_dim)
        """
        all_states = []
        all_actions = []
        
        for traj in trajectories:
            all_states.extend(traj.states)
            all_actions.extend(traj.actions)
        
        states = np.array(all_states)
        actions = np.array(all_actions)
        
        return states, actions
    
    @staticmethod
    def save_dataset(states: np.ndarray, actions: np.ndarray, filepath: str) -> None:
        """데이터셋을 파일로 저장."""
        dataset = {
            "states": states,
            "actions": actions,
            "metadata": {
                "num_samples": len(states),
                "state_dim": states.shape[1],
                "action_dim": actions.shape[1],
            }
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(dataset, f)
        
        print(f"✅ Saved dataset: {len(states)} samples to {filepath}")


def main():
    """메인 함수."""
    from src.envs.panda_env import PandaEnv
    
    env = PandaEnv()
    collector = TeleoperationCollector(env, save_dir="data/demos")
    collector.run()
    
    # Print statistics
    stats = collector.get_statistics()
    print("\n" + "="*60)
    print("Collection Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("="*60)


if __name__ == "__main__":
    main()
