"""
Panda Robot Environment

Franka Emika Panda 7-DoF 로봇 조작 환경.
"""
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any


class PandaEnv:
    """
    Panda 로봇 환경 클래스.
    
    7-DoF 로봇 팔을 제어하고 관찰할 수 있는 환경.
    
    Observation Space:
        - joint_pos: (7,) 관절 위치
        - joint_vel: (7,) 관절 속도
        - ee_pos: (3,) End-effector 위치
        - target_pos: (3,) 타겟 위치
    
    Action Space:
        - (7,) 관절 목표 위치
    
    Example:
        >>> env = PandaEnv()
        >>> obs = env.reset()
        >>> action = np.zeros(7)  # 홈 포지션
        >>> obs, info = env.step(action)
    """
    
    def __init__(self, xml_path: str | None = None, render_mode: str = "human"):
        """
        Args:
            xml_path: MuJoCo XML 파일 경로 (기본: panda_scene.xml)
            render_mode: 렌더링 모드 ("human", "rgb_array", None)
        """
        if xml_path is None:
            xml_path = Path(__file__).parent / "panda_scene.xml"
        
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        
        # Joint indices
        self.joint_names = [f"panda_joint{i}" for i in range(1, 8)]
        self.actuator_names = [f"actuator{i}" for i in range(1, 8)]
        
        # Site indices
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        self.target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site"
        )
        
        # Home position (neutral pose)
        self.home_qpos = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        
        # Renderer (for rgb_array mode)
        self.renderer = None
        if render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
    
    def reset(self, seed: int | None = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        환경 리셋.
        
        Args:
            seed: 랜덤 시드
        
        Returns:
            observation: 관찰값 딕셔너리
            info: 추가 정보
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 로봇을 홈 포지션으로 이동
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.home_qpos
        
        # Forward kinematics 계산
        mujoco.mj_forward(self.model, self.data)
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def step(
        self, action: np.ndarray, n_steps: int = 10
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        환경 스텝 진행.
        
        Args:
            action: (7,) 관절 목표 위치
            n_steps: 물리 시뮬레이션 반복 횟수
        
        Returns:
            observation: 관찰값
            info: 추가 정보
        """
        # Action을 액추에이터에 적용
        self.data.ctrl[:] = action
        
        # 시뮬레이션 진행
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        현재 관찰값 반환.
        
        Returns:
            observation 딕셔너리
        """
        # Joint states
        joint_pos = self.data.qpos[:7].copy()
        joint_vel = self.data.qvel[:7].copy()
        
        # End-effector position
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        
        # Target position
        target_pos = self.data.site_xpos[self.target_site_id].copy()
        
        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "ee_pos": ee_pos,
            "target_pos": target_pos,
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """추가 정보 반환."""
        obs = self._get_obs()
        ee_to_target = np.linalg.norm(obs["ee_pos"] - obs["target_pos"])
        
        return {
            "time": self.data.time,
            "ee_to_target_dist": ee_to_target,
        }
    
    def render(self) -> np.ndarray | None:
        """
        화면 렌더링.
        
        Returns:
            RGB 이미지 (rgb_array 모드) 또는 None (human 모드)
        """
        if self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        return None
    
    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        End-effector의 위치와 회전 반환.
        
        Returns:
            position: (3,) 위치
            rotation: (3, 3) 회전 행렬
        """
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        return ee_pos, ee_mat
    
    def set_joint_positions(self, qpos: np.ndarray) -> None:
        """
        관절 위치 직접 설정 (키네마틱).
        
        Args:
            qpos: (7,) 관절 위치
        """
        self.data.qpos[:7] = qpos
        mujoco.mj_forward(self.model, self.data)
    
    def launch_viewer(self, duration: float | None = None) -> None:
        """
        인터랙티브 뷰어 실행.
        
        Args:
            duration: 실행 시간 (초)
        """
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 135
            
            start_time = self.data.time
            
            # 간단한 데모: 랜덤 관절 움직임
            while viewer.is_running():
                # 매 2초마다 랜덤 목표 생성
                if int(self.data.time) % 2 == 0 and self.data.time - start_time > 0.1:
                    target_qpos = self.home_qpos + np.random.uniform(-0.5, 0.5, 7)
                    self.data.ctrl[:] = target_qpos
                
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                
                if duration and (self.data.time - start_time) >= duration:
                    break


def main():
    """메인 함수 - 환경 데모."""
    print("Panda Robot Environment")
    print("-" * 40)
    
    env = PandaEnv()
    obs, info = env.reset()
    
    print("Initial observation:")
    print(f"  Joint positions: {obs['joint_pos']}")
    print(f"  EE position: {obs['ee_pos']}")
    print(f"  Target position: {obs['target_pos']}")
    print(f"  Distance to target: {info['ee_to_target_dist']:.3f}m")
    
    print("\nLaunching viewer...")
    env.launch_viewer()


if __name__ == "__main__":
    main()
