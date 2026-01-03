"""
Inverse Kinematics Controller

Jacobian 기반 IK 솔버를 사용하여 end-effector를 목표 위치로 이동합니다.
"""
import mujoco
import numpy as np
from typing import Optional, Tuple
import warnings


class IKController:
    """
    역운동학(Inverse Kinematics) 컨트롤러.
    
    Jacobian pseudo-inverse를 사용하여 end-effector를 목표 위치로 이동하는
    관절 속도를 계산합니다.
    
    Args:
        model: MuJoCo 모델
        data: MuJoCo 데이터
        ee_site_name: End-effector site 이름
        damping: Damping factor (특이점 회피)
        step_size: IK 스텝 크기
        max_iterations: 최대 반복 횟수
        tolerance: 수렴 허용 오차 (m)
    
    Example:
        >>> from src.envs.panda_env import PandaEnv
        >>> env = PandaEnv()
        >>> env.reset()
        >>> ik = IKController(env.model, env.data, "ee_site")
        >>> target = np.array([0.5, 0.0, 0.3])
        >>> qpos = ik.solve(target)
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ee_site_name: str = "ee_site",
        damping: float = 1e-4,
        step_size: float = 0.5,
        max_iterations: int = 100,
        tolerance: float = 1e-3,
    ):
        self.model = model
        self.data = data
        self.damping = damping
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Site ID 찾기
        self.ee_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name
        )
        
        # Joint 정보
        self.nq = model.nv  # DoF
        
    def get_ee_position(self) -> np.ndarray:
        """
        현재 end-effector 위치 반환.
        
        Returns:
            position: (3,) End-effector 위치
        """
        return self.data.site_xpos[self.ee_site_id].copy()
    
    def get_jacobian(self) -> np.ndarray:
        """
        End-effector의 Jacobian 행렬 계산.
        
        Returns:
            jacobian: (3, nq) Jacobian 행렬 (position only)
        """
        # Jacobian 버퍼 생성
        jacp = np.zeros((3, self.model.nv))  # Position Jacobian
        jacr = np.zeros((3, self.model.nv))  # Rotation Jacobian
        
        # Jacobian 계산
        mujoco.mj_jacSite(
            self.model, 
            self.data, 
            jacp, 
            jacr, 
            self.ee_site_id
        )
        
        return jacp
    
    def solve(
        self, 
        target_pos: np.ndarray,
        initial_qpos: Optional[np.ndarray] = None,
        position_only: bool = True,
    ) -> Optional[np.ndarray]:
        """
        IK 문제 풀기.
        
        Args:
            target_pos: (3,) 목표 위치
            initial_qpos: (nq,) 초기 관절 위치 (None이면 현재 상태 사용)
            position_only: True면 위치만 고려, False면 자세도 고려
        
        Returns:
            qpos: (nq,) 목표를 달성하는 관절 위치, 실패 시 None
        """
        # 초기 관절 위치 설정
        if initial_qpos is not None:
            self.data.qpos[:self.nq] = initial_qpos
        
        # 반복적으로 IK 풀기
        for iteration in range(self.max_iterations):
            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)
            
            # 현재 위치와 목표 위치 차이
            current_pos = self.get_ee_position()
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)
            
            # 수렴 확인
            if error_norm < self.tolerance:
                return self.data.qpos[:self.nq].copy()
            
            # Jacobian 계산
            J = self.get_jacobian()
            
            # Damped least squares (pseudo-inverse)
            # dq = J^T (J J^T + λI)^-1 error
            JJT = J @ J.T
            damped_inv = np.linalg.inv(JJT + self.damping * np.eye(3))
            dq = J.T @ damped_inv @ error
            
            # 관절 위치 업데이트
            self.data.qpos[:self.nq] += self.step_size * dq
            
            # 관절 한계 적용
            self._enforce_joint_limits()
        
        # 수렴 실패
        warnings.warn(
            f"IK did not converge after {self.max_iterations} iterations. "
            f"Final error: {error_norm:.4f}m"
        )
        return None
    
    def _enforce_joint_limits(self) -> None:
        """관절 한계 적용."""
        for i in range(min(self.nq, self.model.njnt)):
            joint_range = self.model.jnt_range[i]
            if joint_range[0] < joint_range[1]:  # 한계가 정의된 경우
                self.data.qpos[i] = np.clip(
                    self.data.qpos[i],
                    joint_range[0],
                    joint_range[1]
                )
    
    def solve_position_control(
        self,
        target_pos: np.ndarray,
        current_qpos: np.ndarray,
        dt: float = 0.01,
    ) -> np.ndarray:
        """
        위치 제어를 위한 단일 IK 스텝.
        
        실시간 제어에 사용. 한 번의 Jacobian 계산으로 관절 속도를 계산합니다.
        
        Args:
            target_pos: (3,) 목표 위치
            current_qpos: (nq,) 현재 관절 위치
            dt: 시간 간격
        
        Returns:
            qpos_target: (nq,) 목표 관절 위치
        """
        # 현재 상태 설정
        self.data.qpos[:self.nq] = current_qpos
        mujoco.mj_forward(self.model, self.data)
        
        # 위치 오차
        current_pos = self.get_ee_position()
        error = target_pos - current_pos
        
        # Jacobian
        J = self.get_jacobian()
        
        # Damped least squares
        JJT = J @ J.T
        damped_inv = np.linalg.inv(JJT + self.damping * np.eye(3))
        dq = J.T @ damped_inv @ error
        
        # 관절 속도 → 관절 위치
        qpos_target = current_qpos + dq * dt
        
        return qpos_target
    
    def compute_nullspace_control(
        self,
        target_pos: np.ndarray,
        current_qpos: np.ndarray,
        q_desired: np.ndarray,
        null_gain: float = 0.1,
    ) -> np.ndarray:
        """
        Nullspace 제어 (부가 목표 달성).
        
        Primary task: End-effector를 목표 위치로 이동
        Secondary task: 관절을 원하는 위치로 이동 (nullspace에서)
        
        Args:
            target_pos: (3,) 목표 위치
            current_qpos: (nq,) 현재 관절 위치
            q_desired: (nq,) 원하는 관절 위치 (nullspace 목표)
            null_gain: Nullspace 제어 이득
        
        Returns:
            qpos_target: (nq,) 목표 관절 위치
        """
        # 현재 상태
        self.data.qpos[:self.nq] = current_qpos
        mujoco.mj_forward(self.model, self.data)
        
        # Primary task (IK)
        current_pos = self.get_ee_position()
        error = target_pos - current_pos
        
        J = self.get_jacobian()
        JJT = J @ J.T
        damped_inv = np.linalg.inv(JJT + self.damping * np.eye(3))
        
        # Primary task 속도
        dq_primary = J.T @ damped_inv @ error
        
        # Nullspace projector: N = I - J^+ J
        J_pinv = J.T @ damped_inv  # Damped pseudo-inverse
        N = np.eye(self.nq) - J_pinv @ J
        
        # Secondary task (nullspace에서)
        dq_null = null_gain * (q_desired - current_qpos)
        dq_secondary = N @ dq_null
        
        # 결합
        dq_total = dq_primary + dq_secondary
        qpos_target = current_qpos + dq_total
        
        return qpos_target


class CartesianController:
    """
    Cartesian 공간 컨트롤러 (IK 기반).
    
    목표 경로를 따라 end-effector를 이동시킵니다.
    
    Example:
        >>> from src.envs.panda_env import PandaEnv
        >>> env = PandaEnv()
        >>> env.reset()
        >>> controller = CartesianController(env.model, env.data)
        >>> waypoints = [
        ...     np.array([0.4, 0.0, 0.3]),
        ...     np.array([0.5, 0.1, 0.4]),
        ...     np.array([0.3, -0.1, 0.2]),
        ... ]
        >>> trajectory = controller.plan_trajectory(waypoints)
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ee_site_name: str = "ee_site",
    ):
        self.ik_solver = IKController(model, data, ee_site_name)
        self.model = model
        self.data = data
    
    def plan_trajectory(
        self,
        waypoints: list[np.ndarray],
        n_steps: int = 50,
    ) -> list[np.ndarray]:
        """
        Waypoint를 따라 trajectory 생성.
        
        Args:
            waypoints: List of (3,) 위치
            n_steps: Waypoint 사이의 스텝 수
        
        Returns:
            trajectory: List of (nq,) 관절 위치
        """
        trajectory = []
        
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            
            # Linear interpolation
            for t in np.linspace(0, 1, n_steps):
                target = (1 - t) * start + t * end
                qpos = self.ik_solver.solve(target)
                
                if qpos is not None:
                    trajectory.append(qpos)
        
        return trajectory
    
    def execute_trajectory(
        self,
        trajectory: list[np.ndarray],
        callback=None,
    ) -> bool:
        """
        Trajectory 실행.
        
        Args:
            trajectory: List of (nq,) 관절 위치
            callback: 각 스텝마다 호출되는 함수 (선택)
        
        Returns:
            성공 여부
        """
        for i, qpos in enumerate(trajectory):
            self.data.qpos[:len(qpos)] = qpos
            mujoco.mj_forward(self.model, self.data)
            
            if callback:
                callback(i, qpos)
        
        return True


def main():
    """메인 함수 - IK 컨트롤러 테스트."""
    from src.envs.panda_env import PandaEnv
    
    print("IK Controller Test")
    print("=" * 60)
    
    # Environment 생성
    env = PandaEnv()
    env.reset()
    
    # IK Controller 생성
    ik = IKController(env.model, env.data, "ee_site")
    
    # 현재 위치
    current_pos = ik.get_ee_position()
    print(f"Current EE position: {current_pos}")
    
    # 목표 위치
    target_pos = np.array([0.5, 0.2, 0.3])
    print(f"Target position: {target_pos}")
    
    # IK 풀기
    qpos = ik.solve(target_pos)
    
    if qpos is not None:
        print(f"✅ IK solved!")
        print(f"Joint positions: {qpos}")
        
        # 검증
        env.set_joint_positions(qpos)
        final_pos = ik.get_ee_position()
        error = np.linalg.norm(final_pos - target_pos)
        print(f"Final EE position: {final_pos}")
        print(f"Error: {error:.4f}m")
    else:
        print("❌ IK failed to converge")


if __name__ == "__main__":
    main()
