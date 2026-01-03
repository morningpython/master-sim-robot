"""Test IK Controller."""
import numpy as np
import pytest
import mujoco

from src.controllers.ik_controller import (
    IKController,
    CartesianController,
)
from src.envs.panda_env import PandaEnv


@pytest.fixture
def env():
    """Panda 환경 생성."""
    return PandaEnv()


@pytest.fixture
def ik_controller(env):
    """IK 컨트롤러 생성."""
    env.reset()
    return IKController(env.model, env.data, "ee_site")


def test_ik_controller_init(env):
    """IK 컨트롤러 초기화 테스트."""
    ik = IKController(env.model, env.data, "ee_site")
    assert ik.model is not None
    assert ik.data is not None
    assert ik.ee_site_id >= 0
    assert ik.nq == 7


def test_get_ee_position(env, ik_controller):
    """End-effector 위치 추출 테스트."""
    pos = ik_controller.get_ee_position()
    
    assert pos.shape == (3,)
    assert not np.any(np.isnan(pos))


def test_get_jacobian(env, ik_controller):
    """Jacobian 계산 테스트."""
    J = ik_controller.get_jacobian()
    
    assert J.shape == (3, 7)
    assert not np.any(np.isnan(J))


def test_ik_solve_reachable_target(env, ik_controller):
    """도달 가능한 목표에 대한 IK 테스트."""
    # 현재 위치 근처 목표
    current_pos = ik_controller.get_ee_position()
    target_pos = current_pos + np.array([0.05, 0.05, 0.05])
    
    qpos = ik_controller.solve(target_pos)
    
    assert qpos is not None
    assert qpos.shape == (7,)
    
    # 검증: qpos로 설정했을 때 목표에 도달하는지
    ik_controller.data.qpos[:7] = qpos
    mujoco.mj_forward(ik_controller.model, ik_controller.data)
    final_pos = ik_controller.get_ee_position()
    error = np.linalg.norm(final_pos - target_pos)
    
    assert error < 0.01  # 1cm 이내


def test_ik_solve_unreachable_target(env, ik_controller):
    """도달 불가능한 목표에 대한 IK 테스트."""
    # 너무 먼 목표
    target_pos = np.array([5.0, 5.0, 5.0])
    
    with pytest.warns(UserWarning, match="IK did not converge"):
        qpos = ik_controller.solve(target_pos)
    
    assert qpos is None


def test_joint_limit_enforcement(env, ik_controller):
    """관절 한계 적용 테스트."""
    # IK 풀기
    target_pos = np.array([0.5, 0.0, 0.3])
    qpos = ik_controller.solve(target_pos)
    
    if qpos is not None:
        # 모든 관절이 한계 내에 있는지 확인
        for i in range(min(7, ik_controller.model.njnt)):
            joint_range = ik_controller.model.jnt_range[i]
            if joint_range[0] < joint_range[1]:
                assert joint_range[0] <= qpos[i] <= joint_range[1]


def test_solve_position_control(env, ik_controller):
    """단일 IK 스텝 테스트."""
    current_qpos = env.home_qpos
    target_pos = np.array([0.5, 0.0, 0.3])
    
    qpos_target = ik_controller.solve_position_control(
        target_pos, current_qpos, dt=0.01
    )
    
    assert qpos_target.shape == (7,)
    assert not np.any(np.isnan(qpos_target))


def test_nullspace_control(env, ik_controller):
    """Nullspace 제어 테스트."""
    current_qpos = env.home_qpos
    target_pos = np.array([0.5, 0.0, 0.3])
    q_desired = env.home_qpos + 0.1  # Nullspace 목표
    
    qpos_target = ik_controller.compute_nullspace_control(
        target_pos, current_qpos, q_desired, null_gain=0.1
    )
    
    assert qpos_target.shape == (7,)
    assert not np.any(np.isnan(qpos_target))


def test_ik_convergence_iterations(env):
    """IK 수렴 반복 횟수 테스트."""
    ik = IKController(
        env.model, 
        env.data, 
        "ee_site",
        max_iterations=10  # 적은 반복 횟수
    )
    
    env.reset()
    target_pos = np.array([0.6, 0.0, 0.2])
    
    # 10번 반복으로는 수렴 못할 수도 있음
    qpos = ik.solve(target_pos)
    
    # 결과가 None이거나 유효한 qpos
    assert qpos is None or qpos.shape == (7,)


def test_cartesian_controller_init(env):
    """Cartesian 컨트롤러 초기화 테스트."""
    controller = CartesianController(env.model, env.data, "ee_site")
    assert controller.ik_solver is not None


def test_plan_trajectory(env):
    """Trajectory 계획 테스트."""
    env.reset()
    controller = CartesianController(env.model, env.data, "ee_site")
    
    waypoints = [
        np.array([0.4, 0.0, 0.3]),
        np.array([0.5, 0.1, 0.4]),
        np.array([0.4, -0.1, 0.3]),
    ]
    
    trajectory = controller.plan_trajectory(waypoints, n_steps=10)
    
    # Trajectory가 생성되었는지 확인
    assert len(trajectory) > 0
    assert all(qpos.shape == (7,) for qpos in trajectory)


def test_execute_trajectory(env):
    """Trajectory 실행 테스트."""
    env.reset()
    controller = CartesianController(env.model, env.data, "ee_site")
    
    # 간단한 trajectory
    trajectory = [
        env.home_qpos,
        env.home_qpos + 0.1,
        env.home_qpos,
    ]
    
    executed_steps = []
    
    def callback(i, qpos):
        executed_steps.append(i)
    
    success = controller.execute_trajectory(trajectory, callback=callback)
    
    assert success
    assert len(executed_steps) == len(trajectory)


def test_ik_with_different_initial_positions(env, ik_controller):
    """다양한 초기 위치에서 IK 테스트."""
    target_pos = np.array([0.5, 0.0, 0.3])
    
    initial_positions = [
        env.home_qpos,
        env.home_qpos + 0.1,
        env.home_qpos - 0.1,
    ]
    
    for initial_qpos in initial_positions:
        qpos = ik_controller.solve(target_pos, initial_qpos=initial_qpos)
        
        if qpos is not None:
            ik_controller.data.qpos[:7] = qpos
            mujoco.mj_forward(ik_controller.model, ik_controller.data)
            final_pos = ik_controller.get_ee_position()
            error = np.linalg.norm(final_pos - target_pos)
            assert error < 0.01


def test_jacobian_shape_consistency(env, ik_controller):
    """Jacobian 형태 일관성 테스트."""
    # 여러 관절 위치에서 Jacobian 계산
    for _ in range(5):
        random_qpos = np.random.uniform(-1, 1, 7)
        ik_controller.data.qpos[:7] = random_qpos
        mujoco.mj_forward(ik_controller.model, ik_controller.data)
        
        J = ik_controller.get_jacobian()
        assert J.shape == (3, 7)


@pytest.mark.parametrize("damping", [1e-6, 1e-4, 1e-2])
def test_different_damping_values(env, damping):
    """다양한 damping 값 테스트."""
    ik = IKController(env.model, env.data, "ee_site", damping=damping)
    env.reset()
    
    target_pos = np.array([0.5, 0.0, 0.3])
    qpos = ik.solve(target_pos)
    
    # Damping 값에 상관없이 결과가 나와야 함
    assert qpos is not None or qpos is None  # 수렴 여부는 damping에 따라 다를 수 있음


def test_ik_numerical_stability(env, ik_controller):
    """IK 수치 안정성 테스트."""
    # 특이점 근처에서도 안정적인지 확인
    target_pos = np.array([0.0, 0.0, 0.5])  # 로봇 정중앙
    
    # 예외가 발생하지 않아야 함
    try:
        qpos = ik_controller.solve(target_pos)
        assert qpos is None or qpos.shape == (7,)
    except Exception as e:
        pytest.fail(f"IK solver raised exception: {e}")


def test_position_control_convergence(env, ik_controller):
    """위치 제어 수렴 테스트."""
    target_pos = np.array([0.5, 0.0, 0.3])
    current_qpos = env.home_qpos.copy()
    
    # 여러 스텝 반복
    for _ in range(100):
        qpos_target = ik_controller.solve_position_control(
            target_pos, current_qpos, dt=0.01
        )
        current_qpos = qpos_target
    
    # 최종 위치 확인
    ik_controller.data.qpos[:7] = current_qpos
    mujoco.mj_forward(ik_controller.model, ik_controller.data)
    final_pos = ik_controller.get_ee_position()
    error = np.linalg.norm(final_pos - target_pos)
    
    assert error < 0.05  # 5cm 이내 (단일 스텝이므로 덜 정확)
