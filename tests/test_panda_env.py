"""Test Panda Robot Environment."""
import numpy as np
import pytest
import mujoco

from src.envs.panda_env import PandaEnv


def test_env_initialization():
    """환경 초기화 테스트"""
    env = PandaEnv()
    assert env.model is not None
    assert env.data is not None
    assert len(env.joint_names) == 7
    assert len(env.home_qpos) == 7


def test_reset():
    """환경 리셋 테스트"""
    env = PandaEnv()
    obs, info = env.reset(seed=42)
    
    # Observation 구조 확인
    assert "joint_pos" in obs
    assert "joint_vel" in obs
    assert "ee_pos" in obs
    assert "target_pos" in obs
    
    # Shape 확인
    assert obs["joint_pos"].shape == (7,)
    assert obs["joint_vel"].shape == (7,)
    assert obs["ee_pos"].shape == (3,)
    assert obs["target_pos"].shape == (3,)
    
    # Info 확인
    assert "time" in info
    assert "ee_to_target_dist" in info
    assert info["time"] == 0.0


def test_step():
    """환경 스텝 테스트"""
    env = PandaEnv()
    env.reset()
    
    # 홈 포지션으로 액션
    action = env.home_qpos
    obs, info = env.step(action, n_steps=10)
    
    # 관찰값 확인
    assert obs["joint_pos"].shape == (7,)
    assert not np.any(np.isnan(obs["joint_pos"]))
    assert not np.any(np.isnan(obs["ee_pos"]))


def test_home_position():
    """홈 포지션 테스트"""
    env = PandaEnv()
    env.reset()
    
    # 홈 포지션 설정
    env.set_joint_positions(env.home_qpos)
    obs = env._get_obs()
    
    # 관절 위치가 홈 포지션인지 확인
    np.testing.assert_allclose(obs["joint_pos"], env.home_qpos, atol=1e-6)


def test_ee_pose():
    """End-effector 위치 테스트"""
    env = PandaEnv()
    env.reset()
    
    ee_pos, ee_rot = env.get_ee_pose()
    
    # Shape 확인
    assert ee_pos.shape == (3,)
    assert ee_rot.shape == (3, 3)
    
    # 회전 행렬 확인 (직교 행렬)
    identity = np.eye(3)
    np.testing.assert_allclose(
        ee_rot @ ee_rot.T, identity, atol=1e-5
    )


def test_observation_consistency():
    """관찰값 일관성 테스트"""
    env = PandaEnv()
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    
    # 같은 시드로 리셋하면 같은 관찰값
    np.testing.assert_allclose(obs1["joint_pos"], obs2["joint_pos"])
    np.testing.assert_allclose(obs1["ee_pos"], obs2["ee_pos"])


def test_physics_determinism():
    """물리 시뮬레이션 결정성 테스트"""
    env1 = PandaEnv()
    env2 = PandaEnv()
    
    # 같은 초기 상태
    env1.reset(seed=42)
    env2.reset(seed=42)
    
    # 같은 액션
    action = np.array([0.1, -0.5, 0.2, -2.0, 0.1, 1.5, 0.8])
    
    obs1, _ = env1.step(action, n_steps=100)
    obs2, _ = env2.step(action, n_steps=100)
    
    # 결과가 같아야 함
    np.testing.assert_allclose(obs1["joint_pos"], obs2["joint_pos"], atol=1e-10)


def test_joint_limits():
    """관절 한계 확인"""
    env = PandaEnv()
    env.reset()
    
    # 관절 범위 가져오기
    for i, joint_name in enumerate(env.joint_names):
        joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        joint_range = env.model.jnt_range[joint_id]
        assert len(joint_range) == 2  # [min, max]
        assert joint_range[0] < joint_range[1]


def test_render_rgb_array():
    """RGB 렌더링 테스트"""
    env = PandaEnv(render_mode="rgb_array")
    env.reset()
    
    img = env.render()
    assert img is not None
    assert img.shape == (480, 640, 3)
    assert img.dtype == np.uint8


def test_distance_calculation():
    """거리 계산 테스트"""
    env = PandaEnv()
    obs, info = env.reset()
    
    # 수동 거리 계산
    manual_dist = np.linalg.norm(obs["ee_pos"] - obs["target_pos"])
    
    # Info의 거리와 비교
    np.testing.assert_allclose(info["ee_to_target_dist"], manual_dist)


@pytest.mark.parametrize("n_steps", [1, 10, 50, 100])
def test_different_step_counts(n_steps):
    """다양한 스텝 수 테스트"""
    env = PandaEnv()
    env.reset()
    
    action = env.home_qpos
    obs, info = env.step(action, n_steps=n_steps)
    
    assert obs["joint_pos"].shape == (7,)
    assert not np.any(np.isnan(obs["joint_pos"]))
