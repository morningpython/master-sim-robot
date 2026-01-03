"""Test random policy and data generation."""
import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil

from src.utils.random_policy import (
    RandomPolicy,
    SinusoidalPolicy,
    DataGenerator,
)
from src.envs.panda_env import PandaEnv


@pytest.fixture
def temp_dir():
    """임시 디렉토리 생성."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


def test_random_policy_init():
    """RandomPolicy 초기화 테스트."""
    policy = RandomPolicy(action_dim=7, action_scale=0.1)
    assert policy.action_dim == 7
    assert policy.action_scale == 0.1
    assert len(policy.prev_action) == 7


def test_random_policy_predict():
    """RandomPolicy 예측 테스트."""
    policy = RandomPolicy(action_dim=7)
    state = np.zeros(21)
    
    action = policy.predict(state)
    
    assert action.shape == (7,)
    assert not np.any(np.isnan(action))


def test_random_policy_reset():
    """RandomPolicy 리셋 테스트."""
    policy = RandomPolicy()
    
    # 액션 생성
    state = np.zeros(21)
    policy.predict(state)
    
    # 리셋
    policy.reset()
    np.testing.assert_array_equal(policy.prev_action, np.zeros(7))


def test_random_policy_smoothness():
    """RandomPolicy 부드러움 테스트 (momentum)."""
    policy = RandomPolicy(action_dim=7, action_scale=0.01)
    state = np.zeros(21)
    
    actions = []
    for _ in range(10):
        action = policy.predict(state)
        actions.append(action)
    
    # 액션 변화가 크지 않아야 함
    actions = np.array(actions)
    deltas = np.diff(actions, axis=0)
    max_delta = np.abs(deltas).max()
    
    assert max_delta < 0.5  # 부드러운 변화


def test_sinusoidal_policy_init():
    """SinusoidalPolicy 초기화 테스트."""
    policy = SinusoidalPolicy(action_dim=7)
    assert policy.action_dim == 7
    assert len(policy.frequency) == 7
    assert len(policy.amplitude) == 7
    assert policy.time == 0.0


def test_sinusoidal_policy_predict():
    """SinusoidalPolicy 예측 테스트."""
    policy = SinusoidalPolicy(action_dim=7)
    state = np.zeros(21)
    
    action = policy.predict(state, dt=0.01)
    
    assert action.shape == (7,)
    assert not np.any(np.isnan(action))


def test_sinusoidal_policy_periodicity():
    """SinusoidalPolicy 주기성 테스트."""
    # 고정된 주파수/진폭
    frequency = np.ones(7) * 1.0  # 1 Hz
    amplitude = np.ones(7) * 0.5
    
    policy = SinusoidalPolicy(
        action_dim=7,
        frequency=frequency,
        amplitude=amplitude,
    )
    
    state = np.zeros(21)
    
    # 1초 후 (1 period)
    for _ in range(100):  # 100 steps × 0.01s = 1s
        action = policy.predict(state, dt=0.01)
    
    # 시간 확인
    assert abs(policy.time - 1.0) < 0.01


def test_data_generator_init(temp_dir):
    """DataGenerator 초기화 테스트."""
    env = PandaEnv()
    generator = DataGenerator(env, save_dir=temp_dir, policy_type="random")
    
    assert generator.env is not None
    assert generator.save_dir == Path(temp_dir)
    assert isinstance(generator.policy, RandomPolicy)


def test_data_generator_sinusoidal(temp_dir):
    """DataGenerator sinusoidal 정책 테스트."""
    env = PandaEnv()
    generator = DataGenerator(env, save_dir=temp_dir, policy_type="sinusoidal")
    
    assert isinstance(generator.policy, SinusoidalPolicy)


def test_data_generator_invalid_policy(temp_dir):
    """DataGenerator 잘못된 정책 타입 테스트."""
    env = PandaEnv()
    
    with pytest.raises(ValueError, match="Unknown policy type"):
        DataGenerator(env, save_dir=temp_dir, policy_type="invalid")


def test_get_state():
    """상태 벡터 생성 테스트."""
    env = PandaEnv()
    env.reset()
    generator = DataGenerator(env, save_dir="data/test")
    
    state = generator._get_state()
    
    assert state.shape == (21,)
    assert not np.any(np.isnan(state))


def test_generate_trajectory(temp_dir):
    """단일 trajectory 생성 테스트."""
    env = PandaEnv()
    generator = DataGenerator(env, save_dir=temp_dir)
    
    traj = generator.generate_trajectory(steps=50)
    
    assert len(traj) == 50
    assert len(traj.states) == 50
    assert len(traj.actions) == 50
    assert "start_time" in traj.metadata
    assert "end_time" in traj.metadata


def test_generate_multiple_trajectories(temp_dir):
    """여러 trajectory 생성 테스트."""
    env = PandaEnv()
    generator = DataGenerator(env, save_dir=temp_dir)
    
    trajectories = generator.generate(
        num_trajectories=3,
        steps_per_traj=20,
        save=False,
    )
    
    assert len(trajectories) == 3
    assert all(len(traj) == 20 for traj in trajectories)


def test_generate_and_save(temp_dir):
    """Trajectory 생성 및 저장 테스트."""
    env = PandaEnv()
    generator = DataGenerator(env, save_dir=temp_dir)
    
    generator.generate(
        num_trajectories=2,
        steps_per_traj=10,
        save=True,
    )
    
    # 파일 확인
    assert (Path(temp_dir) / "traj_0000.pkl").exists()
    assert (Path(temp_dir) / "traj_0001.pkl").exists()


def test_build_and_save_dataset(temp_dir):
    """데이터셋 빌드 및 저장 테스트."""
    env = PandaEnv()
    generator = DataGenerator(env, save_dir=temp_dir)
    
    # 데이터 생성
    generator.generate(
        num_trajectories=3,
        steps_per_traj=20,
        save=True,
    )
    
    # 데이터셋 빌드
    output_path = Path(temp_dir) / "dataset.pkl"
    states, actions = generator.build_and_save_dataset(str(output_path))
    
    # 검증
    assert states.shape == (60, 21)  # 3 traj × 20 steps
    assert actions.shape == (60, 7)
    assert output_path.exists()


def test_get_statistics():
    """통계 계산 테스트."""
    env = PandaEnv()
    generator = DataGenerator(env, save_dir="data/test")
    
    # 빈 상태
    stats = generator.get_statistics()
    assert stats["num_trajectories"] == 0
    
    # 데이터 생성
    generator.generate(
        num_trajectories=5,
        steps_per_traj=10,
        save=False,
    )
    
    stats = generator.get_statistics()
    assert stats["num_trajectories"] == 5
    assert stats["total_steps"] == 50
    assert stats["avg_length"] == 10.0
    assert stats["state_mean"].shape == (21,)
    assert stats["action_std"].shape == (7,)


def test_action_clipping():
    """액션 클리핑 테스트."""
    env = PandaEnv()
    generator = DataGenerator(env, save_dir="data/test")
    
    # 랜덤 정책으로 여러 스텝 실행
    traj = generator.generate_trajectory(steps=100)
    
    # 모든 액션이 [-π, π] 범위 내
    actions = np.array(traj.actions)
    assert np.all(actions >= -np.pi)
    assert np.all(actions <= np.pi)


@pytest.mark.parametrize("policy_type", ["random", "sinusoidal"])
def test_different_policies(temp_dir, policy_type):
    """다양한 정책 테스트."""
    env = PandaEnv()
    generator = DataGenerator(
        env, 
        save_dir=temp_dir,
        policy_type=policy_type
    )
    
    traj = generator.generate_trajectory(steps=20)
    assert len(traj) == 20


def test_trajectory_consistency():
    """Trajectory 일관성 테스트."""
    env = PandaEnv()
    generator = DataGenerator(env, save_dir="data/test")
    
    traj = generator.generate_trajectory(steps=10)
    
    # States와 actions 길이가 같아야 함
    assert len(traj.states) == len(traj.actions)
    
    # 모든 state가 21차원
    for state in traj.states:
        assert state.shape == (21,)
    
    # 모든 action이 7차원
    for action in traj.actions:
        assert action.shape == (7,)


@pytest.mark.parametrize("steps", [10, 50, 100])
def test_different_trajectory_lengths(temp_dir, steps):
    """다양한 길이의 trajectory 테스트."""
    env = PandaEnv()
    generator = DataGenerator(env, save_dir=temp_dir)
    
    traj = generator.generate_trajectory(steps=steps)
    assert len(traj) == steps
