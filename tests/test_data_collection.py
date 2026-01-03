"""Test data collection tools."""
import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil

from src.utils.data_collection import (
    Trajectory, 
    TeleoperationCollector,
    DatasetBuilder
)
from src.envs.panda_env import PandaEnv


@pytest.fixture
def temp_dir():
    """임시 디렉토리 생성."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_trajectory():
    """샘플 trajectory 생성."""
    traj = Trajectory()
    for i in range(10):
        state = np.random.randn(21)
        action = np.random.randn(7)
        traj.states.append(state)
        traj.actions.append(action)
    traj.metadata = {"test": True}
    return traj


def test_trajectory_creation():
    """Trajectory 생성 테스트."""
    traj = Trajectory()
    assert len(traj) == 0
    assert len(traj.states) == 0
    assert len(traj.actions) == 0


def test_trajectory_save_load(sample_trajectory, temp_dir):
    """Trajectory 저장/로드 테스트."""
    filepath = Path(temp_dir) / "test_traj.pkl"
    
    # Save
    sample_trajectory.save(str(filepath))
    assert filepath.exists()
    
    # Load
    loaded_traj = Trajectory.load(str(filepath))
    assert len(loaded_traj) == len(sample_trajectory)
    np.testing.assert_array_equal(
        np.array(loaded_traj.states), 
        np.array(sample_trajectory.states)
    )
    assert loaded_traj.metadata == sample_trajectory.metadata


def test_teleoperation_collector_init():
    """TeleoperationCollector 초기화 테스트."""
    env = PandaEnv()
    collector = TeleoperationCollector(env, save_dir="data/test")
    
    assert collector.env is not None
    assert collector.save_dir == Path("data/test")
    assert len(collector.trajectories) == 0
    assert not collector.is_recording


def test_get_state():
    """상태 벡터 생성 테스트."""
    env = PandaEnv()
    env.reset()
    collector = TeleoperationCollector(env)
    
    state = collector._get_state()
    
    # Shape 확인
    assert state.shape == (21,)  # 7+7+3+3+1
    
    # NaN 없음
    assert not np.any(np.isnan(state))


def test_handle_keyboard_joint_control():
    """키보드 입력 처리 테스트."""
    env = PandaEnv()
    env.reset()
    collector = TeleoperationCollector(env, delta=0.1)
    
    initial_qpos = collector.target_qpos.copy()
    
    # Q 키 (Joint 1 +)
    collector._handle_keyboard(ord('Q'))
    assert collector.target_qpos[0] == initial_qpos[0] + 0.1
    
    # A 키 (Joint 1 -)
    collector._handle_keyboard(ord('A'))
    assert collector.target_qpos[0] == initial_qpos[0]


def test_handle_keyboard_recording():
    """녹화 제어 테스트."""
    env = PandaEnv()
    env.reset()
    collector = TeleoperationCollector(env)
    
    # 초기 상태
    assert not collector.is_recording
    
    # Space - 녹화 시작
    collector._handle_keyboard(ord(' '))
    assert collector.is_recording
    assert "start_time" in collector.current_traj.metadata
    
    # Space - 녹화 중지
    collector._handle_keyboard(ord(' '))
    assert not collector.is_recording
    assert "end_time" in collector.current_traj.metadata


def test_save_current_trajectory(temp_dir):
    """Trajectory 저장 테스트."""
    env = PandaEnv()
    env.reset()
    collector = TeleoperationCollector(env, save_dir=temp_dir)
    
    # 데이터 추가
    for _ in range(5):
        state = collector._get_state()
        action = collector.target_qpos.copy()
        collector.current_traj.states.append(state)
        collector.current_traj.actions.append(action)
    
    # 저장
    collector._save_current_trajectory()
    
    # 파일 확인
    saved_file = Path(temp_dir) / "traj_0000.pkl"
    assert saved_file.exists()
    
    # 로드 확인
    loaded_traj = Trajectory.load(str(saved_file))
    assert len(loaded_traj) == 5


def test_headless_run(temp_dir):
    """Headless 모드 실행 테스트."""
    env = PandaEnv()
    collector = TeleoperationCollector(env, save_dir=temp_dir)
    
    # 녹화 시작
    collector.is_recording = True
    collector.current_traj.metadata["start_time"] = 0.0
    
    # Headless 실행
    collector.run(headless=True)
    
    # 데이터 수집 확인
    assert len(collector.current_traj) == 100


def test_get_statistics():
    """통계 계산 테스트."""
    env = PandaEnv()
    collector = TeleoperationCollector(env)
    
    # 빈 상태
    stats = collector.get_statistics()
    assert stats["num_trajectories"] == 0
    
    # Trajectory 추가
    for _ in range(3):
        traj = Trajectory()
        for _ in range(10):
            traj.states.append(np.zeros(21))
            traj.actions.append(np.zeros(7))
        collector.trajectories.append(traj)
    
    stats = collector.get_statistics()
    assert stats["num_trajectories"] == 3
    assert stats["total_steps"] == 30
    assert stats["avg_length"] == 10.0


def test_dataset_builder_load_trajectories(temp_dir, sample_trajectory):
    """Trajectory 로드 테스트."""
    # 여러 trajectory 저장
    for i in range(3):
        filepath = Path(temp_dir) / f"traj_{i:04d}.pkl"
        sample_trajectory.save(str(filepath))
    
    # 로드
    trajectories = DatasetBuilder.load_trajectories(temp_dir)
    assert len(trajectories) == 3


def test_dataset_builder_build_dataset():
    """데이터셋 빌드 테스트."""
    trajectories = []
    for _ in range(3):
        traj = Trajectory()
        for _ in range(5):
            traj.states.append(np.random.randn(21))
            traj.actions.append(np.random.randn(7))
        trajectories.append(traj)
    
    states, actions = DatasetBuilder.build_dataset(trajectories)
    
    assert states.shape == (15, 21)  # 3 traj × 5 steps
    assert actions.shape == (15, 7)


def test_dataset_builder_save_load(temp_dir):
    """데이터셋 저장/로드 테스트."""
    states = np.random.randn(100, 21)
    actions = np.random.randn(100, 7)
    
    filepath = Path(temp_dir) / "dataset.pkl"
    DatasetBuilder.save_dataset(states, actions, str(filepath))
    
    assert filepath.exists()
    
    # 로드
    import pickle
    with open(filepath, "rb") as f:
        dataset = pickle.load(f)
    
    np.testing.assert_array_equal(dataset["states"], states)
    np.testing.assert_array_equal(dataset["actions"], actions)
    assert dataset["metadata"]["num_samples"] == 100


def test_joint_limit_clamping():
    """관절 한계 clamping 테스트."""
    env = PandaEnv()
    env.reset()
    collector = TeleoperationCollector(env, delta=5.0)  # 큰 delta
    
    # Joint 1의 한계를 넘어서려고 시도
    for _ in range(10):
        collector._handle_keyboard(ord('Q'))  # +방향
    
    # Joint limit 내에 있는지 확인
    joint_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_JOINT, "panda_joint1"
    )
    limits = env.model.jnt_range[joint_id]
    
    assert limits[0] <= collector.target_qpos[0] <= limits[1]


@pytest.mark.parametrize("num_trajectories", [1, 3, 5])
def test_multiple_trajectories(temp_dir, num_trajectories):
    """여러 trajectory 저장/로드 테스트."""
    env = PandaEnv()
    collector = TeleoperationCollector(env, save_dir=temp_dir)
    
    for _ in range(num_trajectories):
        # 데이터 생성
        for _ in range(10):
            state = collector._get_state()
            action = collector.target_qpos.copy()
            collector.current_traj.states.append(state)
            collector.current_traj.actions.append(action)
        
        # 저장
        collector._save_current_trajectory()
    
    # 로드
    trajectories = DatasetBuilder.load_trajectories(temp_dir)
    assert len(trajectories) == num_trajectories
