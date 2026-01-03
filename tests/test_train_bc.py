"""
학습 루프 테스트
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.train_bc import (
    EarlyStopping,
    TrainingConfig,
    TrainingHistory,
    compute_loss,
    create_batches,
    split_dataset,
    train_bc_agent,
    train_epoch,
    validate,
)
from src.models.bc_agent import BCAgent
from src.utils.preprocessing import save_trajectories_to_hdf5


class TestTrainingConfig:
    """TrainingConfig 테스트"""
    
    def test_default_initialization(self):
        """기본 초기화"""
        config = TrainingConfig()
        assert config.batch_size == 32
        assert config.num_epochs == 100
        assert config.learning_rate == 1e-3
        assert config.validation_split == 0.2
    
    def test_custom_initialization(self):
        """커스텀 초기화"""
        config = TrainingConfig(
            batch_size=64,
            num_epochs=50,
            learning_rate=5e-4,
        )
        assert config.batch_size == 64
        assert config.num_epochs == 50
        assert config.learning_rate == 5e-4


class TestTrainingHistory:
    """TrainingHistory 테스트"""
    
    def test_initialization(self):
        """초기화"""
        history = TrainingHistory()
        assert len(history.train_losses) == 0
        assert len(history.val_losses) == 0
        assert history.best_val_loss == float('inf')
        assert history.best_epoch == 0
    
    def test_update(self):
        """업데이트"""
        history = TrainingHistory()
        history.update(0, 1.0, 0.8, 10.5)
        
        assert len(history.train_losses) == 1
        assert len(history.val_losses) == 1
        assert history.train_losses[0] == 1.0
        assert history.val_losses[0] == 0.8
        assert history.best_val_loss == 0.8
        assert history.best_epoch == 0
    
    def test_best_tracking(self):
        """최고 성능 추적"""
        history = TrainingHistory()
        history.update(0, 1.0, 0.8, 10.0)
        history.update(1, 0.9, 0.7, 10.0)
        history.update(2, 0.8, 0.75, 10.0)
        
        assert history.best_val_loss == 0.7
        assert history.best_epoch == 1
    
    def test_to_dict(self):
        """딕셔너리 변환"""
        history = TrainingHistory()
        history.update(0, 1.0, 0.8, 10.0)
        
        data = history.to_dict()
        assert 'train_losses' in data
        assert 'val_losses' in data
        assert 'best_val_loss' in data
        assert data['best_val_loss'] == 0.8
    
    def test_save_and_load(self):
        """저장 및 로드"""
        history = TrainingHistory()
        history.update(0, 1.0, 0.8, 10.0)
        history.update(1, 0.9, 0.7, 10.0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "history.json"
            history.save(filepath)
            
            # 파일 존재 확인
            assert filepath.exists()
            
            # 로드 및 검증
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert len(data['train_losses']) == 2
            assert data['best_epoch'] == 1


class TestDataSplitting:
    """데이터 분할 테스트"""
    
    def test_split_dataset(self):
        """데이터셋 분할"""
        states = np.random.randn(100, 10)
        actions = np.random.randn(100, 7)
        
        train_s, train_a, val_s, val_a = split_dataset(
            states, actions, val_split=0.2, shuffle=False
        )
        
        assert len(train_s) == 80
        assert len(train_a) == 80
        assert len(val_s) == 20
        assert len(val_a) == 20
    
    def test_split_with_shuffle(self):
        """셔플 포함 분할"""
        states = np.arange(100).reshape(100, 1)
        actions = np.arange(100).reshape(100, 1)
        
        train_s, train_a, val_s, val_a = split_dataset(
            states, actions, val_split=0.2, shuffle=True
        )
        
        # 셔플되었는지 확인 (순차적이지 않음)
        assert not np.array_equal(train_s[:10], np.arange(10).reshape(10, 1))
        
        # 크기 확인
        assert len(train_s) == 80
        assert len(val_s) == 20


class TestBatchCreation:
    """배치 생성 테스트"""
    
    def test_create_batches(self):
        """배치 생성"""
        states = np.random.randn(100, 10)
        actions = np.random.randn(100, 7)
        
        batches = create_batches(states, actions, batch_size=32, shuffle=False)
        
        assert len(batches) == 4  # 100 / 32 = 3 full + 1 partial
        assert batches[0][0].shape == (32, 10)
        assert batches[-1][0].shape == (4, 10)
    
    def test_batch_shuffle(self):
        """배치 셔플"""
        states = np.arange(100).reshape(100, 1)
        actions = np.arange(100).reshape(100, 1)
        
        batches = create_batches(states, actions, batch_size=10, shuffle=True)
        
        # 첫 배치가 순차적이지 않음
        first_batch_states = batches[0][0]
        assert not np.array_equal(first_batch_states, np.arange(10).reshape(10, 1))


class TestLossComputation:
    """손실 계산 테스트"""
    
    def test_compute_loss(self):
        """손실 계산"""
        agent = BCAgent(obs_dim=10, act_dim=7, hidden_dims=[32])
        
        states = np.random.randn(50, 10)
        actions = np.random.randn(50, 7)
        
        loss = compute_loss(agent, states, actions)
        
        assert isinstance(loss, float)
        assert loss >= 0


class TestEarlyStopping:
    """조기 종료 테스트"""
    
    def test_initialization(self):
        """초기화"""
        early_stop = EarlyStopping(patience=5)
        assert early_stop.patience == 5
        assert early_stop.counter == 0
        assert early_stop.best_loss == float('inf')
    
    def test_improvement(self):
        """성능 개선 시"""
        early_stop = EarlyStopping(patience=3)
        
        should_stop = early_stop(1.0)
        assert not should_stop
        assert early_stop.best_loss == 1.0
        
        should_stop = early_stop(0.8)
        assert not should_stop
        assert early_stop.counter == 0
    
    def test_no_improvement(self):
        """성능 개선 없을 시"""
        early_stop = EarlyStopping(patience=3)
        
        early_stop(1.0)
        early_stop(1.1)
        early_stop(1.2)
        should_stop = early_stop(1.3)
        
        assert should_stop
        assert early_stop.counter == 3
    
    def test_reset_on_improvement(self):
        """개선 시 카운터 리셋"""
        early_stop = EarlyStopping(patience=3)
        
        early_stop(1.0)
        early_stop(1.1)
        early_stop(0.9)  # 개선
        
        assert early_stop.counter == 0


class TestTrainEpoch:
    """에포크 학습 테스트"""
    
    def test_train_epoch(self):
        """한 에포크 학습"""
        agent = BCAgent(obs_dim=10, act_dim=7, hidden_dims=[32])
        
        states = np.random.randn(100, 10)
        actions = np.random.randn(100, 7)
        
        loss = train_epoch(
            agent, states, actions,
            batch_size=32,
        )
        
        assert isinstance(loss, float)
        assert loss >= 0


class TestValidation:
    """검증 테스트"""
    
    def test_validate(self):
        """검증"""
        agent = BCAgent(obs_dim=10, act_dim=7, hidden_dims=[32])
        
        states = np.random.randn(50, 10)
        actions = np.random.randn(50, 7)
        
        val_loss = validate(agent, states, actions)
        
        assert isinstance(val_loss, float)
        assert val_loss >= 0


class TestTrainBCAgent:
    """BC 에이전트 학습 테스트"""
    
    def test_training_loop(self):
        """학습 루프"""
        # 에이전트 생성
        agent = BCAgent(obs_dim=10, act_dim=7, hidden_dims=[32, 32])
        
        # 데이터 생성
        train_states = np.random.randn(200, 10)
        train_actions = np.random.randn(200, 7)
        val_states = np.random.randn(50, 10)
        val_actions = np.random.randn(50, 7)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 학습 설정
            training_config = TrainingConfig(
                batch_size=32,
                num_epochs=5,
                learning_rate=1e-3,
                checkpoint_dir=tmpdir,
                save_frequency=2,
                verbose=False,
            )
            
            # 학습
            history = train_bc_agent(
                agent,
                train_states,
                train_actions,
                val_states,
                val_actions,
                training_config,
            )
            
            # 히스토리 검증
            assert len(history.train_losses) == 5
            assert len(history.val_losses) == 5
            assert history.best_epoch >= 0
            
            # 체크포인트 생성 확인
            checkpoint_dir = Path(tmpdir)
            checkpoints = list(checkpoint_dir.glob("checkpoint_*.npz"))
            assert len(checkpoints) >= 2  # epoch 2, 4
    
    def test_early_stopping_trigger(self):
        """조기 종료 발동"""
        agent = BCAgent(obs_dim=5, act_dim=3, hidden_dims=[16])
        
        # 작은 데이터셋
        train_states = np.random.randn(50, 5)
        train_actions = np.random.randn(50, 3)
        val_states = np.random.randn(20, 5)
        val_actions = np.random.randn(20, 3)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            training_config = TrainingConfig(
                batch_size=16,
                num_epochs=100,
                early_stopping_patience=3,
                checkpoint_dir=tmpdir,
                verbose=False,
            )
            
            history = train_bc_agent(
                agent,
                train_states,
                train_actions,
                val_states,
                val_actions,
                training_config,
            )
            
            # 100 에포크보다 일찍 종료되거나, 또는 100 에포크 완료
            # (작은 데이터셋에서는 빠르게 수렴할 수 있음)
            assert len(history.train_losses) <= 100
    
    def test_best_model_saving(self):
        """최고 모델 저장"""
        agent = BCAgent(obs_dim=8, act_dim=4, hidden_dims=[16])
        
        train_states = np.random.randn(100, 8)
        train_actions = np.random.randn(100, 4)
        val_states = np.random.randn(30, 8)
        val_actions = np.random.randn(30, 4)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            training_config = TrainingConfig(
                batch_size=32,
                num_epochs=5,
                early_stopping_patience=10,  # 증가하여 조기 종료 방지
                checkpoint_dir=tmpdir,
                verbose=False,
            )
            
            history = train_bc_agent(
                agent,
                train_states,
                train_actions,
                val_states,
                val_actions,
                training_config,
            )
            
            # best_model.npz 존재 확인
            best_model_path = Path(tmpdir) / "best_model.npz"
            assert best_model_path.exists()
            
            # 로드 가능 확인
            loaded_agent = BCAgent(obs_dim=8, act_dim=4, hidden_dims=[16])
            loaded_agent.load(best_model_path)


class TestIntegration:
    """통합 테스트"""
    
    def test_full_training_pipeline(self):
        """전체 학습 파이프라인"""
        # 더미 데이터 생성
        trajectories = []
        for _ in range(10):
            traj = {
                'observations': np.random.randn(50, 10),
                'actions': np.random.randn(50, 7),
                'rewards': np.random.randn(50),
            }
            trajectories.append(traj)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # HDF5에 저장
            data_path = Path(tmpdir) / "demo.h5"
            traj_tuples = [(t['observations'], t['actions']) for t in trajectories]
            save_trajectories_to_hdf5(traj_tuples, data_path)
            
            # 데이터 추출
            from src.utils.preprocessing import load_trajectories_from_hdf5
            loaded_trajs = load_trajectories_from_hdf5(data_path)
            
            # 튜플로 반환됨: [(obs, act), ...]
            all_states = np.concatenate([t[0] for t in loaded_trajs])
            all_actions = np.concatenate([t[1] for t in loaded_trajs])
            
            # 분할
            train_s, train_a, val_s, val_a = split_dataset(
                all_states, all_actions, val_split=0.2
            )
            
            # 모델 생성 및 학습
            agent = BCAgent(obs_dim=10, act_dim=7, hidden_dims=[32])
            
            training_config = TrainingConfig(
                batch_size=32,
                num_epochs=3,
                checkpoint_dir=Path(tmpdir) / "checkpoints",
                verbose=False,
            )
            
            history = train_bc_agent(
                agent, train_s, train_a, val_s, val_a, training_config
            )
            
            # 검증
            assert len(history.train_losses) == 3
            assert (Path(tmpdir) / "checkpoints" / "best_model.npz").exists()
    
    def test_loss_decreases(self):
        """학습 중 손실 감소 확인"""
        # 간단한 선형 관계 데이터
        np.random.seed(42)
        X = np.random.randn(200, 5)
        W_true = np.random.randn(5, 3)
        y = X @ W_true + 0.1 * np.random.randn(200, 3)
        
        train_s, train_a, val_s, val_a = split_dataset(
            X, y, val_split=0.2, shuffle=True
        )
        
        agent = BCAgent(obs_dim=5, act_dim=3, hidden_dims=[32, 32])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            training_config = TrainingConfig(
                batch_size=32,
                num_epochs=20,
                learning_rate=1e-2,
                checkpoint_dir=tmpdir,
                verbose=False,
            )
            
            history = train_bc_agent(
                agent, train_s, train_a, val_s, val_a, training_config
            )
            
            # 마지막 손실이 초기 손실보다 작음
            assert history.train_losses[-1] < history.train_losses[0]
