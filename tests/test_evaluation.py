"""
평가 유틸리티 테스트
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.envs.panda_env import PandaEnv
from src.models.bc_agent import BCAgent
from src.utils.evaluation import (
    EpisodeMetrics,
    EvaluationResults,
    compare_trajectories,
    compute_mae,
    compute_mse,
    compute_per_dim_metrics,
    evaluate_agent,
    plot_action_predictions,
    plot_evaluation_metrics,
    plot_training_curves,
    run_episode,
)
from src.utils.preprocessing import Normalizer


class TestEpisodeMetrics:
    """EpisodeMetrics 테스트"""
    
    def test_initialization(self):
        """초기화"""
        trajectory = {
            'observations': np.random.randn(100, 20),
            'actions': np.random.randn(100, 7),
            'rewards': np.random.randn(100),
        }
        
        metrics = EpisodeMetrics(
            success=True,
            total_reward=150.5,
            episode_length=100,
            final_distance=0.02,
            mean_action_magnitude=0.5,
            trajectory=trajectory,
        )
        
        assert metrics.success is True
        assert metrics.total_reward == 150.5
        assert metrics.episode_length == 100
        assert metrics.final_distance == 0.02


class TestEvaluationResults:
    """EvaluationResults 테스트"""
    
    def test_initialization(self):
        """초기화"""
        trajectory = {
            'observations': np.random.randn(50, 20),
            'actions': np.random.randn(50, 7),
            'rewards': np.random.randn(50),
        }
        
        episode_metrics = [
            EpisodeMetrics(True, 100.0, 50, 0.01, 0.5, trajectory),
            EpisodeMetrics(False, 80.0, 50, 0.05, 0.6, trajectory),
        ]
        
        results = EvaluationResults(
            success_rate=0.5,
            mean_reward=90.0,
            std_reward=10.0,
            mean_episode_length=50.0,
            mean_final_distance=0.03,
            mean_action_magnitude=0.55,
            episode_metrics=episode_metrics,
        )
        
        assert results.success_rate == 0.5
        assert results.mean_reward == 90.0
        assert len(results.episode_metrics) == 2
    
    def test_to_dict(self):
        """딕셔너리 변환"""
        trajectory = {
            'observations': np.random.randn(50, 20),
            'actions': np.random.randn(50, 7),
            'rewards': np.random.randn(50),
        }
        
        episode_metrics = [
            EpisodeMetrics(True, 100.0, 50, 0.01, 0.5, trajectory),
        ]
        
        results = EvaluationResults(
            success_rate=1.0,
            mean_reward=100.0,
            std_reward=0.0,
            mean_episode_length=50.0,
            mean_final_distance=0.01,
            mean_action_magnitude=0.5,
            episode_metrics=episode_metrics,
        )
        
        data = results.to_dict()
        
        assert 'success_rate' in data
        assert 'mean_reward' in data
        assert data['num_episodes'] == 1
    
    def test_str_representation(self):
        """문자열 표현"""
        trajectory = {
            'observations': np.random.randn(50, 20),
            'actions': np.random.randn(50, 7),
            'rewards': np.random.randn(50),
        }
        
        episode_metrics = [
            EpisodeMetrics(True, 100.0, 50, 0.01, 0.5, trajectory),
        ]
        
        results = EvaluationResults(
            success_rate=1.0,
            mean_reward=100.0,
            std_reward=0.0,
            mean_episode_length=50.0,
            mean_final_distance=0.01,
            mean_action_magnitude=0.5,
            episode_metrics=episode_metrics,
        )
        
        s = str(results)
        assert 'Success Rate' in s
        assert '100.0%' in s


class TestRunEpisode:
    """run_episode 테스트"""
    
    def test_run_episode_without_normalizer(self):
        """정규화 없이 에피소드 실행"""
        env = PandaEnv(render_mode=None)
        agent = BCAgent(obs_dim=20, act_dim=7, hidden_dims=[32])
        
        metrics = run_episode(env, agent, max_steps=10)
        
        assert isinstance(metrics, EpisodeMetrics)
        assert isinstance(metrics.success, (bool, np.bool_))
        assert metrics.episode_length <= 10
        assert 'observations' in metrics.trajectory
        assert 'actions' in metrics.trajectory
    
    def test_run_episode_with_normalizer(self):
        """정규화 포함 에피소드 실행"""
        env = PandaEnv(render_mode=None)
        agent = BCAgent(obs_dim=20, act_dim=7, hidden_dims=[32])
        
        # 정규화기 생성
        normalizer = Normalizer()
        dummy_states = np.random.randn(100, 20)
        normalizer.fit(dummy_states)
        
        metrics = run_episode(env, agent, state_normalizer=normalizer, max_steps=10)
        
        assert isinstance(metrics, EpisodeMetrics)
        assert metrics.episode_length <= 10


class TestEvaluateAgent:
    """evaluate_agent 테스트"""
    
    def test_evaluate_agent(self):
        """에이전트 평가"""
        env = PandaEnv(render_mode=None)
        agent = BCAgent(obs_dim=20, act_dim=7, hidden_dims=[32])
        
        results = evaluate_agent(
            agent,
            env,
            num_episodes=5,
            max_steps=10,
            verbose=False,
        )
        
        assert isinstance(results, EvaluationResults)
        assert len(results.episode_metrics) == 5
        assert 0 <= results.success_rate <= 1
        assert results.mean_episode_length > 0


class TestMetricComputation:
    """메트릭 계산 테스트"""
    
    def test_compute_mse(self):
        """MSE 계산"""
        predictions = np.array([[1.0, 2.0], [3.0, 4.0]])
        targets = np.array([[1.5, 2.5], [3.5, 4.5]])
        
        mse = compute_mse(predictions, targets)
        
        expected = 0.25  # (0.5^2 + 0.5^2 + 0.5^2 + 0.5^2) / 4
        assert abs(mse - expected) < 1e-6
    
    def test_compute_mae(self):
        """MAE 계산"""
        predictions = np.array([[1.0, 2.0], [3.0, 4.0]])
        targets = np.array([[1.5, 2.5], [3.5, 4.5]])
        
        mae = compute_mae(predictions, targets)
        
        expected = 0.5  # (0.5 + 0.5 + 0.5 + 0.5) / 4
        assert abs(mae - expected) < 1e-6
    
    def test_compute_per_dim_metrics(self):
        """차원별 메트릭 계산"""
        predictions = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        targets = np.array([[1.0, 3.0], [3.0, 5.0], [5.0, 7.0]])
        
        per_dim_mse, per_dim_mae = compute_per_dim_metrics(predictions, targets)
        
        assert per_dim_mse.shape == (2,)
        assert per_dim_mae.shape == (2,)
        
        # 첫 번째 차원: 완벽한 예측
        assert per_dim_mse[0] == 0.0
        assert per_dim_mae[0] == 0.0
        
        # 두 번째 차원: 모두 1 차이
        assert abs(per_dim_mse[1] - 1.0) < 1e-6
        assert abs(per_dim_mae[1] - 1.0) < 1e-6


class TestVisualization:
    """시각화 테스트"""
    
    def test_plot_training_curves(self):
        """학습 곡선 플롯"""
        train_losses = [1.0, 0.8, 0.6, 0.5, 0.4]
        val_losses = [1.1, 0.9, 0.7, 0.6, 0.5]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "training_curves.png"
            
            plot_training_curves(train_losses, val_losses, save_path)
            
            assert save_path.exists()
    
    def test_plot_evaluation_metrics(self):
        """평가 메트릭 플롯"""
        trajectory = {
            'observations': np.random.randn(50, 20),
            'actions': np.random.randn(50, 7),
            'rewards': np.random.randn(50),
        }
        
        episode_metrics = [
            EpisodeMetrics(True, 100.0, 50, 0.01, 0.5, trajectory),
            EpisodeMetrics(False, 80.0, 45, 0.08, 0.6, trajectory),
            EpisodeMetrics(True, 120.0, 55, 0.02, 0.4, trajectory),
        ]
        
        results = EvaluationResults(
            success_rate=2/3,
            mean_reward=100.0,
            std_reward=20.0,
            mean_episode_length=50.0,
            mean_final_distance=0.037,
            mean_action_magnitude=0.5,
            episode_metrics=episode_metrics,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "evaluation_metrics.png"
            
            plot_evaluation_metrics(results, save_path)
            
            assert save_path.exists()
    
    def test_plot_action_predictions(self):
        """액션 예측 플롯"""
        expert_actions = np.random.randn(100, 7)
        predicted_actions = expert_actions + 0.1 * np.random.randn(100, 7)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "action_predictions.png"
            
            plot_action_predictions(expert_actions, predicted_actions, save_path=save_path)
            
            assert save_path.exists()
    
    def test_plot_action_predictions_with_names(self):
        """액션 이름 포함 예측 플롯"""
        expert_actions = np.random.randn(50, 3)
        predicted_actions = expert_actions + 0.1 * np.random.randn(50, 3)
        action_names = ['X', 'Y', 'Z']
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "action_predictions_named.png"
            
            plot_action_predictions(
                expert_actions,
                predicted_actions,
                action_dim_names=action_names,
                save_path=save_path,
            )
            
            assert save_path.exists()
    
    def test_compare_trajectories(self):
        """궤적 비교 플롯"""
        expert_traj = {
            'observations': np.random.randn(100, 20),
            'actions': np.random.randn(100, 7),
        }
        
        agent_traj = {
            'observations': np.random.randn(95, 20),
            'actions': np.random.randn(95, 7),
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "trajectory_comparison.png"
            
            compare_trajectories(expert_traj, agent_traj, save_path)
            
            assert save_path.exists()


class TestIntegration:
    """통합 테스트"""
    
    def test_full_evaluation_pipeline(self):
        """전체 평가 파이프라인"""
        # 환경 및 에이전트 생성
        env = PandaEnv(render_mode=None)
        agent = BCAgent(obs_dim=20, act_dim=7, hidden_dims=[32, 32])
        
        # 정규화기 생성
        normalizer = Normalizer()
        dummy_states = np.random.randn(100, 20)
        normalizer.fit(dummy_states)
        
        # 평가
        results = evaluate_agent(
            agent,
            env,
            num_episodes=3,
            state_normalizer=normalizer,
            max_steps=20,
            verbose=False,
        )
        
        # 검증
        assert len(results.episode_metrics) == 3
        assert results.success_rate >= 0.0
        assert results.mean_reward != 0.0
        
        # 시각화
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = Path(tmpdir) / "eval_metrics.png"
            plot_evaluation_metrics(results, plot_path)
            assert plot_path.exists()
    
    def test_metrics_consistency(self):
        """메트릭 일관성 검증"""
        # 예측과 타겟이 동일하면 MSE, MAE 모두 0
        predictions = np.random.randn(50, 7)
        targets = predictions.copy()
        
        mse = compute_mse(predictions, targets)
        mae = compute_mae(predictions, targets)
        
        assert mse < 1e-10
        assert mae < 1e-10
        
        # 차원별 메트릭도 0
        per_dim_mse, per_dim_mae = compute_per_dim_metrics(predictions, targets)
        
        assert np.all(per_dim_mse < 1e-10)
        assert np.all(per_dim_mae < 1e-10)
