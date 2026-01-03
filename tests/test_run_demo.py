"""
데모 실행 스크립트 테스트
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.run_demo import load_normalizers
from src.models.bc_agent import BCAgent
from src.utils.preprocessing import Normalizer


class TestLoadNormalizers:
    """정규화기 로드 테스트"""
    
    def test_load_normalizers(self):
        """정규화기 로드"""
        # 더미 정규화기 생성
        state_normalizer = Normalizer()
        action_normalizer = Normalizer()
        
        state_normalizer.fit(np.random.randn(100, 20))
        action_normalizer.fit(np.random.randn(100, 7))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 저장
            norm_path = Path(tmpdir) / "normalizers.npz"
            np.savez(
                norm_path,
                state_mean=state_normalizer.stats.mean,
                state_std=state_normalizer.stats.std,
                action_mean=action_normalizer.stats.mean,
                action_std=action_normalizer.stats.std,
            )
            
            # 로드
            loaded_state_norm, loaded_action_norm = load_normalizers(norm_path)
            
            # 검증
            assert loaded_state_norm.stats is not None
            assert loaded_action_norm.stats is not None
            assert loaded_state_norm.stats.mean.shape == (20,)
            assert loaded_action_norm.stats.mean.shape == (7,)
            
            np.testing.assert_array_equal(
                loaded_state_norm.stats.mean,
                state_normalizer.stats.mean,
            )


class TestIntegration:
    """통합 테스트"""
    
    def test_model_save_and_load_workflow(self):
        """모델 저장 및 로드 워크플로우"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 모델 생성 및 저장
            agent = BCAgent(obs_dim=20, act_dim=7, hidden_dims=[32, 32])
            model_path = Path(tmpdir) / "model.npz"
            agent.save(model_path)
            
            # 정규화기 생성 및 저장
            state_normalizer = Normalizer()
            action_normalizer = Normalizer()
            
            state_normalizer.fit(np.random.randn(100, 20))
            action_normalizer.fit(np.random.randn(100, 7))
            
            norm_path = Path(tmpdir) / "normalizers.npz"
            np.savez(
                norm_path,
                state_mean=state_normalizer.stats.mean,
                state_std=state_normalizer.stats.std,
                action_mean=action_normalizer.stats.mean,
                action_std=action_normalizer.stats.std,
            )
            
            # 파일 존재 확인
            assert model_path.exists()
            assert norm_path.exists()
            
            # 로드
            loaded_agent = BCAgent(obs_dim=20, act_dim=7, hidden_dims=[32, 32])
            loaded_agent.load(model_path)
            
            loaded_state_norm, loaded_action_norm = load_normalizers(norm_path)
            
            # 예측 테스트
            obs = np.random.randn(20)
            action1 = agent.predict(obs)
            action2 = loaded_agent.predict(obs)
            
            np.testing.assert_array_almost_equal(action1, action2)
    
    def test_normalized_prediction(self):
        """정규화된 예측"""
        # 모델 생성
        agent = BCAgent(obs_dim=20, act_dim=7, hidden_dims=[16])
        
        # 정규화기 생성
        normalizer = Normalizer()
        dummy_states = np.random.randn(100, 20)
        normalizer.fit(dummy_states)
        
        # 정규화된 관찰값으로 예측
        obs = np.random.randn(20)
        obs_norm = normalizer.transform(obs)
        action = agent.predict(obs_norm)
        
        assert action.shape == (7,)
        assert np.all(np.abs(action) <= 1.0)  # tanh output
