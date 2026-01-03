"""BC Agent 테스트."""

import pytest
import numpy as np
from pathlib import Path

from src.models.bc_agent import (
    MLPConfig,
    MLP,
    BCAgent,
    Activation,
)


class TestActivation:
    """활성화 함수 테스트."""

    def test_relu(self):
        """ReLU 테스트."""
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        y = Activation.relu(x)
        expected = np.array([0.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(y, expected)

    def test_relu_derivative(self):
        """ReLU derivative 테스트."""
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        dy = Activation.relu_derivative(x)
        expected = np.array([0.0, 0.0, 1.0, 1.0])
        np.testing.assert_allclose(dy, expected)

    def test_tanh(self):
        """Tanh 테스트."""
        x = np.array([0.0, 1.0])
        y = Activation.tanh(x)
        expected = np.tanh(x)
        np.testing.assert_allclose(y, expected)

    def test_sigmoid(self):
        """Sigmoid 테스트."""
        x = np.array([0.0, 1.0, -1.0])
        y = Activation.sigmoid(x)
        assert np.all((y >= 0) & (y <= 1))

    def test_linear(self):
        """Linear 테스트."""
        x = np.array([1.0, 2.0, 3.0])
        y = Activation.linear(x)
        np.testing.assert_allclose(y, x)


class TestMLPConfig:
    """MLPConfig 테스트."""

    def test_init(self):
        """초기화 테스트."""
        config = MLPConfig(
            input_dim=10,
            hidden_dims=[64, 64],
            output_dim=5,
            activation="relu",
        )

        assert config.input_dim == 10
        assert config.hidden_dims == [64, 64]
        assert config.output_dim == 5
        assert config.activation == "relu"


class TestMLP:
    """MLP 테스트."""

    def test_init(self):
        """초기화 테스트."""
        config = MLPConfig(
            input_dim=10,
            hidden_dims=[64, 32],
            output_dim=5,
        )
        mlp = MLP(config)

        assert len(mlp.weights) == 3  # input->hidden1, hidden1->hidden2, hidden2->output
        assert len(mlp.biases) == 3

        assert mlp.weights[0].shape == (10, 64)
        assert mlp.weights[1].shape == (64, 32)
        assert mlp.weights[2].shape == (32, 5)

    def test_forward(self):
        """Forward pass 테스트."""
        config = MLPConfig(
            input_dim=10,
            hidden_dims=[64],
            output_dim=5,
        )
        mlp = MLP(config)

        x = np.random.randn(32, 10)
        y = mlp.forward(x)

        assert y.shape == (32, 5)

    def test_predict(self):
        """Predict 테스트."""
        config = MLPConfig(
            input_dim=10,
            hidden_dims=[64],
            output_dim=5,
            output_activation="tanh",
        )
        mlp = MLP(config)

        x = np.random.randn(32, 10)
        y = mlp.predict(x)

        assert y.shape == (32, 5)
        # Tanh output in [-1, 1]
        assert np.all(y >= -1.0) and np.all(y <= 1.0)

    def test_train_step(self):
        """학습 스텝 테스트."""
        config = MLPConfig(
            input_dim=10,
            hidden_dims=[32, 32],
            output_dim=5,
            learning_rate=1e-2,
        )
        mlp = MLP(config)

        x = np.random.randn(64, 10)
        y = np.random.randn(64, 5)

        # 초기 loss
        loss1 = mlp.train_step(x, y)

        # 여러 번 학습
        for _ in range(10):
            loss = mlp.train_step(x, y)

        # Loss가 감소해야 함
        assert loss < loss1

    def test_backward(self):
        """Backward pass 테스트."""
        config = MLPConfig(
            input_dim=5,
            hidden_dims=[16],
            output_dim=3,
            learning_rate=1e-2,
        )
        mlp = MLP(config)

        x = np.random.randn(8, 5)
        y_true = np.random.randn(8, 3)

        # Forward
        y_pred = mlp.forward(x)

        # Backward
        loss = mlp.backward(x, y_true, y_pred)

        assert isinstance(loss, (float, np.floating))
        assert loss >= 0

    def test_save_load(self, tmp_path):
        """저장/로드 테스트."""
        config = MLPConfig(
            input_dim=10,
            hidden_dims=[64, 32],
            output_dim=5,
        )
        mlp = MLP(config)

        # 학습
        x = np.random.randn(32, 10)
        y = np.random.randn(32, 5)
        mlp.train_step(x, y)

        # 예측
        y_pred_before = mlp.predict(x)

        # 저장
        save_path = tmp_path / "mlp.npz"
        mlp.save(save_path)

        assert save_path.exists()

        # 새 MLP 생성 및 로드
        mlp2 = MLP(config)
        mlp2.load(save_path)

        # 예측 비교
        y_pred_after = mlp2.predict(x)

        np.testing.assert_allclose(y_pred_before, y_pred_after, rtol=1e-5)

    def test_different_activations(self):
        """다양한 활성화 함수 테스트."""
        for activation in ["relu", "tanh", "sigmoid"]:
            config = MLPConfig(
                input_dim=10,
                hidden_dims=[32],
                output_dim=5,
                activation=activation,
            )
            mlp = MLP(config)

            x = np.random.randn(16, 10)
            y = mlp.forward(x)

            assert y.shape == (16, 5)

    def test_no_hidden_layers(self):
        """Hidden layer 없는 경우."""
        config = MLPConfig(
            input_dim=10,
            hidden_dims=[],
            output_dim=5,
        )
        mlp = MLP(config)

        assert len(mlp.weights) == 1
        assert mlp.weights[0].shape == (10, 5)

        x = np.random.randn(8, 10)
        y = mlp.forward(x)
        assert y.shape == (8, 5)


class TestBCAgent:
    """BCAgent 테스트."""

    def test_init(self):
        """초기화 테스트."""
        agent = BCAgent(obs_dim=20, act_dim=7)

        assert agent.obs_dim == 20
        assert agent.act_dim == 7
        assert agent.policy is not None

    def test_predict_single(self):
        """단일 observation 예측 테스트."""
        agent = BCAgent(obs_dim=20, act_dim=7)

        obs = np.random.randn(20)
        action = agent.predict(obs)

        assert action.shape == (7,)
        # Tanh output
        assert np.all(action >= -1.0) and np.all(action <= 1.0)

    def test_predict_batch(self):
        """배치 observation 예측 테스트."""
        agent = BCAgent(obs_dim=20, act_dim=7)

        obs = np.random.randn(32, 20)
        actions = agent.predict(obs)

        assert actions.shape == (32, 7)

    def test_train(self):
        """학습 테스트."""
        agent = BCAgent(obs_dim=20, act_dim=7, hidden_dims=[64, 64])

        obs = np.random.randn(64, 20)
        actions = np.random.randn(64, 7)

        loss1 = agent.train(obs, actions)

        # 여러 번 학습
        for _ in range(50):
            loss = agent.train(obs, actions)

        # Loss 감소 확인
        assert loss < loss1

    def test_save_load(self, tmp_path):
        """저장/로드 테스트."""
        agent = BCAgent(obs_dim=20, act_dim=7)

        # 학습
        obs = np.random.randn(32, 20)
        actions = np.random.randn(32, 7)
        for _ in range(10):
            agent.train(obs, actions)

        # 예측
        test_obs = np.random.randn(10, 20)
        pred_before = agent.predict(test_obs)

        # 저장
        save_path = tmp_path / "bc_agent.npz"
        agent.save(save_path)

        # 새 에이전트 로드
        agent2 = BCAgent(obs_dim=20, act_dim=7)
        agent2.load(save_path)

        # 예측 비교
        pred_after = agent2.predict(test_obs)

        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5)

    def test_custom_hidden_dims(self):
        """커스텀 hidden dims 테스트."""
        agent = BCAgent(obs_dim=15, act_dim=5, hidden_dims=[128, 64, 32])

        assert len(agent.policy.weights) == 4  # 3 hidden + 1 output

        obs = np.random.randn(16, 15)
        actions = agent.predict(obs)

        assert actions.shape == (16, 5)

    def test_overfitting_small_dataset(self):
        """소규모 데이터셋 오버피팅 테스트."""
        agent = BCAgent(obs_dim=10, act_dim=3, hidden_dims=[32])

        # 작은 데이터셋
        obs = np.random.randn(16, 10)
        actions = np.random.randn(16, 3)

        initial_loss = agent.train(obs, actions)

        # 충분히 학습
        for _ in range(200):
            loss = agent.train(obs, actions)

        # Loss가 감소해야 함 (완벽한 오버피팅은 아닐 수 있음)
        assert loss < initial_loss


class TestIntegration:
    """통합 테스트."""

    def test_full_training_pipeline(self):
        """전체 학습 파이프라인 테스트."""
        # 에이전트 생성
        agent = BCAgent(obs_dim=20, act_dim=7, hidden_dims=[128, 128])

        # 훈련 데이터 생성
        n_samples = 1000
        obs = np.random.randn(n_samples, 20)
        actions = np.random.randn(n_samples, 7)

        # 학습
        losses = []
        batch_size = 64
        n_epochs = 10

        for epoch in range(n_epochs):
            epoch_losses = []
            for i in range(0, n_samples, batch_size):
                batch_obs = obs[i : i + batch_size]
                batch_actions = actions[i : i + batch_size]

                loss = agent.train(batch_obs, batch_actions)
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)

        # Loss 감소 확인
        assert losses[-1] < losses[0]

        # 예측 테스트
        test_obs = np.random.randn(10, 20)
        pred_actions = agent.predict(test_obs)

        assert pred_actions.shape == (10, 7)

    def test_agent_generalization(self):
        """일반화 성능 테스트."""
        agent = BCAgent(obs_dim=5, act_dim=2, hidden_dims=[32, 32])

        # 간단한 함수 학습: action = obs[:2]
        n_train = 500
        obs_train = np.random.randn(n_train, 5)
        actions_train = obs_train[:, :2]  # 처음 2개 차원 복사

        # 학습
        for _ in range(200):
            agent.train(obs_train, actions_train)

        # 테스트
        obs_test = np.random.randn(100, 5)
        actions_test = obs_test[:, :2]

        pred_actions = agent.predict(obs_test)

        # Tanh 활성화 때문에 range가 [-1, 1]로 제한됨
        # 합리적인 일반화 확인 (상관관계)
        correlation = np.corrcoef(pred_actions.flatten(), actions_test.flatten())[0, 1]
        assert correlation > 0.3  # 어느 정도 학습됨
