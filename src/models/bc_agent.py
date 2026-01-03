"""Behavior Cloning Agent.

MLP 기반 정책 네트워크 (NumPy 구현).
PyTorch로 마이그레이션 예정 (Python 3.13 호환성 대기).
"""

from typing import Tuple, Optional, List
from dataclasses import dataclass
import numpy as np
from pathlib import Path


@dataclass
class MLPConfig:
    """MLP 설정."""

    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    activation: str = "relu"  # "relu", "tanh", "sigmoid"
    output_activation: str = "linear"  # "linear", "tanh"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


class Activation:
    """활성화 함수."""

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU."""
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """ReLU derivative."""
        return (x > 0).astype(np.float32)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh."""
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Tanh derivative."""
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Sigmoid derivative."""
        s = Activation.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        """Linear (identity)."""
        return x

    @staticmethod
    def linear_derivative(x: np.ndarray) -> np.ndarray:
        """Linear derivative."""
        return np.ones_like(x)


class MLP:
    """Multi-Layer Perceptron (NumPy 구현)."""

    def __init__(self, config: MLPConfig):
        """초기화.

        Args:
            config: MLP 설정
        """
        self.config = config

        # 레이어 차원
        layer_dims = [config.input_dim] + config.hidden_dims + [config.output_dim]

        # 가중치 초기화 (He initialization)
        self.weights = []
        self.biases = []

        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]

            # He initialization
            std = np.sqrt(2.0 / fan_in)
            w = np.random.randn(fan_in, fan_out) * std
            b = np.zeros(fan_out)

            self.weights.append(w)
            self.biases.append(b)

        # 활성화 함수
        self.activation = getattr(Activation, config.activation)
        self.activation_derivative = getattr(Activation, f"{config.activation}_derivative")
        self.output_activation = getattr(Activation, config.output_activation)
        self.output_activation_derivative = getattr(
            Activation, f"{config.output_activation}_derivative"
        )

        # 학습 관련
        self.lr = config.learning_rate
        self.weight_decay = config.weight_decay

        # Forward pass 캐시
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.

        Args:
            x: (batch_size, input_dim) 입력

        Returns:
            (batch_size, output_dim) 출력
        """
        self.cache["activations"] = [x]
        self.cache["z_values"] = []

        h = x
        for i, (w, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            z = h @ w + b
            self.cache["z_values"].append(z)

            h = self.activation(z)
            self.cache["activations"].append(h)

        # 출력 레이어
        z_out = h @ self.weights[-1] + self.biases[-1]
        self.cache["z_values"].append(z_out)

        output = self.output_activation(z_out)
        self.cache["activations"].append(output)

        return output

    def predict(self, x: np.ndarray) -> np.ndarray:
        """예측 (forward와 동일하지만 캐시 저장 안 함).

        Args:
            x: (batch_size, input_dim)

        Returns:
            (batch_size, output_dim)
        """
        h = x
        for i, (w, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            z = h @ w + b
            h = self.activation(z)

        z_out = h @ self.weights[-1] + self.biases[-1]
        output = self.output_activation(z_out)

        return output

    def backward(
        self, x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Backward pass (MSE loss).

        Args:
            x: (batch_size, input_dim) 입력
            y_true: (batch_size, output_dim) 정답
            y_pred: (batch_size, output_dim) 예측

        Returns:
            loss
        """
        batch_size = x.shape[0]

        # Loss (MSE)
        loss = np.mean((y_pred - y_true) ** 2)

        # Gradient of loss w.r.t. output
        d_output = 2 * (y_pred - y_true) / batch_size

        # Backprop through output activation
        d_z_out = d_output * self.output_activation_derivative(
            self.cache["z_values"][-1]
        )

        # Gradients for weights and biases
        weight_grads = []
        bias_grads = []

        # Output layer
        h_prev = self.cache["activations"][-2]
        dw = h_prev.T @ d_z_out
        db = np.sum(d_z_out, axis=0)

        # Weight decay
        dw += self.weight_decay * self.weights[-1]

        weight_grads.insert(0, dw)
        bias_grads.insert(0, db)

        # Backprop through hidden layers
        d_h = d_z_out @ self.weights[-1].T

        for i in range(len(self.weights) - 2, -1, -1):
            d_z = d_h * self.activation_derivative(self.cache["z_values"][i])

            h_prev = self.cache["activations"][i]
            dw = h_prev.T @ d_z
            db = np.sum(d_z, axis=0)

            # Weight decay
            dw += self.weight_decay * self.weights[i]

            weight_grads.insert(0, dw)
            bias_grads.insert(0, db)

            if i > 0:
                d_h = d_z @ self.weights[i].T

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * weight_grads[i]
            self.biases[i] -= self.lr * bias_grads[i]

        return loss

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """학습 스텝.

        Args:
            x: (batch_size, input_dim)
            y: (batch_size, output_dim)

        Returns:
            loss
        """
        y_pred = self.forward(x)
        loss = self.backward(x, y, y_pred)
        return loss

    def save(self, path: Path):
        """모델 저장.

        Args:
            path: 저장 경로
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            f"weight_{i}": w for i, w in enumerate(self.weights)
        }
        save_dict.update({
            f"bias_{i}": b for i, b in enumerate(self.biases)
        })
        save_dict.update({
            "config_input_dim": self.config.input_dim,
            "config_hidden_dims": np.array(self.config.hidden_dims),
            "config_output_dim": self.config.output_dim,
            "config_activation": self.config.activation,
            "config_output_activation": self.config.output_activation,
            "config_learning_rate": self.config.learning_rate,
            "config_weight_decay": self.config.weight_decay,
            "n_layers": len(self.weights),
        })

        np.savez(path, **save_dict)

    def load(self, path: Path):
        """모델 로드.

        Args:
            path: 로드 경로
        """
        data = np.load(path, allow_pickle=True)

        # Config 복원
        self.config = MLPConfig(
            input_dim=int(data["config_input_dim"]),
            hidden_dims=list(data["config_hidden_dims"]),
            output_dim=int(data["config_output_dim"]),
            activation=str(data["config_activation"]),
            output_activation=str(data["config_output_activation"]),
            learning_rate=float(data["config_learning_rate"]),
            weight_decay=float(data["config_weight_decay"]),
        )

        # 가중치/바이어스 복원
        n_layers = int(data["n_layers"])
        self.weights = [data[f"weight_{i}"] for i in range(n_layers)]
        self.biases = [data[f"bias_{i}"] for i in range(n_layers)]

        # 활성화 함수 재설정
        self.activation = getattr(Activation, self.config.activation)
        self.activation_derivative = getattr(
            Activation, f"{self.config.activation}_derivative"
        )
        self.output_activation = getattr(Activation, self.config.output_activation)
        self.output_activation_derivative = getattr(
            Activation, f"{self.config.output_activation}_derivative"
        )


class BCAgent:
    """Behavior Cloning Agent."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: List[int] = None):
        """초기화.

        Args:
            obs_dim: Observation 차원
            act_dim: Action 차원
            hidden_dims: Hidden layer 차원 리스트
        """
        if hidden_dims is None:
            hidden_dims = [256, 256]

        config = MLPConfig(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=act_dim,
            activation="relu",
            output_activation="tanh",  # Actions typically in [-1, 1]
            learning_rate=1e-3,
            weight_decay=1e-4,
        )

        self.policy = MLP(config)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """액션 예측.

        Args:
            obs: (batch_size, obs_dim) 또는 (obs_dim,)

        Returns:
            (batch_size, act_dim) 또는 (act_dim,) 액션
        """
        single_obs = False
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]
            single_obs = True

        action = self.policy.predict(obs)

        if single_obs:
            action = action[0]

        return action

    def train(self, obs: np.ndarray, actions: np.ndarray) -> float:
        """학습.

        Args:
            obs: (batch_size, obs_dim)
            actions: (batch_size, act_dim)

        Returns:
            loss
        """
        return self.policy.train_step(obs, actions)

    def save(self, path: Path):
        """에이전트 저장."""
        self.policy.save(path)

    def load(self, path: Path):
        """에이전트 로드."""
        self.policy.load(path)


# PyTorch 구현 (Python 3.13 호환 시 활성화 예정)
"""
import torch
import torch.nn as nn

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, act_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)

class BCAgentPyTorch:
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: List[int] = None):
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        self.policy = MLPPolicy(obs_dim, act_dim, hidden_dims)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
    
    def predict(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float()
            action = self.policy(obs_tensor).numpy()
        return action
    
    def train(self, obs: np.ndarray, actions: np.ndarray) -> float:
        obs_tensor = torch.from_numpy(obs).float()
        actions_tensor = torch.from_numpy(actions).float()
        
        pred_actions = self.policy(obs_tensor)
        loss = self.criterion(pred_actions, actions_tensor)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
"""
