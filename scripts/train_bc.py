"""
학습 스크립트: Behavior Cloning 모델 학습

Usage:
    python scripts/train_bc.py --data data/demonstrations.h5 --output models/bc_model.npz
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.models.bc_agent import BCAgent
from src.utils.preprocessing import (
    HDF5Dataset,
    Normalizer,
    TrajectoryDataset,
    load_trajectories_from_hdf5,
)


class TrainingConfig:
    """학습 설정"""
    
    def __init__(
        self,
        batch_size: int = 32,
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = "checkpoints",
        save_frequency: int = 10,
        verbose: bool = True,
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_frequency = save_frequency
        self.verbose = verbose


class TrainingHistory:
    """학습 기록"""
    
    def __init__(self):
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.epoch_times: List[float] = []
        self.best_val_loss: float = float('inf')
        self.best_epoch: int = 0
    
    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        epoch_time: float,
    ):
        """에포크 결과 업데이트"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.epoch_times.append(epoch_time)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epoch_times': self.epoch_times,
            'best_val_loss': float(self.best_val_loss),
            'best_epoch': int(self.best_epoch),
        }
    
    def save(self, filepath: Path):
        """JSON 파일로 저장"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def split_dataset(
    states: np.ndarray,
    actions: np.ndarray,
    val_split: float,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """데이터셋을 train/val로 분할"""
    n_samples = len(states)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    split_idx = int(n_samples * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    return (
        states[train_indices],
        actions[train_indices],
        states[val_indices],
        actions[val_indices],
    )


def create_batches(
    states: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """미니배치 생성"""
    n_samples = len(states)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    batches = []
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        batches.append((states[batch_indices], actions[batch_indices]))
    
    return batches


def compute_loss(
    agent: BCAgent,
    states: np.ndarray,
    actions: np.ndarray,
) -> float:
    """전체 데이터셋에 대한 평균 손실 계산"""
    predictions = agent.predict(states)
    mse = np.mean((predictions - actions) ** 2)
    return float(mse)


def train_epoch(
    agent: BCAgent,
    train_states: np.ndarray,
    train_actions: np.ndarray,
    batch_size: int,
) -> float:
    """한 에포크 학습"""
    batches = create_batches(train_states, train_actions, batch_size, shuffle=True)
    epoch_loss = 0.0
    
    for batch_states, batch_actions in batches:
        loss = agent.train(batch_states, batch_actions)
        epoch_loss += loss * len(batch_states)
    
    return epoch_loss / len(train_states)


def validate(
    agent: BCAgent,
    val_states: np.ndarray,
    val_actions: np.ndarray,
) -> float:
    """검증 데이터셋 평가"""
    return compute_loss(agent, val_states, val_actions)


class EarlyStopping:
    """조기 종료 핸들러"""
    
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss: float) -> bool:
        """조기 종료 여부 판단"""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def train_bc_agent(
    agent: BCAgent,
    train_states: np.ndarray,
    train_actions: np.ndarray,
    val_states: np.ndarray,
    val_actions: np.ndarray,
    config: TrainingConfig,
) -> TrainingHistory:
    """BC 에이전트 학습"""
    history = TrainingHistory()
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)
    
    # 체크포인트 디렉토리 생성
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if config.verbose:
        print(f"Training BC Agent")
        print(f"Train samples: {len(train_states)}, Val samples: {len(val_states)}")
        print(f"Epochs: {config.num_epochs}, Batch size: {config.batch_size}")
        print("-" * 60)
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # 학습
        train_loss = train_epoch(
            agent,
            train_states,
            train_actions,
            config.batch_size,
        )
        
        # 검증
        val_loss = validate(agent, val_states, val_actions)
        
        epoch_time = time.time() - epoch_start
        
        # 로깅
        if config.verbose:
            print(
                f"Epoch {epoch+1:3d}/{config.num_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Time: {epoch_time:.2f}s"
            )
        
        # 체크포인트 저장
        if (epoch + 1) % config.save_frequency == 0:
            checkpoint_path = config.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.npz"
            agent.save(checkpoint_path)
        
        # 최고 모델 저장 (history.update() 전에 확인)
        if val_loss < history.best_val_loss:
            best_model_path = config.checkpoint_dir / "best_model.npz"
            agent.save(best_model_path)
            if config.verbose:
                print(f"  → Best model saved (val_loss: {val_loss:.6f})")
        
        # 히스토리 업데이트
        history.update(epoch, train_loss, val_loss, epoch_time)
        
        # 조기 종료
        if early_stopping(val_loss):
            if config.verbose:
                print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    if config.verbose:
        print("-" * 60)
        print(f"Training complete!")
        print(f"Best epoch: {history.best_epoch+1}, Best val loss: {history.best_val_loss:.6f}")
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train Behavior Cloning agent")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to HDF5 demonstration data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/bc_model.npz",
        help="Output path for trained model",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden layer dimensions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # 시드 설정
    np.random.seed(args.seed)
    
    # 데이터 로드
    print(f"Loading data from {args.data}...")
    trajectories = load_trajectories_from_hdf5(args.data)
    
    # 데이터 추출
    all_states = []
    all_actions = []
    for traj in trajectories:
        all_states.append(traj['observations'])
        all_actions.append(traj['actions'])
    
    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    
    print(f"Loaded {len(trajectories)} trajectories, {len(states)} transitions")
    
    # 정규화
    print("Normalizing data...")
    state_normalizer = Normalizer()
    action_normalizer = Normalizer()
    
    state_normalizer.fit(states)
    action_normalizer.fit(actions)
    
    states_norm = state_normalizer.transform(states)
    actions_norm = action_normalizer.transform(actions)
    
    # Train/Val 분할
    train_states, train_actions, val_states, val_actions = split_dataset(
        states_norm,
        actions_norm,
        args.val_split,
        shuffle=True,
    )
    
    # 모델 생성
    obs_dim = states.shape[1]
    action_dim = actions.shape[1]
    
    agent = BCAgent(
        obs_dim=obs_dim,
        act_dim=action_dim,
        hidden_dims=args.hidden_dims,
    )
    
    # Learning rate와 weight decay 설정
    agent.policy.lr = args.lr
    agent.policy.weight_decay = args.weight_decay
    
    print(f"Created BC agent: {obs_dim} → {args.hidden_dims} → {action_dim}")
    print(f"Learning rate: {args.lr}, Weight decay: {args.weight_decay}")
    
    # 학습 설정
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        validation_split=args.val_split,
        checkpoint_dir=args.checkpoint_dir,
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
    
    # 최종 모델 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(output_path)
    print(f"\nFinal model saved to {output_path}")
    
    # 정규화 파라미터 저장
    norm_path = output_path.parent / f"{output_path.stem}_normalizers.npz"
    np.savez(
        norm_path,
        state_mean=state_normalizer.stats.mean,
        state_std=state_normalizer.stats.std,
        action_mean=action_normalizer.stats.mean,
        action_std=action_normalizer.stats.std,
    )
    print(f"Normalizers saved to {norm_path}")
    
    # 학습 기록 저장
    history_path = output_path.parent / f"{output_path.stem}_history.json"
    history.save(history_path)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
