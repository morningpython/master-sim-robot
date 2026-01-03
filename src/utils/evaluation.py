"""
평가 메트릭 및 유틸리티

환경에서 에이전트를 평가하고 다양한 메트릭을 계산하는 유틸리티
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from src.envs.panda_env import PandaEnv
from src.models.bc_agent import BCAgent
from src.utils.preprocessing import Normalizer


def flatten_observation(obs: Union[Dict, np.ndarray]) -> np.ndarray:
    """관찰값을 1D numpy array로 변환
    
    Args:
        obs: 딕셔너리 또는 numpy array
    
    Returns:
        1D numpy array
    """
    if isinstance(obs, dict):
        # 딕셔너리를 정렬된 키 순서로 flatten
        return np.concatenate([obs[key].flatten() for key in sorted(obs.keys())])
    return obs


@dataclass
class EpisodeMetrics:
    """에피소드 메트릭"""
    
    success: bool
    total_reward: float
    episode_length: int
    final_distance: float
    mean_action_magnitude: float
    trajectory: Dict[str, np.ndarray]


@dataclass
class EvaluationResults:
    """평가 결과"""
    
    success_rate: float
    mean_reward: float
    std_reward: float
    mean_episode_length: float
    mean_final_distance: float
    mean_action_magnitude: float
    episode_metrics: List[EpisodeMetrics]
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'success_rate': float(self.success_rate),
            'mean_reward': float(self.mean_reward),
            'std_reward': float(self.std_reward),
            'mean_episode_length': float(self.mean_episode_length),
            'mean_final_distance': float(self.mean_final_distance),
            'mean_action_magnitude': float(self.mean_action_magnitude),
            'num_episodes': len(self.episode_metrics),
        }
    
    def __str__(self) -> str:
        """문자열 표현"""
        return (
            f"Evaluation Results ({len(self.episode_metrics)} episodes):\n"
            f"  Success Rate: {self.success_rate:.1%}\n"
            f"  Mean Reward: {self.mean_reward:.2f} ± {self.std_reward:.2f}\n"
            f"  Mean Episode Length: {self.mean_episode_length:.1f}\n"
            f"  Mean Final Distance: {self.mean_final_distance:.4f}\n"
            f"  Mean Action Magnitude: {self.mean_action_magnitude:.4f}"
        )


def run_episode(
    env: PandaEnv,
    agent: BCAgent,
    state_normalizer: Optional[Normalizer] = None,
    action_normalizer: Optional[Normalizer] = None,
    action_format: str = "qpos",
    max_steps: int = 500,
    render: bool = False,
    distance_threshold: float = 0.05,
) -> EpisodeMetrics:
    """에피소드 실행 및 메트릭 수집
    
    Args:
        env: 환경
        agent: 에이전트
        state_normalizer: 상태 정규화기 (선택)
        max_steps: 최대 스텝 수
        render: 렌더링 여부
        distance_threshold: 성공 판정 거리 임계값
    
    Returns:
        에피소드 메트릭
    """
    obs, info = env.reset()
    obs = flatten_observation(obs)
    
    states = []
    actions = []
    
    total_reward = 0.0
    step = 0
    
    for step in range(max_steps):
        # 정규화
        if state_normalizer is not None:
            obs_norm = state_normalizer.transform(obs)
        else:
            obs_norm = obs
        
        # 액션 예측
        action_pred = agent.predict(obs_norm)
        # If actions were trained in normalized space, inverse-transform them before applying
        if action_normalizer is not None:
            action_un = action_normalizer.inverse_transform(action_pred)
        else:
            action_un = action_pred

        # Depending on action_format, interpret the un-normalized action
        current_qpos = env.data.qpos[: env.model.nv].copy()
        if action_format == "delta":
            # action_un is delta; apply as qpos_target = current_qpos + delta
            qpos_target = current_qpos[:7] + action_un
            applied_action = qpos_target
        else:
            # action_un is absolute qpos target
            applied_action = action_un

        # 환경 스텝
        next_obs, info = env.step(applied_action)
        next_obs = flatten_observation(next_obs)
        
        # record the applied action for metrics
        actions.append(np.array(applied_action))
        
        # 보상 계산 (목표까지의 거리 감소)
        distance = info['ee_to_target_dist']
        reward = -distance  # 거리가 가까울수록 높은 보상
        
        # 기록
        states.append(obs)
        total_reward += reward
        
        obs = next_obs
        
        if render:
            env.render()
        
        # 목표 도달 시 종료
        if distance < distance_threshold:
            break
    
    # 성공 여부
    final_distance = info.get('ee_to_target_dist', np.inf)
    success = final_distance < distance_threshold
    
    # 액션 크기
    actions_array = np.array(actions)
    if len(actions_array) > 0:
        mean_action_magnitude = np.mean(np.linalg.norm(actions_array, axis=1))
    else:
        mean_action_magnitude = 0.0
    
    # 궤적 저장
    trajectory = {
        'observations': np.array(states),
        'actions': actions_array,
        'rewards': np.full(len(states), total_reward / max(len(states), 1)),
    }
    
    return EpisodeMetrics(
        success=success,
        total_reward=total_reward,
        episode_length=step + 1,
        final_distance=final_distance,
        mean_action_magnitude=mean_action_magnitude,
        trajectory=trajectory,
    )


def evaluate_agent(
    agent: BCAgent,
    env: PandaEnv,
    num_episodes: int = 100,
    state_normalizer: Optional[Normalizer] = None,
    action_normalizer: Optional[Normalizer] = None,
    action_format: str = "qpos",
    max_steps: int = 500,
    render: bool = False,
    verbose: bool = True,
) -> EvaluationResults:
    """에이전트 평가
    
    Args:
        agent: 평가할 에이전트
        env: 환경
        num_episodes: 평가 에피소드 수
        state_normalizer: 상태 정규화기
        max_steps: 에피소드당 최대 스텝
        render: 렌더링 여부
        verbose: 로깅 여부
    
    Returns:
        평가 결과
    """
    episode_metrics = []
    
    if verbose:
        print(f"Evaluating agent for {num_episodes} episodes...")
    
    for i in range(num_episodes):
        metrics = run_episode(
            env,
            agent,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
            action_format=action_format,
            max_steps=max_steps,
            render=render,
        )
        episode_metrics.append(metrics)
        
        if verbose and (i + 1) % 10 == 0:
            successes = sum(m.success for m in episode_metrics)
            print(f"  {i+1}/{num_episodes} episodes | Success: {successes}/{i+1}")
    
    # 통계 계산
    successes = [m.success for m in episode_metrics]
    rewards = [m.total_reward for m in episode_metrics]
    lengths = [m.episode_length for m in episode_metrics]
    distances = [m.final_distance for m in episode_metrics]
    action_mags = [m.mean_action_magnitude for m in episode_metrics]
    
    results = EvaluationResults(
        success_rate=np.mean(successes),
        mean_reward=np.mean(rewards),
        std_reward=np.std(rewards),
        mean_episode_length=np.mean(lengths),
        mean_final_distance=np.mean(distances),
        mean_action_magnitude=np.mean(action_mags),
        episode_metrics=episode_metrics,
    )
    
    if verbose:
        print("\n" + str(results))
    
    return results


def compute_mse(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """MSE 계산
    
    Args:
        predictions: 예측값 (N, D)
        targets: 타겟값 (N, D)
    
    Returns:
        MSE
    """
    return float(np.mean((predictions - targets) ** 2))


def compute_mae(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """MAE 계산
    
    Args:
        predictions: 예측값 (N, D)
        targets: 타겟값 (N, D)
    
    Returns:
        MAE
    """
    return float(np.mean(np.abs(predictions - targets)))


def compute_per_dim_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """차원별 MSE, MAE 계산
    
    Args:
        predictions: 예측값 (N, D)
        targets: 타겟값 (N, D)
    
    Returns:
        (per_dim_mse, per_dim_mae)
    """
    per_dim_mse = np.mean((predictions - targets) ** 2, axis=0)
    per_dim_mae = np.mean(np.abs(predictions - targets), axis=0)
    
    return per_dim_mse, per_dim_mae


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[Path] = None,
):
    """학습 곡선 시각화
    
    Args:
        train_losses: 학습 손실 리스트
        val_losses: 검증 손실 리스트
        save_path: 저장 경로 (선택)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    ax.plot(epochs, val_losses, label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_evaluation_metrics(
    results: EvaluationResults,
    save_path: Optional[Path] = None,
):
    """평가 메트릭 시각화
    
    Args:
        results: 평가 결과
        save_path: 저장 경로 (선택)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 보상 분포
    rewards = [m.total_reward for m in results.episode_metrics]
    axes[0, 0].hist(rewards, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(results.mean_reward, color='red', linestyle='--', 
                       label=f'Mean: {results.mean_reward:.2f}')
    axes[0, 0].set_xlabel('Total Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Reward Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 에피소드 길이 분포
    lengths = [m.episode_length for m in results.episode_metrics]
    axes[0, 1].hist(lengths, bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].axvline(results.mean_episode_length, color='red', linestyle='--',
                       label=f'Mean: {results.mean_episode_length:.1f}')
    axes[0, 1].set_xlabel('Episode Length')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Episode Length Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 최종 거리 분포
    distances = [m.final_distance for m in results.episode_metrics]
    axes[1, 0].hist(distances, bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].axvline(results.mean_final_distance, color='red', linestyle='--',
                       label=f'Mean: {results.mean_final_distance:.4f}')
    axes[1, 0].set_xlabel('Final Distance to Target')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Final Distance Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 성공/실패 파이 차트
    successes = sum(m.success for m in results.episode_metrics)
    failures = len(results.episode_metrics) - successes
    axes[1, 1].pie(
        [successes, failures],
        labels=['Success', 'Failure'],
        autopct='%1.1f%%',
        colors=['#2ecc71', '#e74c3c'],
        startangle=90,
    )
    axes[1, 1].set_title(f'Success Rate: {results.success_rate:.1%}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Evaluation metrics plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_action_predictions(
    expert_actions: np.ndarray,
    predicted_actions: np.ndarray,
    action_dim_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
):
    """액션 예측 비교 시각화
    
    Args:
        expert_actions: 전문가 액션 (N, action_dim)
        predicted_actions: 예측 액션 (N, action_dim)
        action_dim_names: 액션 차원 이름 리스트
        save_path: 저장 경로 (선택)
    """
    action_dim = expert_actions.shape[1]
    
    if action_dim_names is None:
        action_dim_names = [f'Action {i+1}' for i in range(action_dim)]
    
    # 차원별 MSE 계산
    per_dim_mse, per_dim_mae = compute_per_dim_metrics(
        predicted_actions, expert_actions
    )
    
    # 시각화
    n_cols = min(3, action_dim)
    n_rows = (action_dim + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    
    if action_dim == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(action_dim):
        ax = axes[i]
        
        # Scatter plot
        ax.scatter(expert_actions[:, i], predicted_actions[:, i], 
                  alpha=0.3, s=10)
        
        # 대각선 (perfect prediction)
        min_val = min(expert_actions[:, i].min(), predicted_actions[:, i].min())
        max_val = max(expert_actions[:, i].max(), predicted_actions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2, label='Perfect')
        
        ax.set_xlabel(f'Expert {action_dim_names[i]}')
        ax.set_ylabel(f'Predicted {action_dim_names[i]}')
        ax.set_title(f'{action_dim_names[i]}\nMSE: {per_dim_mse[i]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 남은 subplot 숨기기
    for i in range(action_dim, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Action predictions plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_trajectories(
    expert_traj: Dict[str, np.ndarray],
    agent_traj: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
):
    """전문가와 에이전트 궤적 비교
    
    Args:
        expert_traj: 전문가 궤적 {'observations': ..., 'actions': ...}
        agent_traj: 에이전트 궤적
        save_path: 저장 경로 (선택)
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    expert_obs = expert_traj['observations']
    agent_obs = agent_traj['observations']
    
    # End-effector position 비교 (앞 3차원이 위치라고 가정)
    ax = axes[0]
    t_expert = np.arange(len(expert_obs))
    t_agent = np.arange(len(agent_obs))
    
    for i, label in enumerate(['X', 'Y', 'Z']):
        ax.plot(t_expert, expert_obs[:, i], label=f'Expert {label}', 
               linestyle='--', linewidth=2)
        ax.plot(t_agent, agent_obs[:, i], label=f'Agent {label}', 
               linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Position (m)')
    ax.set_title('End-Effector Position Comparison')
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 액션 비교
    ax = axes[1]
    expert_actions = expert_traj['actions']
    agent_actions = agent_traj['actions']
    
    # 액션 노름
    expert_action_norm = np.linalg.norm(expert_actions, axis=1)
    agent_action_norm = np.linalg.norm(agent_actions, axis=1)
    
    ax.plot(t_expert, expert_action_norm, label='Expert Action Magnitude',
           linestyle='--', linewidth=2)
    ax.plot(t_agent, agent_action_norm, label='Agent Action Magnitude',
           linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Action Magnitude')
    ax.set_title('Action Magnitude Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
