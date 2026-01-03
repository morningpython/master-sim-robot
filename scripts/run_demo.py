"""
데모 실행 스크립트: 학습된 BC 에이전트로 데모 실행

Usage:
    python scripts/run_demo.py --model models/bc_model.npz --normalizers models/bc_model_normalizers.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.envs.panda_env import PandaEnv
from src.models.bc_agent import BCAgent
from src.utils.evaluation import evaluate_agent, plot_evaluation_metrics
from src.utils.preprocessing import Normalizer, NormalizationStats


def load_normalizers(path: Path) -> tuple[Normalizer, Normalizer]:
    """정규화기 로드
    
    Args:
        path: 정규화기 파일 경로
    
    Returns:
        (state_normalizer, action_normalizer)
    """
    data = np.load(path)
    
    state_normalizer = Normalizer()
    state_normalizer.stats = NormalizationStats(
        mean=data['state_mean'],
        std=data['state_std'],
        min_val=np.zeros_like(data['state_mean']),
        max_val=np.ones_like(data['state_mean']),
    )
    
    action_normalizer = Normalizer()
    action_normalizer.stats = NormalizationStats(
        mean=data['action_mean'],
        std=data['action_std'],
        min_val=np.zeros_like(data['action_mean']),
        max_val=np.ones_like(data['action_mean']),
    )
    
    return state_normalizer, action_normalizer


def main():
    parser = argparse.ArgumentParser(description="Run BC agent demo")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.npz)",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default=None,
        help="Path to normalizers (.npz)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="demo_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=20,
        help="Observation dimension",
    )
    parser.add_argument(
        "--act-dim",
        type=int,
        default=7,
        help="Action dimension",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden layer dimensions",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes",
    )
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 모델 로드
    print(f"Loading model from {args.model}...")
    agent = BCAgent(
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        hidden_dims=args.hidden_dims,
    )
    agent.load(Path(args.model))
    print(f"Model loaded successfully")
    
    # 정규화기 로드
    state_normalizer = None
    if args.normalizers:
        print(f"Loading normalizers from {args.normalizers}...")
        state_normalizer, _ = load_normalizers(Path(args.normalizers))
        print("Normalizers loaded successfully")
    
    # 환경 생성
    render_mode = "human" if args.render else None
    env = PandaEnv(render_mode=render_mode)
    print(f"Environment created (render={args.render})")
    
    # 평가
    print(f"\nRunning {args.num_episodes} episodes...")
    results = evaluate_agent(
        agent,
        env,
        num_episodes=args.num_episodes,
        state_normalizer=state_normalizer,
        max_steps=args.max_steps,
        render=args.render,
        verbose=True,
    )
    
    # 결과 저장
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # 시각화
    plot_path = output_dir / "evaluation_metrics.png"
    plot_evaluation_metrics(results, plot_path)
    print(f"Metrics plot saved to {plot_path}")
    
    # 요약 출력
    print("\n" + "="*60)
    print(results)
    print("="*60)


if __name__ == "__main__":
    main()
