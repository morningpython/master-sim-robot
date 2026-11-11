"""Small end-to-end smoke test for the public data, training, and demo CLIs."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> None:
    """Run a project CLI and include its output if it fails."""
    subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def test_data_training_and_demo_pipeline(tmp_path):
    """Generate a tiny dataset, train once, then evaluate the saved model."""
    dataset_path = tmp_path / "expert.h5"
    model_path = tmp_path / "bc_model.npz"
    checkpoint_dir = tmp_path / "checkpoints"
    results_dir = tmp_path / "results"

    run_cli(
        "scripts/generate_expert_data.py",
        "--num-trajectories",
        "2",
        "--steps",
        "4",
        "--output",
        str(dataset_path),
        "--action-format",
        "delta",
    )
    run_cli(
        "scripts/train_bc.py",
        "--data",
        str(dataset_path),
        "--output",
        str(model_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--hidden-dims",
        "16",
        "--action-format",
        "delta",
    )

    normalizers_path = tmp_path / "bc_model_normalizers.npz"
    run_cli(
        "scripts/run_demo.py",
        "--model",
        str(model_path),
        "--normalizers",
        str(normalizers_path),
        "--num-episodes",
        "1",
        "--max-steps",
        "2",
        "--hidden-dims",
        "16",
        "--output-dir",
        str(results_dir),
    )

    assert dataset_path.exists()
    assert model_path.exists()
    assert normalizers_path.exists()
    assert (results_dir / "evaluation_results.json").exists()
