import numpy as np
from pathlib import Path
from src.utils.preprocessing import Normalizer
from src.models.bc_agent import BCAgent


def test_default_action_format_saved(tmp_path):
    # Create tiny dataset
    states = np.random.randn(10, 20)
    actions = np.random.randn(10, 7)

    # Save a normalizer file similar to train script
    path = tmp_path / "norms.npz"
    np.savez(path, state_mean=states.mean(0), state_std=states.std(0), action_mean=actions.mean(0), action_std=actions.std(0))

    # Load using run_demo loader and check there's a default action_type fallback
    from scripts.run_demo import load_normalizers

    s, a, action_type = load_normalizers(path)
    assert action_type in ("qpos", "delta")
