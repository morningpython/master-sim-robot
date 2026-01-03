"""Test environment setup."""
import sys
import pytest


def test_python_version():
    """Python 3.11 이상 확인"""
    assert sys.version_info >= (3, 11), "Python 3.11+ required"


def test_mujoco_import():
    """MuJoCo 임포트 확인"""
    try:
        import mujoco
        assert hasattr(mujoco, "__version__")
    except ImportError:
        pytest.fail("MuJoCo not installed")


def test_torch_import():
    """PyTorch 임포트 확인"""
    try:
        import torch
        assert torch.cuda.is_available() or True  # CPU도 허용
    except ImportError:
        pytest.fail("PyTorch not installed")


def test_numpy_import():
    """NumPy 임포트 확인"""
    try:
        import numpy as np
        assert hasattr(np, "__version__")
    except ImportError:
        pytest.fail("NumPy not installed")
