"""Shared pytest fixtures for host-dependent simulation capabilities."""

import os
import sys

import pytest


@pytest.fixture
def requires_renderer():
    """Skip OpenGL rendering tests when this process has no display context."""
    if os.environ.get("CI"):
        pytest.skip("Rendering tests run only in an explicitly configured graphics job")
    if sys.platform == "darwin" and not os.environ.get("DISPLAY"):
        pytest.skip("Offscreen MuJoCo rendering requires an available macOS display")
