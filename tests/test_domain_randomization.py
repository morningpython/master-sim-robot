"""Domain Randomization 기능 테스트."""

import pytest
import numpy as np
from src.envs.panda_env import PandaEnv


def test_domain_randomization_disabled():
    """Domain Randomization이 비활성화된 경우 물리 파라미터가 고정됨."""
    env = PandaEnv(enable_domain_randomization=False)
    
    # 초기 파라미터 기록
    initial_friction = env.model.geom_friction.copy()
    initial_mass = env.model.body_mass.copy()
    
    # 여러 번 리셋해도 동일해야 함
    for _ in range(5):
        env.reset()
        np.testing.assert_array_equal(env.model.geom_friction, initial_friction)
        np.testing.assert_array_equal(env.model.body_mass, initial_mass)


def test_domain_randomization_enabled():
    """Domain Randomization이 활성화된 경우 물리 파라미터가 변경됨."""
    env = PandaEnv(
        enable_domain_randomization=True,
        friction_range=(0.5, 1.5),
        mass_range=(0.8, 1.2),
    )
    
    # 초기 파라미터 기록
    original_friction = env._original_friction.copy()
    original_mass = env._original_mass.copy()
    
    friction_values = []
    mass_values = []
    
    # 여러 번 리셋하여 파라미터 변화 확인
    for _ in range(10):
        env.reset()
        friction_values.append(env.model.geom_friction[0, 0])
        mass_values.append(env.model.body_mass[1])  # body 1 (첫 링크)
    
    # 값들이 다양해야 함 (모두 같지 않아야 함)
    assert len(set(friction_values)) > 1, "Friction should vary across resets"
    assert len(set(mass_values)) > 1, "Mass should vary across resets"
    
    # 값들이 지정된 범위 내에 있어야 함
    for fric in friction_values:
        assert (
            original_friction[0, 0] * 0.5 <= fric <= original_friction[0, 0] * 1.5
        ), f"Friction {fric} out of range"
    
    for mass in mass_values:
        assert (
            original_mass[1] * 0.8 <= mass <= original_mass[1] * 1.2
        ), f"Mass {mass} out of range"


def test_original_params_preserved():
    """원본 물리 파라미터가 백업되어 있는지 확인."""
    env = PandaEnv(enable_domain_randomization=True)
    
    # 원본 파라미터 존재 확인
    assert hasattr(env, "_original_friction")
    assert hasattr(env, "_original_mass")
    
    # 리셋 후에도 원본 파라미터는 그대로
    original_friction = env._original_friction.copy()
    original_mass = env._original_mass.copy()
    
    env.reset()
    
    np.testing.assert_array_equal(env._original_friction, original_friction)
    np.testing.assert_array_equal(env._original_mass, original_mass)


def test_randomization_with_seed():
    """시드를 사용하여 재현 가능한 randomization 확인."""
    env = PandaEnv(enable_domain_randomization=True)
    
    # 같은 시드로 두 번 리셋
    env.reset(seed=42)
    friction_1 = env.model.geom_friction.copy()
    mass_1 = env.model.body_mass.copy()
    
    env.reset(seed=42)
    friction_2 = env.model.geom_friction.copy()
    mass_2 = env.model.body_mass.copy()
    
    # 같은 시드이므로 동일해야 함
    np.testing.assert_array_almost_equal(friction_1, friction_2)
    np.testing.assert_array_almost_equal(mass_1, mass_2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
