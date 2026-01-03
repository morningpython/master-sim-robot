"""Test MuJoCo Basic Viewer."""
import numpy as np
import pytest
import mujoco

from src.envs.basic_viewer import BasicViewer


def test_viewer_initialization():
    """뷰어 초기화 테스트"""
    viewer = BasicViewer()
    assert viewer.model is not None
    assert viewer.data is not None
    assert viewer.model.ngeom >= 2  # floor + box 최소


def test_default_scene_creation():
    """기본 장면 생성 테스트"""
    viewer = BasicViewer()
    
    # 모델 구조 확인
    assert viewer.model.nbody >= 3  # worldbody + box + sphere
    assert viewer.model.ngeom >= 3  # floor + box + sphere
    
    # Gravity 확인
    expected_gravity = np.array([0, 0, -9.81])
    np.testing.assert_allclose(viewer.model.opt.gravity, expected_gravity)


def test_simulation_step():
    """시뮬레이션 스텝 테스트"""
    viewer = BasicViewer()
    
    # 초기 시간
    initial_time = viewer.data.time
    
    # 10 스텝 진행
    viewer.step(n_steps=10)
    
    # 시간이 증가했는지 확인
    assert viewer.data.time > initial_time
    
    # 박스가 낙하했는지 확인 (z 좌표 감소)
    box_id = mujoco.mj_name2id(viewer.model, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_z = viewer.data.xpos[box_id][2]
    assert box_z < 1.0  # 초기 높이 1.0m에서 떨어짐


def test_render():
    """오프스크린 렌더링 테스트"""
    viewer = BasicViewer()
    
    # 이미지 렌더링
    img = viewer.render(width=320, height=240)
    
    # 이미지 속성 확인
    assert img.shape == (240, 320, 3)
    assert img.dtype == np.uint8
    assert img.min() >= 0
    assert img.max() <= 255


def test_reset():
    """리셋 테스트"""
    viewer = BasicViewer()
    
    # 시뮬레이션 진행
    viewer.step(n_steps=100)
    time_after_steps = viewer.data.time
    assert time_after_steps > 0
    
    # 리셋
    viewer.reset()
    
    # 시간이 0으로 돌아갔는지 확인
    assert viewer.data.time == 0


def test_physics_simulation():
    """물리 시뮬레이션 정확성 테스트"""
    viewer = BasicViewer()
    
    # 박스 ID 찾기
    box_id = mujoco.mj_name2id(viewer.model, mujoco.mjtObj.mjOBJ_BODY, "box")
    
    # 초기 위치 저장
    initial_pos = viewer.data.xpos[box_id].copy()
    
    # 1초간 시뮬레이션 (500 steps @ 2ms timestep)
    for _ in range(500):
        viewer.step()
    
    # 박스가 낙하했는지 확인
    final_pos = viewer.data.xpos[box_id]
    assert final_pos[2] < initial_pos[2]  # Z 좌표 감소
    
    # 바닥에 도달했는지 확인 (대략 0.1m 박스 반지름)
    assert final_pos[2] < 0.2


def test_custom_xml_loading():
    """커스텀 XML 로딩 테스트"""
    # 존재하지 않는 파일 - 기본 장면 사용
    viewer = BasicViewer(xml_path="nonexistent.xml")
    assert viewer.model is not None
    
    # 기본 장면이 로드되었는지 확인
    assert viewer.model.ngeom >= 2


@pytest.mark.parametrize("width,height", [
    (320, 240),
    (640, 480),
    (1280, 720),
])
def test_render_different_resolutions(width, height):
    """다양한 해상도 렌더링 테스트"""
    viewer = BasicViewer()
    img = viewer.render(width=width, height=height)
    assert img.shape == (height, width, 3)
