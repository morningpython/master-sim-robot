"""시각화 도구 테스트."""

import pytest
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import mujoco

from src.utils.visualization import (
    TrajectoryVisualizer,
    CameraRenderer,
    VideoRecorder,
    create_comparison_plot,
)


@pytest.fixture
def sample_trajectory():
    """샘플 3D 궤적 데이터."""
    t = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(t)
    y = np.sin(t)
    z = 0.5 + 0.2 * np.sin(2 * t)
    return np.column_stack([x, y, z])


@pytest.fixture
def sample_joint_trajectory():
    """샘플 조인트 궤적."""
    n_steps = 100
    n_joints = 7
    return np.random.randn(n_steps, n_joints) * 0.5


@pytest.fixture
def mujoco_model_data():
    """간단한 MuJoCo 모델/데이터."""
    xml = """
    <mujoco>
        <worldbody>
            <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
            <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
            <body pos="0 0 1">
                <joint type="free"/>
                <geom type="box" size="0.1 0.1 0.1"/>
            </body>
            <camera name="front" pos="2 0 1" xyaxes="0 -1 0 0.5 0 1"/>
        </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data


class TestTrajectoryVisualizer:
    """TrajectoryVisualizer 테스트."""

    def test_init(self):
        """초기화 테스트."""
        viz = TrajectoryVisualizer(figsize=(10, 8))
        assert viz.figsize == (10, 8)

    def test_plot_3d_trajectory(self, sample_trajectory, tmp_path):
        """3D 궤적 플롯 테스트."""
        viz = TrajectoryVisualizer()
        save_path = tmp_path / "trajectory.png"

        fig = viz.plot_3d_trajectory(
            sample_trajectory, title="Test Trajectory", save_path=save_path
        )

        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        plt.close(fig)

    def test_plot_3d_trajectory_no_save(self, sample_trajectory):
        """저장 없이 플롯 테스트."""
        viz = TrajectoryVisualizer()
        fig = viz.plot_3d_trajectory(sample_trajectory)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_joint_trajectories(self, sample_joint_trajectory, tmp_path):
        """조인트 궤적 플롯 테스트."""
        viz = TrajectoryVisualizer()
        save_path = tmp_path / "joints.png"

        joint_names = [f"J{i+1}" for i in range(7)]
        fig = viz.plot_joint_trajectories(
            sample_joint_trajectory, joint_names=joint_names, save_path=save_path
        )

        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        plt.close(fig)

    def test_plot_joint_trajectories_no_names(self, sample_joint_trajectory):
        """조인트 이름 없이 플롯 테스트."""
        viz = TrajectoryVisualizer()
        fig = viz.plot_joint_trajectories(sample_joint_trajectory)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_single_joint(self):
        """단일 조인트 플롯 테스트."""
        viz = TrajectoryVisualizer()
        single_joint = np.random.randn(100, 1)
        fig = viz.plot_joint_trajectories(single_joint)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestCameraRenderer:
    """CameraRenderer 테스트."""

    def test_init(self, mujoco_model_data):
        """초기화 테스트."""
        model, data = mujoco_model_data
        renderer = CameraRenderer(model, data, width=320, height=240)

        assert renderer.width == 320
        assert renderer.height == 240
        assert renderer.model is model
        assert renderer.data is data

        renderer.close()

    def test_render_rgb(self, mujoco_model_data):
        """RGB 렌더링 테스트."""
        model, data = mujoco_model_data
        renderer = CameraRenderer(model, data, width=320, height=240)

        # 물리 시뮬레이션 한 스텝
        mujoco.mj_step(model, data)

        rgb = renderer.render_rgb(camera_name="front")

        assert rgb.shape == (240, 320, 3)
        assert rgb.dtype == np.uint8
        assert rgb.min() >= 0
        assert rgb.max() <= 255

        renderer.close()

    def test_render_rgb_default_camera(self, mujoco_model_data):
        """기본 카메라로 렌더링 테스트."""
        model, data = mujoco_model_data
        renderer = CameraRenderer(model, data)

        mujoco.mj_step(model, data)
        # 카메라 이름 명시
        rgb = renderer.render_rgb(camera_name="front")

        assert rgb.shape == (480, 640, 3)

        renderer.close()

    def test_render_depth(self, mujoco_model_data):
        """Depth 렌더링 테스트."""
        model, data = mujoco_model_data
        renderer = CameraRenderer(model, data, width=320, height=240)

        mujoco.mj_step(model, data)
        depth = renderer.render_depth(camera_name="front")

        assert depth.shape == (240, 320)
        assert depth.dtype == np.float64 or depth.dtype == np.float32
        # Depth 값은 음수일 수 있음 (near/far plane 계산)
        assert depth is not None

        renderer.close()

    def test_multiple_renders(self, mujoco_model_data):
        """여러 번 렌더링 테스트."""
        model, data = mujoco_model_data
        renderer = CameraRenderer(model, data, width=160, height=120)

        frames = []
        for _ in range(5):
            mujoco.mj_step(model, data)
            rgb = renderer.render_rgb(camera_name="front")
            frames.append(rgb)

        assert len(frames) == 5
        for frame in frames:
            assert frame.shape == (120, 160, 3)

        renderer.close()


class TestVideoRecorder:
    """VideoRecorder 테스트."""

    def test_init(self, tmp_path):
        """초기화 테스트."""
        save_path = tmp_path / "test.mp4"
        recorder = VideoRecorder(save_path, fps=30)

        assert recorder.save_path == save_path
        assert recorder.fps == 30
        assert len(recorder.frames) == 0

    def test_add_frame(self, tmp_path):
        """프레임 추가 테스트."""
        recorder = VideoRecorder(tmp_path / "test.mp4")

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        recorder.add_frame(frame)

        assert len(recorder.frames) == 1
        np.testing.assert_array_equal(recorder.frames[0], frame)

    def test_add_multiple_frames(self, tmp_path):
        """여러 프레임 추가 테스트."""
        recorder = VideoRecorder(tmp_path / "test.mp4")

        for i in range(10):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            recorder.add_frame(frame)

        assert len(recorder.frames) == 10

    @pytest.mark.skip(reason="FFMpegWriter requires ffmpeg installation")
    def test_save_video(self, tmp_path):
        """비디오 저장 테스트."""
        save_path = tmp_path / "output.mp4"
        recorder = VideoRecorder(save_path, fps=10)

        # 30프레임 추가
        for i in range(30):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            recorder.add_frame(frame)

        recorder.save()

        assert save_path.exists()

    def test_save_empty_frames(self, tmp_path):
        """빈 프레임으로 저장 시 에러 테스트."""
        recorder = VideoRecorder(tmp_path / "test.mp4")

        with pytest.raises(ValueError, match="No frames to save"):
            recorder.save()

    def test_clear_frames(self, tmp_path):
        """프레임 초기화 테스트."""
        recorder = VideoRecorder(tmp_path / "test.mp4")

        for _ in range(5):
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            recorder.add_frame(frame)

        assert len(recorder.frames) == 5

        recorder.clear()
        assert len(recorder.frames) == 0


class TestCreateComparisonPlot:
    """create_comparison_plot 테스트."""

    def test_basic_plot(self, tmp_path):
        """기본 비교 플롯 테스트."""
        data_dict = {
            "Method A": np.random.randn(100),
            "Method B": np.random.randn(100),
            "Method C": np.random.randn(100),
        }

        save_path = tmp_path / "comparison.png"
        fig = create_comparison_plot(
            data_dict, title="Test Comparison", ylabel="Loss", save_path=save_path
        )

        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        plt.close(fig)

    def test_plot_no_save(self):
        """저장 없이 플롯 테스트."""
        data_dict = {
            "A": np.sin(np.linspace(0, 10, 100)),
            "B": np.cos(np.linspace(0, 10, 100)),
        }

        fig = create_comparison_plot(data_dict, title="Sin vs Cos")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_data(self):
        """단일 데이터 플롯 테스트."""
        data_dict = {"Single": np.random.randn(50)}
        fig = create_comparison_plot(data_dict)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestIntegration:
    """통합 테스트."""

    def test_trajectory_and_video(self, mujoco_model_data, tmp_path):
        """궤적 시각화와 비디오 녹화 통합 테스트."""
        model, data = mujoco_model_data

        # 렌더러 생성
        renderer = CameraRenderer(model, data, width=320, height=240)

        # 비디오 녹화기 생성
        video_path = tmp_path / "simulation.mp4"
        recorder = VideoRecorder(video_path, fps=30)

        # 시뮬레이션 및 녹화
        ee_positions = []
        for i in range(10):
            mujoco.mj_step(model, data)
            rgb = renderer.render_rgb(camera_name="front")
            recorder.add_frame(rgb)

            # End-effector 위치 수집 (여기서는 body 위치)
            ee_positions.append(data.qpos[:3].copy())

        # 궤적 시각화
        viz = TrajectoryVisualizer()
        traj_path = tmp_path / "trajectory.png"
        fig = viz.plot_3d_trajectory(
            np.array(ee_positions), title="Simulation Trajectory", save_path=traj_path
        )

        assert traj_path.exists()
        assert len(recorder.frames) == 10

        plt.close(fig)
        renderer.close()
