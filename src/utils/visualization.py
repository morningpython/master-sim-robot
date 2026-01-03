"""시각화 및 렌더링 유틸리티.

MuJoCo 시뮬레이션의 궤적, RGB/Depth 이미지, 비디오를 생성하는 도구.
"""

from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import mujoco


class TrajectoryVisualizer:
    """3D 궤적 시각화 도구."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """초기화.

        Args:
            figsize: Figure 크기 (width, height)
        """
        self.figsize = figsize

    def plot_3d_trajectory(
        self,
        positions: np.ndarray,
        title: str = "3D End-Effector Trajectory",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """3D 궤적 플롯.

        Args:
            positions: (N, 3) 위치 배열 [x, y, z]
            title: 플롯 제목
            save_path: 저장 경로 (None이면 저장 안함)

        Returns:
            matplotlib Figure 객체
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        # 궤적 플롯
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            "b-",
            linewidth=2,
            alpha=0.6,
            label="Trajectory",
        )

        # 시작/끝 지점 표시
        ax.scatter(
            positions[0, 0],
            positions[0, 1],
            positions[0, 2],
            c="g",
            s=100,
            marker="o",
            label="Start",
        )
        ax.scatter(
            positions[-1, 0],
            positions[-1, 1],
            positions[-1, 2],
            c="r",
            s=100,
            marker="x",
            label="End",
        )

        # 축 레이블
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_joint_trajectories(
        self,
        joint_positions: np.ndarray,
        joint_names: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """조인트 궤적 시계열 플롯.

        Args:
            joint_positions: (N, n_joints) 조인트 위치 배열
            joint_names: 조인트 이름 리스트
            save_path: 저장 경로

        Returns:
            matplotlib Figure 객체
        """
        n_steps, n_joints = joint_positions.shape

        if joint_names is None:
            joint_names = [f"Joint {i+1}" for i in range(n_joints)]

        fig, axes = plt.subplots(
            n_joints, 1, figsize=(12, 2 * n_joints), sharex=True
        )
        if n_joints == 1:
            axes = [axes]

        time_steps = np.arange(n_steps)

        for i, (ax, name) in enumerate(zip(axes, joint_names)):
            ax.plot(time_steps, joint_positions[:, i], linewidth=2)
            ax.set_ylabel(f"{name}\n(rad)", rotation=0, ha="right", va="center")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time Step")
        fig.suptitle("Joint Trajectories", fontsize=14, fontweight="bold")
        fig.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


class CameraRenderer:
    """MuJoCo 카메라 렌더링 도구."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        width: int = 640,
        height: int = 480,
    ):
        """초기화.

        Args:
            model: MuJoCo 모델
            data: MuJoCo 데이터
            width: 렌더링 너비
            height: 렌더링 높이
        """
        self.model = model
        self.data = data
        self.width = width
        self.height = height

        # 렌더러 생성
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def render_rgb(
        self,
        camera_name: Optional[str] = None,
        camera_id: Optional[int] = None,
    ) -> np.ndarray:
        """RGB 이미지 렌더링.

        Args:
            camera_name: 카메라 이름
            camera_id: 카메라 ID (camera_name보다 우선)

        Returns:
            (H, W, 3) RGB 이미지 [0, 255]
        """
        self.renderer.update_scene(self.data, camera=camera_id or camera_name)
        rgb = self.renderer.render()
        return rgb

    def render_depth(
        self,
        camera_name: Optional[str] = None,
        camera_id: Optional[int] = None,
    ) -> np.ndarray:
        """Depth 이미지 렌더링.

        Args:
            camera_name: 카메라 이름
            camera_id: 카메라 ID

        Returns:
            (H, W) Depth 맵 (미터 단위)
        """
        self.renderer.update_scene(self.data, camera=camera_id or camera_name)
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()

        # Depth는 렌더러에서 normalized [0, 1] 범위로 반환될 수 있음
        # MuJoCo extent를 사용하여 실제 거리로 변환
        extent = self.model.stat.extent
        near = self.model.vis.map.znear * extent
        far = self.model.vis.map.zfar * extent

        # Linearize depth
        depth_linear = near / (1.0 - depth * (1.0 - near / far))

        return depth_linear

    def close(self):
        """렌더러 종료."""
        self.renderer.close()


class VideoRecorder:
    """비디오 녹화 도구."""

    def __init__(
        self,
        save_path: Path,
        fps: int = 30,
        codec: str = "libx264",
        bitrate: int = 5000,
    ):
        """초기화.

        Args:
            save_path: 비디오 저장 경로
            fps: Frames per second
            codec: 비디오 코덱
            bitrate: 비트레이트 (kbps)
        """
        self.save_path = Path(save_path)
        self.fps = fps
        self.codec = codec
        self.bitrate = bitrate
        self.frames: List[np.ndarray] = []

        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def add_frame(self, frame: np.ndarray):
        """프레임 추가.

        Args:
            frame: (H, W, 3) RGB 이미지 [0, 255]
        """
        self.frames.append(frame.copy())

    def save(self):
        """비디오 파일 저장."""
        if not self.frames:
            raise ValueError("No frames to save")

        height, width = self.frames[0].shape[:2]

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        im = ax.imshow(self.frames[0])

        def update(frame_idx):
            im.set_data(self.frames[frame_idx])
            return [im]

        anim = FuncAnimation(
            fig, update, frames=len(self.frames), interval=1000 / self.fps, blit=True
        )

        writer = FFMpegWriter(
            fps=self.fps, codec=self.codec, bitrate=self.bitrate
        )
        anim.save(self.save_path, writer=writer)
        plt.close(fig)

        print(f"Video saved to {self.save_path}")

    def clear(self):
        """프레임 버퍼 초기화."""
        self.frames.clear()


def create_comparison_plot(
    data_dict: Dict[str, np.ndarray],
    title: str = "Comparison",
    ylabel: str = "Value",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """여러 데이터 비교 플롯.

    Args:
        data_dict: {label: data} 딕셔너리
        title: 플롯 제목
        ylabel: Y축 레이블
        save_path: 저장 경로

    Returns:
        matplotlib Figure 객체
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for label, data in data_dict.items():
        ax.plot(data, label=label, linewidth=2, alpha=0.8)

    ax.set_xlabel("Time Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
