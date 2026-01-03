"""
MuJoCo Basic Viewer

간단한 MuJoCo 시뮬레이션 뷰어 구현.
바닥과 박스가 있는 기본 장면을 렌더링합니다.
"""
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path


class BasicViewer:
    """
    MuJoCo 기본 뷰어 클래스.
    
    Args:
        xml_path: MuJoCo XML 파일 경로 (없으면 기본 장면 생성)
    
    Example:
        >>> viewer = BasicViewer()
        >>> viewer.launch()  # 뷰어 실행
    """
    
    def __init__(self, xml_path: str | None = None):
        if xml_path and Path(xml_path).exists():
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            # 기본 장면 XML 생성
            self.model = self._create_default_scene()
        
        self.data = mujoco.MjData(self.model)
    
    def _create_default_scene(self) -> mujoco.MjModel:
        """
        기본 장면 생성 (바닥 + 박스).
        
        Returns:
            MuJoCo 모델 객체
        """
        xml_string = """
        <mujoco>
            <option timestep="0.002" gravity="0 0 -9.81"/>
            
            <visual>
                <headlight ambient="0.5 0.5 0.5"/>
            </visual>
            
            <asset>
                <texture name="grid" type="2d" builtin="checker" 
                         width="512" height="512" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3"/>
                <material name="grid" texture="grid" texrepeat="1 1" 
                          texuniform="true" reflectance="0.2"/>
            </asset>
            
            <worldbody>
                <!-- Ground plane -->
                <geom name="floor" type="plane" size="2 2 0.1" 
                      material="grid" condim="3"/>
                
                <!-- Falling box -->
                <body name="box" pos="0 0 1.0">
                    <freejoint/>
                    <geom name="box_geom" type="box" size="0.1 0.1 0.1" 
                          rgba="0.8 0.2 0.2 1" mass="1.0"/>
                </body>
                
                <!-- Static sphere -->
                <body name="sphere" pos="0.5 0 0.5">
                    <geom name="sphere_geom" type="sphere" size="0.15" 
                          rgba="0.2 0.8 0.2 1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        return mujoco.MjModel.from_xml_string(xml_string)
    
    def launch(self, duration: float | None = None) -> None:
        """
        인터랙티브 뷰어 실행.
        
        Args:
            duration: 실행 시간 (초). None이면 무한 실행
        
        Note:
            - ESC: 종료
            - Space: 일시정지/재생
            - 마우스 드래그: 카메라 회전
        """
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 135
            
            start_time = self.data.time
            while viewer.is_running():
                # 물리 시뮬레이션 스텝
                mujoco.mj_step(self.model, self.data)
                
                # 뷰어 동기화
                viewer.sync()
                
                # 지속 시간 체크
                if duration and (self.data.time - start_time) >= duration:
                    break
    
    def render(self, width: int = 640, height: int = 480) -> np.ndarray:
        """
        오프스크린 렌더링 (이미지 생성).
        
        Args:
            width: 이미지 너비
            height: 이미지 높이
        
        Returns:
            RGB 이미지 (H, W, 3), dtype=uint8
        """
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        mujoco.mj_forward(self.model, self.data)
        renderer.update_scene(self.data)
        
        # RGB 이미지 반환
        return renderer.render()
    
    def step(self, n_steps: int = 1) -> None:
        """
        시뮬레이션 스텝 진행.
        
        Args:
            n_steps: 진행할 스텝 수
        """
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)
    
    def reset(self) -> None:
        """시뮬레이션 리셋."""
        mujoco.mj_resetData(self.model, self.data)


def main():
    """메인 함수 - 뷰어 실행 예시."""
    print("MuJoCo Basic Viewer")
    print("Controls:")
    print("  - ESC: Exit")
    print("  - Space: Pause/Resume")
    print("  - Mouse: Rotate camera")
    print("-" * 40)
    
    viewer = BasicViewer()
    viewer.launch()


if __name__ == "__main__":
    main()
