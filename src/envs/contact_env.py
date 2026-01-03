"""접촉 물리 시뮬레이션 환경.

MuJoCo의 접촉 API를 활용하여 접촉 감지, 힘/토크 센서, 충돌 처리를 구현.
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import mujoco

from src.envs.panda_env import PandaEnv


@dataclass
class ContactInfo:
    """접촉 정보."""

    geom1: int  # 첫 번째 geom ID
    geom2: int  # 두 번째 geom ID
    pos: np.ndarray  # 접촉 위치 (3,)
    frame: np.ndarray  # 접촉 프레임 (3, 3) - normal, tangent1, tangent2
    dist: float  # 침투 깊이
    force: np.ndarray  # 접촉력 (6,) - [fx, fy, fz, tx, ty, tz]
    friction: np.ndarray  # 마찰 계수 (5,)


class ContactPhysicsEnv(PandaEnv):
    """접촉 물리를 포함하는 Panda 환경."""

    def __init__(
        self,
        scene_xml_path: Optional[str] = None,
        enable_contact_forces: bool = True,
        contact_force_scale: float = 1.0,
        friction_loss_coefficient: float = 0.01,
    ):
        """초기화.

        Args:
            scene_xml_path: 장면 XML 경로
            enable_contact_forces: 접촉력 활성화 여부
            contact_force_scale: 접촉력 스케일 팩터
            friction_loss_coefficient: 마찰 손실 계수
        """
        super().__init__(scene_xml_path)

        self.enable_contact_forces = enable_contact_forces
        self.contact_force_scale = contact_force_scale
        self.friction_loss_coefficient = friction_loss_coefficient

        # 접촉 정보 저장
        self.contacts: List[ContactInfo] = []

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """환경 스텝 (접촉 감지 포함).

        Args:
            action: 액션 (7,) 조인트 위치 명령

        Returns:
            observation, reward, terminated, truncated, info
        """
        # 액션 적용
        self.data.ctrl[:] = action

        # 물리 시뮬레이션
        mujoco.mj_step(self.model, self.data)

        # 접촉 정보 업데이트
        if self.enable_contact_forces:
            self._update_contacts()

        # Observation 획득
        obs = self._get_obs()

        # 보상 계산 (간단한 거리 기반)
        ee_pos = self.get_ee_pose()[0]
        target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")
        target_pos = self.data.site_xpos[target_site_id]
        distance = np.linalg.norm(ee_pos - target_pos)
        reward = -distance

        # 접촉력 페널티
        if self.enable_contact_forces and len(self.contacts) > 0:
            total_contact_force = sum(
                np.linalg.norm(c.force[:3]) for c in self.contacts
            )
            reward -= self.friction_loss_coefficient * total_contact_force

        # 종료 조건
        terminated = distance < 0.02  # 2cm 이내면 성공
        truncated = self.data.time > 10.0  # 10초 제한

        info = {
            "distance": distance,
            "contacts": len(self.contacts),
            "ee_pos": ee_pos.copy(),
            "target_pos": target_pos.copy(),
        }

        return obs, reward, terminated, truncated, info

    def _update_contacts(self):
        """접촉 정보 업데이트."""
        self.contacts.clear()

        # MuJoCo 접촉 데이터 순회
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # 접촉 깊이가 0보다 작으면 (침투) 유효한 접촉
            if contact.dist < 0:
                # 접촉력 계산
                contact_force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, contact_force)

                # ContactInfo 생성
                contact_info = ContactInfo(
                    geom1=contact.geom1,
                    geom2=contact.geom2,
                    pos=contact.pos.copy(),
                    frame=contact.frame.copy().reshape(3, 3),
                    dist=contact.dist,
                    force=contact_force * self.contact_force_scale,
                    friction=contact.friction.copy(),
                )

                self.contacts.append(contact_info)

    def _get_obs(self) -> np.ndarray:
        """Observation 획득 (접촉 정보 포함).

        Returns:
            observation: [joint_pos(7), joint_vel(7), ee_pos(3), target_pos(3), contact_flag(1), max_contact_force(1)]
        """
        n_joints = 7  # Panda has 7 joints
        joint_pos = self.data.qpos[:n_joints].copy()
        joint_vel = self.data.qvel[:n_joints].copy()
        ee_pos = self.get_ee_pose()[0]
        
        # Target position from site
        target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")
        target_pos = self.data.site_xpos[target_site_id].copy()

        # 접촉 정보
        contact_flag = float(len(self.contacts) > 0)
        max_contact_force = (
            max(np.linalg.norm(c.force[:3]) for c in self.contacts)
            if self.contacts
            else 0.0
        )

        obs = np.concatenate(
            [
                joint_pos,
                joint_vel,
                ee_pos,
                target_pos,
                [contact_flag],
                [max_contact_force],
            ]
        )

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """추가 정보 반환 (override).
        
        Returns:
            info 딕셔너리
        """
        n_joints = 7
        ee_pos = self.get_ee_pose()[0]
        target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")
        target_pos = self.data.site_xpos[target_site_id].copy()
        
        return {
            "joint_pos": self.data.qpos[:n_joints].copy(),
            "joint_vel": self.data.qvel[:n_joints].copy(),
            "ee_pos": ee_pos,
            "target_pos": target_pos,
            "contacts": len(self.contacts),
        }

    def get_contacts(self) -> List[ContactInfo]:
        """현재 접촉 정보 반환.

        Returns:
            접촉 정보 리스트
        """
        return self.contacts.copy()

    def get_contact_by_geom_names(
        self, geom1_name: str, geom2_name: str
    ) -> Optional[ContactInfo]:
        """Geom 이름으로 접촉 정보 검색.

        Args:
            geom1_name: 첫 번째 geom 이름
            geom2_name: 두 번째 geom 이름

        Returns:
            ContactInfo 또는 None
        """
        try:
            geom1_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_name
            )
            geom2_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_name
            )
        except KeyError:
            return None

        for contact in self.contacts:
            if (contact.geom1 == geom1_id and contact.geom2 == geom2_id) or (
                contact.geom1 == geom2_id and contact.geom2 == geom1_id
            ):
                return contact

        return None

    def get_total_contact_force(self) -> np.ndarray:
        """전체 접촉력 계산.

        Returns:
            총 접촉력 (3,) [fx, fy, fz]
        """
        if not self.contacts:
            return np.zeros(3)

        total_force = np.sum([c.force[:3] for c in self.contacts], axis=0)
        return total_force

    def is_in_contact_with(self, geom_name: str) -> bool:
        """특정 geom과 접촉 중인지 확인.

        Args:
            geom_name: Geom 이름

        Returns:
            접촉 여부
        """
        try:
            geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
            )
        except KeyError:
            return False

        for contact in self.contacts:
            if contact.geom1 == geom_id or contact.geom2 == geom_id:
                return True

        return False


class ForceTorqueSensor:
    """힘/토크 센서 시뮬레이션."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        sensor_name: str,
    ):
        """초기화.

        Args:
            model: MuJoCo 모델
            data: MuJoCo 데이터
            sensor_name: 센서 이름 (force 또는 torque 센서)
        """
        self.model = model
        self.data = data
        self.sensor_name = sensor_name

        # 센서 ID 획득
        try:
            self.sensor_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name
            )
            if self.sensor_id == -1:
                raise ValueError(f"Sensor '{sensor_name}' not found in model")
        except Exception:
            raise ValueError(f"Sensor '{sensor_name}' not found in model")

        # 센서 타입 확인
        self.sensor_type = model.sensor_type[self.sensor_id]

        # 센서 데이터 주소
        self.sensor_adr = model.sensor_adr[self.sensor_id]
        self.sensor_dim = model.sensor_dim[self.sensor_id]

    def read(self) -> np.ndarray:
        """센서 값 읽기.

        Returns:
            센서 데이터 (dim,)
        """
        sensor_data = self.data.sensordata[
            self.sensor_adr : self.sensor_adr + self.sensor_dim
        ]
        return sensor_data.copy()

    def read_force(self) -> np.ndarray:
        """힘 센서 값 읽기.

        Returns:
            힘 (3,) [fx, fy, fz]
        """
        if self.sensor_type != mujoco.mjtSensor.mjSENS_FORCE:
            raise ValueError(f"Sensor '{self.sensor_name}' is not a force sensor")

        return self.read()

    def read_torque(self) -> np.ndarray:
        """토크 센서 값 읽기.

        Returns:
            토크 (3,) [tx, ty, tz]
        """
        if self.sensor_type != mujoco.mjtSensor.mjSENS_TORQUE:
            raise ValueError(f"Sensor '{self.sensor_name}' is not a torque sensor")

        return self.read()


def add_contact_visualization(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contacts: List[ContactInfo],
    scale: float = 0.01,
):
    """접촉 정보를 시각화 (디버깅용).

    Args:
        model: MuJoCo 모델
        data: MuJoCo 데이터
        contacts: 접촉 정보 리스트
        scale: 화살표 스케일
    """
    # MuJoCo의 비주얼 API를 사용하여 접촉점과 법선 벡터 표시
    # 실제 구현은 mujoco.viewer를 사용하거나 custom rendering 필요
    pass  # Placeholder for visualization implementation
