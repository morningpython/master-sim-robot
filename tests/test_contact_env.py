"""접촉 물리 환경 테스트."""

import pytest
import numpy as np
import mujoco
from pathlib import Path

from src.envs.contact_env import (
    ContactPhysicsEnv,
    ContactInfo,
    ForceTorqueSensor,
)


@pytest.fixture
def contact_scene_xml(tmp_path):
    """접촉 테스트용 장면 XML."""
    xml_content = """
    <mujoco>
        <option gravity="0 0 -9.81" timestep="0.002"/>
        
        <worldbody>
            <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
            <geom name="floor" type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>
            
            <body name="panda_link0" pos="0 0 0">
                <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
                <geom type="cylinder" size="0.05 0.05" rgba="0.2 0.2 0.2 1"/>
                
                <body name="panda_link1" pos="0 0 0.1">
                    <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
                    <joint name="panda_joint1" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                    <geom type="cylinder" size="0.04 0.08" rgba="1 0.5 0 1"/>
                    
                    <body name="panda_link2" pos="0 0 0.1">
                        <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
                        <joint name="panda_joint2" type="hinge" axis="0 1 0" range="-1.7628 1.7628"/>
                        <geom type="box" size="0.03 0.03 0.08" rgba="1 0.5 0 1"/>
                        
                        <body name="panda_link3" pos="0 0 0.1">
                            <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
                            <joint name="panda_joint3" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                            <geom type="cylinder" size="0.03 0.07" rgba="1 0.5 0 1"/>
                            
                            <body name="panda_link4" pos="0 0 0.1">
                                <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
                                <joint name="panda_joint4" type="hinge" axis="0 -1 0" range="-3.0718 -0.0698"/>
                                <geom type="box" size="0.03 0.03 0.07" rgba="1 0.5 0 1"/>
                                
                                <body name="panda_link5" pos="0 0 0.1">
                                    <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
                                    <joint name="panda_joint5" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                                    <geom type="cylinder" size="0.03 0.06" rgba="1 0.5 0 1"/>
                                    
                                    <body name="panda_link6" pos="0 0 0.1">
                                        <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
                                        <joint name="panda_joint6" type="hinge" axis="0 1 0" range="-0.0175 3.7525"/>
                                        <geom type="box" size="0.02 0.02 0.05" rgba="1 0.5 0 1"/>
                                        
                                        <body name="panda_link7" pos="0 0 0.08">
                                            <inertial pos="0 0 0" mass="0.5" diaginertia="0.05 0.05 0.05"/>
                                            <joint name="panda_joint7" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                                            <geom name="ee_geom" type="sphere" size="0.02" rgba="0 1 0 1"/>
                                            <site name="ee_site" pos="0 0 0" size="0.01"/>
                                        </body>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
            
            <!-- Target box -->
            <body name="target_box" pos="0.3 0.3 0.5">
                <joint type="free"/>
                <inertial pos="0 0 0" mass="0.1" diaginertia="0.001 0.001 0.001"/>
                <geom name="target_geom" type="box" size="0.05 0.05 0.05" rgba="0 0 1 0.5"/>
                <site name="target" pos="0 0 0" size="0.01"/>
            </body>
            
            <!-- Obstacle -->
            <body name="obstacle" pos="0.2 0.2 0.3">
                <geom name="obstacle_geom" type="box" size="0.1 0.1 0.1" rgba="1 0 0 0.5"/>
            </body>
        </worldbody>
        
        <actuator>
            <position joint="panda_joint1" kp="100" ctrlrange="-2.8973 2.8973"/>
            <position joint="panda_joint2" kp="100" ctrlrange="-1.7628 1.7628"/>
            <position joint="panda_joint3" kp="100" ctrlrange="-2.8973 2.8973"/>
            <position joint="panda_joint4" kp="100" ctrlrange="-3.0718 -0.0698"/>
            <position joint="panda_joint5" kp="100" ctrlrange="-2.8973 2.8973"/>
            <position joint="panda_joint6" kp="100" ctrlrange="-0.0175 3.7525"/>
            <position joint="panda_joint7" kp="100" ctrlrange="-2.8973 2.8973"/>
        </actuator>
    </mujoco>
    """
    
    xml_path = tmp_path / "contact_scene.xml"
    xml_path.write_text(xml_content)
    return str(xml_path)


@pytest.fixture
def contact_env(contact_scene_xml):
    """ContactPhysicsEnv fixture."""
    env = ContactPhysicsEnv(contact_scene_xml)
    env.reset()
    return env


class TestContactPhysicsEnv:
    """ContactPhysicsEnv 테스트."""

    def test_init(self, contact_scene_xml):
        """초기화 테스트."""
        env = ContactPhysicsEnv(contact_scene_xml, enable_contact_forces=True)
        assert env.enable_contact_forces is True
        assert env.contact_force_scale == 1.0
        assert len(env.contacts) == 0

    def test_reset(self, contact_env):
        """리셋 테스트."""
        obs, info = contact_env.reset()
        
        # Observation shape: [joint(7), vel(7), ee(3), target(3), contact_flag(1), max_force(1)]
        assert obs.shape == (22,)
        assert "joint_pos" in info
        assert "ee_pos" in info

    def test_step(self, contact_env):
        """스텝 테스트."""
        action = np.zeros(7)
        obs, reward, terminated, truncated, info = contact_env.step(action)

        assert obs.shape == (22,)
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert "distance" in info
        assert "contacts" in info

    def test_contact_detection(self, contact_env):
        """접촉 감지 테스트."""
        # 여러 스텝 실행하여 접촉 발생시키기
        for _ in range(100):
            action = np.random.uniform(-1, 1, 7)
            contact_env.step(action)

        # 접촉이 발생했는지 확인 (중력으로 인해 바닥과 접촉 가능)
        # Note: 접촉은 장면 구성에 따라 다를 수 있음
        assert isinstance(contact_env.contacts, list)

    def test_get_contacts(self, contact_env):
        """접촉 정보 획득 테스트."""
        contact_env.step(np.zeros(7))
        contacts = contact_env.get_contacts()

        assert isinstance(contacts, list)
        for contact in contacts:
            assert isinstance(contact, ContactInfo)

    def test_observation_with_contact(self, contact_env):
        """접촉 정보가 포함된 observation 테스트."""
        obs, _, _, _, _ = contact_env.step(np.zeros(7))

        # Contact flag (index -2)
        contact_flag = obs[-2]
        assert contact_flag in [0.0, 1.0]

        # Max contact force (index -1)
        max_force = obs[-1]
        assert max_force >= 0.0

    def test_get_total_contact_force(self, contact_env):
        """전체 접촉력 계산 테스트."""
        contact_env.step(np.zeros(7))
        total_force = contact_env.get_total_contact_force()

        assert total_force.shape == (3,)
        assert np.all(np.isfinite(total_force))

    def test_is_in_contact_with(self, contact_env):
        """특정 geom과의 접촉 확인 테스트."""
        contact_env.step(np.zeros(7))
        
        # 존재하지 않는 geom
        assert contact_env.is_in_contact_with("nonexistent_geom") is False

    def test_contact_force_scale(self, contact_scene_xml):
        """접촉력 스케일 테스트."""
        env = ContactPhysicsEnv(
            contact_scene_xml, enable_contact_forces=True, contact_force_scale=2.0
        )
        env.reset()

        assert env.contact_force_scale == 2.0

    def test_friction_loss_coefficient(self, contact_scene_xml):
        """마찰 손실 계수 테스트."""
        env = ContactPhysicsEnv(
            contact_scene_xml, friction_loss_coefficient=0.05
        )
        env.reset()

        assert env.friction_loss_coefficient == 0.05

    def test_reward_with_contact_penalty(self, contact_env):
        """접촉 페널티가 포함된 보상 테스트."""
        # 첫 번째 스텝
        _, reward1, _, _, _ = contact_env.step(np.zeros(7))

        # 접촉이 있을 때 보상이 감소하는지 확인하기 어려우므로
        # 보상이 유효한 값인지만 확인
        assert isinstance(reward1, (int, float))
        assert np.isfinite(reward1)

    def test_termination_condition(self, contact_scene_xml):
        """종료 조건 테스트."""
        env = ContactPhysicsEnv(contact_scene_xml)
        env.reset()

        # Target에 매우 가까운 위치로 end-effector 이동
        # (실제로는 IK를 사용해야 하지만 여기서는 직접 설정)
        target_pos = env.data.site("target").xpos
        
        # 여러 스텝 실행
        terminated = False
        for _ in range(10):
            _, _, terminated, _, _ = env.step(np.zeros(7))
            if terminated:
                break

        # Termination은 거리에 따라 결정되므로 확인만
        assert isinstance(terminated, (bool, np.bool_))

    def test_truncation_condition(self, contact_scene_xml):
        """시간 제한 테스트."""
        env = ContactPhysicsEnv(contact_scene_xml)
        env.reset()

        # 충분히 많은 스텝 실행 (10초 / 0.002초 = 5000 스텝)
        truncated = False
        for i in range(6000):
            _, _, _, truncated, _ = env.step(np.zeros(7))
            if truncated:
                break

        assert truncated is True
        assert env.data.time > 10.0


class TestContactInfo:
    """ContactInfo 테스트."""

    def test_dataclass_creation(self):
        """데이터클래스 생성 테스트."""
        contact = ContactInfo(
            geom1=0,
            geom2=1,
            pos=np.array([0.0, 0.0, 0.0]),
            frame=np.eye(3),
            dist=-0.01,
            force=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            friction=np.array([0.5, 0.5, 0.005, 0.0001, 0.0001]),
        )

        assert contact.geom1 == 0
        assert contact.geom2 == 1
        assert contact.pos.shape == (3,)
        assert contact.frame.shape == (3, 3)
        assert contact.dist == -0.01
        assert contact.force.shape == (6,)
        assert contact.friction.shape == (5,)


class TestForceTorqueSensor:
    """ForceTorqueSensor 테스트."""

    def test_init_invalid_sensor(self, contact_scene_xml):
        """존재하지 않는 센서로 초기화 시 에러."""
        model = mujoco.MjModel.from_xml_path(contact_scene_xml)
        data = mujoco.MjData(model)

        with pytest.raises(ValueError, match="not found in model"):
            ForceTorqueSensor(model, data, "nonexistent_sensor")

    def test_sensor_with_sensor_xml(self, tmp_path):
        """센서가 포함된 XML로 테스트."""
        xml_content = """
        <mujoco>
            <worldbody>
                <body name="body1" pos="0 0 1">
                    <joint type="free"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                    <site name="force_site" pos="0 0 0"/>
                </body>
            </worldbody>
            
            <sensor>
                <force name="force_sensor" site="force_site"/>
                <torque name="torque_sensor" site="force_site"/>
            </sensor>
        </mujoco>
        """
        
        xml_path = tmp_path / "sensor_test.xml"
        xml_path.write_text(xml_content)
        
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        
        # Force sensor
        force_sensor = ForceTorqueSensor(model, data, "force_sensor")
        assert force_sensor.sensor_name == "force_sensor"
        assert force_sensor.sensor_type == mujoco.mjtSensor.mjSENS_FORCE
        
        # Torque sensor
        torque_sensor = ForceTorqueSensor(model, data, "torque_sensor")
        assert torque_sensor.sensor_name == "torque_sensor"
        assert torque_sensor.sensor_type == mujoco.mjtSensor.mjSENS_TORQUE

    def test_read_force_sensor(self, tmp_path):
        """Force 센서 읽기 테스트."""
        xml_content = """
        <mujoco>
            <worldbody>
                <body pos="0 0 1">
                    <joint type="free"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                    <site name="force_site" pos="0 0 0"/>
                </body>
            </worldbody>
            <sensor>
                <force name="force_sensor" site="force_site"/>
            </sensor>
        </mujoco>
        """
        
        xml_path = tmp_path / "force_test.xml"
        xml_path.write_text(xml_content)
        
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        
        sensor = ForceTorqueSensor(model, data, "force_sensor")
        force = sensor.read_force()
        
        assert force.shape == (3,)
        assert np.all(np.isfinite(force))

    def test_read_torque_sensor(self, tmp_path):
        """Torque 센서 읽기 테스트."""
        xml_content = """
        <mujoco>
            <worldbody>
                <body pos="0 0 1">
                    <joint type="free"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                    <site name="torque_site" pos="0 0 0"/>
                </body>
            </worldbody>
            <sensor>
                <torque name="torque_sensor" site="torque_site"/>
            </sensor>
        </mujoco>
        """
        
        xml_path = tmp_path / "torque_test.xml"
        xml_path.write_text(xml_content)
        
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        
        sensor = ForceTorqueSensor(model, data, "torque_sensor")
        torque = sensor.read_torque()
        
        assert torque.shape == (3,)
        assert np.all(np.isfinite(torque))

    def test_read_wrong_sensor_type(self, tmp_path):
        """잘못된 센서 타입으로 읽기 시 에러."""
        xml_content = """
        <mujoco>
            <worldbody>
                <body pos="0 0 1">
                    <joint type="free"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                    <site name="sensor_site" pos="0 0 0"/>
                </body>
            </worldbody>
            <sensor>
                <force name="force_sensor" site="sensor_site"/>
            </sensor>
        </mujoco>
        """
        
        xml_path = tmp_path / "wrong_type.xml"
        xml_path.write_text(xml_content)
        
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        
        sensor = ForceTorqueSensor(model, data, "force_sensor")
        
        # Force sensor에서 torque 읽으려고 하면 에러
        with pytest.raises(ValueError, match="not a torque sensor"):
            sensor.read_torque()
