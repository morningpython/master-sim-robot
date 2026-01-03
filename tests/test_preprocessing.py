"""데이터 전처리 테스트."""

import pytest
import numpy as np
from pathlib import Path
import h5py

from src.utils.preprocessing import (
    Normalizer,
    NormalizationStats,
    DataAugmenter,
    HDF5Dataset,
    TrajectoryDataset,
    save_trajectories_to_hdf5,
    load_trajectories_from_hdf5,
)


@pytest.fixture
def sample_data():
    """샘플 데이터."""
    np.random.seed(42)
    data = np.random.randn(100, 10)
    return data


@pytest.fixture
def sample_trajectories():
    """샘플 궤적 데이터."""
    np.random.seed(42)
    trajectories = []
    for _ in range(5):
        T = np.random.randint(50, 100)
        obs = np.random.randn(T, 20)
        act = np.random.randn(T, 7)
        trajectories.append((obs, act))
    return trajectories


class TestNormalizer:
    """Normalizer 테스트."""

    def test_init(self):
        """초기화 테스트."""
        normalizer = Normalizer(method="zscore")
        assert normalizer.method == "zscore"
        assert normalizer.stats is None

    def test_fit_zscore(self, sample_data):
        """Z-score fit 테스트."""
        normalizer = Normalizer(method="zscore")
        normalizer.fit(sample_data)

        assert normalizer.stats is not None
        assert normalizer.stats.mean.shape == (10,)
        assert normalizer.stats.std.shape == (10,)
        np.testing.assert_allclose(normalizer.stats.mean, np.mean(sample_data, axis=0), rtol=1e-5)

    def test_fit_minmax(self, sample_data):
        """MinMax fit 테스트."""
        normalizer = Normalizer(method="minmax")
        normalizer.fit(sample_data)

        assert normalizer.stats is not None
        np.testing.assert_allclose(normalizer.stats.min_val, np.min(sample_data, axis=0), rtol=1e-5)

    def test_transform_zscore(self, sample_data):
        """Z-score transform 테스트."""
        normalizer = Normalizer(method="zscore")
        normalizer.fit(sample_data)
        normalized = normalizer.transform(sample_data)

        # 정규화 후 평균 ~0, 표준편차 ~1
        assert normalized.shape == sample_data.shape
        np.testing.assert_allclose(np.mean(normalized, axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.std(normalized, axis=0), 1.0, atol=1e-5)

    def test_transform_minmax(self, sample_data):
        """MinMax transform 테스트."""
        normalizer = Normalizer(method="minmax")
        normalizer.fit(sample_data)
        normalized = normalizer.transform(sample_data)

        # 정규화 후 범위 [0, 1]
        assert normalized.shape == sample_data.shape
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)

    def test_inverse_transform_zscore(self, sample_data):
        """Z-score 역변환 테스트."""
        normalizer = Normalizer(method="zscore")
        normalized = normalizer.fit_transform(sample_data)
        recovered = normalizer.inverse_transform(normalized)

        np.testing.assert_allclose(recovered, sample_data, rtol=1e-5)

    def test_inverse_transform_minmax(self, sample_data):
        """MinMax 역변환 테스트."""
        normalizer = Normalizer(method="minmax")
        normalized = normalizer.fit_transform(sample_data)
        recovered = normalizer.inverse_transform(normalized)

        np.testing.assert_allclose(recovered, sample_data, rtol=1e-5)

    def test_fit_transform(self, sample_data):
        """fit_transform 테스트."""
        normalizer = Normalizer(method="zscore")
        normalized = normalizer.fit_transform(sample_data)

        assert normalizer.stats is not None
        assert normalized.shape == sample_data.shape

    def test_save_load(self, sample_data, tmp_path):
        """저장/로드 테스트."""
        normalizer = Normalizer(method="zscore")
        normalizer.fit(sample_data)

        save_path = tmp_path / "normalizer.npz"
        normalizer.save(save_path)

        assert save_path.exists()

        # 새로운 normalizer로 로드
        normalizer2 = Normalizer()
        normalizer2.load(save_path)

        assert normalizer2.method == "zscore"
        np.testing.assert_allclose(normalizer2.stats.mean, normalizer.stats.mean)
        np.testing.assert_allclose(normalizer2.stats.std, normalizer.stats.std)

    def test_transform_before_fit(self, sample_data):
        """fit 전에 transform 시 에러."""
        normalizer = Normalizer()

        with pytest.raises(ValueError, match="Must call fit"):
            normalizer.transform(sample_data)

    def test_invalid_method(self, sample_data):
        """잘못된 method."""
        normalizer = Normalizer(method="invalid")
        normalizer.fit(sample_data)

        with pytest.raises(ValueError, match="Unknown method"):
            normalizer.transform(sample_data)


class TestDataAugmenter:
    """DataAugmenter 테스트."""

    def test_init(self):
        """초기화 테스트."""
        augmenter = DataAugmenter(noise_scale=0.01)
        assert augmenter.noise_scale == 0.01

    def test_add_noise(self, sample_data):
        """노이즈 추가 테스트."""
        augmenter = DataAugmenter(noise_scale=0.1)
        noisy_data = augmenter.add_noise(sample_data)

        assert noisy_data.shape == sample_data.shape
        # 노이즈가 추가되어 다름
        assert not np.allclose(noisy_data, sample_data)

    def test_add_noise_custom_scale(self, sample_data):
        """커스텀 스케일 노이즈 테스트."""
        augmenter = DataAugmenter(noise_scale=0.01)
        noisy_data = augmenter.add_noise(sample_data, scale=0.5)

        assert noisy_data.shape == sample_data.shape

    def test_augment_trajectory(self):
        """궤적 증강 테스트."""
        augmenter = DataAugmenter(noise_scale=0.05)

        obs = np.random.randn(100, 20)
        act = np.random.randn(100, 7)

        aug_obs, aug_act = augmenter.augment_trajectory(obs, act)

        assert aug_obs.shape == obs.shape
        assert aug_act.shape == act.shape
        assert not np.allclose(aug_obs, obs)
        assert not np.allclose(aug_act, act)

    def test_random_temporal_crop(self):
        """랜덤 시간 크롭 테스트."""
        augmenter = DataAugmenter()

        obs = np.random.randn(100, 20)
        act = np.random.randn(100, 7)

        crop_length = 50
        crop_obs, crop_act = augmenter.random_temporal_crop(obs, act, crop_length)

        assert crop_obs.shape == (50, 20)
        assert crop_act.shape == (50, 7)

    def test_random_temporal_crop_short_sequence(self):
        """짧은 시퀀스 크롭 테스트."""
        augmenter = DataAugmenter()

        obs = np.random.randn(30, 20)
        act = np.random.randn(30, 7)

        crop_length = 50
        crop_obs, crop_act = augmenter.random_temporal_crop(obs, act, crop_length)

        # 길이가 부족하면 원본 반환
        assert crop_obs.shape == obs.shape
        assert crop_act.shape == act.shape


class TestHDF5Dataset:
    """HDF5Dataset 테스트."""

    def test_create_and_load_trajectory(self, tmp_path):
        """궤적 생성 및 로드 테스트."""
        hdf5_path = tmp_path / "test.h5"

        obs = np.random.randn(100, 20)
        act = np.random.randn(100, 7)

        # 저장
        with HDF5Dataset(hdf5_path, mode="w") as dataset:
            dataset.create_trajectory_group("traj_0", obs, act)

        # 로드
        with HDF5Dataset(hdf5_path, mode="r") as dataset:
            loaded_obs, loaded_act = dataset.get_trajectory("traj_0")

        np.testing.assert_allclose(loaded_obs, obs)
        np.testing.assert_allclose(loaded_act, act)

    def test_list_trajectories(self, tmp_path, sample_trajectories):
        """궤적 리스트 테스트."""
        hdf5_path = tmp_path / "test.h5"

        with HDF5Dataset(hdf5_path, mode="w") as dataset:
            for i, (obs, act) in enumerate(sample_trajectories):
                dataset.create_trajectory_group(f"traj_{i}", obs, act)

        with HDF5Dataset(hdf5_path, mode="r") as dataset:
            traj_ids = dataset.list_trajectories()

        assert len(traj_ids) == 5
        assert "traj_0" in traj_ids

    def test_get_dataset_stats(self, tmp_path, sample_trajectories):
        """데이터셋 통계 테스트."""
        hdf5_path = tmp_path / "test.h5"

        with HDF5Dataset(hdf5_path, mode="w") as dataset:
            for i, (obs, act) in enumerate(sample_trajectories):
                dataset.create_trajectory_group(f"traj_{i}", obs, act)

        with HDF5Dataset(hdf5_path, mode="r") as dataset:
            stats = dataset.get_dataset_stats()

        assert stats["num_trajectories"] == 5
        assert stats["obs_dim"] == 20
        assert stats["act_dim"] == 7
        assert stats["total_steps"] > 0

    def test_context_manager(self, tmp_path):
        """Context manager 테스트."""
        hdf5_path = tmp_path / "test.h5"

        with HDF5Dataset(hdf5_path, mode="w") as dataset:
            assert dataset.file is not None

        # Context 종료 후 파일 닫힘
        assert hdf5_path.exists()

    def test_operation_without_context(self, tmp_path):
        """Context 없이 연산 시 에러."""
        hdf5_path = tmp_path / "test.h5"
        dataset = HDF5Dataset(hdf5_path, mode="w")

        with pytest.raises(ValueError, match="File not opened"):
            dataset.list_trajectories()


class TestTrajectoryDataset:
    """TrajectoryDataset 테스트."""

    def test_init(self, sample_trajectories):
        """초기화 테스트."""
        obs_list = [traj[0] for traj in sample_trajectories]
        act_list = [traj[1] for traj in sample_trajectories]

        dataset = TrajectoryDataset(obs_list, act_list)

        assert len(dataset) == 5

    def test_getitem(self, sample_trajectories):
        """인덱싱 테스트."""
        obs_list = [traj[0] for traj in sample_trajectories]
        act_list = [traj[1] for traj in sample_trajectories]

        dataset = TrajectoryDataset(obs_list, act_list)

        obs, act = dataset[0]
        assert obs.shape == sample_trajectories[0][0].shape
        assert act.shape == sample_trajectories[0][1].shape

    def test_transform(self, sample_trajectories):
        """Transform 함수 테스트."""
        obs_list = [traj[0] for traj in sample_trajectories]
        act_list = [traj[1] for traj in sample_trajectories]

        def dummy_transform(obs, act):
            return obs * 2, act * 2

        dataset = TrajectoryDataset(obs_list, act_list, transform=dummy_transform)

        obs, act = dataset[0]
        np.testing.assert_allclose(obs, obs_list[0] * 2)
        np.testing.assert_allclose(act, act_list[0] * 2)

    def test_from_hdf5(self, tmp_path, sample_trajectories):
        """HDF5에서 로드 테스트."""
        hdf5_path = tmp_path / "test.h5"

        # HDF5 저장
        save_trajectories_to_hdf5(sample_trajectories, hdf5_path)

        # Dataset 로드
        dataset = TrajectoryDataset.from_hdf5(hdf5_path)

        assert len(dataset) == 5

    def test_flatten(self, sample_trajectories):
        """Flatten 테스트."""
        obs_list = [traj[0] for traj in sample_trajectories]
        act_list = [traj[1] for traj in sample_trajectories]

        dataset = TrajectoryDataset(obs_list, act_list)
        all_obs, all_act = dataset.flatten()

        total_steps = sum(len(obs) for obs in obs_list)
        assert all_obs.shape[0] == total_steps
        assert all_act.shape[0] == total_steps

    def test_compute_normalization_stats(self, sample_trajectories):
        """정규화 통계 계산 테스트."""
        obs_list = [traj[0] for traj in sample_trajectories]
        act_list = [traj[1] for traj in sample_trajectories]

        dataset = TrajectoryDataset(obs_list, act_list)
        obs_stats, act_stats = dataset.compute_normalization_stats()

        assert isinstance(obs_stats, NormalizationStats)
        assert isinstance(act_stats, NormalizationStats)
        assert obs_stats.mean.shape == (20,)
        assert act_stats.mean.shape == (7,)


class TestSaveLoadFunctions:
    """저장/로드 함수 테스트."""

    def test_save_trajectories_to_hdf5(self, tmp_path, sample_trajectories):
        """HDF5 저장 테스트."""
        hdf5_path = tmp_path / "trajectories.h5"

        save_trajectories_to_hdf5(sample_trajectories, hdf5_path)

        assert hdf5_path.exists()

        # 저장된 데이터 확인
        with HDF5Dataset(hdf5_path, mode="r") as dataset:
            assert len(dataset.list_trajectories()) == 5

    def test_load_trajectories_from_hdf5(self, tmp_path, sample_trajectories):
        """HDF5 로드 테스트."""
        hdf5_path = tmp_path / "trajectories.h5"

        # 저장
        save_trajectories_to_hdf5(sample_trajectories, hdf5_path)

        # 로드
        loaded_trajectories = load_trajectories_from_hdf5(hdf5_path)

        assert len(loaded_trajectories) == 5

        # 첫 번째 궤적 비교
        np.testing.assert_allclose(
            loaded_trajectories[0][0], sample_trajectories[0][0]
        )
        np.testing.assert_allclose(
            loaded_trajectories[0][1], sample_trajectories[0][1]
        )

    def test_roundtrip(self, tmp_path, sample_trajectories):
        """저장 후 로드 roundtrip 테스트."""
        hdf5_path = tmp_path / "roundtrip.h5"

        # 저장
        save_trajectories_to_hdf5(sample_trajectories, hdf5_path)

        # 로드
        loaded = load_trajectories_from_hdf5(hdf5_path)

        # 모든 궤적 비교
        for i, ((obs1, act1), (obs2, act2)) in enumerate(
            zip(sample_trajectories, loaded)
        ):
            np.testing.assert_allclose(obs1, obs2, err_msg=f"Trajectory {i} obs mismatch")
            np.testing.assert_allclose(act1, act2, err_msg=f"Trajectory {i} act mismatch")


class TestIntegration:
    """통합 테스트."""

    def test_full_pipeline(self, tmp_path):
        """전체 파이프라인 테스트."""
        # 1. 데이터 생성
        np.random.seed(42)
        trajectories = []
        for _ in range(10):
            T = 50
            obs = np.random.randn(T, 20)
            act = np.random.randn(T, 7)
            trajectories.append((obs, act))

        # 2. HDF5 저장
        hdf5_path = tmp_path / "data.h5"
        save_trajectories_to_hdf5(trajectories, hdf5_path)

        # 3. Dataset 로드
        dataset = TrajectoryDataset.from_hdf5(hdf5_path)

        # 4. 정규화 통계 계산
        obs_stats, act_stats = dataset.compute_normalization_stats()

        # 5. Normalizer 생성
        obs_normalizer = Normalizer(method="zscore")
        obs_normalizer.stats = obs_stats

        act_normalizer = Normalizer(method="zscore")
        act_normalizer.stats = act_stats

        # 6. 데이터 정규화
        obs, act = dataset[0]
        norm_obs = obs_normalizer.transform(obs)
        norm_act = act_normalizer.transform(act)

        # 7. 역변환
        recovered_obs = obs_normalizer.inverse_transform(norm_obs)
        recovered_act = act_normalizer.inverse_transform(norm_act)

        np.testing.assert_allclose(recovered_obs, obs, rtol=1e-5)
        np.testing.assert_allclose(recovered_act, act, rtol=1e-5)

    def test_augmentation_pipeline(self):
        """증강 파이프라인 테스트."""
        # 데이터 생성
        obs = np.random.randn(100, 20)
        act = np.random.randn(100, 7)

        # Augmenter
        augmenter = DataAugmenter(noise_scale=0.05)

        # 증강
        aug_obs, aug_act = augmenter.augment_trajectory(obs, act)

        # 크롭
        crop_obs, crop_act = augmenter.random_temporal_crop(aug_obs, aug_act, 50)

        assert crop_obs.shape == (50, 20)
        assert crop_act.shape == (50, 7)
