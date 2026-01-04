"""데이터 전처리 파이프라인.

Normalization, Augmentation, HDF5 저장, DataLoader 구현.
"""

from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import h5py
from tqdm import tqdm


@dataclass
class NormalizationStats:
    """정규화 통계."""

    mean: np.ndarray
    std: np.ndarray
    min_val: np.ndarray
    max_val: np.ndarray


class Normalizer:
    """데이터 정규화."""

    def __init__(self, method: str = "zscore"):
        """초기화.

        Args:
            method: 정규화 방법 ("zscore", "minmax")
        """
        self.method = method
        self.stats: Optional[NormalizationStats] = None

    def fit(self, data: np.ndarray):
        """정규화 통계 계산.

        Args:
            data: (N, D) 데이터
        """
        self.stats = NormalizationStats(
            mean=np.mean(data, axis=0),
            std=np.std(data, axis=0) + 1e-8,  # Avoid division by zero
            min_val=np.min(data, axis=0),
            max_val=np.max(data, axis=0) + 1e-8,
        )

    def transform(self, data: np.ndarray) -> np.ndarray:
        """데이터 정규화.

        Args:
            data: (N, D) 데이터

        Returns:
            정규화된 데이터
        """
        if self.stats is None:
            raise ValueError("Must call fit() before transform()")

        if self.method == "zscore":
            return (data - self.stats.mean) / self.stats.std
        elif self.method == "minmax":
            return (data - self.stats.min_val) / (
                self.stats.max_val - self.stats.min_val
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """정규화 역변환.

        Args:
            data: 정규화된 데이터

        Returns:
            원본 스케일 데이터
        """
        if self.stats is None:
            raise ValueError("Must call fit() before inverse_transform()")

        if self.method == "zscore":
            return data * self.stats.std + self.stats.mean
        elif self.method == "minmax":
            return data * (self.stats.max_val - self.stats.min_val) + self.stats.min_val
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit 후 transform.

        Args:
            data: (N, D) 데이터

        Returns:
            정규화된 데이터
        """
        self.fit(data)
        return self.transform(data)

    def save(self, path: Path):
        """정규화 통계 저장.

        Args:
            path: 저장 경로
        """
        if self.stats is None:
            raise ValueError("No stats to save")

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            method=self.method,
            mean=self.stats.mean,
            std=self.stats.std,
            min_val=self.stats.min_val,
            max_val=self.stats.max_val,
        )

    def load(self, path: Path):
        """정규화 통계 로드.

        Args:
            path: 로드 경로
        """
        data = np.load(path, allow_pickle=True)
        self.method = str(data["method"])
        self.stats = NormalizationStats(
            mean=data["mean"],
            std=data["std"],
            min_val=data["min_val"],
            max_val=data["max_val"],
        )


class DataAugmenter:
    """데이터 증강."""

    def __init__(
        self,
        noise_scale: float = 0.01,
        rotation_range: float = 0.1,
        translation_range: float = 0.05,
        action_noise_scale: float = 0.005,
    ):
        """초기화.

        Args:
            noise_scale: 관측 노이즈 크기
            rotation_range: 회전 범위 (radians)
            translation_range: 이동 범위 (meters)
            action_noise_scale: 액션 노이즈 크기
        """
        self.noise_scale = noise_scale
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.action_noise_scale = action_noise_scale

    def add_noise(self, data: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
        """가우시안 노이즈 추가.

        Args:
            data: (N, D) 데이터
            scale: 노이즈 스케일 (None이면 기본값 사용)

        Returns:
            노이즈가 추가된 데이터
        """
        scale = scale or self.noise_scale
        noise = np.random.randn(*data.shape) * scale
        return data + noise

    def augment_trajectory(
        self, observations: np.ndarray, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """궤적 데이터 증강.

        Args:
            observations: (T, obs_dim) 관찰
            actions: (T, act_dim) 액션

        Returns:
            증강된 (observations, actions)
        """
        # 관측에 노이즈 추가
        aug_obs = self.add_noise(observations, scale=self.noise_scale)
        # 액션에 더 작은 노이즈 추가 (정밀도 유지)
        aug_act = self.add_noise(actions, scale=self.action_noise_scale)

        return aug_obs, aug_act

    def random_temporal_crop(
        self, observations: np.ndarray, actions: np.ndarray, crop_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """랜덤 시간 크롭.

        Args:
            observations: (T, obs_dim)
            actions: (T, act_dim)
            crop_length: 크롭 길이

        Returns:
            크롭된 (observations, actions)
        """
        T = len(observations)
        if T <= crop_length:
            return observations, actions

        start = np.random.randint(0, T - crop_length + 1)
        end = start + crop_length

        return observations[start:end], actions[start:end]

    def augment_dataset(
        self,
        trajectories: List[Tuple[np.ndarray, np.ndarray]],
        n_augment: int = 2,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """데이터셋 전체 증강.

        Args:
            trajectories: 원본 궤적 리스트
            n_augment: 각 궤적당 생성할 증강 데이터 수

        Returns:
            증강된 궤적 리스트 (원본 + 증강)
        """
        augmented = list(trajectories)  # 원본 포함

        for obs, act in tqdm(trajectories, desc="Augmenting data"):
            for _ in range(n_augment):
                aug_obs, aug_act = self.augment_trajectory(obs, act)
                augmented.append((aug_obs, aug_act))

        print(
            f"Dataset augmented: {len(trajectories)} → {len(augmented)} trajectories"
        )
        return augmented


class HDF5Dataset:
    """HDF5 기반 데이터셋."""

    def __init__(self, hdf5_path: Path, mode: str = "r"):
        """초기화.

        Args:
            hdf5_path: HDF5 파일 경로
            mode: 파일 모드 ("r", "w", "a")
        """
        self.hdf5_path = Path(hdf5_path)
        self.mode = mode
        self.file: Optional[h5py.File] = None

    def __enter__(self):
        """Context manager 진입."""
        self.file = h5py.File(self.hdf5_path, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료."""
        if self.file is not None:
            self.file.close()

    def create_trajectory_group(
        self, traj_id: str, observations: np.ndarray, actions: np.ndarray
    ):
        """궤적 그룹 생성.

        Args:
            traj_id: 궤적 ID
            observations: (T, obs_dim)
            actions: (T, act_dim)
        """
        if self.file is None:
            raise ValueError("File not opened. Use context manager.")

        grp = self.file.create_group(traj_id)
        grp.create_dataset("observations", data=observations, compression="gzip")
        grp.create_dataset("actions", data=actions, compression="gzip")
        grp.attrs["length"] = len(observations)

    def get_trajectory(self, traj_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """궤적 데이터 로드.

        Args:
            traj_id: 궤적 ID

        Returns:
            (observations, actions)
        """
        if self.file is None:
            raise ValueError("File not opened. Use context manager.")

        grp = self.file[traj_id]
        observations = np.array(grp["observations"])
        actions = np.array(grp["actions"])

        return observations, actions

    def list_trajectories(self) -> List[str]:
        """모든 궤적 ID 리스트.

        Returns:
            궤적 ID 리스트
        """
        if self.file is None:
            raise ValueError("File not opened. Use context manager.")

        return list(self.file.keys())

    def get_dataset_stats(self) -> Dict[str, any]:
        """데이터셋 통계.

        Returns:
            통계 딕셔너리
        """
        if self.file is None:
            raise ValueError("File not opened. Use context manager.")

        traj_ids = self.list_trajectories()
        total_steps = sum(self.file[tid].attrs["length"] for tid in traj_ids)

        # 첫 번째 궤적에서 차원 정보
        if traj_ids:
            first_obs, first_act = self.get_trajectory(traj_ids[0])
            obs_dim = first_obs.shape[1]
            act_dim = first_act.shape[1]
        else:
            obs_dim, act_dim = 0, 0

        return {
            "num_trajectories": len(traj_ids),
            "total_steps": total_steps,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
        }


class TrajectoryDataset:
    """궤적 데이터셋 (메모리 기반)."""

    def __init__(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray],
        transform: Optional[Callable] = None,
    ):
        """초기화.

        Args:
            observations: 관찰 리스트 [(T1, obs_dim), (T2, obs_dim), ...]
            actions: 액션 리스트 [(T1, act_dim), (T2, act_dim), ...]
            transform: 데이터 변환 함수
        """
        assert len(observations) == len(actions)
        self.observations = observations
        self.actions = actions
        self.transform = transform

    def __len__(self) -> int:
        """데이터셋 크기."""
        return len(self.observations)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """인덱스로 데이터 획득.

        Args:
            idx: 인덱스

        Returns:
            (observation, action)
        """
        obs = self.observations[idx]
        act = self.actions[idx]

        if self.transform:
            obs, act = self.transform(obs, act)

        return obs, act

    @classmethod
    def from_hdf5(
        cls, hdf5_path: Path, transform: Optional[Callable] = None
    ) -> "TrajectoryDataset":
        """HDF5에서 로드.

        Args:
            hdf5_path: HDF5 파일 경로
            transform: 데이터 변환 함수

        Returns:
            TrajectoryDataset
        """
        observations = []
        actions = []

        with HDF5Dataset(hdf5_path, mode="r") as dataset:
            traj_ids = dataset.list_trajectories()

            for traj_id in tqdm(traj_ids, desc="Loading trajectories"):
                obs, act = dataset.get_trajectory(traj_id)
                observations.append(obs)
                actions.append(act)

        return cls(observations, actions, transform)

    def flatten(self) -> Tuple[np.ndarray, np.ndarray]:
        """모든 궤적을 평탄화.

        Returns:
            (all_observations, all_actions) - (N, obs_dim), (N, act_dim)
        """
        all_obs = np.concatenate(self.observations, axis=0)
        all_act = np.concatenate(self.actions, axis=0)
        return all_obs, all_act

    def compute_normalization_stats(self) -> Tuple[NormalizationStats, NormalizationStats]:
        """정규화 통계 계산.

        Returns:
            (obs_stats, act_stats)
        """
        all_obs, all_act = self.flatten()

        obs_normalizer = Normalizer()
        obs_normalizer.fit(all_obs)

        act_normalizer = Normalizer()
        act_normalizer.fit(all_act)

        return obs_normalizer.stats, act_normalizer.stats


def save_trajectories_to_hdf5(
    trajectories: List[Tuple[np.ndarray, np.ndarray]], output_path: Path
):
    """궤적들을 HDF5로 저장.

    Args:
        trajectories: [(observations, actions), ...] 리스트
        output_path: 출력 HDF5 경로
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with HDF5Dataset(output_path, mode="w") as dataset:
        for i, (obs, act) in enumerate(tqdm(trajectories, desc="Saving trajectories")):
            traj_id = f"traj_{i:05d}"
            dataset.create_trajectory_group(traj_id, obs, act)

    print(f"Saved {len(trajectories)} trajectories to {output_path}")


def load_trajectories_from_hdf5(hdf5_path: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    """HDF5에서 궤적 로드.

    Args:
        hdf5_path: HDF5 파일 경로

    Returns:
        [(observations, actions), ...] 리스트
    """
    trajectories = []

    with HDF5Dataset(hdf5_path, mode="r") as dataset:
        traj_ids = dataset.list_trajectories()

        for traj_id in tqdm(traj_ids, desc="Loading trajectories"):
            obs, act = dataset.get_trajectory(traj_id)
            trajectories.append((obs, act))

    print(f"Loaded {len(trajectories)} trajectories from {hdf5_path}")
    return trajectories
