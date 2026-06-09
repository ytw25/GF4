from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


MAX_EVALUATIONS = 60


@dataclass
class BundleAdjustmentResult:
    camera_poses: dict[int, tuple[np.ndarray, np.ndarray]]
    points3d: np.ndarray
    observation_count: int
    initial_mean_error: float
    initial_median_error: float
    final_mean_error: float
    final_median_error: float

    def as_dict(self) -> dict:
        return {
            "observations": self.observation_count,
            "initial_mean_error_px": self.initial_mean_error,
            "initial_median_error_px": self.initial_median_error,
            "final_mean_error_px": self.final_mean_error,
            "final_median_error_px": self.final_median_error,
        }


def bundle_adjustment(
    camera_poses: Mapping[int, tuple[np.ndarray, np.ndarray]],
    points3d: np.ndarray,
    intrinsics: Mapping[int, np.ndarray] | Sequence[np.ndarray],
    features: Sequence[object],
    observations_by_image: Mapping[int, Mapping[int, int]],
    fixed_camera_ids: set[int],
) -> BundleAdjustmentResult:
    """Refine camera poses and 3D points from the tracked 2D observations."""

    points3d = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    camera_indices, point_indices, points_2d = _collect_observations(
        features,
        observations_by_image,
    )
    if len(points_2d) == 0:
        raise ValueError("Bundle adjustment needs at least one observation")

    camera_ids = sorted(set(camera_indices.tolist()))
    point_ids = sorted(set(point_indices.tolist()))
    x0, camera_slices, point_slices = _pack_variables(
        camera_poses,
        points3d,
        camera_ids,
        point_ids,
        fixed_camera_ids,
    )
    if len(x0) == 0:
        raise ValueError("No variables left to optimise")

    def residuals(params: np.ndarray) -> np.ndarray:
        poses, points = _unpack_variables(params, camera_poses, points3d, camera_slices, point_slices)
        errors = []
        for camera_id in camera_ids:
            rows = np.flatnonzero(camera_indices == camera_id)
            projected = _project(
                points[point_indices[rows]],
                np.asarray(intrinsics[camera_id], dtype=np.float64),
                *poses[camera_id],
            )
            errors.append((projected - points_2d[rows]).ravel())
        return np.concatenate(errors)

    initial_errors = _point_errors(residuals(x0))
    result = least_squares(
        residuals,
        x0,
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=MAX_EVALUATIONS,
    )
    final_camera_poses, final_points = _unpack_variables(
        result.x,
        camera_poses,
        points3d,
        camera_slices,
        point_slices,
    )
    final_errors = _point_errors(residuals(result.x))

    return BundleAdjustmentResult(
        camera_poses=final_camera_poses,
        points3d=final_points,
        observation_count=len(points_2d),
        initial_mean_error=float(np.mean(initial_errors)),
        initial_median_error=float(np.median(initial_errors)),
        final_mean_error=float(np.mean(final_errors)),
        final_median_error=float(np.median(final_errors)),
    )


def _collect_observations(
    features: Sequence[object],
    observations_by_image: Mapping[int, Mapping[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_indices = []
    point_indices = []
    points_2d = []

    for camera_id, keypoint_to_point in observations_by_image.items():
        keypoints = features[camera_id].keypoints
        for keypoint_id, point_id in keypoint_to_point.items():
            camera_indices.append(int(camera_id))
            point_indices.append(int(point_id))
            points_2d.append(keypoints[int(keypoint_id)].pt)

    return (
        np.asarray(camera_indices, dtype=np.int64),
        np.asarray(point_indices, dtype=np.int64),
        np.asarray(points_2d, dtype=np.float64).reshape(-1, 2),
    )


def _pack_variables(
    camera_poses: Mapping[int, tuple[np.ndarray, np.ndarray]],
    points3d: np.ndarray,
    camera_ids: list[int],
    point_ids: list[int],
    fixed_camera_ids: set[int],
) -> tuple[np.ndarray, dict[int, slice], dict[int, slice]]:
    blocks = []
    camera_slices = {}
    point_slices = {}
    offset = 0

    for camera_id in camera_ids:
        if camera_id in fixed_camera_ids:
            continue
        R, t = camera_poses[camera_id]
        blocks.append(np.r_[Rotation.from_matrix(R).as_rotvec(), np.asarray(t).reshape(3)])
        camera_slices[camera_id] = slice(offset, offset + 6)
        offset += 6

    for point_id in point_ids:
        blocks.append(points3d[point_id])
        point_slices[point_id] = slice(offset, offset + 3)
        offset += 3

    return np.concatenate(blocks), camera_slices, point_slices


def _unpack_variables(
    params: np.ndarray,
    camera_poses: Mapping[int, tuple[np.ndarray, np.ndarray]],
    points3d: np.ndarray,
    camera_slices: Mapping[int, slice],
    point_slices: Mapping[int, slice],
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], np.ndarray]:
    poses = {
        camera_id: (np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64).reshape(3, 1))
        for camera_id, (R, t) in camera_poses.items()
    }
    points = points3d.copy()

    for camera_id, block_slice in camera_slices.items():
        block = params[block_slice]
        poses[camera_id] = (
            Rotation.from_rotvec(block[:3]).as_matrix(),
            block[3:6].reshape(3, 1),
        )

    for point_id, block_slice in point_slices.items():
        points[point_id] = params[block_slice]

    return poses, points


def _project(points3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    camera_points = (R @ points3d.T).T + np.asarray(t).reshape(1, 3)
    image_points = (K @ camera_points.T).T
    return image_points[:, :2] / image_points[:, 2:3]


def _point_errors(flat_residuals: np.ndarray) -> np.ndarray:
    return np.linalg.norm(flat_residuals.reshape(-1, 2), axis=1)
