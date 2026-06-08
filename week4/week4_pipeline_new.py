"""Week 3 extension: incremental reconstruction over an image set.

This keeps the Week 3 ingredients deliberately visible: choose an initial pair,
build the first two-view reconstruction, then repeatedly pick the remaining
image with the strongest Week 2 pairwise evidence, register it with PnP, and
triangulate extra points from the newly registered view.
"""

# python .\week4\week4_pipeline_new.py `
# >> --image-dir .\week4\hq_set\images `
# >> --output-dir ./week4/output

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np

WEEK3_DIR = Path(__file__).resolve().parents[1] / "week3"
if str(WEEK3_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK3_DIR))

from two_view_utils import (
    ThirdViewResult,
    TwoViewResult,
    camera_center,
    camera_frustum,
    compute_depths,
    compute_reprojection_errors,
    draw_reprojection_overlay,
    draw_single_image_reprojection_overlay,
    ensure_dir,
    estimate_camera_pose_pnp,
    estimate_essential_matrix,
    filter_reconstructed_points,
    make_camera_matrix,
    plot_multi_view_reconstruction,
    plot_patch_cloud_reconstruction,
    plot_two_view_reconstruction,
    recover_relative_pose,
    sample_point_colours,
    save_csv,
    triangulate_points,
    write_ply,
)


DEFAULT_WEEK2_DIR = Path(__file__).resolve().parents[1] / "week2"


def load_week2_module(week2_dir: Path):
    """Load the completed Week 2 sfm_utils.py by path."""
    module_path = Path(week2_dir) / "sfm_utils.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find Week 2 sfm_utils.py: {module_path}")

    spec = importlib.util.spec_from_file_location("week2_sfm_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Week 2 module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GF4 Week 3 sparse reconstruction with incremental multi-view registration."
    )
    parser.add_argument("--image-dir", required=True, type=Path, help="Directory containing the image set.")
    parser.add_argument("--image1", type=Path, default=None, help="Optional initial pair first image.")
    parser.add_argument("--image2", type=Path, default=None, help="Optional initial pair second image.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for outputs.")
    parser.add_argument(
        "--week2-dir",
        type=Path,
        default=DEFAULT_WEEK2_DIR,
        help="Directory containing the completed Week 2 sfm_utils.py.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional maximum number of sorted images to load from the image set.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1600,
        help="Resize images so their long edge is at most this size. Use 0 to disable.",
    )
    parser.add_argument("--max-features", type=int, default=4000)
    parser.add_argument("--ratio", type=float, default=0.7)
    parser.add_argument("--focal-length-px", type=float, default=None)
    parser.add_argument("--principal-point", nargs=2, type=float, metavar=("CX", "CY"), default=None)
    parser.add_argument("--ransac-threshold", type=float, default=1.0)
    parser.add_argument("--confidence", type=float, default=0.999)
    parser.add_argument("--max-reprojection-error", type=float, default=4.0)
    parser.add_argument(
        "--min-triangulation-angle-deg",
        type=float,
        default=1.0,
        help="Minimum ray angle for triangulated points, in degrees. Use 0 to disable.",
    )
    parser.add_argument("--pnp-ransac-threshold", type=float, default=6.0)
    parser.add_argument("--min-pairwise-inliers", type=int, default=20)
    parser.add_argument("--min-pnp-correspondences", type=int, default=30)
    parser.add_argument("--min-initial-inliers", type=int, default=50)
    parser.add_argument("--min-initial-inlier-ratio", type=float, default=0.4)
    parser.add_argument("--min-initial-triangulation-angle-deg", type=float, default=1.5)
    parser.add_argument("--max-initial-median-epipolar-error", type=float, default=1.0)
    parser.add_argument("--min-registration-pnp-inliers", type=int, default=30)
    parser.add_argument("--min-registration-pnp-inlier-ratio", type=float, default=0.25)
    parser.add_argument("--min-registration-coverage", type=float, default=0.2)
    parser.add_argument("--max-registration-median-error", type=float, default=4.0)
    parser.add_argument(
        "--max-new-points-per-pair",
        type=int,
        default=2000,
        help="Cap new triangulation candidates per anchor/new-image pair. Use 0 for no cap.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Write metrics and PLY only, skipping PNG visualisations.",
    )

    args = parser.parse_args()

    if args.max_images is not None and args.max_images < 2:
        parser.error("--max-images must be at least 2")
    if args.max_image_size == 0:
        args.max_image_size = None
    if args.max_features < 1:
        parser.error("--max-features must be positive")
    if not 0.0 < args.ratio < 1.0:
        parser.error("--ratio must be between 0 and 1")
    if args.focal_length_px is not None and args.focal_length_px <= 0:
        parser.error("--focal-length-px must be positive")
    if args.ransac_threshold <= 0:
        parser.error("--ransac-threshold must be positive")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be between 0 and 1")
    if args.max_reprojection_error <= 0:
        parser.error("--max-reprojection-error must be positive")
    if args.min_triangulation_angle_deg < 0:
        parser.error("--min-triangulation-angle-deg cannot be negative")
    if args.pnp_ransac_threshold <= 0:
        parser.error("--pnp-ransac-threshold must be positive")
    if args.min_pairwise_inliers < 0:
        parser.error("--min-pairwise-inliers cannot be negative")
    if args.min_pnp_correspondences < 4:
        parser.error("--min-pnp-correspondences must be at least 4")
    if args.min_initial_inliers < 8:
        parser.error("--min-initial-inliers must be at least 8")
    if not 0.0 <= args.min_initial_inlier_ratio <= 1.0:
        parser.error("--min-initial-inlier-ratio must be between 0 and 1")
    if args.min_initial_triangulation_angle_deg < 0:
        parser.error("--min-initial-triangulation-angle-deg cannot be negative")
    if args.max_initial_median_epipolar_error <= 0:
        parser.error("--max-initial-median-epipolar-error must be positive")
    if args.min_registration_pnp_inliers < 4:
        parser.error("--min-registration-pnp-inliers must be at least 4")
    if not 0.0 <= args.min_registration_pnp_inlier_ratio <= 1.0:
        parser.error("--min-registration-pnp-inlier-ratio must be between 0 and 1")
    if not 0.0 <= args.min_registration_coverage <= 1.0:
        parser.error("--min-registration-coverage must be between 0 and 1")
    if args.max_registration_median_error <= 0:
        parser.error("--max-registration-median-error must be positive")
    if args.max_new_points_per_pair < 0:
        parser.error("--max-new-points-per-pair cannot be negative")
    if (args.image1 is None) != (args.image2 is None):
        parser.error("--image1 and --image2 must be provided together, or both omitted")
    if args.image1 is not None and args.image1 == args.image2:
        parser.error("--image1 and --image2 must be different")

    return args


def _finite_values(values: np.ndarray | list[float] | None) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _median(values: np.ndarray) -> float | None:
    finite = _finite_values(values)
    return float(np.median(finite)) if len(finite) else None


def _mean(values: np.ndarray) -> float | None:
    finite = _finite_values(values)
    return float(np.mean(finite)) if len(finite) else None


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = _finite_values(values)
    return float(np.percentile(finite, percentile)) if len(finite) else None


def _metric(value, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _triangulation_angles(
    points3d: np.ndarray,
    R_a: np.ndarray,
    t_a: np.ndarray,
    R_b: np.ndarray,
    t_b: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return np.empty(0, dtype=np.float64)

    center_a = camera_center(R_a, t_a)
    center_b = camera_center(R_b, t_b)
    rays_a = points - center_a.reshape(1, 3)
    rays_b = points - center_b.reshape(1, 3)
    norms_a = np.linalg.norm(rays_a, axis=1)
    norms_b = np.linalg.norm(rays_b, axis=1)
    valid = (norms_a > 1e-12) & (norms_b > 1e-12)

    angles = np.full(len(points), np.nan, dtype=np.float64)
    cos_angles = np.sum(rays_a[valid] * rays_b[valid], axis=1) / (norms_a[valid] * norms_b[valid])
    angles[valid] = np.degrees(np.arccos(np.clip(cos_angles, -1.0, 1.0)))
    return angles


def _triangulation_angle_mask(
    points3d: np.ndarray,
    R_a: np.ndarray,
    t_a: np.ndarray,
    R_b: np.ndarray,
    t_b: np.ndarray,
    min_angle_deg: float,
) -> np.ndarray:
    points = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    if min_angle_deg <= 0 or len(points) == 0:
        return np.ones(len(points), dtype=bool)

    angles = _triangulation_angles(points, R_a, t_a, R_b, t_b)
    return np.isfinite(angles) & (angles >= min_angle_deg)


def _resolve_initial_indices(args: argparse.Namespace, image_paths: list[Path]) -> list[int] | None:
    if args.image1 is None and args.image2 is None:
        return None

    initial_indices = []
    for requested in (args.image1, args.image2):
        requested_path = Path(requested)
        matches_for_request = [
            idx
            for idx, path in enumerate(image_paths)
            if str(path) == str(requested_path)
            or path.name == requested_path.name
            or path.stem == requested_path.stem
        ]
        if len(matches_for_request) != 1:
            raise ValueError(f"Initial image not found or ambiguous in image set: {requested}")
        initial_indices.append(matches_for_request[0])

    if initial_indices[0] == initial_indices[1]:
        raise ValueError("Initial images must be different")
    return initial_indices


def _get_matches(
    week2,
    features: list,
    image_i: int,
    image_j: int,
    ratio: float,
    match_cache: dict[tuple[int, int], list[cv2.DMatch]],
) -> list[cv2.DMatch]:
    key = (image_i, image_j)
    if key not in match_cache:
        match_cache[key] = week2.match_descriptors(
            features[image_i].descriptors,
            features[image_j].descriptors,
            ratio=ratio,
        )
    return match_cache[key]


def _get_pairwise_row(
    week2,
    features: list,
    image_i: int,
    image_j: int,
    output_dir: Path,
    ratio: float,
    pairwise_cache: dict[tuple[int, int], dict],
) -> dict:
    key = (image_i, image_j)
    if key in pairwise_cache:
        return pairwise_cache[key]

    try:
        analysis = week2.analyse_feature_pair(
            features[image_i],
            features[image_j],
            output_dir / "pairwise",
            ratio=ratio,
            save_figures=False,
        )
        row = analysis.csv_dict()
    except (RuntimeError, ValueError, cv2.error):
        row = {
            "image_i": features[image_i].path.name,
            "image_j": features[image_j].path.name,
            "keypoints_i": len(features[image_i].keypoints),
            "keypoints_j": len(features[image_j].keypoints),
            "raw_matches": 0,
            "filtered_matches": 0,
            "ransac_inliers": 0,
            "inlier_ratio": 0.0,
            "mean_epipolar_error_all": None,
            "median_epipolar_error_all": None,
            "mean_epipolar_error_inliers": None,
            "median_epipolar_error_inliers": None,
            "max_epipolar_error_inliers": None,
            "fundamental_matrix": "",
        }
    pairwise_cache[key] = row
    return row


def _initial_pair_metrics_template(features: list, image_i: int, image_j: int) -> dict:
    return {
        "image_i": features[image_i].path.name,
        "image_j": features[image_j].path.name,
        "filtered_matches": 0,
        "essential_inliers": 0,
        "pose_inliers": 0,
        "inlier_ratio": 0.0,
        "triangulated_points": 0,
        "kept_points": 0,
        "median_triangulation_angle_deg": None,
        "median_epipolar_error_px": None,
        "median_reprojection_error_px": None,
        "connectivity": 0,
        "connectivity_threshold": 0,
        "hard_gate_passed": False,
        "score": 0.0,
        "selected": False,
        "reason": "",
    }


def _initial_pair_hard_gate_reason(metrics: dict, args: argparse.Namespace) -> str | None:
    if metrics["pose_inliers"] < args.min_initial_inliers:
        return "below initial inlier threshold"
    if metrics["inlier_ratio"] < args.min_initial_inlier_ratio:
        return "below initial inlier-ratio threshold"
    if (
        metrics["median_epipolar_error_px"] is None
        or metrics["median_epipolar_error_px"] > args.max_initial_median_epipolar_error
    ):
        return "above initial epipolar-error threshold"
    if (
        metrics["median_triangulation_angle_deg"] is None
        or metrics["median_triangulation_angle_deg"] < args.min_initial_triangulation_angle_deg
    ):
        return "below initial triangulation-angle threshold"
    return None


def _score_initial_pair(
    week2,
    features: list,
    intrinsics: list[np.ndarray],
    image_i: int,
    image_j: int,
    args: argparse.Namespace,
    output_dir: Path,
    match_cache: dict[tuple[int, int], list[cv2.DMatch]],
    pairwise_cache: dict[tuple[int, int], dict],
) -> dict:
    metrics = _initial_pair_metrics_template(features, image_i, image_j)
    pairwise_row = _get_pairwise_row(
        week2,
        features,
        image_i,
        image_j,
        output_dir,
        args.ratio,
        pairwise_cache,
    )
    matches = _get_matches(week2, features, image_i, image_j, args.ratio, match_cache)
    metrics["filtered_matches"] = len(matches)
    metrics["median_epipolar_error_px"] = _metric(
        pairwise_row.get("median_epipolar_error_inliers"),
        default=None,
    )

    if len(matches) < 8:
        metrics["reason"] = "too few Lowe-filtered matches"
        return {"usable": False, "metrics": metrics}

    try:
        pts_i, pts_j = week2.matched_keypoint_coords(
            features[image_i].keypoints,
            features[image_j].keypoints,
            matches,
        )
        E, essential_mask = estimate_essential_matrix(
            pts_i,
            pts_j,
            intrinsics[image_i],
            threshold=args.ransac_threshold,
            confidence=args.confidence,
        )
        R_j, t_j, pose_mask = recover_relative_pose(
            E,
            pts_i,
            pts_j,
            intrinsics[image_i],
            inlier_mask=essential_mask,
        )
        pts_i_pose = pts_i[pose_mask]
        pts_j_pose = pts_j[pose_mask]
        pose_matches = [match for match, keep in zip(matches, pose_mask) if keep]
        initial_points = triangulate_points(pts_i_pose, pts_j_pose, intrinsics[image_i], R_j, t_j)
        errors_i = compute_reprojection_errors(
            initial_points,
            pts_i_pose,
            intrinsics[image_i],
            np.eye(3),
            np.zeros((3, 1)),
        )
        errors_j = compute_reprojection_errors(initial_points, pts_j_pose, intrinsics[image_j], R_j, t_j)
        keep = filter_reconstructed_points(
            initial_points,
            errors_i,
            errors_j,
            R_j,
            t_j,
            max_reprojection_error=args.max_reprojection_error,
        )
        keep &= _triangulation_angle_mask(
            initial_points,
            np.eye(3),
            np.zeros((3, 1)),
            R_j,
            t_j,
            args.min_triangulation_angle_deg,
        )
    except (RuntimeError, ValueError, cv2.error) as exc:
        metrics["reason"] = f"two-view geometry failed: {exc}"
        return {"usable": False, "metrics": metrics}

    pose_inliers = int(np.sum(pose_mask))
    kept_errors = 0.5 * (errors_i[keep] + errors_j[keep])
    triangulation_angles = _triangulation_angles(
        initial_points[keep],
        np.eye(3),
        np.zeros((3, 1)),
        R_j,
        t_j,
    )
    median_angle = _median(triangulation_angles)

    metrics.update(
        {
            "essential_inliers": int(np.sum(essential_mask)),
            "pose_inliers": pose_inliers,
            "inlier_ratio": pose_inliers / len(matches) if matches else 0.0,
            "triangulated_points": len(initial_points),
            "kept_points": int(np.sum(keep)),
            "median_triangulation_angle_deg": median_angle,
            "median_reprojection_error_px": _median(kept_errors),
        }
    )

    if int(np.sum(keep)) == 0 or median_angle is None:
        metrics["reason"] = "no stable triangulated points"
        return {"usable": False, "metrics": metrics}

    hard_gate_reason = _initial_pair_hard_gate_reason(metrics, args)
    if hard_gate_reason is not None:
        metrics["reason"] = hard_gate_reason
        return {"usable": False, "metrics": metrics}

    metrics["hard_gate_passed"] = True
    return {
        "usable": True,
        "metrics": metrics,
        "idx1": image_i,
        "idx2": image_j,
        "matches": matches,
        "pts1": pts_i,
        "pts2": pts_j,
        "E": E,
        "essential_mask": essential_mask,
        "R2": R_j,
        "t2": t_j,
        "pose_mask": pose_mask,
        "pts1_pose": pts_i_pose,
        "pts2_pose": pts_j_pose,
        "pose_matches": pose_matches,
        "initial_points": initial_points,
        "errors1": errors_i,
        "errors2": errors_j,
        "keep": keep,
    }


def _select_initial_pair(
    week2,
    features: list,
    intrinsics: list[np.ndarray],
    args: argparse.Namespace,
    output_dir: Path,
    match_cache: dict[tuple[int, int], list[cv2.DMatch]],
    pairwise_cache: dict[tuple[int, int], dict],
    requested_indices: list[int] | None,
) -> tuple[dict, list[dict]]:
    candidates = []
    candidate_pairs = (
        [tuple(requested_indices)]
        if requested_indices is not None
        else [
            (image_i, image_j)
            for image_i in range(len(features))
            for image_j in range(image_i + 1, len(features))
        ]
    )

    for image_i, image_j in candidate_pairs:
        candidates.append(
            _score_initial_pair(
                week2,
                features,
                intrinsics,
                image_i,
                image_j,
                args,
                output_dir,
                match_cache,
                pairwise_cache,
            )
        )

    for image_i in range(len(features)):
        for image_j in range(image_i + 1, len(features)):
            _get_pairwise_row(
                week2,
                features,
                image_i,
                image_j,
                output_dir,
                args.ratio,
                pairwise_cache,
            )

    adjacency = {feature.path.name: set() for feature in features}
    for row in pairwise_cache.values():
        if int(_metric(row.get("ransac_inliers"), default=0)) < args.min_pairwise_inliers:
            continue
        image_i = row["image_i"]
        image_j = row["image_j"]
        adjacency.setdefault(image_i, set()).add(image_j)
        adjacency.setdefault(image_j, set()).add(image_i)

    for candidate in candidates:
        metrics = candidate["metrics"]
        image_i = metrics["image_i"]
        image_j = metrics["image_j"]
        connected = (adjacency.get(image_i, set()) | adjacency.get(image_j, set())) - {image_i, image_j}
        metrics["connectivity"] = len(connected)
        metrics["connectivity_threshold"] = args.min_pairwise_inliers
        metrics["score"] = metrics["connectivity"] if candidate["usable"] else 0.0

    if requested_indices is not None:
        pair = candidates[0]
        if not pair["usable"]:
            raise ValueError(f"Requested initial pair is not usable: {pair['metrics']['reason']}")
        pair["metrics"]["selected"] = True
        return pair, [pair["metrics"]]

    usable_pairs = [pair for pair in candidates if pair["usable"]]
    top_inlier_pairs = sorted(
        usable_pairs,
        key=lambda pair: pair["metrics"]["pose_inliers"],
        reverse=True,
    )[:3]
    best_pair = max(
        top_inlier_pairs,
        key=lambda pair: (pair["metrics"]["connectivity"], pair["metrics"]["pose_inliers"]),
        default=None,
    )

    if best_pair is None:
        raise ValueError("No usable initial image pair found")

    best_pair["metrics"]["selected"] = True
    return best_pair, [candidate["metrics"] for candidate in candidates]


def _coverage_score(points2d: np.ndarray, image_shape: tuple[int, ...], grid_size: int = 4) -> float:
    points = np.asarray(points2d, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return 0.0

    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        return 0.0

    cols = np.clip((points[:, 0] / width * grid_size).astype(int), 0, grid_size - 1)
    rows = np.clip((points[:, 1] / height * grid_size).astype(int), 0, grid_size - 1)
    occupied = len({(int(row), int(col)) for row, col in zip(rows, cols)})
    return occupied / float(grid_size * grid_size)


def _collect_pnp_correspondences(
    week2,
    features: list,
    candidate_idx: int,
    registered_indices: list[int],
    observations_by_image: dict[int, dict[int, int]],
    points3d: list[np.ndarray],
    match_cache: dict[tuple[int, int], list[cv2.DMatch]],
    ratio: float,
) -> dict:
    candidate_features = features[candidate_idx]
    pnp_candidates = []
    anchor_match_count = 0

    for anchor_idx in registered_indices:
        anchor_matches = _get_matches(week2, features, anchor_idx, candidate_idx, ratio, match_cache)
        anchor_match_count += len(anchor_matches)
        anchor_observations = observations_by_image.get(anchor_idx, {})
        for match in anchor_matches:
            point_id = anchor_observations.get(int(match.queryIdx))
            if point_id is None:
                continue
            new_keypoint_idx = int(match.trainIdx)
            pnp_candidates.append(
                (
                    float(match.distance),
                    int(point_id),
                    new_keypoint_idx,
                    candidate_features.keypoints[new_keypoint_idx].pt,
                )
            )

    pnp_candidates.sort(key=lambda item: item[0])
    used_point_ids = set()
    used_new_keypoints = set()
    pnp_point_ids = []
    new_keypoint_indices = []
    pnp_points = []
    pnp_pts = []
    for _, point_id, new_keypoint_idx, pt in pnp_candidates:
        if point_id in used_point_ids or new_keypoint_idx in used_new_keypoints:
            continue
        if point_id >= len(points3d):
            continue
        used_point_ids.add(point_id)
        used_new_keypoints.add(new_keypoint_idx)
        pnp_point_ids.append(point_id)
        new_keypoint_indices.append(new_keypoint_idx)
        pnp_points.append(points3d[point_id])
        pnp_pts.append(pt)

    return {
        "anchor_match_count": anchor_match_count,
        "pnp_point_ids": pnp_point_ids,
        "new_keypoint_indices": new_keypoint_indices,
        "pnp_points3d": np.asarray(pnp_points, dtype=np.float64).reshape(-1, 3),
        "pts_new": np.asarray(pnp_pts, dtype=np.float64).reshape(-1, 2),
    }


def _registration_metrics_template(features: list, candidate_idx: int, iteration: int) -> dict:
    return {
        "iteration": iteration,
        "image": features[candidate_idx].path.name,
        "pnp_correspondences": 0,
        "pnp_inliers": 0,
        "pnp_inlier_ratio": 0.0,
        "coverage_score": 0.0,
        "mean_track_length": 0.0,
        "median_reprojection_error_px": None,
        "p90_reprojection_error_px": None,
        "hard_gate_passed": False,
        "score": 0.0,
        "selected": False,
        "reason": "",
    }


def _registration_hard_gate_reason(row: dict, args: argparse.Namespace) -> str | None:
    if row["pnp_correspondences"] < args.min_pnp_correspondences:
        return "too few 2D-3D correspondences"
    if row["pnp_inliers"] < args.min_registration_pnp_inliers:
        return "below PnP inlier threshold"
    if row["pnp_inlier_ratio"] < args.min_registration_pnp_inlier_ratio:
        return "below PnP inlier-ratio threshold"
    if (
        row["median_reprojection_error_px"] is None
        or row["median_reprojection_error_px"] > args.max_registration_median_error
    ):
        return "above median reprojection-error threshold"
    if row["coverage_score"] < args.min_registration_coverage:
        return "below coverage threshold"
    return None


def _score_registration_candidate(
    week2,
    features: list,
    intrinsics: list[np.ndarray],
    candidate_idx: int,
    registered_indices: list[int],
    observations_by_image: dict[int, dict[int, int]],
    point_tracks: list[set[tuple[int, int]]],
    points3d: list[np.ndarray],
    match_cache: dict[tuple[int, int], list[cv2.DMatch]],
    args: argparse.Namespace,
    iteration: int,
) -> dict:
    collected = _collect_pnp_correspondences(
        week2,
        features,
        candidate_idx,
        registered_indices,
        observations_by_image,
        points3d,
        match_cache,
        args.ratio,
    )
    row = _registration_metrics_template(features, candidate_idx, iteration)
    pnp_points3d = collected["pnp_points3d"]
    pts_new = collected["pts_new"]
    row["pnp_correspondences"] = len(pnp_points3d)

    if len(pnp_points3d) < args.min_pnp_correspondences:
        row["reason"] = "too few 2D-3D correspondences"
        return {"usable": False, "row": row, **collected}

    try:
        R_new, t_new, pnp_mask = estimate_camera_pose_pnp(
            pnp_points3d,
            pts_new,
            intrinsics[candidate_idx],
            threshold=args.pnp_ransac_threshold,
            confidence=args.confidence,
        )
    except (RuntimeError, ValueError, cv2.error) as exc:
        row["reason"] = f"PnP failed: {exc}"
        return {"usable": False, "row": row, **collected}

    pnp_mask = np.asarray(pnp_mask, dtype=bool).reshape(-1)
    pnp_inliers = int(np.sum(pnp_mask))
    row["pnp_inliers"] = pnp_inliers
    row["pnp_inlier_ratio"] = pnp_inliers / len(pnp_points3d) if len(pnp_points3d) else 0.0

    pnp_errors = compute_reprojection_errors(
        pnp_points3d[pnp_mask],
        pts_new[pnp_mask],
        intrinsics[candidate_idx],
        R_new,
        t_new,
    )
    median_reprojection_error = _median(pnp_errors)
    p90_reprojection_error = _percentile(pnp_errors, 90.0)
    inlier_points = pts_new[pnp_mask]
    coverage = _coverage_score(inlier_points, features[candidate_idx].image.shape)
    track_lengths = [
        len(point_tracks[point_id])
        for point_id, keep in zip(collected["pnp_point_ids"], pnp_mask)
        if keep and point_id < len(point_tracks)
    ]
    mean_track_length = float(np.mean(track_lengths)) if track_lengths else 0.0

    row.update(
        {
            "coverage_score": coverage,
            "mean_track_length": mean_track_length,
            "median_reprojection_error_px": median_reprojection_error,
            "p90_reprojection_error_px": p90_reprojection_error,
        }
    )

    hard_gate_reason = _registration_hard_gate_reason(row, args)
    if hard_gate_reason is not None:
        row["reason"] = hard_gate_reason
        return {
            "usable": False,
            "row": row,
            "R_new": R_new,
            "t_new": t_new,
            "pnp_mask": pnp_mask,
            "pnp_errors": pnp_errors,
            **collected,
        }

    row["hard_gate_passed"] = True
    row["score"] = pnp_inliers
    return {
        "usable": True,
        "row": row,
        "R_new": R_new,
        "t_new": t_new,
        "pnp_mask": pnp_mask,
        "pnp_errors": pnp_errors,
        **collected,
    }


def write_camera_frustums_ply(
    output_path: Path,
    camera_poses: list[tuple[str, np.ndarray, np.ndarray]],
    scale: float | None = None,
) -> None:
    ensure_dir(output_path.parent)

    if not camera_poses:
        output_path.write_text(
            "ply\nformat ascii 1.0\nelement vertex 0\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "element face 0\nproperty list uchar int vertex_indices\nend_header\n",
            encoding="utf-8",
        )
        return

    centers = np.asarray([camera_center(R, t) for _, R, t in camera_poses], dtype=np.float64)
    if scale is None:
        if len(centers) >= 2:
            baselines = [
                np.linalg.norm(centers[i] - centers[j])
                for i in range(len(centers))
                for j in range(i + 1, len(centers))
            ]
            scale = max(0.12, 0.12 * float(np.median(baselines)))
        else:
            scale = 0.12

    palette = np.asarray(
        [
            [88, 166, 255],
            [255, 123, 114],
            [126, 231, 135],
            [210, 168, 255],
            [255, 166, 87],
            [121, 192, 255],
        ],
        dtype=np.uint8,
    )

    vertices: list[tuple[float, float, float, int, int, int]] = []
    faces: list[tuple[int, int, int]] = []
    for camera_idx, (_, R, t) in enumerate(camera_poses):
        colour = palette[camera_idx % len(palette)]
        center = camera_center(R, t)
        frustum = camera_frustum(R, t, scale)
        base = len(vertices)
        for point in [center, *frustum]:
            vertices.append(
                (
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    int(colour[0]),
                    int(colour[1]),
                    int(colour[2]),
                )
            )
        faces.extend(
            [
                (base, base + 1, base + 2),
                (base, base + 2, base + 3),
                (base, base + 3, base + 4),
                (base, base + 4, base + 1),
                (base + 1, base + 2, base + 3),
                (base + 1, base + 3, base + 4),
            ]
        )

    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines.extend(f"{x:.8f} {y:.8f} {z:.8f} {r} {g} {b}" for x, y, z, r, g, b in vertices)
    lines.extend(f"3 {i} {j} {k}" for i, j, k in faces)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> tuple[TwoViewResult, list[ThirdViewResult], list[str]]:
    output_dir = ensure_dir(args.output_dir)
    week2 = load_week2_module(args.week2_dir)

    image_paths = week2.list_image_paths(args.image_dir, max_images=args.max_images)
    requested_initial_indices = _resolve_initial_indices(args, image_paths)

    features = week2.precompute_image_features(
        image_paths,
        max_features=args.max_features,
        max_image_size=args.max_image_size,
    )
    intrinsics = [
        make_camera_matrix(
            feature.image.shape,
            focal_length_px=args.focal_length_px,
            principal_point=None if args.principal_point is None else tuple(args.principal_point),
        )
        for feature in features
    ]

    match_cache: dict[tuple[int, int], list[cv2.DMatch]] = {}
    pairwise_cache: dict[tuple[int, int], dict] = {}
    initial_pair, initial_pair_rows = _select_initial_pair(
        week2,
        features,
        intrinsics,
        args,
        output_dir,
        match_cache,
        pairwise_cache,
        requested_initial_indices,
    )
    save_csv(output_dir / "initial_pair_candidates.csv", initial_pair_rows)

    idx1, idx2 = initial_pair["idx1"], initial_pair["idx2"]
    features1 = features[idx1]
    features2 = features[idx2]
    K = intrinsics[idx1]

    matches12 = initial_pair["matches"]
    pts1 = initial_pair["pts1"]
    pts2 = initial_pair["pts2"]
    E = initial_pair["E"]
    essential_mask = initial_pair["essential_mask"]
    R2 = initial_pair["R2"]
    t2 = initial_pair["t2"]
    pose_mask = initial_pair["pose_mask"]
    pts1_pose = initial_pair["pts1_pose"]
    pts2_pose = initial_pair["pts2_pose"]
    pose_matches = initial_pair["pose_matches"]

    if hasattr(week2, "draw_matches") and not args.skip_figures:
        week2.draw_matches(
            features1.image,
            features1.keypoints,
            features2.image,
            features2.keypoints,
            pose_matches,
            output_dir / "pose_inlier_matches.png",
        )

    initial_points = initial_pair["initial_points"]
    errors1 = initial_pair["errors1"]
    errors2 = initial_pair["errors2"]
    keep = initial_pair["keep"]
    if not np.any(keep):
        raise ValueError("No initial triangulated points survived filtering")

    depths1, depths2 = compute_depths(initial_points, R2, t2)
    positive_depth = (depths1 > 0) & (depths2 > 0)
    kept_points = initial_points[keep]
    kept_colours = sample_point_colours(features1.image, pts1_pose[keep])
    kept_errors = 0.5 * (errors1[keep] + errors2[keep])
    kept_pose_matches = [match for match, keep_point in zip(pose_matches, keep) if keep_point]

    points3d = [point.copy() for point in kept_points]
    point_colours = [colour.copy() for colour in kept_colours]
    observations_by_image: dict[int, dict[int, int]] = {idx1: {}, idx2: {}}
    point_tracks: list[set[tuple[int, int]]] = []
    for point_id, match in enumerate(kept_pose_matches):
        keypoint1 = int(match.queryIdx)
        keypoint2 = int(match.trainIdx)
        observations_by_image[idx1][keypoint1] = point_id
        observations_by_image[idx2][keypoint2] = point_id
        point_tracks.append({(idx1, keypoint1), (idx2, keypoint2)})

    if not args.skip_figures:
        draw_reprojection_overlay(
            features1.image,
            features2.image,
            pts1_pose[keep],
            pts2_pose[keep],
            kept_points,
            K,
            R2,
            t2,
            output_dir / "reprojection_overlay.png",
        )
        plot_two_view_reconstruction(
            kept_points,
            kept_colours,
            R2,
            t2,
            output_dir / "two_view_reconstruction.png",
        )
        plot_patch_cloud_reconstruction(
            kept_points,
            features1.image,
            pts1_pose[keep],
            [
                ("Camera 1", np.eye(3), np.zeros((3, 1))),
                ("Camera 2", R2, t2),
            ],
            output_dir / "two_view_patch_cloud.png",
        )

    two_view_result = TwoViewResult(
        image_i=features1.path.name,
        image_j=features2.path.name,
        filtered_matches=len(matches12),
        essential_inliers=int(np.sum(essential_mask)),
        pose_inliers=int(np.sum(pose_mask)),
        triangulated_points=len(initial_points),
        positive_depth_points=int(np.sum(positive_depth)),
        kept_points=len(points3d),
        median_reprojection_error_px=_median(kept_errors),
        mean_reprojection_error_px=_mean(kept_errors),
        focal_length_px=float(K[0, 0]),
    )

    save_csv(output_dir / "two_view_metrics.csv", [two_view_result.as_dict()])
    np.savetxt(output_dir / "K1.txt", intrinsics[idx1])
    np.savetxt(output_dir / "R1.txt", np.eye(3))
    np.savetxt(output_dir / "t1.txt", np.zeros((3, 1)))
    np.savetxt(output_dir / "K2.txt", intrinsics[idx2])
    np.savetxt(output_dir / "R2.txt", R2)
    np.savetxt(output_dir / "t2.txt", t2)
    np.savetxt(output_dir / "E.txt", E)

    registered_indices = [idx1, idx2]
    camera_order = [idx1, idx2]
    camera_poses: dict[int, tuple[np.ndarray, np.ndarray]] = {
        idx1: (np.eye(3), np.zeros((3, 1))),
        idx2: (R2, t2),
    }
    remaining = [idx for idx in range(len(features)) if idx not in registered_indices]
    registered_results: list[ThirdViewResult] = []
    rejected_diagnostics: list[dict] = []
    new_points_rows = []
    candidate_ranking_rows = []

    while remaining:
        scored_candidates = []
        iteration = len(camera_order) + 1
        for candidate_idx in remaining:
            candidate = _score_registration_candidate(
                week2,
                features,
                intrinsics,
                candidate_idx,
                registered_indices,
                observations_by_image,
                point_tracks,
                points3d,
                match_cache,
                args,
                iteration,
            )
            candidate["candidate_idx"] = candidate_idx
            scored_candidates.append(candidate)
            candidate_ranking_rows.append(candidate["row"])

        usable_candidates = [candidate for candidate in scored_candidates if candidate["usable"]]
        if not usable_candidates:
            for candidate in scored_candidates:
                rejected_diagnostics.append(
                    {
                        "image": candidate["row"]["image"],
                        "reason": candidate["row"]["reason"] or "did not pass registration thresholds",
                        "pnp_correspondences": candidate["row"]["pnp_correspondences"],
                        "pnp_inliers": candidate["row"]["pnp_inliers"],
                        "candidate_score": candidate["row"]["score"],
                    }
                )
            break

        best_candidate = max(usable_candidates, key=lambda candidate: candidate["row"]["pnp_inliers"])
        best_candidate["row"]["selected"] = True
        candidate_idx = best_candidate["candidate_idx"]
        candidate_features = features[candidate_idx]
        anchor_match_count = best_candidate["anchor_match_count"]
        pnp_point_ids = best_candidate["pnp_point_ids"]
        new_keypoint_indices = best_candidate["new_keypoint_indices"]
        pnp_points3d = best_candidate["pnp_points3d"]
        pts_new = best_candidate["pts_new"]
        R_new = best_candidate["R_new"]
        t_new = best_candidate["t_new"]
        pnp_mask = best_candidate["pnp_mask"]
        pnp_errors = best_candidate["pnp_errors"]
        pnp_inliers = int(best_candidate["row"]["pnp_inliers"])

        camera_poses[candidate_idx] = (R_new, t_new)
        observations_by_image[candidate_idx] = {}
        for point_id, keypoint_idx, keep_pnp in zip(pnp_point_ids, new_keypoint_indices, pnp_mask):
            if not keep_pnp:
                continue
            keypoint_idx = int(keypoint_idx)
            point_id = int(point_id)
            observations_by_image[candidate_idx][keypoint_idx] = point_id
            if point_id < len(point_tracks):
                point_tracks[point_id].add((candidate_idx, keypoint_idx))

        previous_registered = registered_indices.copy()
        max_new = args.max_new_points_per_pair or None
        new_points_count = 0

        for anchor_idx in previous_registered:
            match_key = (anchor_idx, candidate_idx)
            if match_key not in match_cache:
                match_cache[match_key] = week2.match_descriptors(
                    features[anchor_idx].descriptors,
                    candidate_features.descriptors,
                    ratio=args.ratio,
                )

            anchor_matches = match_cache[match_key]
            anchor_observations = observations_by_image.setdefault(anchor_idx, {})
            new_observations = observations_by_image.setdefault(candidate_idx, {})
            triangulation_matches = []
            used_anchor_keypoints = set()
            used_candidate_keypoints = set()

            for match in anchor_matches:
                anchor_kp = int(match.queryIdx)
                candidate_kp = int(match.trainIdx)
                if anchor_kp in anchor_observations or candidate_kp in new_observations:
                    continue
                if anchor_kp in used_anchor_keypoints or candidate_kp in used_candidate_keypoints:
                    continue
                used_anchor_keypoints.add(anchor_kp)
                used_candidate_keypoints.add(candidate_kp)
                triangulation_matches.append(match)
                if max_new is not None and len(triangulation_matches) >= max_new:
                    break

            if not triangulation_matches:
                continue

            pts_anchor, pts_candidate = week2.matched_keypoint_coords(
                features[anchor_idx].keypoints,
                candidate_features.keypoints,
                triangulation_matches,
            )

            # Triangulate points in anchor camera axis
            R_anchor, t_anchor = camera_poses[anchor_idx]
            R_rel = R_new @ R_anchor.T
            t_rel = t_new - R_rel @ t_anchor
            triangulated_anchor = triangulate_points(
                pts_anchor,
                pts_candidate,
                intrinsics[anchor_idx],
                R_rel,
                t_rel,
            )
            err_anchor = compute_reprojection_errors(
                triangulated_anchor,
                pts_anchor,
                intrinsics[anchor_idx],
                np.eye(3),
                np.zeros((3, 1)),
            )
            err_candidate = compute_reprojection_errors(
                triangulated_anchor,
                pts_candidate,
                intrinsics[candidate_idx],
                R_rel,
                t_rel,
            )
            keep_new = filter_reconstructed_points(
                triangulated_anchor,
                err_anchor,
                err_candidate,
                R_rel,
                t_rel,
                max_reprojection_error=args.max_reprojection_error,
            )
            keep_new &= _triangulation_angle_mask(
                triangulated_anchor,
                np.eye(3),
                np.zeros((3, 1)),
                R_rel,
                t_rel,
                args.min_triangulation_angle_deg,
            )
            if not np.any(keep_new):
                continue
            
            # Transfer triangulated points from anchor camera axis to world axis
            triangulated_world = (R_anchor.T @ (triangulated_anchor[keep_new] - t_anchor.reshape(1, 3)).T).T
            pts_anchor_kept = pts_anchor[keep_new]
            kept_matches = [match for match, keep_point in zip(triangulation_matches, keep_new) if keep_point]
            colours = sample_point_colours(features[anchor_idx].image, pts_anchor_kept)

            for point, colour, match in zip(triangulated_world, colours, kept_matches):
                anchor_kp = int(match.queryIdx)
                candidate_kp = int(match.trainIdx)
                if anchor_kp in anchor_observations or candidate_kp in new_observations:
                    continue
                point_id = len(points3d)
                points3d.append(np.asarray(point, dtype=np.float64))
                point_colours.append(np.asarray(colour, dtype=np.uint8))
                anchor_observations[anchor_kp] = point_id
                new_observations[candidate_kp] = point_id
                point_tracks.append({(anchor_idx, anchor_kp), (candidate_idx, candidate_kp)})
                new_points_count += 1

        registered_indices.append(candidate_idx)
        camera_order.append(candidate_idx)
        remaining.remove(candidate_idx)
        camera_number = len(camera_order)

        if not args.skip_figures:
            draw_single_image_reprojection_overlay(
                candidate_features.image,
                pts_new[pnp_mask],
                pnp_points3d[pnp_mask],
                intrinsics[candidate_idx],
                R_new,
                t_new,
                output_dir / f"view{camera_number}_reprojection_overlay.png",
            )

        np.savetxt(output_dir / f"K{camera_number}.txt", intrinsics[candidate_idx])
        np.savetxt(output_dir / f"R{camera_number}.txt", R_new)
        np.savetxt(output_dir / f"t{camera_number}.txt", t_new)

        registered_results.append(
            ThirdViewResult(
                image_k=candidate_features.path.name,
                anchor_matches=anchor_match_count,
                pnp_correspondences=len(pnp_points3d),
                pnp_inliers=pnp_inliers,
                median_reprojection_error_px=_median(pnp_errors),
                mean_reprojection_error_px=_mean(pnp_errors),
                focal_length_px=float(intrinsics[candidate_idx][0, 0]),
            )
        )
        new_points_rows.append(
            {
                "image_k": candidate_features.path.name,
                "new_points": new_points_count,
                "total_points": len(points3d),
            }
        )

    pairwise_rows = sorted(pairwise_cache.values(), key=lambda row: (row["image_i"], row["image_j"]))
    save_csv(output_dir / "pairwise_metrics.csv", pairwise_rows)
    if hasattr(week2, "draw_match_graph") and not args.skip_figures:
        week2.draw_match_graph(
            pairwise_rows,
            output_dir / "match_graph.png",
            min_inliers=args.min_pairwise_inliers,
        )
    save_csv(output_dir / "candidate_ranking_metrics.csv", candidate_ranking_rows)
    if registered_results:
        rows = [result.as_dict() for result in registered_results]
        save_csv(output_dir / "registered_view_metrics.csv", rows)
        save_csv(output_dir / "new_points_metrics.csv", new_points_rows)
        if len(registered_results) == 1:
            save_csv(output_dir / "third_view_metrics.csv", rows)
    if rejected_diagnostics:
        save_csv(output_dir / "rejected_view_diagnostics.csv", rejected_diagnostics)

    point_array = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    colour_array = np.asarray(point_colours, dtype=np.uint8).reshape(-1, 3)
    track_lengths = np.asarray([len(track) for track in point_tracks], dtype=np.float64)
    camera_poses_for_plot = [
        (f"Camera {idx + 1}", *camera_poses[image_idx])
        for idx, image_idx in enumerate(camera_order)
    ]
    write_ply(output_dir / "points3d.ply", point_array, colour_array)
    write_camera_frustums_ply(output_dir / "cameras.ply", camera_poses_for_plot)
    save_csv(
        output_dir / "reconstruction_summary.csv",
        [
            {
                "input_images": len(features),
                "registered_images": len(camera_order),
                "rejected_images": len(rejected_diagnostics),
                "points3d": len(point_array),
                "initial_image1": features1.path.name,
                "initial_image2": features2.path.name,
                "initial_pair_score": initial_pair["metrics"]["score"],
                "lowe_ratio": args.ratio,
                "min_triangulation_angle_deg": args.min_triangulation_angle_deg,
                "median_track_length": _median(track_lengths),
                "mean_track_length": _mean(track_lengths),
            }
        ],
    )

    if not args.skip_figures:
        if registered_results:
            plot_multi_view_reconstruction(
                point_array,
                colour_array,
                camera_poses_for_plot,
                output_dir / "multi_view_reconstruction.png",
            )
            plot_patch_cloud_reconstruction(
                point_array,
                features1.image,
                pts1_pose[keep],
                camera_poses_for_plot,
                output_dir / "multi_view_patch_cloud.png",
            )

    return two_view_result, registered_results, [row.get("image", "") for row in rejected_diagnostics if row.get("image")]


def main() -> int:
    args = parse_args()

    try:
        result, registered_results, rejected_images = run(args)
    except NotImplementedError as exc:
        print(f"\nStarter-code TODO reached: {exc}", file=sys.stderr)
        print(
            "Complete the relevant Week 2 matching TODO or Week 3 geometry/visualisation TODO and run again.",
            file=sys.stderr,
        )
        return 2

    print("Sparse reconstruction pipeline complete")
    print(f"  image 1: {result.image_i}")
    print(f"  image 2: {result.image_j}")
    print(f"  Lowe-filtered matches: {result.filtered_matches}")
    print(f"  essential inliers: {result.essential_inliers}")
    print(f"  pose inliers: {result.pose_inliers}")
    print(f"  initial kept 3D points: {result.kept_points}")
    if result.median_reprojection_error_px is not None:
        print(f"  initial median reprojection error: {result.median_reprojection_error_px:.3f} px")
    print(f"  registered extra images: {len(registered_results)}")
    for idx, registered_result in enumerate(registered_results, start=3):
        print(f"  image {idx}: {registered_result.image_k}")
        print(f"    PnP correspondences: {registered_result.pnp_correspondences}")
        print(f"    PnP inliers: {registered_result.pnp_inliers}")
    if rejected_images:
        print(f"  rejected images: {len(rejected_images)}")
        for image_name in rejected_images:
            print(f"    {image_name}")
    print(f"  meshlab point cloud: {args.output_dir / 'points3d.ply'}")
    print(f"  meshlab cameras: {args.output_dir / 'cameras.ply'}")
    print(f"  wrote: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
