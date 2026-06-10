"""Week 3 extension: incremental reconstruction over an image set.

This keeps the Week 3 ingredients deliberately visible: choose an initial pair,
build the first two-view reconstruction, then repeatedly pick the remaining
image with the strongest Week 2 pairwise evidence, register it with PnP, and
triangulate extra points from the newly registered view.
"""

# python .\week3\week3_pipeline_new.py `
# --image-dir .\week4\ `
# --image1  `
# --image2  `
# --output-dir ./week4/output

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np

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
    parser.add_argument("--image1", required=True, type=Path, help="Initial pair first image.")
    parser.add_argument("--image2", required=True, type=Path, help="Initial pair second image.")
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
    parser.add_argument("--min-pnp-inliers", type=int, default=6)
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
    if args.min_pnp_inliers < 4:
        parser.error("--min-pnp-inliers must be at least 4")
    if args.max_new_points_per_pair < 0:
        parser.error("--max-new-points-per-pair cannot be negative")
    if args.image1 == args.image2:
        parser.error("--image1 and --image2 must be different")

    return args


def _median(values: np.ndarray) -> float | None:
    return float(np.median(values)) if len(values) else None


def _mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if len(values) else None


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

    center_a = camera_center(R_a, t_a)
    center_b = camera_center(R_b, t_b)
    rays_a = points - center_a.reshape(1, 3)
    rays_b = points - center_b.reshape(1, 3)
    norms_a = np.linalg.norm(rays_a, axis=1)
    norms_b = np.linalg.norm(rays_b, axis=1)
    valid = (norms_a > 1e-12) & (norms_b > 1e-12)

    cos_angles = np.ones(len(points), dtype=np.float64)
    cos_angles[valid] = np.sum(rays_a[valid] * rays_b[valid], axis=1) / (norms_a[valid] * norms_b[valid])
    angles = np.degrees(np.arccos(np.clip(cos_angles, -1.0, 1.0)))
    return valid & (angles >= min_angle_deg)


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

    idx1, idx2 = initial_indices
    features1 = features[idx1]
    features2 = features[idx2]
    K = intrinsics[idx1]

    match_cache: dict[tuple[int, int], list[cv2.DMatch]] = {}
    pairwise_cache: dict[tuple[int, int], dict] = {}

    matches12 = week2.match_descriptors(features1.descriptors, features2.descriptors, ratio=args.ratio)
    match_cache[(idx1, idx2)] = matches12
    if len(matches12) < 8:
        raise ValueError(f"Need at least 8 Lowe-filtered matches, got {len(matches12)}")

    pts1, pts2 = week2.matched_keypoint_coords(features1.keypoints, features2.keypoints, matches12)
    E, essential_mask = estimate_essential_matrix(
        pts1,
        pts2,
        K,
        threshold=args.ransac_threshold,
        confidence=args.confidence,
    )
    R2, t2, pose_mask = recover_relative_pose(E, pts1, pts2, K, inlier_mask=essential_mask)

    pts1_pose = pts1[pose_mask]
    pts2_pose = pts2[pose_mask]
    pose_matches = [match for match, keep in zip(matches12, pose_mask) if keep]

    if hasattr(week2, "draw_matches") and not args.skip_figures:
        week2.draw_matches(
            features1.image,
            features1.keypoints,
            features2.image,
            features2.keypoints,
            pose_matches,
            output_dir / "pose_inlier_matches.png",
        )

    initial_points = triangulate_points(pts1_pose, pts2_pose, K, R2, t2)
    errors1 = compute_reprojection_errors(initial_points, pts1_pose, K, np.eye(3), np.zeros((3, 1)))
    errors2 = compute_reprojection_errors(initial_points, pts2_pose, K, R2, t2)
    keep = filter_reconstructed_points(
        initial_points,
        errors1,
        errors2,
        R2,
        t2,
        max_reprojection_error=args.max_reprojection_error,
    )
    keep &= _triangulation_angle_mask(
        initial_points,
        np.eye(3),
        np.zeros((3, 1)),
        R2,
        t2,
        args.min_triangulation_angle_deg,
    )
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
    for point_id, match in enumerate(kept_pose_matches):
        observations_by_image[idx1][int(match.queryIdx)] = point_id
        observations_by_image[idx2][int(match.trainIdx)] = point_id

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

    while remaining:
        best_candidate = None
        for candidate_idx in remaining:
            for anchor_idx in registered_indices:
                pair_key = (anchor_idx, candidate_idx)
                if pair_key not in pairwise_cache:
                    analysis = week2.analyse_feature_pair(
                        features[anchor_idx],
                        features[candidate_idx],
                        output_dir / "pairwise",
                        ratio=args.ratio,
                        save_figures=False,
                    )
                    row = analysis.csv_dict()
                    pairwise_cache[pair_key] = row
                else:
                    row = pairwise_cache[pair_key]

                candidate = {
                    "candidate_idx": candidate_idx,
                    "anchor_idx": anchor_idx,
                    "inliers": int(row["ransac_inliers"]),
                    "matches": int(row["filtered_matches"]),
                }
                if best_candidate is None or candidate["inliers"] > best_candidate["inliers"]:
                    best_candidate = candidate

        if best_candidate is None:
            rejected_diagnostics.append({"reason": "no registered anchors"})
            break
        if best_candidate["inliers"] < args.min_pairwise_inliers:
            for idx in remaining:
                rejected_diagnostics.append(
                    {
                        "image": features[idx].path.name,
                        "reason": "best pairwise inliers below threshold",
                    }
                )
            break

        candidate_idx = best_candidate["candidate_idx"]
        candidate_features = features[candidate_idx]

        pnp_candidates = []
        anchor_match_count = 0
        for anchor_idx in registered_indices:
            match_key = (anchor_idx, candidate_idx)
            if match_key not in match_cache:
                match_cache[match_key] = week2.match_descriptors(
                    features[anchor_idx].descriptors,
                    candidate_features.descriptors,
                    ratio=args.ratio,
                )
            anchor_matches = match_cache[match_key]
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
            used_point_ids.add(point_id)
            used_new_keypoints.add(new_keypoint_idx)
            pnp_point_ids.append(point_id)
            new_keypoint_indices.append(new_keypoint_idx)
            pnp_points.append(points3d[point_id])
            pnp_pts.append(pt)

        pnp_points3d = np.asarray(pnp_points, dtype=np.float64).reshape(-1, 3)
        pts_new = np.asarray(pnp_pts, dtype=np.float64).reshape(-1, 2)

        if len(pnp_points3d) < args.min_pnp_correspondences:
            rejected_diagnostics.append(
                {
                    "image": candidate_features.path.name,
                    "reason": "too few 2D-3D correspondences",
                    "best_anchor": features[best_candidate["anchor_idx"]].path.name,
                    "best_pairwise_inliers": best_candidate["inliers"],
                    "pnp_correspondences": len(pnp_points3d),
                }
            )
            remaining.remove(candidate_idx)
            continue

        try:
            R_new, t_new, pnp_mask = estimate_camera_pose_pnp(
                pnp_points3d,
                pts_new,
                intrinsics[candidate_idx],
                threshold=args.pnp_ransac_threshold,
                confidence=args.confidence,
            )
        except (RuntimeError, ValueError, cv2.error) as exc:
            rejected_diagnostics.append(
                {
                    "image": candidate_features.path.name,
                    "reason": "too few PnP inliers",
                    "best_anchor": features[best_candidate["anchor_idx"]].path.name,
                    "best_pairwise_inliers": best_candidate["inliers"],
                    "pnp_correspondences": len(pnp_points3d),
                    "pnp_error": str(exc),
                }
            )
            remaining.remove(candidate_idx)
            continue

        pnp_inliers = int(np.sum(pnp_mask))
        if pnp_inliers < args.min_pnp_inliers:
            rejected_diagnostics.append(
                {
                    "image": candidate_features.path.name,
                    "reason": "too few PnP inliers",
                    "best_anchor": features[best_candidate["anchor_idx"]].path.name,
                    "best_pairwise_inliers": best_candidate["inliers"],
                    "pnp_correspondences": len(pnp_points3d),
                    "pnp_inliers": pnp_inliers,
                }
            )
            remaining.remove(candidate_idx)
            continue

        pnp_errors = compute_reprojection_errors(
            pnp_points3d[pnp_mask],
            pts_new[pnp_mask],
            intrinsics[candidate_idx],
            R_new,
            t_new,
        )

        camera_poses[candidate_idx] = (R_new, t_new)
        observations_by_image[candidate_idx] = {}
        for point_id, keypoint_idx, keep_pnp in zip(pnp_point_ids, new_keypoint_indices, pnp_mask):
            if not keep_pnp:
                continue
            observations_by_image[candidate_idx][int(keypoint_idx)] = int(point_id)

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
                "min_triangulation_angle_deg": args.min_triangulation_angle_deg,
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
    if not args.skip_figures:
        print(f"  interactive view: {args.output_dir / 'interactive_point_cloud.html'}")
    print(f"  meshlab point cloud: {args.output_dir / 'points3d.ply'}")
    print(f"  meshlab cameras: {args.output_dir / 'cameras.ply'}")
    print(f"  wrote: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
