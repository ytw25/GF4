"""Command-line driver for GF4 Week 3 sparse reconstruction.

This script reuses a completed Week 2 sfm_utils.py for SIFT feature detection
and descriptor matching, then calls Week 3 utilities for essential matrix
estimation, pose recovery, triangulation, third-view registration, and
visualisation.

python week4/week4_pipeline.py --image1 path --image2 path 
--image-dir path --output-dir path --week2-dir week2  --week3-dir week3 
 --max-features    --ratio    
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np




DEFAULT_WEEK2_DIR = Path(__file__).resolve().parents[1] / "week2"

DEFAULT_WEEK3_DIR = Path(__file__).resolve().parents[1] / "week3"


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

def load_week3_module(week3_dir:Path):
    """Load the completed Week 3 two_view_utils.py by path."""
    module_path = Path(week3_dir) / "two_view_utils.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find Week 3 two_view_utils.py: {module_path}")

    spec = importlib.util.spec_from_file_location("week3_two_view_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Week 3 module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module 


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GF4 Week 3 sparse reconstruction pipeline."
    )
    parser.add_argument("--image1", required=True, type=Path, help="First image.")
    parser.add_argument("--image2", required=True, type=Path, help="Second image.")
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=None,
        help="Directory containing images for third-view registration.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where metrics and figures will be written.",
    )
    parser.add_argument(
        "--week2-dir",
        type=Path,
        default=DEFAULT_WEEK2_DIR,
        help="Directory containing the completed Week 2 sfm_utils.py.",
    )
    parser.add_argument(
        "--week3-dir",
        type=Path,
        default=DEFAULT_WEEK3_DIR,
        help="Directory containing the completed Week 3 two_view_utils.py.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1600,
        help="Resize images so their long edge is at most this size. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Maximum number of images to load in dataset mode.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=4000,
        help="Maximum number of SIFT features to retain per image.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.75,
        help="Lowe ratio-test threshold passed to the Week 2 matcher.",
    )
    parser.add_argument(
        "--focal-length-px",
        type=float,
        default=None,
        help="Optional focal length in pixels. Default is 1.2 times the image long edge.",
    )
    parser.add_argument(
        "--principal-point",
        nargs=2,
        type=float,
        metavar=("CX", "CY"),
        default=None,
        help="Optional principal point in pixels.",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=1.0,
        help="RANSAC threshold in pixels for essential matrix estimation.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.999,
        help="RANSAC confidence for essential matrix estimation.",
    )
    parser.add_argument(
        "--max-reprojection-error",
        type=float,
        default=4.0,
        help="Maximum reprojection error in pixels for keeping triangulated points.",
    )
    parser.add_argument(
        "--pnp-ransac-threshold",
        type=float,
        default=6.0,
        help="RANSAC reprojection threshold in pixels for third-view PnP.",
    )

    args = parser.parse_args()

    if args.max_image_size == 0:
        args.max_image_size = None
    if args.max_images < 2:
        parser.error("--max-images must be at least 2")
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
    if args.pnp_ransac_threshold <= 0:
        parser.error("--pnp-ransac-threshold must be positive")

    return args


def _median(values: np.ndarray) -> float | None:
    return float(np.median(values)) if len(values) else None


def _mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _merge_2d3d_correspondence_chunks(
    chunks: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    points3d = []
    pts2d = []
    used_pixels = set()
    for chunk_points, chunk_pts in chunks:
        for point, pt in zip(chunk_points, chunk_pts):
            pixel_key = (round(float(pt[0]), 3), round(float(pt[1]), 3))
            if pixel_key in used_pixels:
                continue
            used_pixels.add(pixel_key)
            points3d.append(point)
            pts2d.append(pt)

    if not points3d:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    return np.asarray(points3d, dtype=np.float64), np.asarray(pts2d, dtype=np.float64)


#First two images, then reconstruction for all images in subset using those two images
def run(args: argparse.Namespace):
    week2 = load_week2_module(args.week2_dir)
    week3 = load_week3_module(args.week3_dir)
    
    output_dir = week2.ensure_dir(args.output_dir)

    image_anchors= [args.image1, args.image2]
    print(f"Running pair analysis for {args.image1.name} and {args.image2.name}")
    
    features1, features2 = week2.precompute_image_features(
        image_anchors,
        max_features=args.max_features,
        max_image_size=args.max_image_size,
    )
    matches = week2.match_descriptors(features1.descriptors, features2.descriptors, ratio=args.ratio)
    if len(matches) < 8:
        raise ValueError(f"Need at least 8 Lowe-filtered matches, got {len(matches)}")
    
    pts1, pts2 = week2.matched_keypoint_coords(features1.keypoints, features2.keypoints, matches)
    K = week3.make_camera_matrix(
        features1.image.shape,
        focal_length_px=args.focal_length_px,
        principal_point=None if args.principal_point is None else tuple(args.principal_point),
    )

    E, essential_mask = week3.estimate_essential_matrix(
        pts1,
        pts2,
        K,
        threshold=args.ransac_threshold,
        confidence=args.confidence,
    )
    R, t, pose_mask = week3.recover_relative_pose(E, pts1, pts2, K, inlier_mask=essential_mask)
    
    pts1_pose = pts1[pose_mask]
    pts2_pose = pts2[pose_mask]
    pose_matches = [match for match, keep in zip(matches, pose_mask) if keep]

    if hasattr(week2, "draw_matches"):
        week2.draw_matches(
            features1.image,
            features1.keypoints,
            features2.image,
            features2.keypoints,
            pose_matches,
            output_dir / "pose_inlier_matches.png",
        )
    
    points3d = week3.triangulate_points(pts1_pose, pts2_pose, K, R, t)
    errors1 = week3.compute_reprojection_errors(points3d, pts1_pose, K, np.eye(3), np.zeros((3, 1)))
    errors2 = week3.compute_reprojection_errors(points3d, pts2_pose, K, R, t)
    keep = week3.filter_reconstructed_points(
        points3d,
        errors1,
        errors2,
        R,
        t,
        max_reprojection_error=args.max_reprojection_error,
    )
    depths1, depths2 = week3.compute_depths(points3d, R, t)
    positive_depth = (depths1 > 0) & (depths2 > 0)
    kept_points = points3d[keep]
    kept_colours = week3.sample_point_colours(features1.image, pts1_pose[keep])
    kept_errors = 0.5 * (errors1[keep] + errors2[keep])
    kept_pose_matches = [match for match, keep_point in zip(pose_matches, keep) if keep_point]
    kept_image1_indices = np.array([match.queryIdx for match in kept_pose_matches], dtype=int)
    kept_image2_indices = np.array([match.trainIdx for match in kept_pose_matches], dtype=int)
    week3.draw_reprojection_overlay(
        features1.image,
        features2.image,
        pts1_pose[keep],
        pts2_pose[keep],
        kept_points,
        K,
        R,
        t,
        output_dir / "reprojection_overlay.png",
    )
    week3.plot_two_view_reconstruction(
        kept_points,
        kept_colours,
        R,
        t,
        output_dir / "two_view_reconstruction.png",
    )
    week3.plot_patch_cloud_reconstruction(
        kept_points,
        features1.image,
        pts1_pose[keep],
        [
            ("Camera 1", np.eye(3), np.zeros((3, 1))),
            ("Camera 2", R, t),
        ],
        output_dir / "two_view_patch_cloud.png",
    )
    week3.write_ply(output_dir / "points3d.ply", kept_points, kept_colours)

    result = week3.TwoViewResult(
        image_i=features1.path.name,
        image_j=features2.path.name,
        filtered_matches=len(matches),
        essential_inliers=int(np.sum(essential_mask)),
        pose_inliers=int(np.sum(pose_mask)),
        triangulated_points=len(points3d),
        positive_depth_points=int(np.sum(positive_depth)),
        kept_points=int(np.sum(keep)),
        median_reprojection_error_px=_median(kept_errors),
        mean_reprojection_error_px=_mean(kept_errors),
        focal_length_px=float(K[0, 0]),
    )

    week2.save_csv(output_dir / "two_view_metrics.csv", [result.as_dict()])
    np.savetxt(output_dir / "K.txt", K)
    np.savetxt(output_dir / "R.txt", R)
    np.savetxt(output_dir / "t.txt", t)
    np.savetxt(output_dir / "E.txt", E)

    print("Sparse reconstruction pipeline complete")
    print(f"  image 1: {result.image_i}")
    print(f"  image 2: {result.image_j}")
    print(f"  Lowe-filtered matches: {result.filtered_matches}")
    print(f"  essential inliers: {result.essential_inliers}")
    print(f"  pose inliers: {result.pose_inliers}")
    print(f"  kept 3D points: {result.kept_points}", f'unique: {len(np.unique(kept_points, axis=0))}')
    if result.median_reprojection_error_px is not None:
        print(f"  median reprojection error: {result.median_reprojection_error_px:.3f} px")
    print(f"  wrote: {args.output_dir}")


    third_result = None
    if args.image_dir is not None:
        image_paths = week2.list_image_paths(args.image_dir, max_images=args.max_images)
        print(f"Loaded {len(image_paths)} images")
        print(f"Running reconstruction using two-view anchors {result.image_i} and {result.image_j}")

        features = week2.precompute_image_features(
            image_paths,
            max_features=args.max_features,
            max_image_size=args.max_image_size,
        )

        rows=[]
        cameras=[("Camera 1", np.eye(3), np.zeros((3, 1))),
                    ("Camera 2", R, t)]
        i=2
        for features3 in features:
            print('-----------------------------------------------------')
            i+=1
            image_name = f"{features3.path.stem}"
            print(f"Analysing {features3.path.name}")

            K3 = week3.make_camera_matrix(
                features3.image.shape,
                focal_length_px=args.focal_length_px,
                principal_point=None if args.principal_point is None else tuple(args.principal_point),
            )
            matches13 = week2.match_descriptors(
                features1.descriptors,
                features3.descriptors,
                ratio=args.ratio,
            )
            matches23 = week2.match_descriptors(
                features2.descriptors,
                features3.descriptors,
                ratio=args.ratio,
            )
            if len(matches13) + len(matches23) < 6:
                print("Need at least 6 Lowe-filtered matches from reconstructed images to image 3",f"got {len(matches13) + len(matches23)}")
                break
                
            '''
            if hasattr(week2, "draw_matches"):
                week2.draw_matches(
                    features1.image,
                    features1.keypoints,
                    features3.image,
                    features3.keypoints,
                    matches13,
                    output_dir / "image1_image3_matches.png",
                )
                week2.draw_matches(
                    features2.image,
                    features2.keypoints,
                    features3.image,
                    features3.keypoints,
                    matches23,
                    output_dir / "image2_image3_matches.png",
                )
            '''

            pnp_points13, pts3_from1 = week3.build_2d3d_correspondences(
                kept_image1_indices,
                kept_points,
                matches13,
                features3.keypoints,
            )
            pnp_points23, pts3_from2 = week3.build_2d3d_correspondences(
                kept_image2_indices,
                kept_points,
                matches23,
                features3.keypoints,
            )
            pnp_points3d, pts3 = _merge_2d3d_correspondence_chunks(
                [(pnp_points13, pts3_from1), (pnp_points23, pts3_from2)]
            )
            if len(pnp_points3d) < 6:
                print("Need at least 6 Lowe-filtered matches from reconstructed images to image 3",f"got {len(pnp_points3d)}")
            else :
                R3, t3, pnp_mask = week3.estimate_camera_pose_pnp(
                    pnp_points3d,
                    pts3,
                    K3,
                    threshold=args.pnp_ransac_threshold,
                    confidence=args.confidence,
                )
                pnp_points = pnp_points3d[pnp_mask]
                pts3_inliers = pts3[pnp_mask]
                pnp_errors = week3.compute_reprojection_errors(pnp_points, pts3_inliers, K3, R3, t3)
                week3.draw_single_image_reprojection_overlay(
                    features3.image,
                    pts3_inliers,
                    pnp_points,
                    K3,
                    R3,
                    t3,
                    output_dir/'reprojection' / f"third_view_reprojection_overlay_{image_name}.png",
                )
                

                third_result = week3.ThirdViewResult(
                    image_k=features3.path.name,
                    anchor_matches=len(matches13) + len(matches23),
                    pnp_correspondences=len(pnp_points3d),
                    pnp_inliers=int(np.sum(pnp_mask)),
                    median_reprojection_error_px=_median(pnp_errors),
                    mean_reprojection_error_px=_mean(pnp_errors),
                    focal_length_px=float(K3[0, 0]),
                )

                rows.append(third_result.as_dict())
                cameras.append((f"Camera {i}", R3, t3))
                print(f"  image 3: {third_result.image_k}")
                print(f"  anchor-to-image-3 Lowe-filtered matches: {third_result.anchor_matches}")
                print(f"  image 3 PnP correspondences: {third_result.pnp_correspondences}")
                print(f"  image 3 PnP inliers: {third_result.pnp_inliers}")
                if third_result.median_reprojection_error_px is not None:
                    print(
                        "  image 3 median reprojection error: "
                        f"{third_result.median_reprojection_error_px:.3f} px"
                    )

        week2.save_csv(output_dir / "third_view_metrics.csv", rows)
        week3.plot_multi_view_reconstruction(
                kept_points,
                kept_colours,
                cameras,
                output_dir / f"third_view_reconstruction.png",
            )
        week3.plot_patch_cloud_reconstruction(
                kept_points,
                features1.image,
                pts1_pose[keep],
                cameras,
                output_dir / "three_view_patch_cloud.png",
            )

        
    print(f"  wrote: {args.output_dir}")

    return result, third_result


def main() -> int:
    args = parse_args()

    try:
        run(args)
    except NotImplementedError as exc:
        return 2      

    return 0
    


if __name__ == "__main__":
    raise SystemExit(main())
