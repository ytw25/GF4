from pathlib import Path
import sys

import numpy as np

from sfm_utils import (
    detect_sift_features,
    draw_keypoints,
    draw_matches,
    load_image,
    match_descriptors,
    precompute_image_features,
    raw_descriptor_matches,
    count_raw_matches,
    compute_epipolar_errors,
    draw_epipolar_lines,
    estimate_fundamental_ransac,
    matched_keypoint_coords,
)


ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "IMG_01.jpg"
IMAGE2_PATH = ROOT / "IMG_02.jpg"
KEYPOINT_OUTPUT_PATH = ROOT / "week2" / "output" / "test_detect_sift_IMG_01.jpg"
MATCH_OUTPUT_PATH = ROOT / "week2" / "output" / "test_match_descriptors_IMG_01_IMG_02.jpg"
RAW_MATCH_OUTPUT_PATH = ROOT / "week2" / "output" / "test_raw_matches_IMG_01_IMG_02.jpg"
EPIPOLAR_OUTPUT_PATH = ROOT / "week2" / "output" / "test_epipolar_lines_IMG_01_IMG_02.jpg"


def test_detect_sift_features_on_img_01(max_features: int = 4000) -> None:
    image = load_image(IMAGE_PATH, max_size=1200)
    keypoints, descriptors = detect_sift_features(image, max_features=max_features)

    assert len(keypoints) > 0, "No SIFT keypoints were detected on IMG_01.jpg"
    assert len(keypoints) <= max_features, "detect_sift_features returned too many keypoints"
    assert isinstance(descriptors, np.ndarray), "Descriptors should be a NumPy array"
    assert descriptors.shape == (
        len(keypoints),
        128,
    ), "SIFT descriptors should have one 128D row per keypoint"
    assert descriptors.dtype == np.float32, "OpenCV SIFT descriptors should be float32"

    draw_keypoints(image, keypoints, KEYPOINT_OUTPUT_PATH)

    print("detect_sift_features smoke test passed")
    print(f"  image       : {IMAGE_PATH}")
    print(f"  keypoints   : {len(keypoints)}")
    print(f"  descriptors : {descriptors.shape}, {descriptors.dtype}")
    print(f"  visualisation saved to: {KEYPOINT_OUTPUT_PATH}")


def test_match_descriptors_on_img_01_img_02(
    max_features: int = 4000,
    ratio: float = 0.75,
) -> None:
    features1, features2 = precompute_image_features(
        [IMAGE_PATH, IMAGE2_PATH],
        max_features=max_features,
        max_image_size=1200,
    )

    raw_matches = raw_descriptor_matches(features1.descriptors, features2.descriptors)
    raw_match_count = count_raw_matches(features1.descriptors, features2.descriptors)
    matches = match_descriptors(features1.descriptors, features2.descriptors, ratio=ratio)

    assert isinstance(raw_matches, list), "raw_descriptor_matches should return a list"
    assert len(raw_matches) > 0, "No raw nearest-neighbour matches were found"
    assert raw_match_count == len(raw_matches), "count_raw_matches should count raw matches"
    assert all(hasattr(match, "queryIdx") for match in raw_matches), (
        "raw_descriptor_matches should return cv2.DMatch objects"
    )
    assert raw_matches == sorted(raw_matches, key=lambda match: match.distance), (
        "Raw matches should be sorted by descriptor distance"
    )
    assert isinstance(matches, list), "match_descriptors should return a list"
    assert len(matches) > 0, "No Lowe-ratio matches were found between IMG_01 and IMG_02"
    assert all(hasattr(match, "queryIdx") for match in matches), (
        "match_descriptors should return cv2.DMatch objects, not nested lists"
    )
    assert all(0 <= match.queryIdx < len(features1.descriptors) for match in matches)
    assert all(0 <= match.trainIdx < len(features2.descriptors) for match in matches)
    assert matches == sorted(matches, key=lambda match: match.distance), (
        "Matches should be sorted by descriptor distance"
    )
    assert match_descriptors(np.empty((0, 128), dtype=np.float32), features2.descriptors) == []
    assert match_descriptors(features1.descriptors, np.empty((0, 128), dtype=np.float32)) == []
    assert raw_descriptor_matches(np.empty((0, 128), dtype=np.float32), features2.descriptors) == []
    assert raw_descriptor_matches(features1.descriptors, np.empty((0, 128), dtype=np.float32)) == []
    assert count_raw_matches(np.empty((0, 128), dtype=np.float32), features2.descriptors) == 0
    assert count_raw_matches(features1.descriptors, np.empty((0, 128), dtype=np.float32)) == 0
    assert len(raw_matches) >= len(matches), (
        "Raw nearest-neighbour matches should be at least as many as Lowe-filtered matches"
    )

    draw_matches(
        features1.image,
        features1.keypoints,
        features2.image,
        features2.keypoints,
        raw_matches,
        RAW_MATCH_OUTPUT_PATH,
        max_draw=400,
    )
    draw_matches(
        features1.image,
        features1.keypoints,
        features2.image,
        features2.keypoints,
        matches,
        MATCH_OUTPUT_PATH,
        max_draw=400,
    )

    print("match_descriptors smoke test passed")
    print(f"  image 1          : {IMAGE_PATH}")
    print(f"  image 2          : {IMAGE2_PATH}")
    print(f"  keypoints image 1: {len(features1.keypoints)}")
    print(f"  keypoints image 2: {len(features2.keypoints)}")
    print(f"  raw matches      : {raw_match_count}")
    print(f"  filtered matches : {len(matches)}")
    print(f"  raw visualisation saved to: {RAW_MATCH_OUTPUT_PATH}")
    print(f"  visualisation saved to: {MATCH_OUTPUT_PATH}")


def test_compute_epipolar_errors() -> None:
    F = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    pts1 = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    pts2_on_line = np.array([[5.0, 20.0], [15.0, 40.0]], dtype=np.float64)
    pts2_one_pixel_away = np.array([[5.0, 21.0], [15.0, 39.0]], dtype=np.float64)

    zero_errors = compute_epipolar_errors(F, pts1, pts2_on_line)
    one_pixel_errors = compute_epipolar_errors(F, pts1, pts2_one_pixel_away)

    assert np.allclose(zero_errors, 0.0), "Points on epipolar lines should have zero error"
    assert np.allclose(one_pixel_errors, 1.0), (
        "A one-pixel vertical offset should give one-pixel epipolar error"
    )
    assert compute_epipolar_errors(F, np.empty((0, 2)), np.empty((0, 2))).shape == (0,)

    print("compute_epipolar_errors smoke test passed")


def test_epipolar_errors_for_filtered_and_inlier_matches(
    max_features: int = 4000,
    ratio: float = 0.75,
) -> None:
    features1, features2 = precompute_image_features(
        [IMAGE_PATH, IMAGE2_PATH],
        max_features=max_features,
        max_image_size=1200,
    )
    matches = match_descriptors(features1.descriptors, features2.descriptors, ratio=ratio)
    pts1, pts2 = matched_keypoint_coords(features1.keypoints, features2.keypoints, matches)
    F, inlier_mask = estimate_fundamental_ransac(pts1, pts2)

    all_errors = compute_epipolar_errors(F, pts1, pts2)
    inlier_errors = all_errors[inlier_mask]
    inlier_ratio = len(inlier_errors) / len(matches) if matches else 0.0

    assert all_errors.shape == (len(matches),), "There should be one error per filtered match"
    assert inlier_mask.shape == (len(matches),), "RANSAC mask should align with filtered matches"
    assert len(inlier_errors) > 0, "RANSAC should find at least one inlier"
    assert len(inlier_errors) <= len(all_errors), "Inliers are a subset of filtered matches"
    assert np.median(inlier_errors) <= np.median(all_errors), (
        "RANSAC inlier errors should normally be no worse than all filtered-match errors"
    )
    assert 0.0 <= inlier_ratio <= 1.0, "Inlier ratio should be between 0 and 1"

    draw_epipolar_lines(
        features1.image,
        features2.image,
        pts1[inlier_mask],
        pts2[inlier_mask],
        F,
        EPIPOLAR_OUTPUT_PATH,
        max_lines=20,
    )

    print("filtered-vs-inlier epipolar error test passed")
    print(f"  filtered matches      : {len(matches)}")
    print(f"  RANSAC inliers        : {len(inlier_errors)}")
    print(f"  inlier ratio          : {inlier_ratio:.3f}")
    print(f"  median all error      : {np.median(all_errors):.3f} px")
    print(f"  median inlier error   : {np.median(inlier_errors):.3f} px")
    print(f"  epipolar visualisation saved to: {EPIPOLAR_OUTPUT_PATH}")


def main() -> int:
    try:
        test_detect_sift_features_on_img_01()
        test_match_descriptors_on_img_01_img_02()
        test_compute_epipolar_errors()
        test_epipolar_errors_for_filtered_and_inlier_matches()
    except Exception as exc:
        print(f"Week 2 smoke test failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
