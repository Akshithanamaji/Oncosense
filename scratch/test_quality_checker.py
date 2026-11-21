"""Comprehensive test suite for Mammogram Quality Checker."""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
from PIL import Image

from src.quality import check_mammogram_quality


def run_tests():
    print("--- 1. Testing Real Test Mammogram ---")
    test_files = list((ROOT / "data" / "processed" / "cbis_ddsm" / "test").rglob("*.png"))
    assert len(test_files) > 0, "Expected at least 1 test image"
    real_sample = test_files[0]
    rep_real = check_mammogram_quality(real_sample)
    print(f"Result: {rep_real.status} (Score: {rep_real.overall_score}/100)")
    assert rep_real.status in ("PASSED", "WARNING")
    assert rep_real.is_acceptable is True
    assert rep_real.preprocessed_image is not None
    print("[PASS] Real mammogram passed quality screening!")

    print("\n--- 2. Testing Corrupted / Empty File ---")
    rep_empty = check_mammogram_quality(b"")
    print(f"Empty File: {rep_empty.status} (Score: {rep_empty.overall_score})")
    assert rep_empty.status == "REJECTED"
    assert rep_empty.is_acceptable is False

    rep_garbage = check_mammogram_quality(b"CORRUPT_NOT_AN_IMAGE_STREAM_1234567890")
    print(f"Corrupt File: {rep_garbage.status} (Score: {rep_garbage.overall_score})")
    assert rep_garbage.status == "REJECTED"
    assert rep_garbage.is_acceptable is False
    print("[PASS] Corrupt/unreadable files successfully intercepted!")

    print("\n--- 3. Testing Unsuitable Micro Dimensions ---")
    micro = np.random.randint(50, 200, (20, 20), dtype=np.uint8)
    _, micro_buf = cv2.imencode(".png", micro)
    rep_micro = check_mammogram_quality(micro_buf.tobytes())
    print(f"Micro Image (20x20): {rep_micro.status} (Score: {rep_micro.overall_score})")
    assert rep_micro.status == "REJECTED"
    assert any("Resolution is extremely low" in e for e in rep_micro.errors)
    print("[PASS] Unsuitable micro-dimensions successfully intercepted!")

    print("\n--- 4. Testing Solid Black / Zero Contrast ---")
    black = np.zeros((384, 384), dtype=np.uint8)
    _, black_buf = cv2.imencode(".png", black)
    rep_black = check_mammogram_quality(black_buf.tobytes())
    print(f"Solid Black: {rep_black.status} (Score: {rep_black.overall_score})")
    assert rep_black.status == "REJECTED"
    print("[PASS] Zero contrast / blank image successfully intercepted!")

    print("\n--- 5. Testing Severe Blur (Defocussed Scan) ---")
    base = cv2.imread(str(real_sample), cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(base, (55, 55), 30)
    _, blur_buf = cv2.imencode(".png", blurred)
    rep_blur = check_mammogram_quality(blur_buf.tobytes())
    print(f"Severe Blur: {rep_blur.status} (Score: {rep_blur.overall_score})")
    assert rep_blur.status == "REJECTED"
    assert any("blur" in e.lower() for e in rep_blur.errors)
    print("[PASS] Severely blurred input successfully intercepted!")

    print("\n=======================================================")
    print("ALL QUALITY CHECKER TESTS PASSED!")
    print("=======================================================")


if __name__ == "__main__":
    run_tests()

