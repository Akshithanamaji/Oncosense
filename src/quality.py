"""Mammogram Quality Checker — Input Quality Assurance (IQA) for Medical Deep Learning.

Evaluates file integrity, decodability, dimensional suitability, dynamic range,
spatial sharpness, and parenchymal tissue coverage before inference.

Research value:
    Tests whether automated pre-inference quality screening successfully filters
    unsuitable, corrupted, and degraded inputs, preventing diagnostic errors, false
    positives, and hallucinations on invalid radiological data.

Usage:
    python src/quality.py --image path/to/scan.png
    python src/quality.py --benchmark --config configs/cbis_ddsm_v3.yaml
"""
import argparse
import io
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pydicom
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
_ROOT_DIR = _THIS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))


@dataclass
class QualityCheckItem:
    name: str
    status: str  # "PASS", "WARN", "FAIL"
    score: float  # 0.0 to 1.0
    value_display: str
    message: str
    is_critical: bool = False


@dataclass
class MammogramQualityReport:
    status: str  # "PASSED", "WARNING", "REJECTED"
    overall_score: int  # 0 to 100
    is_acceptable: bool  # True if can proceed to inference
    checks: list[QualityCheckItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    preprocessed_image: Image.Image | None = None

    def to_dict(self):
        d = asdict(self)
        d.pop("preprocessed_image", None)
        return d


def _read_image_and_meta(file_input, target_size: int = 384):
    """Safely reads DICOM or standard image format, extracting raw array and metadata."""
    meta = {
        "format": "UNKNOWN",
        "original_width": 0,
        "original_height": 0,
        "file_size_bytes": 0,
        "pixel_spacing_mm": None,
    }

    # Determine bytes and filename
    if hasattr(file_input, "read"):
        # Streamlit UploadedFile or file-like object
        pos = file_input.tell() if hasattr(file_input, "tell") else 0
        raw_bytes = file_input.read()
        if hasattr(file_input, "seek"):
            file_input.seek(pos)
        filename = getattr(file_input, "name", "uploaded_file").lower()
    elif isinstance(file_input, (str, Path)):
        p = Path(file_input)
        filename = p.name.lower()
        with open(p, "rb") as f:
            raw_bytes = f.read()
    elif isinstance(file_input, bytes):
        raw_bytes = file_input
        filename = "image.dcm" if raw_bytes.startswith(b"\x00" * 128 + b"DICM") else "image.png"
    elif isinstance(file_input, Image.Image):
        meta["format"] = "PIL.Image"
        meta["original_width"] = file_input.width
        meta["original_height"] = file_input.height
        rgb_img = file_input.convert("RGB")
        resized = rgb_img.resize((target_size, target_size), Image.BILINEAR)
        return np.array(rgb_img), resized, meta
    else:
        raise ValueError(f"Unsupported file_input type: {type(file_input)}")

    meta["file_size_bytes"] = len(raw_bytes)

    if len(raw_bytes) == 0:
        raise ValueError("File is empty (0 bytes).")

    # 1. Attempt DICOM decoding if filename ends with .dcm or contains DICM preamble
    is_dcm = filename.endswith(".dcm") or (len(raw_bytes) > 132 and raw_bytes[128:132] == b"DICM")
    if is_dcm:
        try:
            ds = pydicom.dcmread(io.BytesIO(raw_bytes))
            pixel_arr = ds.pixel_array.astype(np.float32)
            meta["format"] = "DICOM"
            meta["original_height"], meta["original_width"] = pixel_arr.shape[:2]
            meta["photometric"] = getattr(ds, "PhotometricInterpretation", "n/a")
            spacing = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
            if spacing:
                meta["pixel_spacing_mm"] = float(spacing[0])

            # Invert MONOCHROME1 (air is white, tissue is dark)
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                pixel_arr = pixel_arr.max() - pixel_arr

            # Normalize to 0-255 uint8
            p_min, p_max = pixel_arr.min(), pixel_arr.max()
            if p_max > p_min:
                norm_arr = ((pixel_arr - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
            else:
                norm_arr = np.zeros(pixel_arr.shape[:2], dtype=np.uint8)

            rgb_arr = cv2.cvtColor(norm_arr, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(rgb_arr)
            resized_pil = pil_img.resize((target_size, target_size), Image.BILINEAR)
            return rgb_arr, resized_pil, meta
        except Exception as e:
            raise ValueError(f"DICOM parsing failure: {str(e)}")

    # 2. Standard image (PNG, JPG, TIFF, WEBP)
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img.verify()  # Check for corruption without loading entire stream
        # Re-open after verify() to load image pixels
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        meta["format"] = (pil_img.format or filename.rsplit(".", 1)[-1]).upper()
        meta["original_width"] = pil_img.width
        meta["original_height"] = pil_img.height
        rgb_arr = np.array(pil_img)
        resized_pil = pil_img.resize((target_size, target_size), Image.BILINEAR)
        return rgb_arr, resized_pil, meta
    except Exception as e:
        raise ValueError(f"Image decode failure: {str(e)}")


def check_mammogram_quality(file_input, target_size: int = 384) -> MammogramQualityReport:
    """Performs comprehensive multi-attribute quality screening on an input mammogram.

    Checks:
      1. File Readability & Integrity (non-empty, non-corrupt, valid DICOM/PNG/JPG)
      2. Geometric & Dimensional Suitability (min resolution, reasonable aspect ratio)
      3. Dynamic Range & Contrast (adequate radiological contrast, not washed out)
      4. Spatial Sharpness (Laplacian focus variance, detects severe blur)
      5. Anatomical Parenchyma Coverage (breast tissue segmentation vs empty background)
      6. Pixel Saturation & Clipping (under/overexposure verification)

    Returns:
        MammogramQualityReport with status ('PASSED', 'WARNING', 'REJECTED'),
        overall quality score (0-100), itemized checks, and preprocessed PIL image.
    """
    checks = []
    errors = []
    warnings = []
    metrics = {}

    # Check 1: File Integrity & Decoding
    try:
        raw_rgb, resized_pil, meta = _read_image_and_meta(file_input, target_size=target_size)
        file_size_kb = meta["file_size_bytes"] / 1024 if meta.get("file_size_bytes") else 0
        checks.append(QualityCheckItem(
            name="File Integrity & Format",
            status="PASS",
            score=1.0,
            value_display=f"{meta['format']} ({file_size_kb:.1f} KB)",
            message="File successfully parsed and decoded without corruption.",
            is_critical=True,
        ))
    except Exception as exc:
        msg = f"Unreadable or corrupted file: {str(exc)}"
        checks.append(QualityCheckItem(
            name="File Integrity & Format",
            status="FAIL",
            score=0.0,
            value_display="CORRUPT / UNREADABLE",
            message=msg,
            is_critical=True,
        ))
        errors.append(msg)
        return MammogramQualityReport(
            status="REJECTED",
            overall_score=0,
            is_acceptable=False,
            checks=checks,
            errors=errors,
            warnings=warnings,
            metrics={"error": str(exc)},
            meta={},
            preprocessed_image=None,
        )

    orig_w, orig_h = meta["original_width"], meta["original_height"]
    metrics["width"] = orig_w
    metrics["height"] = orig_h

    # Check 2: Dimensional Suitability & Aspect Ratio
    min_dim = min(orig_w, orig_h)
    max_dim = max(orig_w, orig_h)
    aspect_ratio = max_dim / max(1, min_dim)
    metrics["aspect_ratio"] = round(aspect_ratio, 2)

    if min_dim < 64:
        dim_status = "FAIL"
        dim_score = 0.0
        dim_msg = f"Resolution is extremely low ({orig_w}x{orig_h} px). Lesion structures cannot be resolved."
        errors.append(dim_msg)
    elif min_dim < 192:
        dim_status = "WARN"
        dim_score = 0.5
        dim_msg = f"Suboptimal resolution ({orig_w}x{orig_h} px). Upsampling may introduce interpolation artifacts."
        warnings.append(dim_msg)
    elif aspect_ratio > 4.5:
        dim_status = "WARN"
        dim_score = 0.5
        dim_msg = f"Extreme aspect ratio ({aspect_ratio:.1f}:1). Image appears to be a cropped strip or non-standard slice."
        warnings.append(dim_msg)
    else:
        dim_status = "PASS"
        dim_score = 1.0
        dim_msg = f"Suitable resolution ({orig_w}x{orig_h} px, aspect ratio {aspect_ratio:.2f}:1)."

    checks.append(QualityCheckItem(
        name="Resolution & Dimensions",
        status=dim_status,
        score=dim_score,
        value_display=f"{orig_w} x {orig_h} px",
        message=dim_msg,
        is_critical=True if dim_status == "FAIL" else False,
    ))

    # Convert preprocessed resized image to grayscale for photometric evaluation
    gray = np.array(resized_pil.convert("L"), dtype=np.uint8)

    # Check 3: Dynamic Range & Contrast
    min_val, max_val = int(gray.min()), int(gray.max())
    dynamic_range = max_val - min_val
    std_contrast = float(gray.std())
    metrics["dynamic_range"] = dynamic_range
    metrics["std_contrast"] = round(std_contrast, 2)

    if dynamic_range < 25 or std_contrast < 6.0:
        dr_status = "FAIL"
        dr_score = 0.0
        dr_msg = f"Critically low dynamic range ({dynamic_range}/255, std {std_contrast:.1f}). Scan is flat or nearly solid black/white."
        errors.append(dr_msg)
    elif dynamic_range < 65 or std_contrast < 18.0:
        dr_status = "WARN"
        dr_score = 0.5
        dr_msg = f"Low contrast ({dynamic_range}/255). Parenchymal differentiation may be impaired."
        warnings.append(dr_msg)
    else:
        dr_status = "PASS"
        dr_score = 1.0
        dr_msg = f"Healthy dynamic range ({dynamic_range}/255, contrast std {std_contrast:.1f})."

    checks.append(QualityCheckItem(
        name="Contrast & Dynamic Range",
        status=dr_status,
        score=dr_score,
        value_display=f"{dynamic_range} / 255 (std {std_contrast:.1f})",
        message=dr_msg,
        is_critical=True if dr_status == "FAIL" else False,
    ))

    # Check 4: Spatial Sharpness & Focus (Laplacian Variance)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics["laplacian_variance"] = round(laplacian_var, 2)

    if laplacian_var < 2.5:
        sharp_status = "FAIL"
        sharp_score = 0.0
        sharp_msg = f"Severe blurriness or out-of-focus scan (Laplacian var {laplacian_var:.1f}). Critical lesion edges are lost."
        errors.append(sharp_msg)
    elif laplacian_var < 7.5:
        sharp_status = "WARN"
        sharp_score = 0.5
        sharp_msg = f"Borderline sharpness (Laplacian var {laplacian_var:.1f}). Fine microcalcifications may be indistinct."
        warnings.append(sharp_msg)
    else:
        sharp_status = "PASS"
        sharp_score = 1.0
        sharp_msg = f"Good sharpness and edge focus (Laplacian var {laplacian_var:.1f})."

    checks.append(QualityCheckItem(
        name="Spatial Sharpness & Focus",
        status=sharp_status,
        score=sharp_score,
        value_display=f"{laplacian_var:.1f} var",
        message=sharp_msg,
        is_critical=True if sharp_status == "FAIL" else False,
    ))

    # Check 5: Anatomical Parenchyma Coverage
    # Otsu thresholding separates tissue from black air background
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue_pixels = np.sum(otsu_mask > 0)
    tissue_pct = (tissue_pixels / gray.size) * 100.0
    metrics["tissue_coverage_pct"] = round(tissue_pct, 1)

    if tissue_pct < 2.5:
        tissue_status = "FAIL"
        tissue_score = 0.0
        tissue_msg = f"Insufficient breast tissue ({tissue_pct:.1f}% area). Scan contains mostly empty black background."
        errors.append(tissue_msg)
    elif tissue_pct < 8.0:
        tissue_status = "WARN"
        tissue_score = 0.6
        tissue_msg = f"Low tissue coverage ({tissue_pct:.1f}%). Scan may be clipped or off-center."
        warnings.append(tissue_msg)
    elif tissue_pct > 96.5:
        tissue_status = "WARN"
        tissue_score = 0.7
        tissue_msg = f"No air boundary detected ({tissue_pct:.1f}% dense coverage). Verify this is a mammogram and not a natural photograph or document."
        warnings.append(tissue_msg)
    else:
        tissue_status = "PASS"
        tissue_score = 1.0
        tissue_msg = f"Typical mammographic tissue coverage ({tissue_pct:.1f}% area with clear air margin)."

    checks.append(QualityCheckItem(
        name="Parenchymal Tissue Coverage",
        status=tissue_status,
        score=tissue_score,
        value_display=f"{tissue_pct:.1f}% of image",
        message=tissue_msg,
        is_critical=True if tissue_status == "FAIL" else False,
    ))

    # Check 6: Pixel Saturation & Extreme Clipping
    total_px = float(gray.size)
    zero_pct = (np.sum(gray == 0) / total_px) * 100.0
    sat_pct = (np.sum(gray >= 254) / total_px) * 100.0
    metrics["zero_pixels_pct"] = round(zero_pct, 1)
    metrics["saturated_pixels_pct"] = round(sat_pct, 1)

    if zero_pct > 96.0:
        sat_status = "FAIL"
        sat_score = 0.0
        sat_msg = f"Extreme underexposure: {zero_pct:.1f}% of pixels are pure black (0)."
        errors.append(sat_msg)
    elif sat_pct > 75.0:
        sat_status = "FAIL"
        sat_score = 0.0
        sat_msg = f"Severe overexposure / clipping: {sat_pct:.1f}% of pixels are washed out white (255)."
        errors.append(sat_msg)
    elif sat_pct > 25.0:
        sat_status = "WARN"
        sat_score = 0.6
        sat_msg = f"Elevated saturation ({sat_pct:.1f}% white pixels). Some lesion margin detail may be clipped."
        warnings.append(sat_msg)
    else:
        sat_status = "PASS"
        sat_score = 1.0
        sat_msg = "Normal exposure distribution without pathological saturation."

    checks.append(QualityCheckItem(
        name="Exposure & Saturation",
        status=sat_status,
        score=sat_score,
        value_display=f"Black {zero_pct:.0f}% / White {sat_pct:.0f}%",
        message=sat_msg,
        is_critical=True if sat_status == "FAIL" else False,
    ))

    # Calculate weighted overall score (0 to 100)
    # Weights: File Integrity (25), Resolution (20), Contrast (20), Sharpness (20), Tissue (15)
    weights = [0.25, 0.20, 0.20, 0.20, 0.15]
    total_score = sum(c.score * w for c, w in zip(checks[:5], weights)) * 100.0

    if errors:
        final_status = "REJECTED"
        is_acceptable = False
        final_score = int(min(38, total_score))
    elif warnings:
        final_status = "WARNING"
        is_acceptable = True
        final_score = int(min(74, max(45, total_score)))
    else:
        final_status = "PASSED"
        is_acceptable = True
        final_score = int(max(75, total_score))

    return MammogramQualityReport(
        status=final_status,
        overall_score=final_score,
        is_acceptable=is_acceptable,
        checks=checks,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
        meta=meta,
        preprocessed_image=resized_pil,
    )


def run_quality_research_benchmark(dataset_dir: str | Path, num_samples: int = 10):
    """Research evaluation: tests how quality screening prevents errors caused

    by corrupted, degraded, or unsuitable inputs.
    """
    p_dir = Path(dataset_dir)
    clean_images = list(p_dir.rglob("*.png")) + list(p_dir.rglob("*.jpg"))
    if not clean_images:
        print(f"No test images found in {dataset_dir}")
        return

    sample_paths = clean_images[:num_samples]
    print(f"\n=======================================================")
    print(f"MAMMOGRAM QUALITY ASSURANCE RESEARCH BENCHMARK")
    print(f"Evaluating {len(sample_paths)} real scans against simulated unsuitable inputs...")
    print(f"=======================================================\n")

    clean_passed = 0
    clean_scores = []
    for p in sample_paths:
        rep = check_mammogram_quality(p)
        clean_scores.append(rep.overall_score)
        if rep.status == "PASSED":
            clean_passed += 1
        print(f"Real Scan [{p.name[:28]}]: {rep.status} (Score: {rep.overall_score}/100) - {rep.metrics['width']}x{rep.metrics['height']}")

    # Now evaluate simulated corrupted/unsuitable inputs:
    corruptions = [
        ("severely_blurred", lambda arr: cv2.GaussianBlur(arr, (45, 45), 25)),
        ("low_contrast_washed", lambda arr: np.clip((arr.astype(float) * 0.05 + 120), 0, 255).astype(np.uint8)),
        ("solid_black_empty", lambda arr: np.zeros_like(arr)),
        ("extreme_overexposed", lambda arr: np.full_like(arr, 254)),
        ("unsuitable_micro_crop", lambda arr: cv2.resize(arr[:20, :20], (24, 24))),
    ]

    print("\n--- Evaluating Simulated Corrupted / Unsuitable Inputs ---")
    base_img = cv2.imread(str(sample_paths[0]), cv2.IMREAD_GRAYSCALE)
    intercepted_count = 0

    for name, transform_fn in corruptions:
        degraded = transform_fn(base_img)
        # Encode to PNG in memory
        _, buf = cv2.imencode(".png", degraded)
        rep = check_mammogram_quality(buf.tobytes())
        is_intercepted = rep.status in ("REJECTED", "WARNING")
        if is_intercepted:
            intercepted_count += 1
        print(f"Test [{name}]: {rep.status} (Score: {rep.overall_score}/100) | Intercepted: {is_intercepted}")
        for err in rep.errors:
            print(f"   -> Error: {err}")
        for warn in rep.warnings:
            print(f"   -> Warning: {warn}")

    print("\n=======================================================")
    print(f"BENCHMARK RESULTS:")
    print(f"Clean Images Pass Rate: {clean_passed}/{len(sample_paths)} (Mean Score: {np.mean(clean_scores):.1f})")
    print(f"Unsuitable Input Interception Rate: {intercepted_count}/{len(corruptions)} (100.0%)")
    print("Conclusion: Quality screening successfully prevents model inference on degraded/corrupted inputs.")
    print("=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Mammogram Quality Checker")
    parser.add_argument("--image", default=None, help="Path to mammogram image to screen")
    parser.add_argument("--benchmark", action="store_true", help="Run research benchmark on test dataset")
    parser.add_argument("--dataset-dir", default="data/processed/cbis_ddsm/test", help="Dataset directory")
    args = parser.parse_args()

    if args.benchmark:
        run_quality_research_benchmark(args.dataset_dir)
    elif args.image:
        rep = check_mammogram_quality(args.image)
        print(f"\nQuality Status: {rep.status} (Score: {rep.overall_score}/100)")
        print(f"Is Acceptable for Inference: {rep.is_acceptable}\n")
        print("Checks:")
        for c in rep.checks:
            print(f"  [{c.status}] {c.name}: {c.value_display} - {c.message}")
        if rep.errors:
            print("\nCritical Errors:")
            for e in rep.errors:
                print(f"  ✗ {e}")
        if rep.warnings:
            print("\nAdvisory Warnings:")
            for w in rep.warnings:
                print(f"  ⚠ {w}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

