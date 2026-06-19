"""Lightweight computer-vision pre-flight checks on claim images."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat


BLUR_THRESHOLD = 35.0
DARK_THRESHOLD = 45.0
BRIGHT_THRESHOLD = 215.0
GLARE_FRACTION = 0.12
MIN_DIMENSION = 180


@dataclass
class ImagePreflightResult:
    image_id: str
    path: str
    blur_score: float = 0.0
    is_blurry: bool = False
    brightness: float = 0.0
    is_dark: bool = False
    is_glare: bool = False
    width: int = 0
    height: int = 0
    is_low_resolution: bool = False
    aspect_ratio: float = 1.0
    is_extreme_crop: bool = False
    perceptual_hash: str = ""
    usable: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreflightReport:
    images: list[ImagePreflightResult] = field(default_factory=list)
    duplicate_groups: list[list[str]] = field(default_factory=list)
    suggested_risk_flags: list[str] = field(default_factory=list)
    all_unusable: bool = False
    summary_text: str = ""

    def to_dict(self) -> dict:
        return {
            "images": [img.to_dict() for img in self.images],
            "duplicate_groups": self.duplicate_groups,
            "suggested_risk_flags": self.suggested_risk_flags,
            "all_unusable": self.all_unusable,
            "summary_text": self.summary_text,
        }


def _average_hash(img: Image.Image, size: int = 8) -> str:
    gray = img.convert("L").resize((size, size))
    pixels = list(gray.getdata())
    avg = sum(pixels) / max(len(pixels), 1)
    return "".join("1" if pixel >= avg else "0" for pixel in pixels)


def _hamming(a: str, b: str) -> int:
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


def _blur_score(img: Image.Image) -> float:
    gray = img.convert("L")
    if max(gray.size) > 640:
        gray = gray.copy()
        gray.thumbnail((640, 640))
    edges = gray.filter(ImageFilter.FIND_EDGES)
    pixels = list(edges.getdata())
    if not pixels:
        return 0.0
    mean = sum(pixels) / len(pixels)
    return sum((pixel - mean) ** 2 for pixel in pixels) / len(pixels)


def _glare_fraction(img: Image.Image) -> float:
    gray = img.convert("L")
    pixels = list(gray.getdata())
    if not pixels:
        return 0.0
    bright = sum(1 for pixel in pixels if pixel >= BRIGHT_THRESHOLD)
    return bright / len(pixels)


def analyze_image(image_id: str, image_path: Path) -> ImagePreflightResult:
    """Run pre-flight checks on a single image file."""
    result = ImagePreflightResult(image_id=image_id, path=str(image_path))
    try:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            result.width, result.height = rgb.size
            result.aspect_ratio = result.width / max(result.height, 1)
            result.is_low_resolution = min(result.width, result.height) < MIN_DIMENSION
            result.is_extreme_crop = result.aspect_ratio > 3.5 or result.aspect_ratio < 0.28

            stat = ImageStat.Stat(rgb.convert("L"))
            result.brightness = float(stat.mean[0])
            result.is_dark = result.brightness < DARK_THRESHOLD
            result.is_glare = _glare_fraction(rgb) >= GLARE_FRACTION

            result.blur_score = _blur_score(rgb)
            result.is_blurry = result.blur_score < BLUR_THRESHOLD
            result.perceptual_hash = _average_hash(rgb)

            result.usable = not (
                result.is_blurry and result.is_dark and result.is_low_resolution
            )
    except OSError:
        result.usable = False
        result.is_low_resolution = True
    return result


def analyze_images(entries: list[tuple[str, Path]]) -> PreflightReport:
    """Analyze all images for a claim and produce a combined pre-flight report."""
    report = PreflightReport()
    if not entries:
        report.all_unusable = True
        report.summary_text = "No readable images were supplied."
        return report

    for image_id, path in entries:
        report.images.append(analyze_image(image_id, path))

    hash_groups: dict[str, list[str]] = {}
    for image in report.images:
        if image.perceptual_hash:
            hash_groups.setdefault(image.perceptual_hash, []).append(image.image_id)

    for group in hash_groups.values():
        if len(group) > 1:
            report.duplicate_groups.append(group)

    flags: set[str] = set()
    blurry_count = sum(1 for image in report.images if image.is_blurry)
    dark_or_glare_count = sum(
        1 for image in report.images if image.is_dark or image.is_glare
    )
    crop_count = sum(
        1
        for image in report.images
        if image.is_extreme_crop or image.is_low_resolution
    )

    if blurry_count == len(report.images):
        flags.add("blurry_image")
    elif blurry_count:
        flags.add("blurry_image")

    if dark_or_glare_count:
        flags.add("low_light_or_glare")

    if crop_count:
        flags.add("cropped_or_obstructed")

    if report.duplicate_groups:
        flags.add("possible_manipulation")

    usable_count = sum(1 for image in report.images if image.usable)
    report.all_unusable = usable_count == 0
    report.suggested_risk_flags = sorted(flags)

    lines = []
    for image in report.images:
        issues = []
        if image.is_blurry:
            issues.append(f"blur={image.blur_score:.1f}")
        if image.is_dark or image.is_glare:
            issues.append(f"brightness={image.brightness:.0f}")
        if image.is_low_resolution or image.is_extreme_crop:
            issues.append(f"size={image.width}x{image.height}")
        status = ", ".join(issues) if issues else "acceptable quality"
        lines.append(f"- {image.image_id}: {status}")

    if report.duplicate_groups:
        lines.append(f"- duplicate image groups: {report.duplicate_groups}")

    report.summary_text = "\n".join(lines)
    return report
