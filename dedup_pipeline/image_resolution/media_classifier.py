"""Phase 1 heuristic media classifier: portrait vs identity-document layout.

Uses only Pillow (already required). OpenCV is optional and unused in Phase 1 so
the base test environment can run without the face stack. The classifier is
conservative: when signals conflict or are weak the label is UNKNOWN.

This is not a calibrated accuracy claim. Evaluate on an authorised labelled set
before treating the heuristic as reliable in production operations.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from PIL import Image, ImageFilter, ImageStat


class MediaClass(str, Enum):
    IDENTITY_DOCUMENT = "IDENTITY_DOCUMENT"
    CANDIDATE_PORTRAIT = "CANDIDATE_PORTRAIT"
    UNKNOWN = "UNKNOWN"


# Confident decisions only. Below these floors the result is UNKNOWN.
DOCUMENT_CONFIDENCE_MIN = 0.70
PORTRAIT_CONFIDENCE_MIN = 0.70
# Horizontal luminance bands at/above this level are document-like. A portrait
# is accepted despite them only when the frame is clearly taller than wide.
BAND_CONFLICT = 28.0
PORTRAIT_ASPECT_MAX_WITH_BANDS = 0.95


@dataclass(frozen=True)
class MediaClassification:
    label: MediaClass
    confidence: float
    reason: str
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label.value,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "signals": self.signals,
        }


def _skin_mask(pixels: list[tuple[int, int, int]]) -> list[bool]:
    out = []
    for r, g, b in pixels:
        # Loose YCbCr-ish skin gate used only as a layout cue, not as identity evidence.
        out.append(r > 95 and g > 40 and b > 20 and r > g and r > b and abs(r - g) > 15)
    return out


def _largest_blob_fraction(mask: list[bool], width: int, height: int) -> float:
    """4-connected component size of the largest True region, as a fraction of pixels."""
    if not mask or width < 1 or height < 1:
        return 0.0
    seen = [False] * len(mask)
    best = 0

    def idx(x: int, y: int) -> int:
        return y * width + x

    for y in range(height):
        for x in range(width):
            start = idx(x, y)
            if not mask[start] or seen[start]:
                continue
            stack = [start]
            seen[start] = True
            size = 0
            while stack:
                current = stack.pop()
                size += 1
                cy = current // width
                cx = current % width
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < width and 0 <= ny < height:
                        n = ny * width + nx
                        if mask[n] and not seen[n]:
                            seen[n] = True
                            stack.append(n)
            best = max(best, size)
    return best / float(len(mask))


def _band_contrast(gray_rows: list[float]) -> float:
    if len(gray_rows) < 6:
        return 0.0
    third = max(1, len(gray_rows) // 3)
    bands = [
        sum(gray_rows[:third]) / third,
        sum(gray_rows[third:2 * third]) / max(1, len(gray_rows[third:2 * third])),
        sum(gray_rows[2 * third:]) / max(1, len(gray_rows[2 * third:])),
    ]
    return max(bands) - min(bands)


def classify_media_image(data: bytes, width: Optional[int] = None, height: Optional[int] = None) -> MediaClassification:
    """Classify decoded image bytes. Call only after format validation succeeded."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            rgb = image.convert("RGB")
            if width is None or height is None:
                width, height = rgb.size
            # Work on a small grid so the heuristic stays cheap and deterministic.
            sample_w, sample_h = 64, 64
            sample = rgb.resize((sample_w, sample_h), Image.Resampling.BILINEAR)
            gray = sample.convert("L")
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_mean = float(ImageStat.Stat(edges).mean[0])
            luma_std = float(ImageStat.Stat(gray).stddev[0])
            pixels = [sample.getpixel((x, y)) for y in range(sample_h) for x in range(sample_w)]
            aspect = float(width) / float(height) if height else 1.0

            # Horizontal mean luminance profile (document header/footer bands).
            row_means = []
            for y in range(sample_h):
                row = pixels[y * sample_w:(y + 1) * sample_w]
                row_means.append(sum(p[0] * 0.299 + p[1] * 0.587 + p[2] * 0.114 for p in row) / sample_w)
            band_contrast = _band_contrast(row_means)

            skin = _skin_mask(pixels)
            skin_frac = sum(1 for bit in skin if bit) / float(len(skin))
            # Centre vs corners: portraits put the face near the middle.
            centre = []
            corners = []
            for y in range(sample_h):
                for x in range(sample_w):
                    bit = skin[y * sample_w + x]
                    if sample_w * 0.25 <= x < sample_w * 0.75 and sample_h * 0.15 <= y < sample_h * 0.75:
                        centre.append(bit)
                    if (x < sample_w * 0.2 or x >= sample_w * 0.8) and (y < sample_h * 0.2 or y >= sample_h * 0.8):
                        corners.append(bit)
            centre_skin = (sum(centre) / len(centre)) if centre else 0.0
            corner_skin = (sum(corners) / len(corners)) if corners else 0.0
            blob = _largest_blob_fraction(skin, sample_w, sample_h)

            # Count coarse skin peaks along x to flag group-like layouts.
            col_skin = [0.0] * sample_w
            for y in range(sample_h):
                for x in range(sample_w):
                    if skin[y * sample_w + x]:
                        col_skin[x] += 1.0
            peaks = 0
            active = False
            for value in col_skin:
                if value > sample_h * 0.12:
                    if not active:
                        peaks += 1
                        active = True
                else:
                    active = False
    except Exception as exc:  # noqa: BLE001 - any decode/layout failure is UNKNOWN
        return MediaClassification(
            MediaClass.UNKNOWN, 0.0,
            f"classification failed ({type(exc).__name__}); media left for manual review",
            {"error": type(exc).__name__},
        )

    signals = {
        "aspect_ratio": round(aspect, 4),
        "luma_std": round(luma_std, 4),
        "edge_mean": round(edge_mean, 4),
        "band_contrast": round(band_contrast, 4),
        "skin_fraction": round(skin_frac, 4),
        "centre_skin": round(centre_skin, 4),
        "corner_skin": round(corner_skin, 4),
        "largest_skin_blob": round(blob, 4),
        "skin_column_peaks": peaks,
        "width": width,
        "height": height,
    }

    if luma_std < 8.0 or edge_mean < 2.0:
        return MediaClassification(
            MediaClass.UNKNOWN, 0.55,
            "image is nearly blank or has almost no structure; not safe for face matching",
            signals,
        )

    if peaks >= 3 and skin_frac > 0.08:
        return MediaClassification(
            MediaClass.UNKNOWN, 0.65,
            "several skin-like regions suggest a group or busy scene; left for manual review",
            signals,
        )

    document_score = 0.0
    document_reasons: list[str] = []
    # ID cards are usually landscape and show strong horizontal banding / structured edges.
    if 1.35 <= aspect <= 1.95:
        document_score += 0.35
        document_reasons.append("landscape card-like aspect")
    if band_contrast >= 28.0:
        document_score += 0.30
        document_reasons.append("strong horizontal luminance bands")
    if edge_mean >= 18.0 and skin_frac < 0.22:
        document_score += 0.25
        document_reasons.append("dense edge structure with limited skin area")
    if corner_skin < 0.05 and centre_skin < 0.18 and edge_mean >= 14.0:
        document_score += 0.15
        document_reasons.append("little central face-like colour")

    portrait_score = 0.0
    portrait_reasons: list[str] = []
    if 0.55 <= aspect <= 1.15:
        portrait_score += 0.25
        portrait_reasons.append("near-square or portrait aspect")
    if centre_skin >= 0.18 and centre_skin > corner_skin + 0.08:
        portrait_score += 0.35
        portrait_reasons.append("skin-like colour concentrated in the centre")
    if 0.12 <= blob <= 0.55:
        portrait_score += 0.25
        portrait_reasons.append("one dominant skin-like region of face-like size")
    if skin_frac >= 0.10 and edge_mean < 35.0:
        portrait_score += 0.15
        portrait_reasons.append("moderate skin area without document-like edge density")

    # Prefer UNKNOWN when both fire or neither is confident.
    if document_score >= DOCUMENT_CONFIDENCE_MIN and document_score >= portrait_score + 0.10:
        return MediaClassification(
            MediaClass.IDENTITY_DOCUMENT, min(0.95, document_score),
            "heuristic identity-document layout: " + "; ".join(document_reasons),
            signals,
        )
    if portrait_score >= PORTRAIT_CONFIDENCE_MIN and portrait_score >= document_score + 0.10:
        # D1: a coloured header/footer band plus a central skin-like block can look like a
        # portrait. Require a clearly taller-than-wide frame before trusting portrait over bands.
        if band_contrast >= BAND_CONFLICT and aspect >= PORTRAIT_ASPECT_MAX_WITH_BANDS:
            return MediaClassification(
                MediaClass.UNKNOWN, max(document_score, portrait_score, 0.50),
                "document-like horizontal bands conflict with portrait evidence; left for manual review",
                {**signals, "document_score": round(document_score, 4), "portrait_score": round(portrait_score, 4)},
            )
        return MediaClassification(
            MediaClass.CANDIDATE_PORTRAIT, min(0.95, portrait_score),
            "heuristic candidate-portrait layout: " + "; ".join(portrait_reasons),
            signals,
        )
    return MediaClassification(
        MediaClass.UNKNOWN, max(document_score, portrait_score, 0.40),
        "signals are weak or conflicting; left for manual review instead of face matching",
        {**signals, "document_score": round(document_score, 4), "portrait_score": round(portrait_score, 4)},
    )
