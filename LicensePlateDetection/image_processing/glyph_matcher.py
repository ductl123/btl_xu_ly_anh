"""Image features shared by all 30 character classes.

Scores combine pixel overlap, symmetric edge distance, gradient orientations,
original aspect ratio and symmetry. They are not calibrated probabilities.
Reference images must be explicitly labelled; OCR predictions are never learned.
"""
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHARS = "0123456789ABCDEFGHKLMNPSTUVXYZ"
FONT_NAMES = (
    "bahnschrift.ttf", "arialbd.ttf", "ARIALNB.TTF", "impact.ttf",
    "consolab.ttf", "calibrib.ttf", "timesbd.ttf",
)


def tight(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("Character mask must be two-dimensional")
    mask = mask > 0
    ys, xs = np.where(mask)
    if not len(xs):
        return np.zeros((2, 2), dtype=bool)
    return mask[ys.min():ys.max()+1, xs.min():xs.max()+1]


def norm(mask):
    mask = tight(mask)
    return cv2.resize(mask.astype(np.uint8)*255, (24, 40),
                      interpolation=cv2.INTER_AREA) > 100


def features(mask):
    original = tight(mask)
    aspect = original.shape[1] / original.shape[0]
    normalized = norm(original)
    padded = np.pad(normalized.astype(np.uint8)*255, 1)
    edge = cv2.morphologyEx(padded, cv2.MORPH_GRADIENT,
                            np.ones((3, 3), np.uint8)) > 0
    distance = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)
    gray = padded.astype(np.float32) / 255
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude = np.hypot(gx, gy)
    orientation = (np.arctan2(gy, gx) % np.pi) * 9 / np.pi
    cells = []
    for row in np.array_split(np.arange(gray.shape[0]), 5):
        for col in np.array_split(np.arange(gray.shape[1]), 3):
            angles = orientation[np.ix_(row, col)].ravel()
            weights = magnitude[np.ix_(row, col)].ravel()
            lower = np.floor(angles).astype(int) % 9
            fraction = angles - np.floor(angles)
            histogram = (
                np.bincount(lower, weights*(1-fraction), minlength=9)
                + np.bincount((lower+1) % 9, weights*fraction, minlength=9)
            )
            cells.extend(histogram / (np.linalg.norm(histogram)+1e-6))
    hog = np.array(cells, dtype=np.float32)
    hog /= np.linalg.norm(hog) + 1e-6
    pixels = normalized.astype(float)
    structure = np.array([
        np.abs(pixels-pixels[:, ::-1]).mean(),
        np.abs(pixels-pixels[::-1]).mean(),
        np.abs(pixels-pixels[::-1, ::-1]).mean(),
        pixels.mean(),
    ])
    return normalized, edge, distance, hog, aspect, structure


class TemplateMatcher:
    CHARS_ALL = CHARS
    CHARS_DIGITS = CHARS[:10]
    CHARS_LETTERS = CHARS[10:]

    def __init__(self, reference_dir=None, font_names=None):
        self.templates = {char: [] for char in CHARS}
        self.items = []
        self.sources = []
        self.reference_count = 0
        self._cache = OrderedDict()
        for name in FONT_NAMES if font_names is None else font_names:
            path = Path("C:/Windows/Fonts") / name
            if not path.is_file():
                continue
            font = ImageFont.truetype(str(path), 64)
            for char in CHARS:
                image = Image.new("L", (130, 120))
                ImageDraw.Draw(image).text((18, 5), char, font=font, fill=255)
                original = np.array(image)
                for thickness in [-1, 0, 1, 2, 3]:
                    if thickness < 0:
                        variant = cv2.erode(original, np.ones((2, 2), np.uint8))
                    elif thickness > 0:
                        size = 2*thickness+1
                        variant = cv2.dilate(original, np.ones((size, size), np.uint8))
                    else:
                        variant = original
                    for shear in [-.16, 0, .16]:
                        transform = np.float32([[1, shear, -shear*60], [0, 1, 0]])
                        mask = cv2.warpAffine((variant > 100).astype(np.uint8),
                                              transform, (130, 120)) > 0
                        self._add(char, mask, "font:"+name)
        reference_dir = Path(reference_dir) if reference_dir is not None else (
            Path(__file__).resolve().parents[1] / "data/character_templates"
        )
        for char in CHARS:
            for path in sorted((reference_dir/char).glob("*.png")):
                with Image.open(path) as image:
                    mask = np.array(image.convert("L")) > 127
                if not mask.any():
                    raise ValueError(f"Empty labelled reference: {path}")
                for shear in [-.08, 0, .08]:
                    padded = np.pad(mask, ((4, 4), (12, 12)))
                    height, width = padded.shape
                    transform = np.float32([[1, shear, -shear*height/2], [0, 1, 0]])
                    variant = cv2.warpAffine(padded.astype(np.uint8), transform,
                                             (width, height)) > 0
                    self._add(char, variant, "reference:"+path.name)
                self.reference_count += 1
        if any(not samples for samples in self.templates.values()):
            raise RuntimeError("Missing glyph classes; install a supported font or labelled references")
        self.labels = np.array([item[0] for item in self.items])
        self.ns = np.array([item[1][0] for item in self.items])
        self.es = np.array([item[1][1] for item in self.items])
        self.ds = np.array([item[1][2] for item in self.items])
        self.hs = np.array([item[1][3] for item in self.items])
        self.ars = np.array([item[1][4] for item in self.items])
        self.structs = np.array([item[1][5] for item in self.items])

    def _add(self, char, mask, source):
        self.items.append((char, features(mask)))
        self.templates[char].append(mask)
        self.sources.append(source)

    def match_details(self, patch, is_digit_only=False, is_letter_only=False):
        if is_digit_only and is_letter_only:
            raise ValueError("Conflicting character restrictions")
        original = tight(patch)
        if not original.any():
            return {"char": "?", "score": 0., "raw_char": "?", "proposed_char": "?",
                    "rule_applied": False, "top_candidates": [], "margin": 0.,
                    "needs_review": True, "backend": "shape-v2"}
        key = (original.shape, np.packbits(original).tobytes(), is_digit_only, is_letter_only)
        if key in self._cache:
            self._cache.move_to_end(key)
            # Callers must not be able to mutate another character's diagnostics.
            return self._copy_result(self._cache[key])
        n, edge, distance, hog, aspect, structure = features(original)
        overlap = (self.ns & n).sum(axis=(1, 2))
        union = (self.ns | n).sum(axis=(1, 2))
        iou = overlap / np.maximum(union, 1)
        chamfer = .5 * (
            self.ds[:, edge].mean(axis=1)
            + (self.es*distance).sum(axis=(1, 2)) / np.maximum(self.es.sum(axis=(1, 2)), 1)
        )
        boundary = np.exp(-chamfer/2.)
        direction = np.einsum("ij,j->i", self.hs, hog)
        proportions = np.exp(-np.abs(np.log(aspect/self.ars)))
        symmetry = np.exp(-np.abs(self.structs-structure).mean(axis=1)*6.)
        similarities = (.25*iou + .25*boundary + .25*direction
                        + .10*proportions + .15*symmetry)
        scores = {}
        for char in CHARS:
            if (is_digit_only and not char.isdigit()) or (is_letter_only and char.isdigit()):
                continue
            nearest = np.sort(similarities[self.labels == char])[-3:]
            # Exact exemplars must remain consistent. Otherwise require support
            # from several nearby shapes instead of a single accidental match.
            scores[char] = float(nearest[-1] if nearest[-1] >= .985 else nearest.mean())
        ranked = sorted(scores.items(), key=lambda item: -item[1])
        char, score = ranked[0]
        margin = score-ranked[1][1] if len(ranked) > 1 else score
        result = {
            "char": char, "score": round(score, 4), "raw_char": char,
            "proposed_char": char, "rule_applied": False,
            "margin": round(margin, 4), "needs_review": bool(score < .65 or margin < .025),
            "backend": "shape-v2",
            "class_scores": {ch: round(sc, 4) for ch, sc in ranked},
            "top_candidates": [{"char": ch, "score": round(sc, 4)} for ch, sc in ranked[:3]],
        }
        self._cache[key] = result
        if len(self._cache) > 512:
            self._cache.popitem(last=False)
        return self._copy_result(result)

    @staticmethod
    def _copy_result(result):
        return dict(result, class_scores=dict(result.get("class_scores", {})),
                    top_candidates=[dict(item) for item in result["top_candidates"]])

    def match(self, patch, is_digit_only=False, is_letter_only=False):
        details = self.match_details(patch, is_digit_only, is_letter_only)
        return details["char"], details["score"]
