"""Cryptographic and perceptual hashing helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fftpack import dct


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: Path, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """64-bit DCT perceptual hash (pHash).

    The image is converted to grayscale, resized to
    ``hash_size * highfreq_factor`` square, DCT-transformed, and the top-left
    ``hash_size`` coefficients (excluding DC) are compared with their median.
    """
    image_size = hash_size * highfreq_factor
    with Image.open(path) as image:
        gray = image.convert("L").resize((image_size, image_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float64)
    coefficients = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = coefficients[:hash_size, :hash_size]
    median = np.median(low[1:, :])
    bits = (low > median).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()
