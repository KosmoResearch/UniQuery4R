"""Image loading / preprocessing for UniQuery4R inference."""

from __future__ import annotations

import glob
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from torchvision import transforms as TF

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.PNG")


def collect_image_paths(input_path: str) -> List[str]:
    """Expand a file, directory, or glob pattern into a sorted list of images."""
    if os.path.isdir(input_path):
        paths: List[str] = []
        for ext in IMAGE_EXTENSIONS:
            paths.extend(glob.glob(os.path.join(input_path, ext)))
        return sorted(paths)
    if any(ch in input_path for ch in "*?[]"):
        return sorted(glob.glob(input_path))
    if os.path.isfile(input_path):
        return [input_path]
    raise FileNotFoundError(f"Input not found: {input_path}")


def load_and_preprocess_images(
    image_path_list: Sequence[str],
    max_size: int = 504,
    patch_size: int = 14,
    mode: str = "keep_ratio",
    num_workers: int = 16,
) -> torch.Tensor:
    """Load images and preprocess them for UniQuery4R inference.

    Uses a keep-ratio resize (no upscaling) with sides divisible by
    ``patch_size`` and outputs float tensors in [0, 1] with shape [N, 3, H, W];
    the encoder applies ImageNet normalization internally.

    Images are read/resized in a thread pool (IO + decode bound); results preserve
    the input order. Set ``num_workers <= 1`` to load sequentially.
    """
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    if mode not in ("keep_ratio", "square"):
        raise ValueError("mode must be 'keep_ratio' or 'square'")
    if max_size % patch_size != 0:
        raise ValueError("max_size must be divisible by patch_size")

    to_tensor = TF.ToTensor()

    def process_one(image_path: str) -> torch.Tensor:
        image = _load_rgb_image(image_path)
        width, height = image.size

        if mode == "square":
            target_h = target_w = max_size
        else:
            target_h, target_w = _keep_ratio_target_shape(height, width, max_size, patch_size)

        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        # ToTensor -> [0, 1]; the encoder applies ImageNet normalization internally.
        return to_tensor(image)

    workers = min(max(1, int(num_workers)), len(image_path_list))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            images = list(executor.map(process_one, image_path_list))
    else:
        images = [process_one(p) for p in image_path_list]

    shapes = {(tensor.shape[1], tensor.shape[2]) for tensor in images}

    if len(shapes) > 1:
        warnings.warn(
            f"Found images with different shapes: {shapes}; padding to a common size.",
            stacklevel=2,
        )
        images = _pad_images_to_common_size(images, shapes, pad_value=0.0)

    return torch.stack(images)


def _load_rgb_image(image_path: str) -> Image.Image:
    with Image.open(image_path) as image:
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        return image.convert("RGB")


def _keep_ratio_target_shape(
    height: int,
    width: int,
    max_size: int,
    patch_size: int,
) -> Tuple[int, int]:
    # Match ResizeKeepRatio(low_resolution=True): do not upscale small images.
    if width < max_size and height < max_size:
        new_width = int(width / patch_size) * patch_size
        new_height = int(height / patch_size) * patch_size
        new_width = max(patch_size, new_width)
        new_height = max(patch_size, new_height)
        return new_height, new_width

    if width >= height:
        new_width = max_size
        new_height = int(round(height * (new_width / width) / patch_size)) * patch_size
    else:
        new_height = max_size
        new_width = int(round(width * (new_height / height) / patch_size)) * patch_size

    new_width = max(patch_size, new_width)
    new_height = max(patch_size, new_height)
    return new_height, new_width


def _pad_images_to_common_size(images, shapes, pad_value: float = 0.0):
    max_height = max(shape[0] for shape in shapes)
    max_width = max(shape[1] for shape in shapes)

    padded = []
    for image in images:
        h_padding = max_height - image.shape[1]
        w_padding = max_width - image.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            image = torch.nn.functional.pad(
                image,
                (pad_left, pad_right, pad_top, pad_bottom),
                value=pad_value,
            )
        padded.append(image)
    return padded
