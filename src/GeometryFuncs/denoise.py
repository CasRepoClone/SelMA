"""Patch denoising utilities.

Supports Gaussian blur (fast) and Non-Local Means (higher quality)
denoising of extracted image patches before feature encoding.
"""

import cv2
import numpy as np

from config import settings


def denoise_patch_gaussian(patch, kernel_size=None):
    kernel_size = kernel_size or settings.DENOISE_GAUSSIAN_KERNEL
    try:
        return cv2.GaussianBlur(patch, (int(kernel_size), int(kernel_size)), 0)
    except cv2.error:
        # Rare OpenCV C++ exception on certain pixel patterns; return as-is
        return patch


def denoise_patch_nlmeans(patch, h=None, template_window=None, search_window=None):
    h = h or settings.DENOISE_H
    template_window = template_window or settings.DENOISE_TEMPLATE_WINDOW
    search_window = search_window or settings.DENOISE_SEARCH_WINDOW

    if len(patch.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(
            patch, None, h, h,
            template_window, search_window
        )
    else:
        return cv2.fastNlMeansDenoising(
            patch, None, h,
            template_window, search_window
        )


def denoise_patch(patch, method=None, **kwargs):
    method = method or settings.DENOISE_METHOD
    if method == "gaussian":
        return denoise_patch_gaussian(patch, **kwargs)
    elif method == "nlmeans":
        return denoise_patch_nlmeans(patch, **kwargs)
    else:
        raise ValueError(f"Unknown denoise method: {method}")


def denoise_patches(patches, **kwargs):
    method = kwargs.get('method') or settings.DENOISE_METHOD
    if method == "gaussian":
        kernel_size = kwargs.get('kernel_size') or settings.DENOISE_GAUSSIAN_KERNEL
        ks = int(kernel_size)
        result = []
        for p in patches:
            try:
                result.append(cv2.GaussianBlur(p, (ks, ks), 0))
            except cv2.error:
                result.append(p)
        return result
    else:
        # Pass method explicitly; filter it from kwargs to avoid duplication
        fwd_kwargs = {k: v for k, v in kwargs.items() if k != 'method'}
        return [denoise_patch(p, method=method, **fwd_kwargs) for p in patches]
