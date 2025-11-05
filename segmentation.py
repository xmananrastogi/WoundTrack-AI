"""Segmentation Module"""
import numpy as np
import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk, binary_closing, binary_opening, remove_small_objects
from typing import Tuple

def apply_entropy_filter(image: np.ndarray, disk_size: int = 10) -> np.ndarray:
    if image.max() <= 1.0:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)
    entropy_img = entropy(image_uint8, disk(disk_size))
    entropy_img = entropy_img.astype(np.float32)
    if entropy_img.max() > entropy_img.min():
        entropy_img = (entropy_img - entropy_img.min()) / (entropy_img.max() - entropy_img.min())
    return entropy_img

def otsu_threshold(image: np.ndarray) -> Tuple[np.ndarray, float]:
    if image.max() <= 1.0:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)
    threshold_value, binary_mask = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask = binary_mask > 0
    return binary_mask, threshold_value / 255.0

def morphological_operations(binary_mask: np.ndarray, closing_size: int = 5, opening_size: int = 3, remove_small: bool = True, min_size: int = 100) -> np.ndarray:
    selem_close = disk(closing_size)
    mask = binary_closing(binary_mask, selem_close)
    selem_open = disk(opening_size)
    mask = binary_opening(mask, selem_open)
    if remove_small:
        mask = remove_small_objects(mask, min_size=min_size)
    return mask

def detect_wound_contours(binary_mask: np.ndarray) -> Tuple[list, np.ndarray]:
    mask_uint8 = (binary_mask.astype(np.uint8) * 255)
    contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def calculate_wound_area(binary_mask: np.ndarray) -> int:
    return np.sum(binary_mask)

def segment_wound_from_array(image: np.ndarray, disk_size: int = 10, apply_morph: bool = True) -> Tuple[np.ndarray, int]:
    from preprocessing import normalize_intensity, apply_gaussian_blur
    if image.max() > 1.0:
        image = normalize_intensity(image)
    image = apply_gaussian_blur(image)
    entropy_img = apply_entropy_filter(image, disk_size=disk_size)
    binary_mask, _ = otsu_threshold(entropy_img)
    wound_mask = ~binary_mask
    if apply_morph:
        wound_mask = morphological_operations(wound_mask)
    wound_area = calculate_wound_area(wound_mask)
    return wound_mask, wound_area

def segment_wound(image_path: str, disk_size: int = 10, normalize: bool = True, apply_morph: bool = True) -> Tuple[np.ndarray, int]:
    from preprocessing import load_and_preprocess_image
    original, processed = load_and_preprocess_image(image_path, normalize=normalize, blur=True)
    entropy_img = apply_entropy_filter(processed, disk_size=disk_size)
    binary_mask, _ = otsu_threshold(entropy_img)
    wound_mask = ~binary_mask
    if apply_morph:
        wound_mask = morphological_operations(wound_mask)
    wound_area = calculate_wound_area(wound_mask)
    return wound_mask, wound_area

if __name__ == "__main__":
    print("Segmentation module loaded!")
