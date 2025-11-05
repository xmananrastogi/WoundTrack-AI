"""Preprocessing Module"""
import numpy as np
import cv2

def normalize_intensity(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    i_min = np.min(image)
    i_max = np.max(image)
    if i_max == i_min:
        return np.zeros_like(image)
    return (image - i_min) / (i_max - i_min)

def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def enhance_contrast(image: np.ndarray, method: str = 'clahe') -> np.ndarray:
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
    else:
        image_uint8 = image
    if method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image_uint8)
    elif method == 'equalize':
        enhanced = cv2.equalizeHist(image_uint8)
    else:
        raise ValueError(f"Unknown method: {method}")
    return enhanced.astype(np.float32) / 255.0

def load_and_preprocess_image(image_path: str, normalize: bool = True, blur: bool = True, kernel_size: int = 5) -> tuple:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load {image_path}")
    original = image.copy()
    processed = image.astype(np.float32)
    if normalize:
        processed = normalize_intensity(processed)
    if blur:
        processed = apply_gaussian_blur(processed, kernel_size=kernel_size)
    return original, processed

if __name__ == "__main__":
    print("Preprocessing module loaded!")
