"""General utility functions for the document verification system"""

import cv2
import numpy as np
from PIL import Image
import os
from typing import Union, Optional

def load_image(file_path: str) -> Optional[np.ndarray]:
    """Load image from file path"""
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        # Load image using OpenCV
        image = cv2.imread(file_path)
        if image is None:
            # Try with PIL as fallback
            pil_image = Image.open(file_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return image
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

def save_image(image: np.ndarray, file_path: str) -> bool:
    """Save image to file path"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save image
        success = cv2.imwrite(file_path, image)
        if success:
            print(f"Image saved to: {file_path}")
        else:
            print(f"Failed to save image to: {file_path}")
        
        return success
    except Exception as e:
        print(f"Error saving image to {file_path}: {e}")
        return False

def resize_image(image: np.ndarray, target_size: tuple = (800, 600)) -> np.ndarray:
    """Resize image while maintaining aspect ratio"""
    try:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return resized
    except Exception as e:
        print(f"Error resizing image: {e}")
        return image

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale"""
    try:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    except Exception as e:
        print(f"Error converting to grayscale: {e}")
        return image

def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """Enhance image quality for better OCR"""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Apply sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    except Exception as e:
        print(f"Error enhancing image: {e}")
        return image

def validate_image_format(image: np.ndarray) -> bool:
    """Validate if image is in correct format"""
    try:
        if image is None:
            return False
        
        if len(image.shape) not in [2, 3]:
            return False
        
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            return False
        
        return True
    except Exception as e:
        print(f"Error validating image format: {e}")
        return False

def get_image_info(image: np.ndarray) -> dict:
    """Get basic information about the image"""
    try:
        if image is None:
            return {}
        
        info = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min_value': float(np.min(image)),
            'max_value': float(np.max(image)),
            'mean_value': float(np.mean(image)),
            'is_grayscale': len(image.shape) == 2
        }
        
        return info
    except Exception as e:
        print(f"Error getting image info: {e}")
        return {}

def create_thumbnail(image: np.ndarray, size: tuple = (200, 200)) -> np.ndarray:
    """Create thumbnail of the image"""
    try:
        return resize_image(image, size)
    except Exception as e:
        print(f"Error creating thumbnail: {e}")
        return image

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to 0-1 range"""
    try:
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            return image.astype(np.float32) / 65535.0
        else:
            return image.astype(np.float32)
    except Exception as e:
        print(f"Error normalizing image: {e}")
        return image

def denormalize_image(image: np.ndarray, dtype: np.dtype = np.uint8) -> np.ndarray:
    """Denormalize image back to original range"""
    try:
        if dtype == np.uint8:
            return (image * 255).astype(np.uint8)
        elif dtype == np.uint16:
            return (image * 65535).astype(np.uint16)
        else:
            return image.astype(dtype)
    except Exception as e:
        print(f"Error denormalizing image: {e}")
        return image

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by given angle"""
    try:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        return rotated
    except Exception as e:
        print(f"Error rotating image: {e}")
        return image

def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """Crop image to specified region"""
    try:
        h, w = image.shape[:2]
        
        # Ensure coordinates are within bounds
        x = max(0, min(x, w))
        y = max(0, min(y, h))
        width = min(width, w - x)
        height = min(height, h - y)
        
        return image[y:y+height, x:x+width]
    except Exception as e:
        print(f"Error cropping image: {e}")
        return image

def merge_images_horizontally(images: list) -> np.ndarray:
    """Merge multiple images horizontally"""
    try:
        if not images:
            return None
        
        # Resize all images to same height
        target_height = min(img.shape[0] for img in images)
        resized_images = []
        
        for img in images:
            if img.shape[0] != target_height:
                scale = target_height / img.shape[0]
                new_width = int(img.shape[1] * scale)
                resized = cv2.resize(img, (new_width, target_height))
                resized_images.append(resized)
            else:
                resized_images.append(img)
        
        # Concatenate horizontally
        merged = np.hstack(resized_images)
        
        return merged
    except Exception as e:
        print(f"Error merging images: {e}")
        return None

def merge_images_vertically(images: list) -> np.ndarray:
    """Merge multiple images vertically"""
    try:
        if not images:
            return None
        
        # Resize all images to same width
        target_width = min(img.shape[1] for img in images)
        resized_images = []
        
        for img in images:
            if img.shape[1] != target_width:
                scale = target_width / img.shape[1]
                new_height = int(img.shape[0] * scale)
                resized = cv2.resize(img, (target_width, new_height))
                resized_images.append(resized)
            else:
                resized_images.append(img)
        
        # Concatenate verticallybb
        merged = np.vstack(resized_images)
        
        return merged
    except Exception as e:
        print(f"Error merging images: {e}")
        return None
