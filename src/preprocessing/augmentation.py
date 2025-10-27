"""Data augmentation module for robust verification"""

import cv2
import numpy as np
import albumentations as A
from albumentations import Compose

class DataAugmentor:
    def __init__(self, config=None):
        self.config = config or {}
        self.augmentation_pipeline = self.create_pipeline()
    
    def create_pipeline(self):
        """Create augmentation pipeline for training"""
        return Compose([
            A.Rotate(limit=15, p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-10, 10),
                shear=(-5, 5),
                p=0.5
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.OpticalDistortion(distort_limit=0.05, p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        ])
    
    def augment(self, image):
        """Apply augmentation to image"""
        augmented = self.augmentation_pipeline(image=image)
        return augmented['image']
    
    def generate_augmented_samples(self, image, num_samples=10):
        """Generate multiple augmented versions of an image"""
        samples = []
        for _ in range(num_samples):
            augmented = self.augment(image)
            samples.append(augmented)
        return samples
    
    def apply_rotation(self, image, angle):
        """Apply specific rotation to image"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), 
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
        return rotated
    
    def apply_mirror(self, image, axis='horizontal'):
        """Apply mirror transformation"""
        if axis == 'horizontal':
            return cv2.flip(image, 1)
        elif axis == 'vertical':
            return cv2.flip(image, 0)
        else:
            return cv2.flip(image, -1) 