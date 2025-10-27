"""Statistical validation using CNN for forgery detection"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple

class CNNForgeryDetector:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.input_shape = config.get('input_shape', (224, 224, 3))
        
    def build_model(self):
        """Build CNN model for forgery detection"""
        model = models.Sequential([
            # Conv Block 1
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Conv Block 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Conv Block 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Conv Block 4
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.get('learning_rate', 0.001)),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        self.model = model
        return model
    
    def preprocess_image(self, image):
        """Preprocess image for CNN input"""
        # Resize to model input shape
        img = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        
        # Convert to RGB if grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Normalize
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def detect_forgery(self, image) -> Tuple[float, bool]:
        """Detect if document is forged"""
        if self.model is None:
            # For demo, use image analysis to provide reasonable confidence
            return self.analyze_image_quality(image)
        
        # Preprocess image
        processed_img = self.preprocess_image(image)
        
        # Predict
        prediction = self.model.predict(processed_img, verbose=0)[0][0]
        
        # Determine if forged (1 = forged, 0 = authentic)
        is_forged = prediction > self.config.get('threshold', 0.5)
        
        return float(prediction), is_forged
    
    def analyze_image_quality(self, image) -> Tuple[float, bool]:
        """Analyze image quality to provide confidence score"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate image quality metrics
            # 1. Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 2. Contrast
            contrast = gray.std()
            
            # 3. Brightness distribution
            brightness_mean = gray.mean()
            brightness_std = gray.std()
            
            # 4. Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Calculate confidence based on quality metrics
            confidence = 0.5  # Base confidence
            
            # Adjust based on sharpness
            if laplacian_var > 100:
                confidence += 0.2
            elif laplacian_var < 20:
                confidence -= 0.3
            
            # Adjust based on contrast
            if contrast > 50:
                confidence += 0.1
            elif contrast < 20:
                confidence -= 0.2
            
            # Adjust based on brightness
            if 80 <= brightness_mean <= 180:
                confidence += 0.1
            else:
                confidence -= 0.1
            
            # Adjust based on edge density
            if 0.05 <= edge_density <= 0.3:
                confidence += 0.1
            else:
                confidence -= 0.1
            
            # Ensure confidence is between 0 and 1
            confidence = max(0.0, min(1.0, confidence))
            
            # Determine if potentially forged (low confidence = potentially forged)
            is_forged = confidence < 0.4
            
            return confidence, is_forged
            
        except Exception as e:
            print(f"Error in image quality analysis: {e}")
            return 0.5, False
    
    def extract_features(self, image):
        """Extract statistical features from image"""
        features = {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Texture features
        features['mean_intensity'] = np.mean(gray)
        features['std_intensity'] = np.std(gray)
        features['contrast'] = gray.max() - gray.min()
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # Frequency domain features
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        features['freq_energy'] = np.sum(magnitude_spectrum)
        
        return features