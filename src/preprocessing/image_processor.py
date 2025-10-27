"""Image preprocessing module for document verification"""

import cv2
import numpy as np
from PIL import Image
import imutils

class ImagePreprocessor:
    def __init__(self):
        self.target_size = (1200, 1600)
        
    def preprocess(self, image):
        """Main preprocessing pipeline"""
        if isinstance(image, np.ndarray):
            img = image
        else:
            img = np.array(image)
            
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
            
        # Apply preprocessing steps
        gray = self.denoise(gray)
        gray = self.correct_skew(gray)
        gray = self.enhance_contrast(gray)
        binary = self.binarize(gray)
        
        return binary, gray
    
    def denoise(self, image):
        """Remove noise from image"""
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    def correct_skew(self, image):
        """Correct image skew/rotation"""
        # Detect edges
        edges = cv2.Canny(image, 50, 200, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = (theta * 180 / np.pi) - 90
                if -45 <= angle <= 45:
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                rotated = imutils.rotate_bound(image, median_angle)
                return rotated
        
        return image
    
    def enhance_contrast(self, image):
        """Enhance image contrast using CLAHE"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def binarize(self, image):
        """Apply adaptive thresholding for binarization"""
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    
    def detect_orientation(self, image):
        """Detect if image needs rotation (0, 90, 180, 270 degrees)"""
        # Use OCR to detect text orientation
        import pytesseract
        
        try:
            osd = pytesseract.image_to_osd(image)
            rotation = int(osd.split('\n')[2].split(':')[1].strip())
            
            if rotation != 0:
                image = imutils.rotate_bound(image, -rotation)
                
        except Exception as e:
            print(f"Orientation detection failed: {e}")
            
        return image
    
    def extract_regions(self, image):
        """Extract specific regions of interest from marksheet"""
        regions = {}
        h, w = image.shape[:2]
        
        # Define regions based on typical SPPU marksheet layout
        regions['header'] = image[0:int(h*0.15), :]
        regions['student_info'] = image[int(h*0.15):int(h*0.25), :]
        regions['grades_table'] = image[int(h*0.25):int(h*0.85), :]
        regions['footer'] = image[int(h*0.85):, :]
        
        return regions
