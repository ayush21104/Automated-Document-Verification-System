# config.py
"""Configuration file for the Document Verification System"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
SAMPLE_DIR = os.path.join(DATA_DIR, 'sample_marksheets')
AUGMENTED_DIR = os.path.join(DATA_DIR, 'augmented_data')

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, SAMPLE_DIR, AUGMENTED_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# OCR Configuration
TESSERACT_CONFIG = '--oem 3 --psm 6'
OCR_LANGUAGES = 'eng'

# SPPU Marksheet Specific Rules
SPPU_CONFIG = {
    'university_name': 'SAVITRIBAI PHULE PUNE UNIVERSITY',
    'institute_codes': ['95', '96', '97', '98', '99'],  # Valid institute codes
    'valid_grades': ['O', 'A+', 'A', 'B+', 'B', 'C', 'P', 'F', 'AB', 'AC'],
    'grade_points': {
        'O': 10, 'A+': 9, 'A': 8, 'B+': 7, 'B': 6, 'C': 5, 'P': 4, 'F': 0, 'AB': 0, 'AC': 0
    },
    'prn_pattern': r'^\d{8}[A-Z]?$',  # Updated PRN pattern for SPPU
    'seat_number_pattern': r'^F\d{9}$',  # Seat number pattern
    'branch_codes': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'],  # Valid branch codes
    'document_number_pattern': r'^\d{2}-\d{7}$',  # Document number pattern
    'sgpa_range': (0.0, 10.0),
    'credit_range': (0, 30),
    'subject_codes_pattern': r'^\d{6}$',  # Updated for 6-digit course codes
    'course_types': ['TH', 'PR', 'TW', 'AC'],  # Theory, Practical, Tutorial, Audit Course
    'valid_exam_periods': ['JAN/FEB', 'MAR/APR', 'MAY/JUN', 'JUL/AUG', 'SEP/OCT', 'OCT/NOV', 'NOV/DEC']
}

# Validation Thresholds
VALIDATION_THRESHOLDS = {
    'ocr_confidence': 0.60,  # Further reduced
    'rule_match_score': 0.65,  # Further reduced
    'cnn_forgery_threshold': 0.75,
    'isolation_forest_threshold': -0.5,
    'nlp_similarity_threshold': 0.60,  # Further reduced
    'overall_confidence_high': 0.65,  # For "Verified" - much more lenient
    'overall_confidence_medium': 0.40  # For "Needs Review" - much more lenient
}

# CNN Model Configuration
CNN_CONFIG = {
    'input_shape': (224, 224, 3),
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001
}

# Isolation Forest Configuration
ISOLATION_FOREST_CONFIG = {
    'n_estimators': 100,
    'contamination': 0.1,
    'random_state': 42
}

# NLP Configuration
NLP_CONFIG = {
    'model_name': 'en_core_web_sm',
    'subject_keywords': [
        'database', 'computation', 'programming', 'systems', 'networks',
        'security', 'laboratory', 'practice', 'seminar', 'skills'
    ]
}

# Augmentation Configuration
AUGMENTATION_CONFIG = {
    'rotation_range': (-15, 15),
    'scale_range': (0.9, 1.1),
    'brightness_range': (0.8, 1.2),
    'noise_probability': 0.3,
    'blur_probability': 0.2
}

# Database Configuration
DB_CONFIG = {
    'db_path': os.path.join(DATA_DIR, 'verification_records.db'),
    'table_name': 'verification_history'
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.path.join(BASE_DIR, 'verification.log')
}
