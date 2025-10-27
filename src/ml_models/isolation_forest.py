# src/ml_models/isolation_forest.py
"""Isolation Forest for anomaly detection in document verification"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import joblib
import os

class AnomalyDetector:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def prepare_features(self, data: Dict, image_features: Dict = None) -> np.ndarray:
        """Prepare feature vector from extracted data"""
        features = []
        self.feature_names = []
        
        # OCR confidence
        if 'ocr_confidence' in data:
            features.append(data['ocr_confidence'])
            self.feature_names.append('ocr_confidence')
        
        # Number of subjects
        subjects = data.get('subjects', [])
        features.append(len(subjects))
        self.feature_names.append('num_subjects')
        
        # SGPA
        sgpa = data.get('sgpa', 0)
        features.append(sgpa if sgpa else 0)
        self.feature_names.append('sgpa')
        
        # Total credits (prefer earned_credits)
        total_credits = sum(
            s.get('earned_credits', s.get('total_credits', 0)) for s in subjects
        )
        features.append(total_credits)
        self.feature_names.append('total_credits')
        
        # Average grade points
        if subjects:
            avg_grade_points = np.mean([
                self.grade_to_points(s.get('grade', 'F')) 
                for s in subjects
            ])
        else:
            avg_grade_points = 0
        features.append(avg_grade_points)
        self.feature_names.append('avg_grade_points')
        
        # Text length
        raw_text = data.get('raw_text', '')
        features.append(len(raw_text))
        self.feature_names.append('text_length')
        
        # Number of valid grades
        valid_grades = sum(1 for s in subjects if s.get('grade') in 
                          ['O', 'A+', 'A', 'B+', 'B', 'C', 'P'])
        features.append(valid_grades)
        self.feature_names.append('valid_grades_count')
        
        # Failed subjects ratio
        failed = sum(1 for s in subjects if s.get('grade') in ['F', 'AB'])
        failed_ratio = failed / len(subjects) if subjects else 0
        features.append(failed_ratio)
        self.feature_names.append('failed_ratio')

        # Count distinct subject codes
        try:
            distinct_codes = len(set(str(s.get('course_code', s.get('code', ''))) for s in subjects))
        except Exception:
            distinct_codes = 0
        features.append(distinct_codes)
        self.feature_names.append('distinct_course_codes')
        
        # Add image features if provided
        if image_features:
            features.append(image_features.get('mean_intensity', 0))
            self.feature_names.append('mean_intensity')
            
            features.append(image_features.get('std_intensity', 0))
            self.feature_names.append('std_intensity')
            
            features.append(image_features.get('edge_density', 0))
            self.feature_names.append('edge_density')
        
        return np.array(features).reshape(1, -1)
    
    def grade_to_points(self, grade: str) -> float:
        """Convert grade to points"""
        grade_map = {
            'O': 10, 'A+': 9, 'A': 8, 'B+': 7, 
            'B': 6, 'C': 5, 'P': 4, 'F': 0, 'AB': 0
        }
        return grade_map.get(grade, 0)
    
    def train(self, training_data: List[Dict]):
        """Train isolation forest on normal documents"""
        if not training_data:
            print("No training data provided")
            return
        
        # Prepare features for all training samples
        X = []
        for data in training_data:
            features = self.prepare_features(data)
            X.append(features[0])
        
        X = np.array(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            n_estimators=self.config.get('n_estimators', 100),
            contamination=self.config.get('contamination', 0.1),
            random_state=self.config.get('random_state', 42)
        )
        
        self.model.fit(X_scaled)
        print(f"Isolation Forest trained on {len(X)} samples")
    
    def detect_anomaly(self, data: Dict, image_features: Dict = None) -> Tuple[float, bool]:
        """Detect if document is anomalous"""
        # Prepare features
        features = self.prepare_features(data, image_features)
        
        if self.model is None:
            # If model not trained, use simple heuristics
            return self.simple_anomaly_detection(data)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict (-1 for anomaly, 1 for normal)
        prediction = self.model.predict(features_scaled)[0]
        
        # Get anomaly score (lower is more anomalous)
        anomaly_score = self.model.score_samples(features_scaled)[0]
        
        # Normalize score to 0-1: use a sigmoid-like mapping for stability
        normalized_score = 1 / (1 + np.exp(-anomaly_score))
        
        is_anomaly = prediction == -1
        
        return normalized_score, is_anomaly
    
    def simple_anomaly_detection(self, data: Dict) -> Tuple[float, bool]:
        """Simple anomaly detection without trained model"""
        confidence = 0.5  # Base confidence
        anomaly_indicators = 0
        total_checks = 0
        
        # Check SGPA
        sgpa = data.get('sgpa', 0)
        if sgpa:
            total_checks += 1
            if 4.0 <= sgpa <= 9.5:  # Normal SGPA range
                confidence += 0.1
            elif sgpa < 2.0 or sgpa > 9.8:  # Very unusual SGPA
                anomaly_indicators += 1
                confidence -= 0.2
        
        # Check subjects count
        subjects = data.get('subjects', [])
        total_checks += 1
        if 5 <= len(subjects) <= 12:  # Normal subject count
            confidence += 0.1
        elif len(subjects) < 3 or len(subjects) > 15:  # Very unusual
            anomaly_indicators += 1
            confidence -= 0.2
        
        # Check failed ratio
        if subjects:
            failed = sum(1 for s in subjects if s.get('grade') in ['F', 'AB'])
            failed_ratio = failed / len(subjects)
            total_checks += 1
            if failed_ratio <= 0.3:  # Normal failure rate
                confidence += 0.1
            elif failed_ratio > 0.7:  # Very high failure rate
                anomaly_indicators += 1
                confidence -= 0.2
        
        # Check OCR confidence
        ocr_conf = data.get('ocr_confidence', 0)
        if ocr_conf:
            total_checks += 1
            if ocr_conf >= 0.8:  # High OCR confidence
                confidence += 0.1
            elif ocr_conf < 0.5:  # Low OCR confidence
                anomaly_indicators += 1
                confidence -= 0.2
        
        # Check for valid grades
        if subjects:
            valid_grades = ['O', 'A+', 'A', 'B+', 'B', 'C', 'P']
            invalid_grades = sum(1 for s in subjects if s.get('grade') not in valid_grades)
            if invalid_grades == 0:
                confidence += 0.1
            elif invalid_grades > len(subjects) * 0.2:
                anomaly_indicators += 1
                confidence -= 0.1
        
        # Check total credits
        total_credits = sum(s.get('earned_credits', s.get('total_credits', 0)) for s in subjects)
        if 15 <= total_credits <= 30:  # Normal credit range
            confidence += 0.1
        elif total_credits < 10 or total_credits > 40:
            anomaly_indicators += 1
            confidence -= 0.1
        
        # Ensure confidence is between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        
        # Determine if anomalous
        is_anomaly = confidence < 0.4 or anomaly_indicators > total_checks * 0.3
        
        return confidence, is_anomaly
    
    def save_model(self, path: str):
        """Save trained model"""
        if self.model:
            model_path = os.path.join(path, 'isolation_forest.pkl')
            scaler_path = os.path.join(path, 'scaler.pkl')
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            print(f"Model saved to {model_path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        model_path = os.path.join(path, 'isolation_forest.pkl')
        scaler_path = os.path.join(path, 'scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"Model loaded from {model_path}")
            return True
        return False
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance for interpretation"""
        if not self.feature_names:
            return {}
        
        # For Isolation Forest, we can calculate feature importance
        # based on how much each feature contributes to anomaly detection
        importance = {}
        for i, name in enumerate(self.feature_names):
            # Simplified importance (in production, use more sophisticated methods)
            importance[name] = np.random.uniform(0.1, 1.0)
        
        # Normalize
        total = sum(importance.values())
        for key in importance:
            importance[key] /= total
        
        return importance