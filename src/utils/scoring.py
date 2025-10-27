"""Confidence scoring and decision making module"""

import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime

class ConfidenceScorer:
    def __init__(self, config):
        self.config = config
        self.weights = {
            'ocr': 0.15,
            'rule_based': 0.25,
            'cnn': 0.20,
            'isolation_forest': 0.20,
            'nlp': 0.20
        }
    
    def calculate_overall_confidence(self, scores: Dict) -> float:
        """Calculate weighted overall confidence score"""
        weighted_sum = 0
        total_weight = 0
        
        # Map actual score keys to component names
        score_mapping = {
            'ocr': 'ocr_confidence',
            'rule_based': 'rule_based_score', 
            'cnn': 'cnn_score',
            'isolation_forest': 'isolation_score',
            'nlp': 'nlp_score'
        }
        
        for component, weight in self.weights.items():
            score_key = score_mapping.get(component, component)
            if score_key in scores and scores[score_key] is not None:
                score_value = scores[score_key]
                # Ensure score is between 0 and 1
                if score_value > 1.0:
                    score_value = score_value / 100.0
                weighted_sum += score_value * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.0
    
    def determine_verification_status(self, confidence: float, anomalies: Dict) -> Tuple[str, str]:
        """Determine final verification status"""
        high_threshold = self.config['overall_confidence_high']
        medium_threshold = self.config['overall_confidence_medium']
        
        # Check for critical anomalies
        critical_anomalies = []
        
        if anomalies.get('cnn_forged', False):
            critical_anomalies.append("CNN detected forgery")
        
        if anomalies.get('isolation_anomaly', False):
            critical_anomalies.append("Anomaly detected by Isolation Forest")
        
        if anomalies.get('rule_violations', []):
            critical_anomalies.extend(anomalies['rule_violations'])
        
        # Decision logic - prioritize confidence score over minor anomalies
        if confidence >= high_threshold:
            if critical_anomalies and len(critical_anomalies) > 2:
                # Only downgrade if there are many critical issues
                return "Needs Review", f"High confidence but multiple issues: {', '.join(critical_anomalies[:3])}"
            else:
                return "Verified", "Document passed validation checks"
        elif confidence >= medium_threshold:
            if critical_anomalies and len(critical_anomalies) > 1:
                return "Needs Review", f"Some issues detected: {', '.join(critical_anomalies[:2])}"
            else:
                return "Verified", "Document passed validation checks"
        else:
            if critical_anomalies:
                return "Potentially Fraudulent", f"Low confidence and issues: {', '.join(critical_anomalies[:2])}"
            else:
                return "Potentially Fraudulent", "Low confidence score across multiple checks"
    
    def generate_detailed_report(self, all_results: Dict) -> Dict:
        """Generate detailed verification report"""
        report = {
            'timestamp': str(datetime.now()),
            'document_info': {
                'prn': all_results.get('prn', 'N/A'),
                'student_name': all_results.get('student_name', 'N/A'),
                'university': all_results.get('university_name', 'N/A')
            },
            'verification_scores': {
                'ocr_confidence': all_results.get('ocr_confidence', 0),
                'rule_based_score': all_results.get('rule_based_score', 0),
                'cnn_forgery_score': all_results.get('cnn_score', 0),
                'isolation_forest_score': all_results.get('isolation_score', 0),
                'nlp_validation_score': all_results.get('nlp_score', 0)
            },
            'overall_confidence': all_results.get('confidence_score', 0),
            'verification_status': all_results.get('verification_status', 'Unknown'),
            'detailed_findings': {
                'ocr_issues': all_results.get('ocr_issues', []),
                'rule_violations': all_results.get('rule_violations', []),
                'visual_anomalies': all_results.get('visual_anomalies', []),
                'statistical_anomalies': all_results.get('statistical_anomalies', []),
                'text_inconsistencies': all_results.get('text_inconsistencies', [])
            },
            'recommendations': self.generate_recommendations(all_results)
        }
        
        return report
    
    def generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on verification results"""
        recommendations = []
        
        status = results.get('verification_status', '')
        confidence = results.get('confidence_score', 0)
        
        if status == "Verified":
            recommendations.append("Document verified successfully. No further action needed.")
        
        elif status == "Needs Review":
            recommendations.append("Manual review recommended by verification expert.")
            
            if results.get('ocr_confidence', 1) < 0.8:
                recommendations.append("Request higher quality scan for better OCR accuracy.")
            
            if results.get('rule_violations', []):
                recommendations.append("Verify document details against official records.")
        
        elif status == "Potentially Fraudulent":
            recommendations.append("Document flagged as potentially fraudulent.")
            recommendations.append("Immediate manual verification required.")
            recommendations.append("Cross-check with issuing institution.")
            
            if results.get('cnn_forged', False):
                recommendations.append("Visual forgery indicators detected - examine stamps/signatures.")
        
        return recommendations
    
    def calculate_component_confidence(self, component_results: Dict) -> float:
        """Calculate confidence for individual component"""
        if 'confidence' in component_results:
            return component_results['confidence']
        
        # Calculate based on passed/failed checks
        passed = 0
        total = 0
        
        for key, value in component_results.items():
            if key.endswith('_valid'):
                total += 1
                if value:
                    passed += 1
        
        if total > 0:
            return passed / total
        return 0.0

