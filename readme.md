# Comprehensive Automated Document Verification

**Automated Document Verification System** for marksheets/transcripts.  
Combines OCR, rule-based validation, and machine-learning forgery detection to determine authenticity and provide interpretable evidence.

---

## Features
- OCR extraction of student details and marks
- Rule-based validators (credits, grade scale, PRN format, etc.)
- Image & text feature extraction
- ML forgery detection: CNN-based image forgery detector + IsolationForest anomaly detection
- Decision fusion producing a verification verdict and per-component scores
- Unit tests for critical modules

---

## Repo layout (high level)
