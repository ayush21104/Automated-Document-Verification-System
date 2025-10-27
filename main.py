# main.py
"""
Comprehensive Document Verification System for SPPU Engineering Marksheets
Main Streamlit Application
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys
import tempfile
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import configuration
from config import (
    SPPU_CONFIG, VALIDATION_THRESHOLDS, CNN_CONFIG,
    ISOLATION_FOREST_CONFIG, NLP_CONFIG, DB_CONFIG, MODELS_DIR
)

# Import modules
from src.preprocessing.image_processor import ImagePreprocessor
from src.ocr.text_extractor import SPPUMarksheetOCR
from src.validation.nlp_validator import NLPValidator
from src.validation.rule_based import RuleBasedValidator
from src.validation.statistical import CNNForgeryDetector
from src.ml_models.isolation_forest import AnomalyDetector
from src.utils.database import VerificationDatabase
from src.utils.scoring import ConfidenceScorer
from src.utils.visualization import visualize_results, create_verification_summary_card, display_subject_analysis
from src.utils.utils import load_image, save_image, enhance_image_quality, get_image_info

# Page configuration
st.set_page_config(
    page_title="SPPU Document Verification System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .verification-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def ensure_bgr_image(image):
    """Ensure image has 3 channels for BGR display in Streamlit"""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        return image

def ensure_rgb_image(image):
    """Ensure image has 3 channels for RGB display in Streamlit"""
    if image is None:
        return None
    
    # Handle tuple input (unpack if needed)
    if isinstance(image, tuple):
        image = image[0]  # Take first element
    
    if not hasattr(image, 'shape'):
        return image
    
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Convert BGR to RGB for proper display
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        return image

def initialize_components():
    """Initialize all verification components"""
    try:
        # Initialize components
        image_processor = ImagePreprocessor()
        ocr_engine = SPPUMarksheetOCR(SPPU_CONFIG)
        nlp_validator = NLPValidator(NLP_CONFIG)
        rule_validator = RuleBasedValidator(SPPU_CONFIG)
        cnn_detector = CNNForgeryDetector(CNN_CONFIG)
        anomaly_detector = AnomalyDetector(ISOLATION_FOREST_CONFIG)
        # Try loading pre-trained Isolation Forest model if available
        try:
            anomaly_detector.load_model(MODELS_DIR)
        except Exception:
            pass
        database = VerificationDatabase(DB_CONFIG)
        scorer = ConfidenceScorer(VALIDATION_THRESHOLDS)
        
        return {
            'image_processor': image_processor,
            'ocr_engine': ocr_engine,
            'nlp_validator': nlp_validator,
            'rule_validator': rule_validator,
            'cnn_detector': cnn_detector,
            'anomaly_detector': anomaly_detector,
            'database': database,
            'scorer': scorer
        }
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        return None

def process_document_verification(uploaded_file, components):
    """Process document verification pipeline"""
    
    try:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("Could not load image. Please try a different file.")
            return None
        
        # Display original image
        st.subheader("📄 Original Document")
        original_display = ensure_rgb_image(image)
        st.image(original_display)
        
        # Image preprocessing
        with st.spinner("Preprocessing image..."):
            try:
                processed_result = components['image_processor'].preprocess(image)
                # Handle tuple return (binary, gray) from preprocess method
                if isinstance(processed_result, tuple):
                    processed_image = processed_result[0]  # Use binary image
                else:
                    processed_image = processed_result
                
                # Ensure we have a valid numpy array
                if not isinstance(processed_image, np.ndarray):
                    st.error("Image preprocessing failed - invalid output format")
                    return None
                    
                enhanced_image = enhance_image_quality(processed_image)
            except Exception as e:
                st.error(f"Error during image preprocessing: {e}")
                return None
        
        # Display processed image
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔧 Preprocessed Image (Binary)")
            processed_display = ensure_rgb_image(processed_image)
            st.image(processed_display)
            
            # Also show grayscale version if available
            if isinstance(processed_result, tuple) and len(processed_result) > 1:
                st.subheader("🔧 Preprocessed Image (Grayscale)")
                gray_display = ensure_rgb_image(processed_result[1])
                st.image(gray_display)
                
        with col2:
            st.subheader("✨ Enhanced Image")
            enhanced_display = ensure_rgb_image(enhanced_image)
            st.image(enhanced_display)
        
        # OCR Text Extraction
        with st.spinner("Extracting text using OCR..."):
            raw_text = components['ocr_engine'].extract_text(enhanced_image)
            ocr_data = components['ocr_engine'].extract_with_confidence(enhanced_image)
            structured_data = components['ocr_engine'].parse_marksheet_structure(raw_text)
        
        # Display extracted text
        st.subheader("📝 Extracted Text")
        with st.expander("View Raw OCR Text"):
            st.text(raw_text)
        
        # Display structured data
        if structured_data:
            st.subheader("📊 Structured Data")
            col1, col2 = st.columns(2)
            with col1:
                st.json(structured_data)
            with col2:
                if 'subjects' in structured_data:
                    display_subject_analysis(structured_data['subjects'])
        
        # Validation Pipeline
        verification_results = {}
        
        # OCR Confidence
        with st.spinner("Calculating OCR confidence..."):
            ocr_confidence = components['ocr_engine'].calculate_confidence(ocr_data)
            verification_results['ocr_confidence'] = ocr_confidence
        
        # Rule-based Validation
        with st.spinner("Performing rule-based validation..."):
            rule_score, rule_details = components['rule_validator'].validate(structured_data)
            verification_results['rule_based_score'] = rule_score
            verification_results['rule_violations'] = rule_details.get('details', [])
        
        # CNN Forgery Detection
        with st.spinner("Running CNN forgery detection..."):
            cnn_score, is_forged = components['cnn_detector'].detect_forgery(enhanced_image)
            verification_results['cnn_score'] = cnn_score
            verification_results['cnn_forged'] = is_forged
        
        # Isolation Forest Anomaly Detection
        with st.spinner("Running anomaly detection..."):
            image_features = components['cnn_detector'].extract_features(enhanced_image)
            # Combine structured data with additional context needed by anomaly detector
            anomaly_input = {
                **(structured_data or {}),
                'ocr_confidence': verification_results.get('ocr_confidence', 0),
                'raw_text': raw_text or ''
            }
            isolation_score, is_anomaly = components['anomaly_detector'].detect_anomaly(
                anomaly_input, image_features
            )
            verification_results['isolation_score'] = isolation_score
            verification_results['isolation_anomaly'] = is_anomaly
        
        # NLP Validation
        with st.spinner("Performing NLP validation..."):
            # Add raw_text to structured_data for NLP validation
            nlp_input = structured_data.copy() if structured_data else {}
            nlp_input['raw_text'] = raw_text
            nlp_score, nlp_details = components['nlp_validator'].validate_text_consistency(nlp_input)
            verification_results['nlp_score'] = nlp_score
            verification_results['text_inconsistencies'] = nlp_details.get('details', [])
        
        # Calculate overall confidence
        with st.spinner("Calculating overall confidence..."):
            overall_confidence = components['scorer'].calculate_overall_confidence(verification_results)
            verification_results['confidence_score'] = overall_confidence
        
        # Determine verification status
        status, reason = components['scorer'].determine_verification_status(
            overall_confidence, verification_results
        )
        verification_results['verification_status'] = status
        verification_results['verification_reason'] = reason
        
        # Add metadata
        verification_results.update({
            'timestamp': datetime.now().isoformat(),
            'prn': structured_data.get('prn', 'N/A'),
            'student_name': structured_data.get('student_name', 'N/A'),
            'university_name': structured_data.get('university_name', 'N/A'),
            'sgpa': structured_data.get('sgpa', 'N/A'),
            'total_credits': sum(s.get('credits', 0) for s in structured_data.get('subjects', [])),
            'raw_data': structured_data,
            'detailed_findings': {
                'ocr_issues': [],
                'rule_violations': verification_results.get('rule_violations', []),
                'visual_anomalies': ['CNN detected potential forgery'] if is_forged else [],
                'statistical_anomalies': ['Isolation Forest detected anomaly'] if is_anomaly else [],
                'text_inconsistencies': verification_results.get('text_inconsistencies', [])
            },
            'recommendations': components['scorer'].generate_recommendations(verification_results)
        })
        
        # Save to database
        components['database'].save_verification_result(verification_results)
        
        return verification_results
        
    except Exception as e:
        st.error(f"Error during verification: {e}")
        st.exception(e)
        return None

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">📋 SPPU Document Verification System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🔧 System Controls")
    
    # Initialize components
    if 'components' not in st.session_state:
        with st.spinner("Initializing verification system..."):
            st.session_state.components = initialize_components()
    
    if st.session_state.components is None:
        st.error("Failed to initialize system components. Please check the configuration.")
        return
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Verification", "📊 System Statistics", "🔍 Verification History", "⚙️ System Settings"])
    
    with tab1:
        st.header("Document Verification")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload SPPU Marksheet Image",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Upload a clear image of the SPPU marksheet for verification"
        )
        
        if uploaded_file is not None:
            # Process verification
            if st.button("🔍 Start Verification", type="primary"):
                verification_results = process_document_verification(uploaded_file, st.session_state.components)
                
                if verification_results:
                    # Display results
                    st.markdown("---")
                    st.header("🎯 Verification Results")
                    
                    # Summary card
                    create_verification_summary_card(verification_results)
                    
                    # Detailed visualizations
                    visualize_results(verification_results)
                    
                    # Download results
                    st.subheader("📥 Download Results")
                    results_json = json.dumps(verification_results, indent=2, default=str)
                    st.download_button(
                        label="Download Verification Report (JSON)",
                        data=results_json,
                        file_name=f"verification_report_{verification_results.get('prn', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    with tab2:
        st.header("System Statistics")
        
        # Get statistics from database
        try:
            stats = st.session_state.components['database'].get_statistics()
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Verifications", stats.get('total_verifications', 0))
            with col2:
                avg_conf = stats.get('average_scores', {}).get('overall', 0)
                st.metric("Average Confidence", f"{avg_conf:.1%}" if avg_conf else "N/A")
            with col3:
                status_dist = stats.get('status_distribution', {})
                verified_count = status_dist.get('Verified', 0)
                st.metric("Verified Documents", verified_count)
            with col4:
                fraud_count = status_dist.get('Potentially Fraudulent', 0)
                st.metric("Flagged Documents", fraud_count)
            
            # Status distribution chart
            if status_dist:
                st.subheader("📊 Verification Status Distribution")
                df_status = pd.DataFrame(list(status_dist.items()), columns=['Status', 'Count'])
                fig = px.pie(df_status, values='Count', names='Status', 
                           title="Document Verification Status Distribution")
                st.plotly_chart(fig)
            
            # Average scores by component
            avg_scores = stats.get('average_scores', {})
            if avg_scores:
                st.subheader("📈 Average Scores by Component")
                components = ['ocr', 'rule_based', 'cnn', 'isolation_forest', 'nlp']
                scores = [avg_scores.get(comp, 0) for comp in components]
                
                df_scores = pd.DataFrame({
                    'Component': [comp.replace('_', ' ').title() for comp in components],
                    'Score': scores
                })
                
                fig = px.bar(df_scores, x='Component', y='Score',
                           title="Average Verification Scores by Component")
                fig.update_layout(yaxis_title="Average Score", xaxis_title="Verification Component")
                st.plotly_chart(fig)
        
        except Exception as e:
            st.error(f"Error loading statistics: {e}")
    
    with tab3:
        st.header("Verification History")
        
        try:
            # Get verification history
            history_df = st.session_state.components['database'].get_verification_history(100)
            
            if not history_df.empty:
                # Display history table
                st.subheader("📋 Recent Verifications")
                st.dataframe(history_df)
                
                # Filter options
                st.subheader("🔍 Filter History")
                col1, col2 = st.columns(2)
                with col1:
                    status_filter = st.selectbox("Filter by Status", 
                                               ["All"] + list(history_df['verification_status'].unique()))
                with col2:
                    date_range = st.date_input("Date Range", value=[])
                
                # Apply filters
                filtered_df = history_df.copy()
                if status_filter != "All":
                    filtered_df = filtered_df[filtered_df['verification_status'] == status_filter]
                
                if filtered_df is not None and not filtered_df.empty:
                    st.subheader("📊 Filtered Results")
                    st.dataframe(filtered_df)
            else:
                st.info("No verification history available.")
        
        except Exception as e:
            st.error(f"Error loading verification history: {e}")
    
    with tab4:
        st.header("System Settings")
        
        # Configuration display
        st.subheader("⚙️ Current Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            st.json(SPPU_CONFIG)
        with col2:
            st.json(VALIDATION_THRESHOLDS)
        
        # Model status
        st.subheader("🤖 Model Status")
        components = st.session_state.components
        
        # Dynamic model readiness status
        cnn_ready = components['cnn_detector'].model is not None
        iso_ready = components['anomaly_detector'].model is not None
        model_status = {
            "Image Processor": "✅ Ready",
            "OCR Engine": "✅ Ready", 
            "NLP Validator": "✅ Ready",
            "Rule-based Validator": "✅ Ready",
            "CNN Forgery Detector": "✅ Ready" if cnn_ready else "⚠️ Demo Mode",
            "Isolation Forest": "✅ Ready" if iso_ready else "⚠️ Heuristic Mode",
            "Database": "✅ Ready",
            "Confidence Scorer": "✅ Ready"
        }
        
        for model, status in model_status.items():
            st.write(f"**{model}:** {status}")
        
        # System information
        st.subheader("ℹ️ System Information")
        st.write(f"**Database Path:** {DB_CONFIG['db_path']}")
        st.write(f"**Models Directory:** {os.path.join(os.path.dirname(__file__), 'data', 'models')}")
        st.write(f"**Sample Directory:** {os.path.join(os.path.dirname(__file__), 'data', 'sample-marksheets')}")

if __name__ == "__main__":
    main()

