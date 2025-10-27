"""Visualization utilities for verification results"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List

def visualize_results(verification_results: Dict):
    """Create comprehensive visualization of verification results"""
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overall Scores", "🔍 Component Analysis", "📈 Historical Trends", "📋 Detailed Report"])
    
    with tab1:
        visualize_overall_scores(verification_results)
    
    with tab2:
        visualize_component_analysis(verification_results)
    
    with tab3:
        visualize_historical_trends()
    
    with tab4:
        visualize_detailed_report(verification_results)

def visualize_overall_scores(results: Dict):
    """Visualize overall confidence scores"""
    
    # Overall confidence gauge
    confidence = results.get('confidence_score', 0)
    status = results.get('verification_status', 'Unknown')
    
    # Color based on status
    color_map = {
        'Verified': 'green',
        'Needs Review': 'orange', 
        'Potentially Fraudulent': 'red'
    }
    color = color_map.get(status, 'gray')
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Overall Confidence Score"},
        delta = {'reference': 85},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 60], 'color': "lightgray"},
                {'range': [60, 85], 'color': "yellow"},
                {'range': [85, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Status display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Verification Status", status)
    with col2:
        st.metric("Confidence Score", f"{confidence:.2%}")
    with col3:
        st.metric("PRN", results.get('prn', 'N/A'))

def visualize_component_analysis(results: Dict):
    """Visualize individual component scores"""
    
    # Component scores
    components = {
        'OCR Confidence': results.get('ocr_confidence', 0),
        'Rule-based Validation': results.get('rule_based_score', 0),
        'CNN Forgery Detection': results.get('cnn_score', 0),
        'Isolation Forest': results.get('isolation_score', 0),
        'NLP Validation': results.get('nlp_score', 0)
    }
    
    # Create bar chart
    df = pd.DataFrame(list(components.items()), columns=['Component', 'Score'])
    df['Score'] = df['Score'] * 100  # Convert to percentage
    
    fig = px.bar(df, x='Component', y='Score', 
                 title="Component-wise Verification Scores",
                 color='Score',
                 color_continuous_scale=['red', 'yellow', 'green'])
    
    fig.update_layout(
        yaxis_title="Score (%)",
        xaxis_title="Verification Component",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Component details
    st.subheader("Component Details")
    for component, score in components.items():
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(component)
        with col2:
            st.progress(score)
        with col3:
            st.write(f"{score:.1%}")

def visualize_historical_trends():
    """Visualize historical verification trends"""
    
    try:
        # This would typically load from database
        # For demo, create sample data
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        verified = np.random.poisson(15, 30)
        needs_review = np.random.poisson(5, 30)
        fraudulent = np.random.poisson(2, 30)
        
        df = pd.DataFrame({
            'Date': dates,
            'Verified': verified,
            'Needs Review': needs_review,
            'Potentially Fraudulent': fraudulent
        })
        
        # Ensure all numeric columns are actually numeric
        numeric_cols = ['Verified', 'Needs Review', 'Potentially Fraudulent']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    except Exception as e:
        st.error(f"Error creating historical trends data: {e}")
        return
    
    try:
        # Stacked area chart
        fig = px.area(df, x='Date', y=['Verified', 'Needs Review', 'Potentially Fraudulent'],
                      title="Verification Trends Over Time",
                      color_discrete_map={
                          'Verified': 'green',
                          'Needs Review': 'orange',
                          'Potentially Fraudulent': 'red'
                      })
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating trends chart: {e}")
    
    try:
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate totals for numeric columns only
        numeric_cols = ['Verified', 'Needs Review', 'Potentially Fraudulent']
        total_verifications = df[numeric_cols].sum().sum()
        
        with col1:
            st.metric("Total Verifications", total_verifications)
        with col2:
            if total_verifications > 0:
                success_rate = (df['Verified'].sum() / total_verifications * 100)
                st.metric("Success Rate", f"{success_rate:.1f}%")
            else:
                st.metric("Success Rate", "0.0%")
        with col3:
            if total_verifications > 0:
                review_rate = (df['Needs Review'].sum() / total_verifications * 100)
                st.metric("Review Rate", f"{review_rate:.1f}%")
            else:
                st.metric("Review Rate", "0.0%")
        with col4:
            if total_verifications > 0:
                fraud_rate = (df['Potentially Fraudulent'].sum() / total_verifications * 100)
                st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
            else:
                st.metric("Fraud Rate", "0.0%")
    except Exception as e:
        st.error(f"Error calculating summary statistics: {e}")

def visualize_detailed_report(results: Dict):
    """Display detailed verification report"""
    
    st.subheader("📋 Detailed Verification Report")
    
    # Document information
    with st.expander("📄 Document Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**PRN:** {results.get('prn', 'N/A')}")
            st.write(f"**Student Name:** {results.get('student_name', 'N/A')}")
            st.write(f"**University:** {results.get('university_name', 'N/A')}")
        with col2:
            st.write(f"**SGPA:** {results.get('sgpa', 'N/A')}")
            st.write(f"**Total Credits:** {results.get('total_credits', 'N/A')}")
            st.write(f"**Verification Date:** {results.get('timestamp', 'N/A')}")
    
    # Verification findings
    with st.expander("🔍 Verification Findings"):
        findings = results.get('detailed_findings', {})
        
        # OCR Issues
        if findings.get('ocr_issues'):
            st.error("**OCR Issues:**")
            for issue in findings['ocr_issues']:
                st.write(f"• {issue}")
        
        # Rule Violations
        if findings.get('rule_violations'):
            st.warning("**Rule Violations:**")
            for violation in findings['rule_violations']:
                st.write(f"• {violation}")
        
        # Visual Anomalies
        if findings.get('visual_anomalies'):
            st.error("**Visual Anomalies:**")
            for anomaly in findings['visual_anomalies']:
                st.write(f"• {anomaly}")
        
        # Statistical Anomalies
        if findings.get('statistical_anomalies'):
            st.warning("**Statistical Anomalies:**")
            for anomaly in findings['statistical_anomalies']:
                st.write(f"• {anomaly}")
        
        # Text Inconsistencies
        if findings.get('text_inconsistencies'):
            st.warning("**Text Inconsistencies:**")
            for inconsistency in findings['text_inconsistencies']:
                st.write(f"• {inconsistency}")
    
    # Recommendations
    with st.expander("💡 Recommendations"):
        recommendations = results.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.info("No specific recommendations available.")
    
    # Raw data (collapsible)
    with st.expander("🔧 Raw Verification Data"):
        st.json(results)

def create_verification_summary_card(results: Dict):
    """Create a summary card for the verification result"""
    
    status = results.get('verification_status', 'Unknown')
    confidence = results.get('confidence_score', 0)
    
    # Status color
    if status == 'Verified':
        status_color = '🟢'
        bg_color = 'rgba(0, 255, 0, 0.1)'
    elif status == 'Needs Review':
        status_color = '🟡'
        bg_color = 'rgba(255, 255, 0, 0.1)'
    else:
        status_color = '🔴'
        bg_color = 'rgba(255, 0, 0, 0.1)'
    
    # Create custom HTML for the card
    card_html = f"""
    <div style="
        background-color: {bg_color};
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid {'green' if status == 'Verified' else 'orange' if status == 'Needs Review' else 'red'};
        margin: 10px 0;
    ">
        <h3>{status_color} Verification Result: {status}</h3>
        <p><strong>Confidence Score:</strong> {confidence:.1%}</p>
        <p><strong>PRN:</strong> {results.get('prn', 'N/A')}</p>
        <p><strong>Student:</strong> {results.get('student_name', 'N/A')}</p>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)

def display_subject_analysis(subjects: List[Dict]):
    """Display detailed subject analysis"""
    
    if not subjects:
        st.warning("No subject data available")
        return
    
    st.subheader("📚 Subject Analysis")
    
    # Create DataFrame for subjects
    df = pd.DataFrame(subjects)
    
    # Display subjects table
    st.dataframe(df, use_container_width=True)
    
    # Grade distribution
    if 'grade' in df.columns:
        grade_counts = df['grade'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart for grade distribution
            fig = px.pie(values=grade_counts.values, names=grade_counts.index,
                        title="Grade Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart for grade distribution
            fig = px.bar(x=grade_counts.index, y=grade_counts.values,
                        title="Grade Count")
            st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Subjects", len(subjects))
    with col2:
        if 'credits' in df.columns:
            st.metric("Total Credits", df['credits'].sum())
    with col3:
        if 'grade' in df.columns:
            passed = len(df[df['grade'].isin(['O', 'A+', 'A', 'B+', 'B', 'C', 'P'])])
            st.metric("Passed Subjects", passed)
    with col4:
        if 'grade' in df.columns:
            failed = len(df[df['grade'].isin(['F', 'AB'])])
            st.metric("Failed Subjects", failed)
