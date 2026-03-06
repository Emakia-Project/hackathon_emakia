"""
Streamlit Dashboard for the Emakia Validator Agent.

This module provides a web-based interface for interacting with the validation agent.
"""

import streamlit as st
import asyncio
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

# Add src to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.main import EmakiaValidatorAgent
from src.config.model_config import load_config


def init_agent():
    """Initialize the Emakia Validator Agent."""
    try:
        config = load_config()
        agent = EmakiaValidatorAgent(config)
        return agent
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        return None


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Emakia Validator Agent",
        page_icon="🔍",
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-success { color: #28a745; }
    .status-error { color: #dc3545; }
    .status-warning { color: #ffc107; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Emakia Validator Agent</h1>', unsafe_allow_html=True)
    
    # Initialize agent
    if 'agent' not in st.session_state:
        st.session_state.agent = init_agent()
    
    if st.session_state.agent is None:
        st.error("Agent initialization failed. Please check your configuration.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        st.subheader("Model Settings")
        config = st.session_state.agent.config
        default_provider = config.get('models', {}).get('default', 'openai')
        selected_provider = st.selectbox(
            "Model Provider",
            list(config.get('models', {}).get('providers', {}).keys()),
            index=list(config.get('models', {}).get('providers', {}).keys()).index(default_provider) if default_provider in config.get('models', {}).get('providers', {}) else 0
        )
        
        # Content type selection
        st.subheader("Content Settings")
        content_type = st.selectbox(
            "Content Type",
            ["text", "image", "video"],
            index=0
        )
        
        # Health check
        st.subheader("System Health")
        if st.button("Check Health"):
            health = st.session_state.agent.health_check()
            st.json(health)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Content Validation", "Batch Processing", "Metrics", "API Documentation"])
    
    with tab1:
        st.header("Content Validation")
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            content = st.text_area(
                "Enter content to validate",
                height=200,
                placeholder="Enter your content here..."
            )
        
        with col2:
            st.subheader("Validation Options")
            
            # Validation threshold
            validation_threshold = st.slider(
                "Validation Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.1
            )
            
            # Confidence threshold
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1
            )
        
        # Validation button
        if st.button("Validate Content", type="primary"):
            if content.strip():
                with st.spinner("Validating content..."):
                    try:
                        # Run validation
                        result = asyncio.run(st.session_state.agent.validate_content(content, content_type))
                        
                        # Display results
                        st.success("Validation completed!")
                        
                        # Results in columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Validation Results")
                            
                            # Overall status
                            status = result.get('status', 'unknown')
                            status_color = {
                                'success': 'status-success',
                                'error': 'status-error',
                                'validation_failed': 'status-warning',
                                'content_flagged': 'status-warning'
                            }.get(status, 'status-warning')
                            
                            st.markdown(f'<p class="{status_color}"><strong>Status:</strong> {status}</p>', unsafe_allow_html=True)
                            
                            # Validation details
                            validation = result.get('validation', {})
                            st.write(f"**Valid:** {validation.get('is_valid', False)}")
                            st.write(f"**Confidence:** {validation.get('confidence', 0.0):.2f}")
                            st.write(f"**Model Provider:** {validation.get('model_provider', 'unknown')}")
                            
                            # Violations
                            violations = validation.get('violations', [])
                            if violations:
                                st.write("**Violations:**")
                                for violation in violations:
                                    st.write(f"- {violation}")
                            
                            # Suggestions
                            suggestions = validation.get('suggestions', [])
                            if suggestions:
                                st.write("**Suggestions:**")
                                for suggestion in suggestions:
                                    st.write(f"- {suggestion}")
                        
                        with col2:
                            st.subheader("Classification Results")
                            
                            # Classification details
                            classification = result.get('classification', {})
                            st.write(f"**Category:** {classification.get('category', 'unknown')}")
                            st.write(f"**Confidence:** {classification.get('confidence', 0.0):.2f}")
                            st.write(f"**Threshold Met:** {classification.get('threshold_met', False)}")
                            st.write(f"**Model Provider:** {classification.get('model_provider', 'unknown')}")
                            
                            # Reasoning
                            reasoning = classification.get('reasoning', '')
                            if reasoning:
                                st.write("**Reasoning:**")
                                st.write(reasoning)
                        
                        # Raw results (expandable)
                        with st.expander("Raw Results"):
                            st.json(result)
                            
                    except Exception as e:
                        st.error(f"Validation failed: {str(e)}")
            else:
                st.warning("Please enter content to validate.")
    
    with tab2:
        st.header("Batch Processing")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload file with content (one item per line)",
            type=['txt', 'csv', 'json']
        )
        
        if uploaded_file is not None:
            # Read file content
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    contents = df.iloc[:, 0].tolist()  # First column
                elif uploaded_file.name.endswith('.json'):
                    data = json.load(uploaded_file)
                    contents = data if isinstance(data, list) else [data]
                else:
                    contents = uploaded_file.read().decode('utf-8').splitlines()
                
                st.write(f"Loaded {len(contents)} items")
                
                # Batch processing options
                col1, col2 = st.columns(2)
                
                with col1:
                    batch_size = st.number_input("Batch Size", min_value=1, max_value=100, value=10)
                
                with col2:
                    if st.button("Process Batch", type="primary"):
                        with st.spinner(f"Processing {len(contents)} items..."):
                            try:
                                # Process in batches
                                results = []
                                for i in range(0, len(contents), batch_size):
                                    batch = contents[i:i + batch_size]
                                    batch_results = asyncio.run(st.session_state.agent.batch_validate(batch, content_type))
                                    results.extend(batch_results)
                                
                                st.success(f"Batch processing completed! Processed {len(results)} items.")
                                
                                # Display summary
                                st.subheader("Batch Processing Summary")
                                
                                # Create summary dataframe
                                summary_data = []
                                for i, result in enumerate(results):
                                    validation = result.get('validation', {})
                                    classification = result.get('classification', {})
                                    
                                    summary_data.append({
                                        'Item': i + 1,
                                        'Status': result.get('status', 'unknown'),
                                        'Valid': validation.get('is_valid', False),
                                        'Category': classification.get('category', 'unknown'),
                                        'Confidence': classification.get('confidence', 0.0)
                                    })
                                
                                df_summary = pd.DataFrame(summary_data)
                                st.dataframe(df_summary)
                                
                                # Download results
                                results_json = json.dumps(results, indent=2, default=str)
                                st.download_button(
                                    label="Download Results",
                                    data=results_json,
                                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                                
                            except Exception as e:
                                st.error(f"Batch processing failed: {str(e)}")
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with tab3:
        st.header("Metrics & Analytics")
        
        # Get metrics
        metrics = st.session_state.agent.get_metrics()
        
        # Display metrics in cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Validations", metrics.get('counters', {}).get('validations_text', 0))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Classifications", metrics.get('counters', {}).get('classifications_text', 0))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Errors", metrics.get('counters', {}).get('total_errors', 0))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            success_rate = 0.0
            if metrics.get('validation_summary', {}).get('text', {}).get('total_validations', 0) > 0:
                success_rate = metrics['validation_summary']['text']['success_rate']
            st.metric("Success Rate", f"{success_rate:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Validation Summary")
            validation_summary = metrics.get('validation_summary', {})
            if validation_summary:
                for content_type, data in validation_summary.items():
                    st.write(f"**{content_type.title()}:**")
                    st.write(f"- Total: {data.get('total_validations', 0)}")
                    st.write(f"- Success Rate: {data.get('success_rate', 0.0):.1%}")
                    st.write(f"- Avg Confidence: {data.get('avg_confidence', 0.0):.2f}")
        
        with col2:
            st.subheader("Classification Summary")
            classification_summary = metrics.get('classification_summary', {})
            if classification_summary:
                for content_type, data in classification_summary.items():
                    st.write(f"**{content_type.title()}:**")
                    st.write(f"- Total: {data.get('total_classifications', 0)}")
                    st.write(f"- Avg Confidence: {data.get('avg_confidence', 0.0):.2f}")
                    
                    # Category distribution
                    category_dist = data.get('category_distribution', {})
                    if category_dist:
                        st.write("Category Distribution:")
                        for category, count in category_dist.items():
                            st.write(f"- {category}: {count}")
    
    with tab4:
        st.header("API Documentation")
        
        st.subheader("Usage Examples")
        
        # Python example
        st.write("**Python Example:**")
        st.code("""
from src.main import EmakiaValidatorAgent

# Initialize agent
agent = EmakiaValidatorAgent()

# Validate content
result = await agent.validate_content("Your content here", "text")
print(result)
        """, language="python")
        
        # CLI example
        st.write("**CLI Example:**")
        st.code("""
python src/main.py --content "Your content here" --type text
        """, language="bash")
        
        # API endpoints (if implemented)
        st.subheader("API Endpoints")
        st.write("""
        - `POST /validate` - Validate content
        - `POST /classify` - Classify content
        - `GET /health` - Health check
        - `GET /metrics` - Get metrics
        """)
        
        # Configuration
        st.subheader("Configuration")
        st.write("""
        The agent can be configured using the `src/config/model_config.yaml` file.
        Key configuration options include:
        - Model providers and API keys
        - Validation thresholds
        - Classification categories
        - Logging settings
        """)


if __name__ == "__main__":
    main()
