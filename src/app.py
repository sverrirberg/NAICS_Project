import streamlit as st
import pandas as pd
from openai import OpenAI
import time
from datetime import timedelta
import os
from dotenv import load_dotenv

# Import from our modules
from src.models.translation import translate_text
from src.models.classification import get_naics_code
from src.models.unspsc import get_unspsc_code, find_best_unspsc_match
from src.utils.config import load_config, save_config
from src.utils.helpers import format_time, plot_confidence_distribution
from src.data.processors import load_unspsc_data, process_data

# Initialize OpenAI client with API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def main():
    st.set_page_config(
        page_title="NAICS & UNSPSC Class Engine",
        page_icon="static/images/favicon-32x32.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Configuration sidebar
    with st.sidebar:
        st.image("static/images/Logo_Greind_Horizontal.png", width=100, use_container_width=True)
        st.header("Settings")
        
        # Load existing configuration
        if 'config' not in st.session_state:
            st.session_state.config = load_config()
            
        # Model selection
        st.subheader("Model Selection")
        st.session_state.config['translation_model'] = st.selectbox(
            "Translation Model",
            ["gpt-3.5-turbo", "gpt-4"],
            index=0 if st.session_state.config['translation_model'] == "gpt-3.5-turbo" else 1
        )
        
        st.session_state.config['classification_model'] = st.selectbox(
            "NAICS Classification Model",
            ["gpt-3.5-turbo", "gpt-4"],
            index=0 if st.session_state.config['classification_model'] == "gpt-3.5-turbo" else 1
        )
        
        st.session_state.config['unspsc_model'] = st.selectbox(
            "UNSPSC Classification Model",
            ["gpt-3.5-turbo", "gpt-4"],
            index=0 if st.session_state.config['unspsc_model'] == "gpt-3.5-turbo" else 1
        )
        
        # Confidence thresholds
        st.subheader("Confidence Thresholds")
        st.session_state.config['confidence_threshold'] = st.slider(
            "NAICS Confidence Threshold (%)",
            min_value=0,
            max_value=100,
            value=st.session_state.config['confidence_threshold']
        )
        
        st.session_state.config['unspsc_confidence_threshold'] = st.slider(
            "UNSPSC Confidence Threshold (%)",
            min_value=0,
            max_value=100,
            value=st.session_state.config['unspsc_confidence_threshold']
        )
        
        # Save/Load configuration
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Configuration"):
                save_config(st.session_state.config)
                st.success("Configuration saved successfully")
        with col2:
            if st.button("Load Configuration"):
                st.session_state.config = load_config()
                st.success("Configuration loaded successfully")
    
    # Main content
    st.title("NAICS & UNSPSC Class Engine")
    
    # File upload section
    st.markdown("""
    **CSV File Requirements:**
    - The CSV file must contain exactly two columns:
      - `description`: Contains the procurement descriptions
      - `account`: Contains the account information
    - The file should be in UTF-8 encoding
    - The first row should contain the column headers
    """)
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Validate the CSV structure
            required_columns = ['description', 'account']
            if not all(col in df.columns for col in required_columns):
                st.error(f"CSV file must contain these columns: {', '.join(required_columns)}")
                return
                
            # Show the first few rows
            st.subheader("Preview of Uploaded Data")
            st.dataframe(df.head())
            
            # Process the data
            if st.button("Process Data"):
                start_time = time.time()
                
                # Initialize progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Load UNSPSC data
                unspsc_df = load_unspsc_data()
                if unspsc_df is None:
                    return
                    
                # Process the data
                processed_df = process_data(df, st.session_state.config)
                if processed_df is None:
                    return
                    
                # Save the results
                processed_df.to_csv('processed_results.csv', index=False)
                
                # Show processing time
                end_time = time.time()
                processing_time = end_time - start_time
                st.success(f"Processing completed in {format_time(processing_time)}")
                
                # Show the results
                st.subheader("Results")
                st.dataframe(processed_df)
                
                # Show confidence distributions
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_confidence_distribution(processed_df, 'naics_confidence'))
                with col2:
                    st.plotly_chart(plot_confidence_distribution(processed_df, 'unspsc_confidence'))
                    
                # Download button
                st.download_button(
                    label="Download Results",
                    data=processed_df.to_csv(index=False),
                    file_name="processed_results.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main() 