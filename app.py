import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import time
from datetime import timedelta
import os
from dotenv import load_dotenv
import re
from fuzzywuzzy import fuzz
import numpy as np
import plotly.express as px

# Initialize OpenAI client with API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load configuration
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            'translation_model': 'gpt-3.5-turbo',
            'classification_model': 'gpt-3.5-turbo',
            'unspsc_model': 'gpt-3.5-turbo',
            'confidence_threshold': 85,
            'unspsc_confidence_threshold': 85
        }

# Save configuration
def save_config(config):
    with open('config.json', 'w') as f:
        json.dump(config, f)

# Initialize session state for configuration
if 'config' not in st.session_state:
    st.session_state.config = load_config()

# Hlaða inn UNSPSC CSV skránni
@st.cache_data
def load_unspsc_data():
    try:
        unspsc_df = pd.read_csv('unspsc.csv')
        # Staðfesta að nauðsynlegir dálkar séu til
        required_columns = ['code', 'description']
        missing_columns = [col for col in required_columns if col not in unspsc_df.columns]
        if missing_columns:
            st.error(f"UNSPSC skrá vantar eftirfarandi dálka: {', '.join(missing_columns)}")
            return None
        return unspsc_df
    except FileNotFoundError:
        st.warning("UNSPSC skrá fannst ekki. Vinsamlegast setjið unspsc.csv skrána í sömu möppu og app.py")
        return None
    except Exception as e:
        st.error(f"Villa við að hlaða UNSPSC skrá: {str(e)}")
        return None

def translate_text(text, model="gpt-3.5-turbo"):
    """Translates text to English using OpenAI API"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional translator from Icelandic to English. Translate the following text accurately while maintaining its meaning and context."},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

def validate_naics_code(code):
    """Validates if a NAICS code is in the correct format"""
    if not code:
        return False
    # NAICS codes are 2-6 digits
    return bool(re.match(r'^\d{2,6}$', str(code)))

def get_naics_code(text, model="gpt-3.5-turbo"):
    """Gets NAICS code for description with improved accuracy"""
    try:
        if not isinstance(text, str) or not text.strip():
            return None, None, 0
            
        # Enhanced system prompt for better classification
        system_prompt = """You are an expert in NAICS (North American Industry Classification System) classification. 
        Your task is to analyze the given text and determine the most appropriate NAICS code and description.
        
        Guidelines:
        1. NAICS codes are 2-6 digits long
        2. Choose the most specific code that matches the description
        3. Consider both the primary activity and any significant secondary activities
        4. If unsure, choose a more general code rather than an incorrect specific one
        5. Provide a confidence score based on how well the description matches the NAICS code
        
        Format your response exactly as:
        NAICS Code: [code]
        NAICS Description: [description]
        Confidence: [percentage]%
        
        Example:
        NAICS Code: 541511
        NAICS Description: Custom Computer Programming Services
        Confidence: 95%"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please classify the following business activity:\n\n{text}"}
            ],
            temperature=0.3
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse response using regex
        code_match = re.search(r'NAICS Code:\s*(\d+)', response_text)
        desc_match = re.search(r'NAICS Description:\s*(.*?)(?=Confidence:|$)', response_text, re.DOTALL)
        conf_match = re.search(r'Confidence:\s*(\d+)%', response_text)
        
        code = code_match.group(1) if code_match else None
        description = desc_match.group(1).strip() if desc_match else None
        confidence = int(conf_match.group(1)) if conf_match else 0
        
        # Validate the NAICS code
        if not validate_naics_code(code):
            return None, None, 0
            
        return code, description, confidence
    except Exception as e:
        st.error(f"NAICS classification error: {str(e)}")
        return None, None, 0

def get_unspsc_code(procurement_desc, naics_desc, model="gpt-3.5-turbo"):
    """Gets UNSPSC code based on procurement description and NAICS description"""
    try:
        if not isinstance(procurement_desc, str) or not procurement_desc.strip():
            return None, None, 0
            
        # Enhanced system prompt for UNSPSC classification
        system_prompt = """You are an expert in UNSPSC (United Nations Standard Products and Services Code) classification.
        Your task is to analyze the given procurement description and NAICS classification to determine the most appropriate UNSPSC code and description.
        
        CRITICAL INSTRUCTIONS:
        1. You must ONLY use existing, valid UNSPSC codes from the official UNSPSC system
        2. Do NOT generate or make up codes - they must be real UNSPSC codes
        3. The code must exactly match the description you provide
        4. If you are not certain about the exact code, do not provide one
        5. UNSPSC codes are hierarchical:
           - First 2 digits: Segment (e.g., 43 for IT)
           - First 4 digits: Family (e.g., 4321 for Software)
           - First 6 digits: Class (e.g., 432115 for Business Software)
        
        Guidelines:
        1. Choose the most specific 6-digit code that matches the description
        2. Consider both the procurement description and NAICS classification for context
        3. If unsure about the exact code, do not provide one
        4. Provide a confidence score based on how well the description matches the UNSPSC code
        5. The code and description must be from the official UNSPSC system
        
        Format your response exactly as:
        UNSPSC Code: [6-digit code]
        UNSPSC Description: [description]
        Confidence: [percentage]%
        
        Example of valid UNSPSC codes:
        UNSPSC Code: 432115
        UNSPSC Description: Business application software
        Confidence: 95%
        
        UNSPSC Code: 441115
        UNSPSC Description: Desktop computers
        Confidence: 95%"""
        
        context = f"""Procurement Description: {procurement_desc}
        NAICS Classification: {naics_desc}"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please classify the following procurement:\n\n{context}"}
            ],
            temperature=0.3
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse response using regex
        code_match = re.search(r'UNSPSC Code:\s*(\d{6})', response_text)
        desc_match = re.search(r'UNSPSC Description:\s*(.*?)(?=Confidence:|$)', response_text, re.DOTALL)
        conf_match = re.search(r'Confidence:\s*(\d+)%', response_text)
        
        code = code_match.group(1) if code_match else None
        description = desc_match.group(1).strip() if desc_match else None
        confidence = int(conf_match.group(1)) if conf_match else 0
        
        return code, description, confidence
    except Exception as e:
        st.error(f"UNSPSC classification error: {str(e)}")
        return None, None, 0

def get_unspsc_description(text, model="gpt-3.5-turbo"):
    """Gets UNSPSC description for the text"""
    try:
        if not isinstance(text, str) or not text.strip():
            return None
            
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": """You are an expert in UNSPSC classification. 
                For the given text, provide a detailed UNSPSC description that matches the official UNSPSC terminology.
                Format your response as:
                UNSPSC Description: [detailed description]"""},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        
        response_text = response.choices[0].message.content.strip()
        unspsc_desc_match = re.search(r'UNSPSC Description: (.+?)(?:\n|$)', response_text)
        return unspsc_desc_match.group(1) if unspsc_desc_match else None
    except Exception as e:
        st.error(f"UNSPSC description error: {str(e)}")
        return None

def parse_unspsc_response(response):
    """Parses UNSPSC response into three separate columns"""
    try:
        # Find UNSPSC code
        code_match = re.search(r'UNSPSC Code:\s*(\d+)', response)
        code = code_match.group(1) if code_match else None
        
        # Find description
        desc_match = re.search(r'UNSPSC Description:\s*(.*?)(?=Confidence:|$)', response, re.DOTALL)
        description = desc_match.group(1).strip() if desc_match else None
        
        # Find confidence
        conf_match = re.search(r'Confidence:\s*(\d+)%', response)
        confidence = conf_match.group(1) if conf_match else None
        
        return code, description, confidence
    except Exception as e:
        st.error(f"Error processing UNSPSC response: {str(e)}")
        return None, None, None

def find_best_unspsc_match(description, unspsc_df):
    if unspsc_df is None:
        return None, None, 0
    
    # Check if description is valid
    if not isinstance(description, str) or not description.strip():
        return None, None, 0
    
    # Search in all descriptions
    best_match = None
    best_score = 0
    best_code = None
    
    try:
        for _, row in unspsc_df.iterrows():
            if not isinstance(row['description'], str):
                continue
                
            score = fuzz.token_sort_ratio(description.lower(), row['description'].lower())
            if score > best_score:
                best_score = score
                best_match = row['description']
                best_code = row['code']
    except Exception as e:
        st.error(f"Error in UNSPSC matching: {str(e)}")
        return None, None, 0
    
    return best_code, best_match, best_score

def is_english(text):
    """Checks if text is already in English"""
    try:
        # Simple check for common Icelandic characters
        icelandic_chars = ['á', 'é', 'í', 'ó', 'ú', 'ý', 'þ', 'æ', 'ö', 'ð']
        return not any(char in text.lower() for char in icelandic_chars)
    except:
        return False

def format_time(seconds):
    """Formats seconds into HH:MM:SS"""
    return str(timedelta(seconds=int(seconds)))

def plot_confidence_distribution(df, column):
    """Creates a histogram of confidence levels"""
    fig = px.histogram(df, x=column, 
                      title=f'Dreifing á {column}',
                      labels={'value': 'Öryggi (%)', 'count': 'Fjöldi'},
                      nbins=20)
    fig.update_layout(bargap=0.2)
    return fig

def main():
    st.title("NAICS and UNSPSC Classification System")
    
    # Configuration sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Load existing configuration
        if st.button("Load Configuration", key="load_config_button"):
            loaded_config = load_config()
            if loaded_config:
                st.session_state.config = loaded_config
                st.success("Configuration loaded successfully")
            else:
                st.warning("No configuration found")
        
        # Model selection
        st.session_state.config['translation_model'] = st.selectbox(
            "Select Translation Model",
            ["gpt-3.5-turbo", "gpt-4"],
            index=["gpt-3.5-turbo", "gpt-4"].index(st.session_state.config['translation_model'])
        )
        
        st.session_state.config['classification_model'] = st.selectbox(
            "Select NAICS Classification Model",
            ["gpt-3.5-turbo", "gpt-4"],
            index=["gpt-3.5-turbo", "gpt-4"].index(st.session_state.config['classification_model'])
        )
        
        st.session_state.config['unspsc_model'] = st.selectbox(
            "Select UNSPSC Classification Model",
            ["gpt-3.5-turbo", "gpt-4"],
            index=["gpt-3.5-turbo", "gpt-4"].index(st.session_state.config['unspsc_model'])
        )
        
        # Set confidence thresholds
        st.session_state.config['confidence_threshold'] = st.slider(
            "NAICS Confidence Threshold (%)",
            min_value=0,
            max_value=100,
            value=st.session_state.config['confidence_threshold'],
            step=5
        )
        
        st.session_state.config['unspsc_confidence_threshold'] = st.slider(
            "UNSPSC Confidence Threshold (%)",
            min_value=0,
            max_value=100,
            value=st.session_state.config['unspsc_confidence_threshold'],
            step=5
        )
        
        # Save configuration
        if st.button("Save Configuration", key="save_config_button"):
            save_config(st.session_state.config)
            st.success("Configuration saved successfully")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check for required columns
            required_columns = ['procurement_description', 'account']
            if 'procurment_description' in df.columns:  # Handle common misspelling
                df = df.rename(columns={'procurment_description': 'procurement_description'})
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"The CSV file must contain these columns: {', '.join(missing_columns)}")
                return
            
            if st.button("Process File"):
                total_rows = len(df)
                
                # Add progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                stopwatch = st.empty()
                
                # Add new columns for translations and classifications
                df['procurement_description_en'] = ''
                df['account_en'] = ''
                df['naics_code'] = ''
                df['naics_description'] = ''
                df['naics_confidence'] = 0
                df['unspsc_code'] = ''
                df['unspsc_description'] = ''
                df['unspsc_confidence'] = 0
                
                start_time = time.time()
                
                # Translation phase
                status_text.text("Phase 1/3: Translating descriptions...")
                for i, row in df.iterrows():
                    progress = (i + 1) / (total_rows * 3)  # First third for translation
                    stopwatch.text(f"Time elapsed: {format_time(time.time() - start_time)}")
                    
                    # Check if procurement description is already in English
                    if is_english(str(row['procurement_description'])):
                        df.at[i, 'procurement_description_en'] = str(row['procurement_description'])
                    else:
                        df.at[i, 'procurement_description_en'] = translate_text(
                            str(row['procurement_description']),
                            model=st.session_state.config['translation_model']
                        )
                    
                    # Check if account description is already in English
                    if is_english(str(row['account'])):
                        df.at[i, 'account_en'] = str(row['account'])
                    else:
                        df.at[i, 'account_en'] = translate_text(
                            str(row['account']),
                            model=st.session_state.config['translation_model']
                        )
                    
                    progress_bar.progress(progress)
                
                # NAICS Classification phase
                status_text.text("Phase 2/3: Classifying with NAICS codes...")
                for i, row in df.iterrows():
                    progress = 1/3 + (i + 1) / (total_rows * 3)  # Second third for NAICS
                    stopwatch.text(f"Time elapsed: {format_time(time.time() - start_time)}")
                    
                    # Combine translated descriptions for better context
                    combined_text = f"Procurement Description: {row['procurement_description_en']}\nAccount Classification: {row['account_en']}"
                    
                    # Get NAICS classification
                    code, description, confidence = get_naics_code(
                        combined_text,
                        model=st.session_state.config['classification_model']
                    )
                    
                    # Only update if confidence meets threshold
                    if confidence >= st.session_state.config['confidence_threshold']:
                        df.at[i, 'naics_code'] = code
                        df.at[i, 'naics_description'] = description
                        df.at[i, 'naics_confidence'] = confidence
                    
                    progress_bar.progress(progress)
                
                # UNSPSC Classification phase
                status_text.text("Phase 3/3: Classifying with UNSPSC codes...")
                for i, row in df.iterrows():
                    progress = 2/3 + (i + 1) / (total_rows * 3)  # Final third for UNSPSC
                    stopwatch.text(f"Time elapsed: {format_time(time.time() - start_time)}")
                    
                    # Get UNSPSC classification using both procurement description and NAICS description
                    code, description, confidence = get_unspsc_code(
                        row['procurement_description_en'],
                        row['naics_description'],
                        model=st.session_state.config['unspsc_model']
                    )
                    
                    # Only update if confidence meets threshold
                    if confidence >= st.session_state.config['unspsc_confidence_threshold']:
                        df.at[i, 'unspsc_code'] = code
                        df.at[i, 'unspsc_description'] = description
                        df.at[i, 'unspsc_confidence'] = confidence
                    
                    progress_bar.progress(progress)
                
                # Processing complete
                end_time = time.time()
                processing_time = end_time - start_time
                
                status_text.text("Processing complete!")
                stopwatch.text(f"Total processing time: {format_time(processing_time)}")
                st.success(f"Processing completed in {processing_time:.2f} seconds!")
                
                # Display results
                st.write("### Results Preview")
                st.write(df)
                
                # Save results
                csv = df.to_csv(index=False)
                st.markdown("---")  # Add a horizontal line for better separation
                st.markdown("### Download Results")
                st.download_button(
                    label="⬇️ Download CSV File",
                    data=csv,
                    file_name="classified_results.csv",
                    mime="text/csv",
                    help="Click to download the processed results as a CSV file"
                )
                st.markdown("---")  # Add a horizontal line for better separation

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 