import pandas as pd
import streamlit as st

@st.cache_data
def load_unspsc_data():
    """
    Load UNSPSC data from CSV file.
    
    Returns:
        pd.DataFrame: The UNSPSC data
    """
    try:
        return pd.read_csv('unspsc_optimized.csv')
    except FileNotFoundError:
        st.error("UNSPSC data file not found")
        return None

def process_data(df, config):
    """
    Process the input data.
    
    Args:
        df (pd.DataFrame): The input data
        config (dict): The configuration
        
    Returns:
        pd.DataFrame: The processed data
    """
    try:
        # Add new columns
        df['description_en'] = df['description'].apply(lambda x: translate_text(x, config['translation_model']))
        df['account_en'] = df['account'].apply(lambda x: translate_text(x, config['translation_model']))
        
        # Get NAICS codes
        naics_results = df['description_en'].apply(lambda x: get_naics_code(x, config['classification_model']))
        df['naics_code'] = naics_results.apply(lambda x: x[0])
        df['naics_description'] = naics_results.apply(lambda x: x[1])
        df['naics_confidence'] = naics_results.apply(lambda x: x[2])
        
        # Get UNSPSC codes
        unspsc_results = df.apply(lambda x: get_unspsc_code(x['description_en'], x['naics_description'], config['unspsc_model']), axis=1)
        df['unspsc_code'] = unspsc_results.apply(lambda x: x[0])
        df['unspsc_description'] = unspsc_results.apply(lambda x: x[1])
        df['unspsc_confidence'] = unspsc_results.apply(lambda x: x[2])
        
        return df
        
    except Exception as e:
        st.error(f"Data processing error: {str(e)}")
        return None
