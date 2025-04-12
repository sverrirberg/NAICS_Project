from openai import OpenAI
import streamlit as st
import pandas as pd
from fuzzywuzzy import fuzz
import re

def get_unspsc_code(procurement_desc, naics_desc, model="gpt-3.5-turbo"):
    """
    Get UNSPSC code for a given procurement description and NAICS description.
    
    Args:
        procurement_desc (str): The procurement description
        naics_desc (str): The NAICS description
        model (str): The OpenAI model to use
        
    Returns:
        tuple: (code, description, confidence)
    """
    try:
        prompt = f"""
        Analyze this procurement description and NAICS description to determine the most appropriate UNSPSC code.
        Return the response in this exact format:
        UNSPSC Code: [8-digit code]
        Description: [brief description]
        Confidence: [percentage 0-100]
        
        Procurement Description:
        {procurement_desc}
        
        NAICS Description:
        {naics_desc}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a UNSPSC code classifier. Your task is to analyze procurement and NAICS descriptions and return the most appropriate UNSPSC code with confidence level."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        # Parse the response
        response_text = response.choices[0].message.content.strip()
        code_match = re.search(r'UNSPSC Code: (\d{8})', response_text)
        desc_match = re.search(r'Description: (.*?)(?:\n|$)', response_text)
        conf_match = re.search(r'Confidence: (\d+)', response_text)
        
        if code_match and desc_match and conf_match:
            code = code_match.group(1)
            description = desc_match.group(1).strip()
            confidence = int(conf_match.group(1))
            return code, description, confidence
            
        return None, None, 0
        
    except Exception as e:
        st.error(f"UNSPSC classification error: {str(e)}")
        return None, None, 0

def find_best_unspsc_match(description, unspsc_df):
    """
    Find the best matching UNSPSC code using fuzzy matching.
    
    Args:
        description (str): The description to match
        unspsc_df (pd.DataFrame): The UNSPSC data
        
    Returns:
        tuple: (code, description, confidence)
    """
    try:
        best_match = None
        best_score = 0
        
        for _, row in unspsc_df.iterrows():
            score = fuzz.ratio(description.lower(), row['description'].lower())
            if score > best_score:
                best_score = score
                best_match = row
                
        if best_match is not None and best_score >= 70:
            return best_match['code'], best_match['description'], best_score
            
        return None, None, 0
        
    except Exception as e:
        st.error(f"UNSPSC matching error: {str(e)}")
        return None, None, 0
