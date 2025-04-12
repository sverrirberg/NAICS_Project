from openai import OpenAI
import streamlit as st
import re

def validate_naics_code(code):
    """
    Validate if a code is a valid NAICS code.
    
    Args:
        code (str): The code to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    # NAICS codes are 2-6 digits
    return bool(re.match(r'^\d{2,6}$', str(code)))

def get_naics_code(text, model="gpt-3.5-turbo"):
    """
    Get NAICS code for a given text using OpenAI's API.
    
    Args:
        text (str): The text to classify
        model (str): The OpenAI model to use
        
    Returns:
        tuple: (code, description, confidence)
    """
    try:
        prompt = f"""
        Analyze this procurement description and determine the most appropriate NAICS code.
        Return the response in this exact format:
        NAICS Code: [6-digit code]
        Description: [brief description]
        Confidence: [percentage 0-100]
        
        Description to analyze:
        {text}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a NAICS code classifier. Your task is to analyze procurement descriptions and return the most appropriate NAICS code with confidence level."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        # Parse the response
        response_text = response.choices[0].message.content.strip()
        code_match = re.search(r'NAICS Code: (\d{2,6})', response_text)
        desc_match = re.search(r'Description: (.*?)(?:\n|$)', response_text)
        conf_match = re.search(r'Confidence: (\d+)', response_text)
        
        if code_match and desc_match and conf_match:
            code = code_match.group(1)
            description = desc_match.group(1).strip()
            confidence = int(conf_match.group(1))
            
            if validate_naics_code(code):
                return code, description, confidence
                
        return None, None, 0
        
    except Exception as e:
        st.error(f"NAICS classification error: {str(e)}")
        return None, None, 0
