from openai import OpenAI
import streamlit as st

def translate_text(text, model="gpt-3.5-turbo"):
    """
    Translate text to English using OpenAI's API.
    
    Args:
        text (str): The text to translate
        model (str): The OpenAI model to use
        
    Returns:
        str: The translated text
    """
    try:
        # First check if the text is already in English
        language_check_prompt = f"""
        Is this text in English? Answer with only 'yes' or 'no':
        {text}
        """
        
        language_check_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a language detector. Your task is to determine if the given text is in English. Answer with only 'yes' or 'no'."},
                {"role": "user", "content": language_check_prompt}
            ],
            temperature=0
        )
        
        is_english = language_check_response.choices[0].message.content.strip().lower() == 'yes'
        
        if is_english:
            return text
            
        # If not English, proceed with translation
        translation_prompt = f"""
        Translate the following text to English. 
        Return only the translated text, nothing else:
        {text}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a translator. Your task is to translate text to English. Return only the translated text, nothing else."},
                {"role": "user", "content": translation_prompt}
            ],
            temperature=0
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text
