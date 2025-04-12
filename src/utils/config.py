import json
import streamlit as st

def load_config():
    """
    Load configuration from config.json file.
    
    Returns:
        dict: The configuration
    """
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            # Ensure all required keys are present
            default_config = {
                'translation_model': 'gpt-3.5-turbo',
                'classification_model': 'gpt-3.5-turbo',
                'unspsc_model': 'gpt-3.5-turbo',
                'confidence_threshold': 85,
                'unspsc_confidence_threshold': 85
            }
            # Update with any missing keys
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            return config
    except FileNotFoundError:
        return default_config

def save_config(config):
    """
    Save configuration to config.json file.
    
    Args:
        config (dict): The configuration to save
    """
    with open('config.json', 'w') as f:
        json.dump(config, f)
