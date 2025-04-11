# NAICS and UNSPSC Classification System

This application processes procurement descriptions and accounts, translating them into English and classifying them using NAICS and UNSPSC codes.

## Features

- Translation of Icelandic procurement descriptions and accounts to English
- NAICS code classification with confidence scores
- UNSPSC code classification with confidence scores
- Configurable confidence thresholds
- Progress tracking and time elapsed display
- CSV file processing and results export

## Requirements

- Python 3.8+
- OpenAI API key
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd naics_project
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Upload a CSV file containing:
   - procurement_description
   - account

3. Configure the settings in the sidebar:
   - Select models for translation and classification
   - Set confidence thresholds
   - Save/load configurations

4. Click "Process File" to start the classification process

5. Download the results as a CSV file

## Configuration

The application allows you to:
- Choose between GPT-3.5 and GPT-4 models
- Set confidence thresholds for NAICS and UNSPSC classifications
- Save and load configurations

## Output

The processed CSV file will contain:
- Original procurement description and account
- English translations
- NAICS codes and descriptions with confidence scores
- UNSPSC codes and descriptions with confidence scores 