import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()

# Set OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def process_row(description, account):
    """Processes a single row by translating and classifying in one API call"""
    try:
        prompt = f"""
        You are a procurement classification expert. Your task is to:
        1. Translate the following procurement description and account classification to English
        2. Determine the most appropriate NAICS code for the translated content
        3. Only return a NAICS code if you are 85% confident
        
        Original Description: {description}
        Original Account: {account}
        
        Return the response in this exact format:
        Translated Description: [translated description]
        Translated Account: [translated account]
        NAICS Code: [code]
        NAICS Description: [description]
        Confidence: [percentage]
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a procurement classification expert with expertise in translation and NAICS classification."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse the response
        response_text = response.choices[0].message.content
        
        # Extract translated content
        trans_desc_match = re.search(r'Translated Description:\s*(.*?)(?=Translated Account:|$)', response_text, re.DOTALL)
        trans_account_match = re.search(r'Translated Account:\s*(.*?)(?=NAICS Code:|$)', response_text, re.DOTALL)
        
        # Extract NAICS information
        code_match = re.search(r'NAICS Code:\s*(\d+)', response_text)
        desc_match = re.search(r'NAICS Description:\s*(.*?)(?=Confidence:|$)', response_text, re.DOTALL)
        conf_match = re.search(r'Confidence:\s*(\d+)%', response_text)
        
        # Return all values
        return (
            trans_desc_match.group(1).strip() if trans_desc_match else None,
            trans_account_match.group(1).strip() if trans_account_match else None,
            code_match.group(1) if code_match else None,
            desc_match.group(1).strip() if desc_match else None,
            conf_match.group(1) if conf_match else None
        )
    except Exception as e:
        st.error(f"Error processing row: {str(e)}")
        return None, None, None, None, None

def main():
    st.title("NAICS Classification System")
    st.write("Upload a CSV file with procurement description and account classification")

    uploaded_file = st.file_uploader("Select CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Try to read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Show CSV file information
            st.write("CSV File Information:")
            st.write(f"Number of rows: {len(df)}")
            st.write("Columns in file:")
            st.write(df.columns.tolist())
            
            # Check if required columns exist
            required_columns = ['procurement_description', 'account']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            # Check for alternative column name
            if 'procurment_description' in df.columns:
                df = df.rename(columns={'procurment_description': 'procurement_description'})
                st.info("Renamed column 'procurment_description' to 'procurement_description'")
                missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.error("Columns in file:")
                st.write(df.columns.tolist())
                return
            
            if st.button("Start Processing"):
                with st.spinner("Processing..."):
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Add new columns
                    total_rows = len(df)
                    df['procurement_description_en'] = ""
                    df['account_en'] = ""
                    df['NAICS Code'] = ""
                    df['NAICS Description'] = ""
                    df['Confidence'] = ""
                    
                    # Process each row
                    for i, row in df.iterrows():
                        status_text.text(f"Processing row {i+1} of {total_rows}")
                        trans_desc, trans_account, code, description, confidence = process_row(
                            row['procurement_description'],
                            row['account']
                        )
                        
                        # Update DataFrame
                        df.at[i, 'procurement_description_en'] = trans_desc
                        df.at[i, 'account_en'] = trans_account
                        df.at[i, 'NAICS Code'] = code
                        df.at[i, 'NAICS Description'] = description
                        df.at[i, 'Confidence'] = confidence
                        
                        progress_bar.progress((i + 1) / total_rows)
                    
                    # Show results
                    st.success("Processing complete!")
                    st.dataframe(df)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name="naics_results.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.error("Please check that the CSV file is in the correct format.")

if __name__ == "__main__":
    main() 