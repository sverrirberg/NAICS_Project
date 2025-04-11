import pandas as pd
import re
from typing import List, Tuple
import numpy as np

def clean_description(text):
    """Hreinsar og staðlar lýsingartexta."""
    if pd.isna(text):
        return ""
    # Breytum í lágstafi
    text = str(text).lower()
    # Fjarlægjum sérstafi og auka bil
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Fjarlægjum algeng for- og viðskeyti
    text = re.sub(r'^(the|a|an)\s+', '', text)
    text = re.sub(r'\s+(inc|ltd|llc|co|corp|corporation)$', '', text)
    return text

def create_optimized_unspsc(input_file, output_file):
    """Les inn UNSPSC skrá, hreinsar og hagræðir gögnin."""
    # Lesum inn upprunalegu skrána
    df = pd.read_csv(input_file)
    
    # Hreinsum kóðadálkinn
    df['code'] = df['code'].astype(str)
    # Fjarlægjum .0 úr kóðunum
    df['code'] = df['code'].str.replace('.0', '')
    # Breytum í heiltölur, en látum nan gildin vera nan
    df['code'] = pd.to_numeric(df['code'], errors='coerce')
    
    # Hreinsum lýsingardálkinn
    df['description'] = df['description'].apply(clean_description)
    
    # Fjarlægjum dulritar lýsingar
    df = df.drop_duplicates(subset=['description'])
    
    # Vistum hagræddu skrána
    df.to_csv(output_file, index=False)
    
    # Prentum tölfræði
    print(f"Upprunalegt fjöldi færslna: {len(df)}")
    print(f"Fjöldi einstakra hreinsaðra lýsinga: {df['description'].nunique()}")
    print("\nDæmi um hreinsaðar lýsingar:")
    print(df[['code', 'description']].head())

if __name__ == "__main__":
    create_optimized_unspsc('unspsc.csv', 'unspsc_optimized.csv') 