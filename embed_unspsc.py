import pandas as pd
from sentence_transformers import SentenceTransformer
import pinecone
import os
from dotenv import load_dotenv
from tqdm import tqdm  # For progress tracking
import logging
from typing import List, Tuple, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_environment() -> None:
    """Load environment variables and initialize Pinecone."""
    try:
        load_dotenv()
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")
    except Exception as e:
        logger.error(f"Failed to initialize: {str(e)}")
        raise

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Load and clean UNSPSC CSV data."""
    try:
        df = pd.read_csv(file_path)
        if not all(col in df.columns for col in ["code", "description"]):
            raise ValueError("Required columns 'code' and 'description' not found in CSV")
        
        df = df.dropna(subset=["code", "description"])
        df["code"] = df["code"].astype(str).str.replace(".0", "", regex=False)
        df["text"] = df["code"] + " - " + df["description"]
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def create_embeddings(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[List[float]]:
    """Create embeddings for the given texts."""
    try:
        model = SentenceTransformer(model_name)
        embeddings = []
        for text in tqdm(texts, desc="Creating embeddings"):
            embedding = model.encode(text).tolist()
            embeddings.append(embedding)
        return embeddings
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise

def setup_pinecone_index(index_name: str, dimension: int) -> pinecone.Index:
    """Set up Pinecone index."""
    try:
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=dimension, metric="cosine")
        return pinecone.Index(index_name)
    except Exception as e:
        logger.error(f"Error setting up Pinecone index: {str(e)}")
        raise

def upload_to_pinecone(index: pinecone.Index, items: List[Tuple], batch_size: int = 100) -> None:
    """Upload embeddings to Pinecone in batches."""
    try:
        for i in tqdm(range(0, len(items), batch_size), desc="Uploading to Pinecone"):
            batch = items[i:i+batch_size]
            index.upsert(vectors=batch)
    except Exception as e:
        logger.error(f"Error uploading to Pinecone: {str(e)}")
        raise

def main():
    try:
        # Initialize
        load_environment()
        
        # Load and prepare data
        df = load_and_clean_data("unspsc.csv")
        
        # Create embeddings
        embeddings = create_embeddings(df["text"].tolist())
        df["embedding"] = embeddings
        
        # Setup Pinecone
        index = setup_pinecone_index("unspsc-codes", dimension=384)
        
        # Prepare items for upload
        items = [
            (row["code"], row["embedding"], {"description": row["description"]})
            for _, row in df.iterrows()
        ]
        
        # Upload to Pinecone
        upload_to_pinecone(index, items)
        
        logger.info("Successfully completed embedding and uploading process")
        
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 