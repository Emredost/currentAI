"""
Data download module for the Smart Meters in London project.
Downloads the dataset from Kaggle and extracts it into the raw data directory.
"""
import os
import subprocess
import logging
from src.utils.config import RAW_DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_download.log"),
        logging.StreamHandler()
    ]
)

def download_from_kaggle():
    """
    Downloads the dataset from Kaggle using the Kaggle API.
    
    Returns:
        bool: True if successful, False otherwise
    """
    kaggle_dataset = "jeanmidev/smart-meters-in-london"
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    try:
        # Run the Kaggle API command to download the dataset
        logging.info(f"Downloading dataset from Kaggle: {kaggle_dataset}")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", kaggle_dataset, "-p", RAW_DATA_DIR],
            check=True
        )
        
        # Extract the downloaded zip file
        zip_file = os.path.join(RAW_DATA_DIR, "smart-meters-in-london.zip")
        if os.path.exists(zip_file):
            logging.info(f"Extracting dataset to {RAW_DATA_DIR}")
            subprocess.run(
                ["unzip", zip_file, "-d", RAW_DATA_DIR],
                check=True
            )
            logging.info("Extraction complete")
            return True
        else:
            logging.error(f"Downloaded zip file not found: {zip_file}")
            return False
            
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing command: {e}")
        logging.error("Make sure Kaggle API is configured correctly with your API key")
        return False
    except FileNotFoundError:
        logging.error("Kaggle CLI or unzip command not found")
        logging.error("Please install Kaggle CLI with: pip install kaggle")
        return False
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    # Download the dataset
    success = download_from_kaggle()
    
    if success:
        logging.info(f"Dataset downloaded and extracted successfully to {RAW_DATA_DIR}.")
    else:
        logging.error("Failed to download or extract the dataset.")
