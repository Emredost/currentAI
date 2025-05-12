"""
Startup script to initialize the project and run the web app.
"""
import os
import argparse
import subprocess
import logging
from src.pipelines.pipeline import SmartMeterPipeline
from src.models.train import train_electricity_forecast_models
from src.data.load_data import load_processed_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("startup.log"),
        logging.StreamHandler()
    ]
)

def process_data(sample_size=None):
    """Process the raw data and create analysis dataset"""
    logging.info("Initializing data processing pipeline...")
    
    try:
        pipeline = SmartMeterPipeline()
        
        # Try to load processed data
        try:
            pipeline.load_processed_data(['hourly', 'weather', 'household'])
            logging.info("Successfully loaded processed data!")
        except Exception as e:
            logging.warning(f"Could not load processed data: {e}")
            logging.info("Processing raw data instead...")
            
            # Load and process raw data
            pipeline.load_raw_data(['household_info', 'weather_hourly', 'halfhourly_dataset'])
            
            pipeline.preprocess_household_data()
            pipeline.preprocess_weather_data(hourly=True)
            
            # Use sample size if specified
            if sample_size:
                pipeline.preprocess_consumption_data(halfhourly=True, sample_size=int(sample_size))
            else:
                pipeline.preprocess_consumption_data(halfhourly=True)
                
            logging.info("Creating analysis dataset...")
            pipeline.create_analysis_dataset()
            
        logging.info("Data processing complete!")
        return True
    except Exception as e:
        logging.error(f"Error processing data: {e}")
        return False

def train_models():
    """Train forecasting models on the processed data"""
    logging.info("Training forecasting models...")
    
    try:
        # Load processed data
        data = load_processed_data('hourly')
        
        # Train models
        result = train_electricity_forecast_models(data)
        
        logging.info(f"Model training complete! Best model: {result['best_model']}")
        return True
    except Exception as e:
        logging.error(f"Error training models: {e}")
        return False

def run_webapp():
    """Run the Streamlit web app"""
    logging.info("Starting Streamlit web app...")
    
    try:
        subprocess.run(["streamlit", "run", "app.py"])
        return True
    except Exception as e:
        logging.error(f"Error running web app: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Meter Analytics Startup Script")
    parser.add_argument("--process", action="store_true", help="Process raw data")
    parser.add_argument("--train", action="store_true", help="Train forecasting models")
    parser.add_argument("--sample", type=int, help="Sample size for processing (optional)")
    parser.add_argument("--webapp", action="store_true", help="Run the web app")
    parser.add_argument("--all", action="store_true", help="Run all steps (process, train, webapp)")
    
    args = parser.parse_args()
    
    if args.all:
        # Run everything
        process_success = process_data(args.sample)
        if process_success:
            train_success = train_models()
            if train_success:
                run_webapp()
    else:
        # Run individual steps as requested
        if args.process:
            process_data(args.sample)
            
        if args.train:
            train_models()
            
        if args.webapp:
            run_webapp()
            
        # If no arguments provided, just run the webapp
        if not (args.process or args.train or args.webapp or args.all):
            logging.info("No specific steps requested, launching web app...")
            run_webapp() 