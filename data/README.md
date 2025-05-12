# Data Directory

This directory contains all data for the Smart Meters in London project, organized into subdirectories.

## Directory Structure

- `raw/`: Original unmodified data from the London Smart Meters dataset
- `processed/`: Cleaned and preprocessed data files ready for analysis
- `external/`: Additional external data sources (if any)

## Data Sources

The primary data comes from the "Smart meters in London" dataset on Kaggle:
https://www.kaggle.com/jeanmidev/smart-meters-in-london

### Raw Data Files

- `informations_households.csv`: Metadata about each household
- `acorn_details.csv`: Information about ACORN demographic categories
- `uk_bank_holidays.csv`: UK bank holidays during the measurement period
- `weather_daily_darksky.csv`: Daily weather data
- `weather_hourly_darksky.csv`: Hourly weather data
- `daily_dataset/`: Directory containing daily smart meter readings
- `halfhourly_dataset/`: Directory containing half-hourly smart meter readings

### Processed Data Files

- `household_info_processed.csv`: Cleaned household information
- `cleaned_weather_hourly.csv`: Preprocessed hourly weather data
- `cleaned_halfhourly_data.csv`: Cleaned half-hourly electricity consumption data
- `analysis_dataset.csv`: Integrated dataset joining consumption with weather and household data

## Data Versioning

Data is versioned using DVC (Data Version Control). To get the latest version of the data:

```
dvc pull
```

## Usage Notes

- Large data files are not stored in GitHub and should be downloaded using `src/download_data.py`
- Most data processing is automated via the data pipeline defined in `dvc.yaml`
- Use `src/pipelines/pipeline.py` for programmatic data loading and processing 