# Smart Meters in London - Data Science Project

An advanced data science project for analyzing and forecasting electricity consumption from smart meters in London.

## Project Overview

This project analyzes household electricity consumption in London using smart meter data from 2011-2014. We investigate consumption patterns, seasonal variations, weather impacts, and build forecasting models to predict future consumption.

### Key Features

- **Data Processing Pipeline**: Robust pipeline for cleaning and preprocessing large electricity datasets
- **Time Series Analysis**: Tools for analyzing consumption patterns over time
- **Weather Impact Analysis**: Assess how weather conditions affect electricity usage
- **Machine Learning Models**: LSTM, GRU, CNN, and Random Forest models for consumption forecasting
- **Interactive Dashboard**: Streamlit web application for data exploration and model visualization

## Project Structure

```
.
├── app.py                      # Streamlit web application (main dashboard)
├── run.py                      # Main startup script
├── run.sh                      # Shell script wrapper
├── dvc.yaml                    # DVC pipeline configuration
├── data/                       # Data directory
│   ├── processed/              # Cleaned & processed data
│   └── raw/                    # Original raw data
├── models/                     # Trained forecasting models
├── notebooks/                  # Jupyter notebooks for exploration
│   └── visualizations/         # Visualization outputs
├── src/                        # Source code
│   ├── data/                   # Data loading and preprocessing
│   ├── models/                 # Model training and evaluation
│   ├── pipelines/              # Data processing pipelines
│   ├── analysis/               # Analysis and visualization tools
│   └── utils/                  # Utility functions and config
├── docs/                       # Project documentation
│   ├── README.md               # Documentation overview
│   ├── app_usage_guide.md      # User guide for the web application
│   ├── developer_guide.md      # Guide for developers
│   └── api_reference.md        # API reference for key modules
└── tests/                      # Unit tests
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip or conda for package management

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd smart-meters-london
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the dataset:
   ```
   python -m src.download_data
   ```
   
   Or manually download from [Smart meters in London](https://www.kaggle.com/jeanmidev/smart-meters-in-london)

### Running the Pipeline

Using the command-line interface:

```bash
# Process data, train models, and run the web app
./run.sh --all

# Or run components individually
./run.sh --process  # Process the raw data
./run.sh --train    # Train forecasting models
./run.sh --webapp   # Run the Streamlit dashboard
```

Or using Python:

```bash
python run.py --all
```

You can also use DVC to run the pipeline:
```bash
dvc repro
```

### Using the Web Dashboard

The Streamlit dashboard provides interactive visualization and forecasting:

```bash
streamlit run app.py
```

The dashboard will be available at http://localhost:8501 and includes:
- Data overview and statistics
- Time series visualizations
- Weather impact analysis
- Consumption pattern insights
- Interactive forecasting

## Model Performance

Our models achieve the following performance metrics on test data:

| Model | RMSE | MAE | R² |
|-------|------|-----|---|
| LSTM | 0.114 | 0.087 | 0.83 |
| GRU | 0.118 | 0.089 | 0.82 |
| CNN | 0.125 | 0.093 | 0.80 |
| Random Forest | 0.132 | 0.097 | 0.78 |

## Documentation

Detailed documentation is available in the `docs/` directory:

- `docs/README.md`: Overview of project documentation
- `docs/app_usage_guide.md`: Guide for using the web application
- `docs/developer_guide.md`: Guide for developers contributing to the project
- `docs/api_reference.md`: API reference for key modules and classes

## Contributing

Contributions are welcome! Please follow these steps:

1. Create a fork of the repository
2. Create a feature branch
3. Make your changes
4. Run the tests (`pytest tests/`)
5. Submit a pull request

Please follow our code style guidelines and add appropriate tests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Dataset provided by UK Power Networks
- Weather data from Dark Sky API
- ACORN classification system for household categorization