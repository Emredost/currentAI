# Streamlit Web Application Guide

## Overview

The Smart Meter Analytics Dashboard is an interactive web application built with Streamlit that provides insights into household electricity consumption data. The application allows users to:

- View and explore smart meter data
- Analyze time series patterns
- Examine weather impacts on consumption
- View consumption patterns by time and household characteristics
- Generate electricity consumption forecasts

## Running the Application

There are several ways to run the application:

### Method 1: Using run.py

```
python run.py --webapp
```

### Method 2: Using run.sh script

```
./run.sh --webapp
```

### Method 3: Direct Streamlit command

```
streamlit run app.py
```

## Dashboard Sections

The dashboard contains the following tabs:

### 1. Data Overview
- View sample datasets
- See basic statistics for consumption and weather data
- View distribution of ACORN household groups

### 2. Time Series Analysis
- Select individual households
- View daily and monthly consumption patterns
- Analyze day-of-week trends
- Filter by custom date ranges

### 3. Weather Impact
- Explore the relationship between temperature and energy consumption
- View temperature bins and average consumption
- Analyze seasonal consumption patterns

### 4. Consumption Patterns
- Examine hourly consumption patterns
- Compare weekday vs. weekend usage
- View consumption by ACORN demographic group

### 5. Forecast
- Select a household for forecasting
- Choose a forecasting model
- Generate predictions for future consumption
- View forecast statistics and visualizations

## Setup and Data Processing

When first running the application, if required data is not found, the dashboard will offer to initialize the data pipeline. This will:

1. Process raw data files
2. Create the necessary datasets for analysis
3. Prepare the data for forecasting models

This initial setup may take some time depending on your system and the size of the data files. 