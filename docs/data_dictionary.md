# Data Dictionary: Smart Meters in London

This document describes the datasets used in this project, including field definitions, data formats, and relationships between datasets.

## Raw Datasets

### 1. Household Information (`informations_households.csv`)

Contains metadata about the households participating in the smart meter trial.

| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| LCLid | Unique identifier for each household | String | "MAC000001" |
| stdorToU | Standard or Time of Use tariff | String | "Std" or "ToU" |
| Acorn | ACORN classification (socio-economic segmentation) | String | "ACORN-A" |
| Acorn_grouped | Simplified ACORN classification | String | "Affluent" |

### 2. Half-Hourly Consumption Data (`halfhourly_dataset`)

The primary dataset of electricity consumption recorded at 30-minute intervals.

| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| LCLid | Unique identifier for each household | String | "MAC000001" |
| tstp | Timestamp for the measurement | DateTime | "2013-01-01 00:00:00" |
| energy(kWh/hh) | Energy consumption in kilowatt-hours per half hour | Float | 0.285 |

### 3. Weather Data (`weather_hourly_darksky.csv`)

Hourly weather observations from the Dark Sky API.

| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| time | Timestamp for the observation | DateTime | "2011-01-01 00:00:00" |
| visibility | Visibility in miles | Float | 10.0 |
| temperature | Temperature in Celsius | Float | 8.5 |
| dewPoint | Dew point in Celsius | Float | 5.2 |
| pressure | Sea-level air pressure in millibars | Float | 1012.5 |
| cloudCover | Percentage of sky occluded by clouds (0-1) | Float | 0.85 |
| humidity | Relative humidity (0-1) | Float | 0.74 |
| windSpeed | Wind speed in miles per hour | Float | 7.2 |
| windBearing | Wind direction in degrees | Integer | 180 |
| precipIntensity | Precipitation intensity in inches per hour | Float | 0.015 |
| precipProbability | Probability of precipitation (0-1) | Float | 0.35 |
| icon | Weather condition summary | String | "partly-cloudy-night" |

### 4. UK Bank Holidays (`uk_bank_holidays.csv`)

List of UK bank holidays during the study period.

| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| date | Holiday date | Date | "2011-01-01" |
| holiday | Holiday name | String | "New Year's Day" |

### 5. ACORN Details (`acorn_details.csv`)

Detailed descriptions of the ACORN classification categories.

| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| ACORN | ACORN category code | String | "ACORN-A" |
| Description | Full description of the category | String | "Wealthy Executives" |
| Group | Higher-level grouping | String | "Affluent" |

## Processed Datasets

### 1. Cleaned Half-Hourly Data (`cleaned_halfhourly_data.csv`)

The processed version of the half-hourly consumption data with outliers removed, missing values handled, and additional features.

| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| LCLid | Household identifier | String | "MAC000001" |
| tstp | Timestamp | DateTime | "2013-01-01 00:00:00" |
| energy(kWh/hh) | Energy consumption | Float | 0.285 |
| hour | Hour of day | Integer | 0-23 |
| day | Day of week | Integer | 0-6 |
| month | Month | Integer | 1-12 |
| is_weekend | Weekend indicator | Boolean | True/False |
| is_holiday | Holiday indicator | Boolean | True/False |
| season | Season of year | String | "Winter" |

### 2. Cleaned Weather Data (`cleaned_weather_hourly.csv`)

Processed weather data with missing values handled and aligned with consumption data timestamps.

| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| time | Timestamp | DateTime | "2011-01-01 00:00:00" |
| temperature | Temperature in Celsius | Float | 8.5 |
| humidity | Relative humidity | Float | 0.74 |
| windSpeed | Wind speed in mph | Float | 7.2 |
| cloudCover | Cloud cover percentage | Float | 0.85 |
| precipIntensity | Precipitation intensity | Float | 0.015 |
| hour | Hour of day | Integer | 0-23 |
| day | Day of week | Integer | 0-6 |
| month | Month | Integer | 1-12 |
| season | Season of year | String | "Winter" |

### 3. Household Information (`household_info_processed.csv`)

Processed household metadata with additional aggregated statistics.

| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| LCLid | Household identifier | String | "MAC000001" |
| stdorToU | Tariff type | String | "Std" |
| Acorn_grouped | ACORN group | String | "Affluent" |
| avg_daily_consumption | Average daily consumption | Float | 8.75 |
| peak_hour_avg | Average consumption during peak hours | Float | 0.45 |
| off_peak_avg | Average consumption during off-peak hours | Float | 0.22 |
| weekend_avg | Average weekend consumption | Float | 9.2 |
| weekday_avg | Average weekday consumption | Float | 8.5 |

### 4. Analysis Dataset (`analysis_dataset.csv`)

A comprehensive dataset joining consumption, weather, and household information for analysis purposes.

| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| LCLid | Household identifier | String | "MAC000001" |
| tstp | Timestamp | DateTime | "2013-01-01 00:00:00" |
| energy(kWh/hh) | Energy consumption | Float | 0.285 |
| temperature | Temperature in Celsius | Float | 8.5 |
| humidity | Relative humidity | Float | 0.74 |
| windSpeed | Wind speed in mph | Float | 7.2 |
| Acorn_grouped | ACORN group | String | "Affluent" |
| stdorToU | Tariff type | String | "Std" |
| hour | Hour of day | Integer | 0-23 |
| day | Day of week | Integer | 0-6 |
| month | Month | Integer | 1-12 |
| is_weekend | Weekend indicator | Boolean | True/False |
| is_holiday | Holiday indicator | Boolean | True/False |
| season | Season of year | String | "Winter" |

## Data Relationships

- The `LCLid` field connects household information to consumption data
- The `tstp` and `time` fields allow joining consumption data with weather data
- Temporal features (hour, day, month) enable time-based analysis across datasets

## Data Quality Notes

- Half-hourly consumption data contains some gaps and outliers that have been processed
- Weather data is complete for the period but measured at hourly intervals
- Not all households have complete data for the entire study period
- ACORN classification is available for all households in the dataset 