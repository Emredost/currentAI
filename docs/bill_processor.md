# Bill Processor User Guide

The Bill Processor in the Smart Meter Analytics Dashboard allows users to easily analyze electricity consumption patterns and get forecasts based on either manually entered data or uploaded electricity bills.

## Setup Instructions

### Prerequisites

To use the OCR (Optical Character Recognition) bill upload functionality, ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

You also need to install Tesseract OCR:
- Windows: Download and install from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- macOS: `brew install tesseract`
- Linux: `sudo apt install tesseract-ocr`

### Configuration

For custom bill processing, you may need to adjust the regular expressions in the code to match your specific bill format. These are located in the `app.py` file in the Bill Processor tab section.

## Using the Bill Processor

### Method 1: Manual Data Entry

1. Navigate to the "Bill Processor" tab in the dashboard.
2. In the left panel, enter:
   - Household ID (optional)
   - Daily consumption data for the past 7 days
   - Additional information like ACORN group, household size, tariff type, and property type
3. Click "Generate Forecast" to process the data.
4. Review the results, including:
   - 7-day consumption forecast
   - Visualization of historical and forecasted data
   - Statistical insights and recommendations

### Method 2: Bill Upload

1. Navigate to the "Bill Processor" tab in the dashboard.
2. In the right panel, upload your electricity bill in PDF or image format (JPG, JPEG, PNG).
3. The system will attempt to extract:
   - Total consumption
   - Billing period
   - Customer ID
   - Daily average consumption
4. Verify the extracted information and make any necessary corrections.
5. Add additional information if needed.
6. Click "Generate Forecast" to process the data.
7. Review the results, including:
   - 7-day consumption forecast
   - Bill vs. forecast comparison
   - Cost projections
   - Personalized recommendations

## Understanding the Results

### Forecast Graph

The forecast graph shows:
- Historical data (solid line) - either manually entered or derived from bill
- Forecasted data (dashed red line) - predicted consumption for the next 7 days

### Insights Section

This section provides:
- Average historical and forecasted consumption
- Percentage change in consumption trends
- Weekend vs. weekday patterns (if applicable)
- Personalized recommendations based on consumption patterns

### Bill Analysis (Bill Upload Only)

When using bill upload, you'll also get:
- Comparison between your last bill and projected bill
- Cost projections based on current rates
- Specific recommendations based on bill trends

## Troubleshooting

### OCR Issues

If the system fails to extract data from your bill:
1. Try uploading a clearer image or better-quality PDF
2. Ensure the bill has clear text that mentions consumption values
3. Use the manual entry method as an alternative

### Forecast Errors

If you encounter forecast errors:
1. Ensure your data contains reasonable values (no zeroes or extreme values)
2. Check that you have entered data for all 7 days
3. Verify that the models have been trained (`python run.py --train`)

## Additional Resources

- For more information on the forecasting models used, refer to the [Model Architecture](model_architecture.md) documentation.
- To understand the data fields, check the [Data Dictionary](data_dictionary.md). 