"""
Streamlit demo for the Bill Processor functionality.
This lightweight version focuses only on the bill processor feature
and does not require the full dataset.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import random

# Set page configuration
st.set_page_config(
    page_title="Smart Meter Bill Processor Demo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("⚡ Smart Meter Bill Processor Demo")
st.markdown("""
This demo allows you to test the bill processor functionality without requiring the full dataset.
You can upload electricity bills or manually enter consumption data to generate forecasts and insights.
""")

# Create mock data and models
class MockPipeline:
    def create_sequences(self, df, lookback, target_col, forecast_steps):
        # Create a simple mock sequence for demonstration
        n_features = len(df.columns)
        X = np.random.random((1, lookback, n_features))
        y = None
        return X, y

class MockModel:
    def predict(self, X):
        # Generate realistic looking predictions based on the input
        # For a week of predictions
        predictions = []
        last_value = 10 + random.random() * 5  # Start with a base value
        
        for _ in range(7):
            # Add some random variation
            variation = (random.random() - 0.5) * 2  # -1 to 1
            last_value += variation
            predictions.append(max(0, last_value))  # Ensure no negative values
            
        return np.array(predictions).reshape(1, -1)

# Create mock models dictionary
models = {
    'lstm': MockModel(),
    'data_info': {
        'target_scaler': type('DummyScaler', (), {
            'inverse_transform': lambda self, x: x  # Identity function
        })()
    }
}

# Bill Processor Tab
st.header("📄 Smart Meter Bill Processor")

st.markdown("""
This tool allows you to upload electricity bills or manually enter consumption data to generate forecasts and insights.
Choose one of the methods below to get started.
""")

# Create two columns for the input methods
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Manual Data Entry")
    
    # Manual form inputs
    with st.form("manual_entry_form"):
        household_id = st.text_input("Household ID (Optional)", placeholder="e.g., MAC000073")
        
        # Recent consumption data
        st.write("Recent Daily Consumption (kWh)")
        recent_days = []
        daily_values = []
        
        # Add 7 days of consumption input fields
        for i in range(7):
            day_col, value_col = st.columns([1, 1])
            with day_col:
                day = st.date_input(f"Day {i+1}", value=datetime.now() - timedelta(days=7-i))
                recent_days.append(day)
            with value_col:
                value = st.number_input(f"kWh {i+1}", min_value=0.0, max_value=100.0, value=10.0, format="%.2f")
                daily_values.append(value)
        
        # Additional features for better prediction
        st.write("Additional Information (Optional)")
        
        col_a, col_b = st.columns(2)
        with col_a:
            acorn_group = st.selectbox("ACORN Group", 
                options=["A", "B", "C", "D", "E", "F", "G", "U"], index=0)
            household_size = st.slider("Household Size", min_value=1, max_value=6, value=2)
        
        with col_b:
            tariff = st.selectbox("Tariff Type", 
                options=["Standard", "Time of Use (ToU)"], index=0)
            property_type = st.selectbox("Property Type", 
                options=["Flat", "Terraced", "Semi-detached", "Detached"], index=0)
        
        submit_button = st.form_submit_button("Generate Forecast")
    
    if submit_button:
        with st.spinner("Processing data and generating forecast..."):
            try:
                # Process input data
                input_data = {
                    'household_id': household_id if household_id else "MANUAL_INPUT",
                    'days': recent_days,
                    'consumption': daily_values,
                    'acorn_group': acorn_group,
                    'tariff': tariff,
                    'household_size': household_size,
                    'property_type': property_type
                }
                
                # Create a DataFrame from the input data
                df = pd.DataFrame({
                    'date': input_data['days'],
                    'energy_sum': input_data['consumption']
                })
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Prepare features for prediction
                # Add temporal features
                df['hour'] = 12  # Mid-day as default
                df['day_of_week'] = df.index.dayofweek
                df['month'] = df.index.month
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
                df['season'] = df['month'].apply(lambda x: 1 if x in [12, 1, 2] else 
                                               2 if x in [3, 4, 5] else 
                                               3 if x in [6, 7, 8] else 4)
                
                # Get the forecast
                model_choice = 'lstm'  # Default to LSTM
                    
                # Display loading information
                st.info(f"Using {model_choice.upper()} model for forecasting...")
                
                # Create pipeline for preprocessing
                try:
                    pipeline = MockPipeline()
                    X_seq, _ = pipeline.create_sequences(df, 
                                                      lookback=7, 
                                                      target_col='energy_sum',
                                                      forecast_steps=7)
                    
                    # Make prediction
                    predictions = models[model_choice].predict(X_seq)
                    
                    # Display the forecast
                    forecast_dates = [max(input_data['days']) + timedelta(days=i+1) for i in range(7)]
                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Forecasted Consumption (kWh)': predictions.flatten()[:7]
                    })
                    
                    st.success("Forecast successfully generated!")
                    st.dataframe(forecast_df)
                    
                    # Plot the results
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Historical data
                    ax.plot(input_data['days'], input_data['consumption'], 
                          marker='o', linestyle='-', label='Historical Data')
                    
                    # Forecast data
                    ax.plot(forecast_dates, predictions.flatten()[:7], 
                          marker='o', linestyle='--', color='red', label='Forecast')
                    
                    ax.set_title('Electricity Consumption Forecast')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Daily Consumption (kWh)')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    # Format x-axis dates
                    fig.autofmt_xdate()
                    
                    st.pyplot(fig)
                    
                    # Add insights
                    st.subheader("Insights")
                    
                    # Calculate statistics
                    avg_historical = np.mean(input_data['consumption'])
                    avg_forecast = np.mean(predictions.flatten()[:7])
                    pct_change = ((avg_forecast - avg_historical) / avg_historical) * 100
                    
                    # Create three columns
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Avg. Historical Consumption", f"{avg_historical:.2f} kWh", "Past 7 days")
                    c2.metric("Avg. Forecasted Consumption", f"{avg_forecast:.2f} kWh", "Next 7 days")
                    c3.metric("Consumption Trend", f"{pct_change:.1f}%", 
                            f"{'↑' if pct_change > 0 else '↓'} from previous week")
                    
                    # Add more detailed insights
                    st.write("### Consumption Patterns")
                    
                    if pct_change > 10:
                        st.warning("⚠️ Your consumption is projected to increase significantly. Consider reviewing usage patterns.")
                    elif pct_change < -10:
                        st.success("✅ Your consumption is projected to decrease significantly. Great job!")
                    
                    # Day of the week patterns
                    weekday_avg = np.mean([v for i, v in enumerate(input_data['consumption']) 
                                        if input_data['days'][i].weekday() < 5])
                    weekend_avg = np.mean([v for i, v in enumerate(input_data['consumption']) 
                                        if input_data['days'][i].weekday() >= 5])
                    
                    if weekend_avg > weekday_avg * 1.2:
                        st.info("ℹ️ Your weekend consumption is significantly higher than weekdays.")
                    
                    # Recommendations
                    st.write("### Recommendations")
                    recommendations = []
                    
                    if pct_change > 5:
                        recommendations.append("Consider monitoring heavy appliance usage during peak hours.")
                        recommendations.append("Check for devices that might be left on standby.")
                    
                    if weekend_avg > weekday_avg * 1.5:
                        recommendations.append("Your weekend usage is very high. Consider energy-saving activities.")
                    
                    if avg_historical > 15:  # Arbitrary threshold
                        recommendations.append("Your overall consumption is higher than average. Consider an energy audit.")
                    
                    if not recommendations:
                        recommendations.append("Your consumption patterns look efficient. Keep up the good work!")
                    
                    for i, rec in enumerate(recommendations):
                        st.write(f"{i+1}. {rec}")
                
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

with col2:
    st.subheader("📊 Bill Upload")
    
    # File uploader for bills
    uploaded_file = st.file_uploader("Upload electricity bill (PDF or image)", 
                                  type=["pdf", "jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded file
        file_details = {"Filename": uploaded_file.name, 
                       "File size": f"{uploaded_file.size / 1024:.2f} KB", 
                       "File type": uploaded_file.type}
        st.write(file_details)
        
        # Check if we have the necessary libraries
        try:
            import pytesseract
            from PIL import Image
            import io
            import re
            import pdfplumber
            
            ocr_ready = True
        except ImportError:
            ocr_ready = False
            st.warning("OCR processing libraries not available. Please install pytesseract and pdfplumber for automatic bill processing.")
            st.info("You can still use the manual entry form on the left.")
        
        # Process the uploaded file if OCR is available
        if ocr_ready:
            with st.spinner("Extracting data from bill..."):
                try:
                    text = ""
                    
                    # Handle PDF files
                    if uploaded_file.type == "application/pdf":
                        with pdfplumber.open(io.BytesIO(uploaded_file.getvalue())) as pdf:
                            for page in pdf.pages:
                                text += page.extract_text() + "\n"
                    
                    # Handle image files
                    elif uploaded_file.type.startswith("image"):
                        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
                        text = pytesseract.image_to_string(image)
                    
                    # Show extracted text
                    with st.expander("Extracted Text"):
                        st.text(text)
                    
                    # Process the extracted text to find consumption data
                    # These patterns may need adjustment based on actual bill formats
                    patterns = {
                        'consumption': r'total\s+consumption\s*:?\s*(\d+\.?\d*)\s*kwh',
                        'date_range': r'billing\s+period\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})\s*to\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
                        'customer_id': r'customer\s+(id|number)\s*:?\s*([A-Za-z0-9]+)',
                        'daily_avg': r'daily\s+average\s*:?\s*(\d+\.?\d*)\s*kwh',
                    }
                    
                    extracted_data = {}
                    
                    for key, pattern in patterns.items():
                        match = re.search(pattern, text.lower())
                        if match:
                            if key == 'date_range':
                                extracted_data['start_date'] = match.group(1)
                                extracted_data['end_date'] = match.group(2)
                            else:
                                extracted_data[key] = match.group(1) if key != 'customer_id' else match.group(2)
                    
                    # If no data was extracted, create some sample data
                    if not extracted_data:
                        st.info("No specific data was found in the bill. Using sample data for demonstration.")
                        extracted_data = {
                            'customer_id': 'SAMPLE123',
                            'consumption': '325.5',
                            'daily_avg': '10.85'
                        }
                    
                    # Display extracted data
                    st.success("Data extracted successfully!")
                    st.json(extracted_data)
                    
                    # Create form for verification and correction
                    with st.form("extracted_data_form"):
                        st.write("Please verify the extracted information and make corrections if needed:")
                        
                        # Fields with default values from extraction
                        customer_id = st.text_input("Customer ID", 
                                                 value=extracted_data.get('customer_id', ''))
                        consumption = st.number_input("Total Consumption (kWh)", 
                                                   value=float(extracted_data.get('consumption', 0)))
                        daily_avg = st.number_input("Daily Average (kWh)", 
                                                 value=float(extracted_data.get('daily_avg', 0)))
                        
                        # Date range
                        col_start, col_end = st.columns(2)
                        with col_start:
                            start_date = st.date_input("Billing Start Date", 
                                                    value=datetime.now() - timedelta(days=30))
                        with col_end:
                            end_date = st.date_input("Billing End Date", 
                                                  value=datetime.now())
                        
                        # Optional fields for better prediction
                        st.write("Additional Information (Optional)")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            acorn_group = st.selectbox("ACORN Group", 
                                options=["A", "B", "C", "D", "E", "F", "G", "U"], index=0,
                                key="ocr_acorn")
                            household_size = st.slider("Household Size", min_value=1, max_value=6, value=2,
                                                    key="ocr_household_size")
                        
                        with col_b:
                            tariff = st.selectbox("Tariff Type", 
                                options=["Standard", "Time of Use (ToU)"], index=0,
                                key="ocr_tariff")
                            property_type = st.selectbox("Property Type", 
                                options=["Flat", "Terraced", "Semi-detached", "Detached"], index=0,
                                key="ocr_property")
                        
                        submit_ocr = st.form_submit_button("Generate Forecast")
                    
                    if submit_ocr:
                        with st.spinner("Processing data and generating forecast..."):
                            try:
                                # Calculate date range and create synthetic daily data
                                date_range = (end_date - start_date).days
                                if date_range <= 0:
                                    st.error("End date must be after start date")
                                else:
                                    # Create synthetic daily data based on the bill's average
                                    if daily_avg == 0 and date_range > 0:
                                        daily_avg = consumption / date_range
                                    
                                    # Create dates for the last 7 days
                                    recent_days = [end_date - timedelta(days=i) for i in range(7, 0, -1)]
                                    
                                    # Create synthetic consumption values with some variation
                                    np.random.seed(42)  # For reproducibility
                                    variations = np.random.normal(1, 0.1, 7)  # 10% variation
                                    daily_values = [daily_avg * var for var in variations]
                                    
                                    # Create the input data
                                    input_data = {
                                        'household_id': customer_id if customer_id else "BILL_UPLOAD",
                                        'days': recent_days,
                                        'consumption': daily_values,
                                        'acorn_group': acorn_group,
                                        'tariff': tariff,
                                        'household_size': household_size,
                                        'property_type': property_type
                                    }
                                    
                                    # Create a DataFrame from the input data
                                    df = pd.DataFrame({
                                        'date': input_data['days'],
                                        'energy_sum': input_data['consumption']
                                    })
                                    df['date'] = pd.to_datetime(df['date'])
                                    df.set_index('date', inplace=True)
                                    
                                    # Prepare features for prediction
                                    # Add temporal features
                                    df['hour'] = 12  # Mid-day as default
                                    df['day_of_week'] = df.index.dayofweek
                                    df['month'] = df.index.month
                                    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
                                    df['season'] = df['month'].apply(lambda x: 1 if x in [12, 1, 2] else 
                                                                  2 if x in [3, 4, 5] else 
                                                                  3 if x in [6, 7, 8] else 4)
                                    
                                    # Get the forecast
                                    model_choice = 'lstm'  # Default to LSTM
                                        
                                    # Display loading information
                                    st.info(f"Using {model_choice.upper()} model for forecasting...")
                                    
                                    # Create pipeline for preprocessing
                                    pipeline = MockPipeline()
                                    X_seq, _ = pipeline.create_sequences(df, 
                                                                      lookback=7, 
                                                                      target_col='energy_sum',
                                                                      forecast_steps=7)
                                    
                                    # Make prediction
                                    predictions = models[model_choice].predict(X_seq)
                                    
                                    # Display the forecast
                                    forecast_dates = [end_date + timedelta(days=i+1) for i in range(7)]
                                    forecast_df = pd.DataFrame({
                                        'Date': forecast_dates,
                                        'Forecasted Consumption (kWh)': predictions.flatten()[:7]
                                    })
                                    
                                    st.success("Forecast successfully generated!")
                                    st.dataframe(forecast_df)
                                    
                                    # Plot the results
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    
                                    # Historical data
                                    ax.plot(input_data['days'], input_data['consumption'], 
                                          marker='o', linestyle='-', label='Extracted Data')
                                    
                                    # Forecast data
                                    ax.plot(forecast_dates, predictions.flatten()[:7], 
                                          marker='o', linestyle='--', color='red', label='Forecast')
                                    
                                    ax.set_title('Electricity Consumption Forecast from Bill Data')
                                    ax.set_xlabel('Date')
                                    ax.set_ylabel('Daily Consumption (kWh)')
                                    ax.grid(True, alpha=0.3)
                                    ax.legend()
                                    
                                    # Format x-axis dates
                                    fig.autofmt_xdate()
                                    
                                    st.pyplot(fig)
                                    
                                    # Add insights
                                    st.subheader("Insights")
                                    
                                    # Calculate statistics
                                    avg_historical = np.mean(input_data['consumption'])
                                    avg_forecast = np.mean(predictions.flatten()[:7])
                                    pct_change = ((avg_forecast - avg_historical) / avg_historical) * 100
                                    
                                    # Create three columns
                                    c1, c2, c3 = st.columns(3)
                                    c1.metric("Avg. Bill Consumption", f"{daily_avg:.2f} kWh", "Daily average")
                                    c2.metric("Avg. Forecasted Consumption", f"{avg_forecast:.2f} kWh", "Next 7 days")
                                    c3.metric("Consumption Trend", f"{pct_change:.1f}%", 
                                            f"{'↑' if pct_change > 0 else '↓'} from bill average")
                                    
                                    # Add more detailed insights
                                    st.write("### Bill vs. Forecast Analysis")
                                    
                                    total_forecast = avg_forecast * 30  # Assuming a month
                                    bill_difference = ((total_forecast - consumption) / consumption) * 100
                                    
                                    if abs(bill_difference) > 10:
                                        st.warning(f"⚠️ Your projected monthly consumption would be {total_forecast:.2f} kWh, which is {abs(bill_difference):.1f}% {'higher' if bill_difference > 0 else 'lower'} than your last bill.")
                                    else:
                                        st.info(f"ℹ️ Your projected monthly consumption would be {total_forecast:.2f} kWh, similar to your last bill ({consumption:.2f} kWh).")
                                    
                                    # Cost projection (assuming a simple rate)
                                    rate = 0.15  # £/kWh - This should be customizable in a real app
                                    current_cost = consumption * rate
                                    projected_cost = total_forecast * rate
                                    
                                    st.write(f"### Cost Projection (at £{rate:.2f}/kWh)")
                                    cost_cols = st.columns(2)
                                    cost_cols[0].metric("Current Bill Cost", f"£{current_cost:.2f}")
                                    cost_cols[1].metric("Projected Bill Cost", f"£{projected_cost:.2f}", 
                                                    f"{(projected_cost - current_cost):.2f} ({(projected_cost - current_cost) / current_cost * 100:.1f}%)")
                                    
                                    # Recommendations
                                    st.write("### Recommendations")
                                    recommendations = []
                                    
                                    if bill_difference > 10:
                                        recommendations.append("Your consumption is trending upward. Consider energy-saving measures.")
                                        recommendations.append("Review usage of major appliances during peak hours.")
                                    elif bill_difference < -10:
                                        recommendations.append("Your consumption is trending downward. Great job!")
                                        recommendations.append("Continue your current energy-saving practices.")
                                    else:
                                        recommendations.append("Your consumption is stable. Consider setting a reduction goal.")
                                    
                                    if consumption > 300:  # Arbitrary threshold
                                        recommendations.append("Your overall consumption is higher than average. Consider an energy audit.")
                                    
                                    for i, rec in enumerate(recommendations):
                                        st.write(f"{i+1}. {rec}")
                            except Exception as e:
                                st.error(f"Error generating forecast: {str(e)}")
                                st.info("Please try the manual entry form instead.")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    st.info("Please try the manual entry form instead.")
        else:
            st.info("Please use the manual entry form to input your consumption data.")

# Add footer
st.markdown("---")
st.markdown("Smart Meter Bill Processor Demo | For testing and demonstration purposes only")
st.markdown("Note: This demo uses synthetic predictions and doesn't require the full model.")

# Sidebar
with st.sidebar:
    st.header("About this Demo")
    st.markdown("""
    This is a standalone demo of the Bill Processor functionality.
    
    It allows you to:
    - Manually enter electricity consumption data
    - Upload electricity bills (if OCR libraries are installed)
    - Generate forecasts based on your data
    - See insights and recommendations
    
    **Note**: This demo uses synthetic predictions for demonstration purposes.
    """)
    
    st.header("Required Dependencies")
    
    # Check for required dependencies
    try:
        import pytesseract
        st.success("✅ pytesseract is installed")
    except ImportError:
        st.error("❌ pytesseract is not installed")
        
    try:
        import pdfplumber
        st.success("✅ pdfplumber is installed")
    except ImportError:
        st.error("❌ pdfplumber is not installed")
    
    try:
        from PIL import Image
        st.success("✅ PIL/Pillow is installed")
    except ImportError:
        st.error("❌ PIL/Pillow is not installed")
        
    st.markdown("""
    To install missing dependencies:
    ```
    pip install pytesseract pdfplumber Pillow
    ```
    
    For OCR functionality, you also need to install Tesseract OCR.
    """) 