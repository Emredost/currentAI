# Example Files for Bill Processing

This directory contains example files that can be used to test the bill processing functionality.

## Sample Bill Files

1. `sample_bill.jpg` - A sample electricity bill image that can be used to test the OCR functionality.

## How to Use the Sample Files

1. Navigate to the Bill Processor tab in the dashboard
2. Upload the sample bill file using the file upload component
3. The system will attempt to extract consumption data from the bill
4. You can then verify the extracted information and generate a forecast

## Adding Your Own Sample Files

You can add your own sample bill files to this directory for testing purposes. Currently supported formats include:

- PDF files
- Image files (JPG, JPEG, PNG)

Note that OCR extraction works best with clear, high-resolution images where text is crisp and easily readable.

## Sample Bill Format

The sample bill includes the following information that the system attempts to extract:

- Customer ID: Customer123456
- Total consumption: 320.5 kWh
- Billing period: 01/01/2023 to 31/01/2023
- Daily average: 10.34 kWh

If you're creating your own sample bill for testing, including these fields in a clearly formatted way will help the OCR engine extract them correctly.

## Troubleshooting

If the OCR engine fails to extract data from your sample bill:

1. Check that the text is clearly visible and not distorted
2. Ensure the file is not corrupted
3. Try converting to a different format (e.g., from PDF to JPG)
4. Adjust the regular expressions in the code to match your bill format 