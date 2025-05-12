# Smart Meter Analytics Presentation

This directory contains the presentation materials for demonstrating the Smart Meter Analytics project to electricity companies and stakeholders.

## Contents

- `slides.md`: Presentation slides in Markdown format (for use with Marp, reveal.js, or similar)
- `demo_script.md`: Detailed script for giving the presentation and demonstration
- `images/`: Directory containing images and diagrams for the presentation
- `handout.pdf`: One-page handout summarizing key findings and business value

## Presentation Setup

### Requirements

- A working installation of the Smart Meter Analytics project
- Processed data loaded into the application
- Trained models available
- Streamlit dashboard running and accessible

### Presentation Flow

1. **Introduction**: Use slides 1-3
2. **Data Processing**: Use slides 4-5 and demonstrate the data loading in the dashboard
3. **Key Findings**: Use slides 6-8 and show the corresponding visualizations in the dashboard
4. **Models**: Use slides 9-11 and demonstrate the forecasting tab in the dashboard
5. **Business Applications**: Use slides 12-14 to discuss practical applications
6. **Q&A and Next Steps**: Use final slides and refer to the handout

## Dashboard Demonstration Guide

For a successful dashboard demonstration, prepare these scenarios in advance:

1. **Data Overview Tab**:
   - Highlight the number of households, time range, and data completeness
   - Show the distribution of ACORN groups

2. **Time Series Analysis Tab**:
   - Pre-select a household with interesting patterns (e.g., MAC000127)
   - Show daily and monthly consumption trends
   - Demonstrate weekday vs. weekend patterns

3. **Weather Impact Tab**:
   - Show the temperature vs. consumption scatter plot
   - Demonstrate the seasonal patterns chart
   - Highlight how humidity affects summer consumption

4. **Consumption Patterns Tab**:
   - Show hourly patterns across different household types
   - Demonstrate the impact of tariff types on peak consumption

5. **Forecast Tab**:
   - Select a household with good data quality (e.g., MAC000073)
   - Compare forecasts from different models
   - Demonstrate how forecast accuracy varies with prediction horizon

## Preparation Checklist

- [ ] Review the entire demo script and practice the presentation
- [ ] Ensure all images are properly placed in the `images/` directory
- [ ] Test the Streamlit dashboard with the specific examples mentioned above
- [ ] Prepare for potential questions using the Q&A section in the demo script
- [ ] Print handouts for all attendees

## Customization

The presentation can be customized for specific electricity companies:

1. Update the `slides.md` file to include company-specific information
2. Modify the demonstration examples to highlight aspects most relevant to the company
3. Adjust the business applications section to address their specific challenges
4. Update contact information and next steps based on the specific engagement

## Converting Slides

To convert the Markdown slides to a PowerPoint or PDF presentation:

1. **Using Marp**:
   ```
   marp slides.md --pdf
   ```

2. **Using Pandoc**:
   ```
   pandoc -t revealjs -s slides.md -o slides.html
   ```

3. **Using reveal.js**:
   - Install reveal.js
   - Configure to use the markdown file
   - Serve the presentation through a web server 