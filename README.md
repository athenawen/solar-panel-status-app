# Solar Panel Status App

## Overview
This app is a Streamlit-based tool for detecting underperforming or overperforming solar panels using daily generation and weather sensor data.  
It applies monotonic regression with outlier detection to compare actual panel output against expected performance given irradiation/irradiance data.

## Features
- Data ingestion from multiple solar plants’ CSV files (generation and weather data)  
- Automatic sensor column detection (`IRRADIANCE`, `IRRADIATION`, or equivalent)  
- Isotonic regression model to estimate expected panel yield  
- Robust z-score method to detect anomalies (underperformance or overperformance)  
- Visualization of panel performance vs. expected output  
- Flagging system that stores details for each anomaly, including:
  - Date  
  - Plant name  
  - Source key (panel identifier)  
  - Measured yield and expected yield  
  - Residuals and z-score  
  - Performance label  
- Metadata display (e.g., panel location, model, manufacturer)  
- Interactive "Schedule a Check" button with confirmation message  

## Data Requirements
The app expects the following CSV files in the working directory:
- `Plant_1_Generation_Data.csv`
- `Plant_1_Weather_Sensor_Data.csv`
- `Plant_2_Generation_Data.csv`
- `Plant_2_Weather_Sensor_Data.csv`
- Original Dataset Used: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data/data

**Generation Data CSV** must include:
- `DATE_TIME` (timestamp)
- `DAILY_YIELD`
- `SOURCE_KEY` (panel identifier)

**Weather Data CSV** must include:
- `DATE_TIME`
- Irradiance or irradiation column (name detected automatically)

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/solar-panel-status-app.git
cd solar-panel-status-app
```

2. Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate       # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the app
```bash
streamlit run app.py
```


