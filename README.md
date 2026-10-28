# Chennai Temperature Prediction Using Machine Learning

## Project Overview

This project uses historical daily weather data for the Chennai region to predict the next day's temperature using machine learning.

Three regression models are implemented and compared:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

The project also includes a Streamlit application for interactive demonstration.

## Dataset

- **Source:** NASA POWER
- **Location:** Chennai region, India
- **Period:** 1 January 2005 – 31 December 2024
- **Frequency:** Daily weather observations

The dataset contains meteorological variables such as temperature, relative humidity, wind speed, atmospheric pressure, and precipitation.

NASA POWER data portal: https://power.larc.nasa.gov/

## Project Files

- `temperature_model.py` — Data preprocessing, model training, evaluation, and result generation.
- `demo.py` — Demonstrates predictions from the three trained models.
- `app.py` — Streamlit application for interactive use.
- `requirements.txt` — Required Python libraries.

## Models

The models are evaluated using regression metrics, including Mean Absolute Error (MAE) and R-squared (R²).

## Installation

Install Python, then install the required libraries:

```bash
pip install -r requirements.txt
```

## Running the Project

Train the models:

```bash
python temperature_model.py
```

Run the demonstration:

```bash
python demo.py
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

Run the commands from the project directory. The training script must be run first if the application requires its generated models and result files.

## Limitations

Predictions depend on the available historical weather data and selected input features. Future predictions are estimates and should not be treated as official weather forecasts.

## Author

**Basil George**

VIT Chennai  
B.Tech — Mechatronics and Automation

## Repository

https://github.com/Basilgmat/temperature-prediction
