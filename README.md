# House Price Prediction

House Price Prediction estimates residential property prices using features such as area, bedrooms, bathrooms, stories, parking, and various amenities. The project focuses on building a clean pipeline, robust evaluation, and a modern Streamlit UI for predictions.

## Features
- **End-to-End Pipeline**: Automated data preprocessing and model training using `scikit-learn` Pipelines.
- **Advanced Modeling**: Uses `RandomForestRegressor` with `GridSearchCV` for hyperparameter tuning.
- **Modern UI**: Interactive web interface built with `Streamlit` for real-time price estimation.
- **Categorical Support**: Handles categorical features like furnishing status, air conditioning, and location preference.

## Problem Definition
- **Problem Statement**: Predict the sale price of a house using tabular features describing its physical characteristics and amenities.
- **Type**: Supervised Learning (Regression)
- **Target Variable**: `SalePrice`
- **Evaluation Metrics**: R² (Coefficient of Determination), MAE (Mean Absolute Error), RMSE (Root Mean Squared Error).

## Dataset Features
The model uses the following features for prediction:
- `area`: Total area in square feet.
- `bedrooms`: Number of bedrooms.
- `bathrooms`: Number of bathrooms.
- `stories`: Number of floors/stories.
- `mainroad`: Proximity to the main road (yes/no).
- `guestroom`: Availability of a guestroom (yes/no).
- `basement`: Availability of a basement (yes/no).
- `hotwaterheating`: Availability of hot water heating (yes/no).
- `airconditioning`: Availability of air conditioning (yes/no).
- `parking`: Number of parking spaces.
- `prefarea`: Preferred area status (yes/no).
- `furnishingstatus`: Furnishing status (furnished, semi-furnished, unfurnished).

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/neeraj214/House-Price-Prediction.git
   cd House-Price-Prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**:
   ```bash
   python main.py
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## Project Structure
- `src/`: Core logic for preprocessing, training, and evaluation.
- `models/`: Saved model pipelines and evaluation results.
- `data/`: Raw and processed data storage.
- `app.py`: Streamlit web application.
- `main.py`: CLI entry point for training.
