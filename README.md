# House_Price_Prediction_Using_LinearRegression

## Supervised Machine Learning Model

## Overview

This project implements a supervised machine learning model to predict house prices using a linear regression approach. Leveraging a dataset with features such as area, number of bedrooms, bathrooms, and additional amenities, the model is trained to provide accurate price predictions. The solution includes data preprocessing, exploratory data analysis, model training, evaluation, and a deployable Streamlit web application for real-time predictions.

## Dataset

The dataset, stored as `Housing.csv` in the `data` folder, contains the following features:
- `price`: The target variable representing house price (in some unit, e.g., thousands).
- `area`: House size in square feet.
- `bedrooms`: Number of bedrooms.
- `bathrooms`: Number of bathrooms.
- `stories`: Number of stories in the house.
- `mainroad`: Binary indicator (1 = yes, 0 = no) for main road access.
- `airconditioning`: Binary indicator (1 = yes, 0 = no) for air conditioning.
- `parking`: Number of parking spaces.
- `prefarea`: Binary indicator (1 = yes, 0 = no) for preferred area.
- `furnishingstatus`: Categorical indicator (e.g., 0 = unfurnished, 1 = semi-furnished, 2 = furnished).

A cleaned version, `cleaned_housing.csv`, is generated after preprocessing to handle missing values and encode categorical variables. Each row represents a unique house with its attributes.

## Methodology

1. **Data Preprocessing**:
   - Loaded `Housing.csv` using pandas.
   - Handled missing values through imputation or removal.
   -`handled the duplicates values.

2. **Exploratory Data Analysis (EDA)**:
   - Performed in `exploratory_data_analysis.ipynb` using matplotlib and seaborn.
   - Analyzed correlations (e.g., `area` vs. `price`) and visualized distributions to identify trends.
    - Visualized categorical features to understand their impact on house prices.
    - Plotted relationships between numerical features and the target variable.
    - Used pair plots and heatmaps to explore feature interactions and correlations.
    - Generated summary statistics to understand data distribution and feature significance.
    - Identified outliers and anomalies in the dataset.
    - Created visualizations to illustrate key findings, such as price distributions and feature correlations.


3. **Feature Selection**:
   - Selected key features (e.g., `area`, `bedrooms`, `bathrooms`) based on correlation analysis.
   - Retained binary features (`mainroad`, `airconditioning`, `prefarea`) for their predictive power.

4. **Model Training**:
   - Split data into training (80%) and testing (20%) sets.
   - Trained a linear regression model using scikit-learn in `model_training.ipynb`.
   - Saved the model as `linear_regression_model.pkl` and scaler as `scaler.pkl`.

5. **Model Evaluation**:
   - Assessed performance with metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared.
   - Validated model generalizability on the test set.

6. **Deployment**:
   - Integrated the model into a Streamlit app (`app.py`) for user-friendly predictions.
   - Users can input house features to receive predicted prices.

## Requirements

- **Python 3.x**
- **Libraries**:
  - `pandas` for data manipulation
  - `numpy` for numerical operations
  - `scikit-learn` for machine learning
  - `matplotlib` and `seaborn` for visualization
  - `streamlit` for web app
  - `pickle` for serialization

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ArunPandeyLaudari/HousePricePrediction_LinearRegression_Model.git
   cd House_Price_Prediction_Using_LinearRegression
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Activate the virtual environment (as above).
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open the local URL (e.g., `http://localhost:8501`) in your browser to use the app.

## Input Features
- `area`: Numeric input for house size in square feet.
- `bedrooms`: Numeric input for the number of bedrooms.
- `bathrooms`: Numeric input for the number of bathrooms.
- `stories`: Numeric input for the number of stories.
- `mainroad`: Select from a dropdown (Yes/No) for main road access.
- `airconditioning`: Select from a dropdown (Yes/No) for air conditioning.
- `parking`: Numeric input for the number of parking spaces.
- `prefarea`: Select from a dropdown (Yes/No) for preferred area.
- `furnishingstatus`: Select from a dropdown (Unfurnished/Semi-furnished/Furnished).
## Output
- Predicted house price displayed in the app after inputting the features.


## Project Structure

- `data/`: Contains `Housing.csv` (raw) and `cleaned_housing.csv` (processed).
- `notebook/`: Includes `exploratory_data_analysis.ipynb` and `model_training.ipynb`.
- `linear_regression_model.pkl`: Trained linear regression model.
- `scaler.pkl`: Saved scaler for data normalization.
- `app.py`: Streamlit application.
- `requirements.txt`: Dependency list.
- `README.md`: Project documentation.
- `.gitignore`: Version control exclusions.

## Contributions

- Implemented data preprocessing and EDA.
- Developed and trained the linear regression model.
- Created and deployed the Streamlit prediction app.

## Future Improvements

- Incorporate advanced models (e.g., Random Forest, Gradient Boosting) for comparison.
- Add cross-validation to enhance model robustness.
- Expand the Streamlit app with visualizations and input validation.
- Implement hyperparameter tuning for better model performance.

## Developer
Arun Pandey Laudari