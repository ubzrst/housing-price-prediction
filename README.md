# Housing Price Prediction Model

This project implements a machine learning pipeline to predict housing prices based on various features such as area, number of bedrooms, and location.

The core of the prediction engine is a Gradient Boosted Trees regressor. Gradient Boosting is an ensemble technique that builds models sequentially, with each new model attempting to correct the errors of the previous ones. This results in a powerful predictor capable of capturing complex non-linear relationships in the data.

1.  `HousingDataPreprocessor`:
    -   Responsible for loading the raw CSV data.
    -   Handles preprocessing steps such as mapping binary 'yes/no' variables to 0/1 and creating dummy variables for categorical features like 'furnishingstatus'.
    -   Splits the data into training and testing sets.

2.  `HousingModel`:
    -   Encapsulates the `GradientBoostingRegressor`.
    -   Provides methods for training (`train`), prediction (`predict`), and evaluation (`evaluate`).
    -   Calculates metrics like Root Mean Squared Error (RMSE) and R-squared (R2).

3.  `HousingVisualizer`:
    -   Generates plots to help interpret the model's performance.
    -   Produces a Feature Importance plot, Actual vs. Predicted price scatter plot, and a Residuals histogram.

## How to Run
1.  Set up the environment
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  Run the pipeline
    ```bash
    python main.py
    ```

3.  View Results
    The script will output evaluation metrics to the console and save visualizations to `housing_analysis.png`.
