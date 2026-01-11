import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import os

class HousingDataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def load_data(self):
        """Loads data from CSV."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")
        self.data = pd.read_csv(self.filepath)
        print(f"Data loaded successfully. Shape: {self.data.shape}")

    def preprocess(self):
        """Clean and prepare data for training."""
        if self.data is None:
            self.load_data()

        binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                       'airconditioning', 'prefarea']
        
        for col in binary_cols:
            if col in self.data.columns:
                self.data[col] = self.data[col].apply(lambda x: 1 if x == 'yes' else 0)

        if 'furnishingstatus' in self.data.columns:
            status = pd.get_dummies(self.data['furnishingstatus'], drop_first=True)
            self.data = pd.concat([self.data, status], axis=1)
            self.data.drop(['furnishingstatus'], axis=1, inplace=True)

        if 'price' not in self.data.columns:
             raise ValueError("Target column 'price' not found in dataset.")
             
        X = self.data.drop('price', axis=1)
        y = self.data['price']
        
        self.feature_names = X.columns.tolist()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("Data preprocessing complete.")
        return self.X_train, self.X_test, self.y_train, self.y_test

class HousingModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )

    def train(self, X_train, y_train):
        """Train the model."""
        print("Training Gradient Boosted Trees model...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        print(f"Model Evaluation:")
        print(f"RMSE: {rmse:,.2f}")
        print(f"R2 Score: {r2:.4f}")
        return predictions, r2

class HousingVisualizer:
    def __init__(self, save_path="housing_analysis.png"):
        self.save_path = save_path

    def plot_results(self, model, feature_names, y_test, y_pred):
        """Create visualizations for analysis."""
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        sns.barplot(
            x=[importances[i] for i in indices], 
            y=[feature_names[i] for i in indices], 
            ax=axes[0], 
            palette="viridis",
            hue=[feature_names[i] for i in indices],
            legend=False
        )
        axes[0].set_title("Feature Importance")
        axes[0].set_xlabel("Importance Score")

        sns.scatterplot(x=y_test, y=y_pred, ax=axes[1], alpha=0.6, color='b')
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[1].set_title("Actual vs Predicted Prices")
        axes[1].set_xlabel("Actual Price")
        axes[1].set_ylabel("Predicted Price")
        
        residuals = y_pred - y_test
        sns.histplot(residuals, kde=True, ax=axes[2], color='purple')
        axes[2].set_title("Residuals Distribution")
        axes[2].set_xlabel("Residuals")

        plt.tight_layout()
        plt.savefig(self.save_path)
        print(f"Visualizations saved to {self.save_path}")

def main():
    filepath = 'housing.csv'
    
    processor = HousingDataPreprocessor(filepath)
    X_train, X_test, y_train, y_test = processor.preprocess()
    
    model_wrapper = HousingModel(n_estimators=200, learning_rate=0.05, max_depth=4)
    model_wrapper.train(X_train, y_train)
    
    y_pred, r2 = model_wrapper.evaluate(X_test, y_test)
    
    visualizer = HousingVisualizer()
    visualizer.plot_results(model_wrapper.model, processor.feature_names, y_test, y_pred)

if __name__ == "__main__":
    main()
