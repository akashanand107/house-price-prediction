# 🏠 House Price Prediction Project

A professional machine learning project to predict house prices using XGBoost. This project demonstrates data preprocessing, model training, evaluation, and visualization.

## 🚀 Features
- **Centralized Preprocessing**: Consistent data cleaning and encoding logic.
- **XGBoost Training**: High-performance regression model.
- **Detailed Evaluation**: MAE, R2 scores, and train/test accuracy tracking.
- **Rich Visualizations**: Scatter plots, residual analysis, and feature importance.
- **Interactive Predictor**: Command-line tool for individual house price estimations.

## 📂 Project Structure
- `preprocessing.py`: Core logic for data cleaning and feature alignment.
- `regression_model.py`: Script to train and save the XGBoost model.
- `visualise.py`: Generates performance graphs and insights.
- `predict.py`: Interactive CLI tool for user predictions.
- `train.csv`: Dataset used for training and testing.

## 🛠️ Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage
1. **Train the Model**:
   ```bash
   python regression_model.py
   ```
2. **Visualize Results**:
   ```bash
   python visualise.py
   ```
3. **Make Predictions**:
   ```bash
   python predict.py
   ```

## 📊 Model Performance
The model uses **XGBoost Regressor** and evaluates performance using Mean Absolute Error (MAE) and R2 Score. Check `visualise.py` for comparative charts between actual and predicted prices.
