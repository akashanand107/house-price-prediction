import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from preprocessing import load_and_preprocess_data

def train_model():
    """Trains the XGBoost model and saves the artifacts."""
    print("🚀 Loading and Preprocessing Data...")
    X, y = load_and_preprocess_data("train.csv", is_training=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"📊 Dataset split: Train={len(X_train)}, Test={len(X_test)}")

    # Initialize and train model
    print("⚙️ Training XGBoost Regressor...")
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluation
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    print("\n✅ Model Performance:")
    print(f"   - Mean Absolute Error: ${mae:,.2f}")
    print(f"   - R2 Score: {r2:.4f}")
    print(f"   - Train Accuracy: {model.score(X_train, y_train):.4f}")
    print(f"   - Test Accuracy: {model.score(X_test, y_test):.4f}")

    # Save artifacts
    print("\n💾 Saving model and feature columns...")
    joblib.dump(model, "house_price_model.pkl")
    joblib.dump(X.columns, "model_columns.pkl")
    print("✨ Model saved successfully!")

if __name__ == "__main__":
    train_model()
