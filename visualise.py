import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preprocessing import load_and_preprocess_data, align_features

def run_visualisation():
    print("📈 Generating Visualisations...")
    
    # 1. Load and Preprocess
    X, y = load_and_preprocess_data("train.csv", is_training=True)
    
    # Align columns exactly with the saved model's expectations
    model_columns = joblib.load("model_columns.pkl")
    X = align_features(X, model_columns)

    # 2. Split and Load Model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = joblib.load("house_price_model.pkl")
    y_pred = model.predict(X_test)

    # 3. Plotting
    # --- Actual vs Predicted Scatter ---
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    plt.show()

    # --- Residual Histogram ---
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=40, edgecolor='black')
    plt.xlabel("Prediction Error")
    plt.title("Residual Error Distribution")
    plt.show()

    # --- Feature Importance ---
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": model_columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(15)

    plt.figure(figsize=(12, 8))
    plt.barh(importance_df["Feature"], importance_df["Importance"], color='skyblue')
    plt.xlabel("Importance Score")
    plt.title("Top 15 Most Important Features")
    plt.gca().invert_yaxis()
    plt.show()

    # --- Comparison Bar (First 20 Houses) ---
    comparison_df = pd.DataFrame({
        "Actual": y_test.values[:20],
        "Predicted": y_pred[:20]
    })
    comparison_df.plot(kind="bar", figsize=(12, 6))
    plt.title("Actual vs Predicted Prices (Sample of 20 Houses)")
    plt.ylabel("House Price ($)")
    plt.legend(["Actual", "Predicted"])
    plt.show()

if __name__ == "__main__":
    run_visualisation()
