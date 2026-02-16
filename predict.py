import pandas as pd
import joblib
from preprocessing import align_features

def predict_house_price():
    print("🏠 Welcome to the House Price Prediction Tool")
    print("-------------------------------------------")

    try:
        # Load saved model and columns
        model = joblib.load("house_price_model.pkl")
        model_columns = joblib.load("model_columns.pkl")
    except FileNotFoundError:
        print("❌ Model files not found! Please run 'regression_model.py' first to train the model.")
        return

    # Take user inputs
    print("\nPlease enter the details of the house:")
    try:
        lot_area = float(input("   Lot Area (sq ft, e.g., 9000): "))
        overall_qual = int(input("   Overall Quality (1-10): "))
        gr_liv_area = float(input("   Living Area (sq ft, e.g., 1800): "))
        garage_cars = int(input("   Garage Cars (e.g., 2): "))
        year_built = int(input("   Year Built (e.g., 2005): "))
    except ValueError:
        print("❌ Invalid input! Please enter numeric values.")
        return

    # Create input DataFrame
    input_data = {
        "LotArea": lot_area,
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "GarageCars": garage_cars,
        "YearBuilt": year_built
    }
    input_df = pd.DataFrame([input_data])

    # Handle dummy variables (even if none for this simple input, it's good practice)
    input_df = pd.get_dummies(input_df)

    # Align columns with training data
    input_df = align_features(input_df, model_columns)

    # Predict
    prediction = model.predict(input_df)
    
    print("-" * 43)
    print(f"✅ Estimated House Price: ${prediction[0]:,.2f}")
    print("-" * 43)

if __name__ == "__main__":
    predict_house_price()
