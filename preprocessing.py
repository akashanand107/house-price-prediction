import pandas as pd
import joblib

def load_and_preprocess_data(filepath="train.csv", is_training=True):
    """
    Loads and preprocesses the house price dataset.
    
    Args:
        filepath (str): Path to the CSV file.
        is_training (bool): If True, treats the data as training data (drops 'Id' and returns 'SalePrice').
        
    Returns:
        pd.DataFrame: Preprocessed features.
        pd.Series (optional): Target variable if is_training is True.
    """
    data = pd.read_csv(filepath)
    
    # Drop Id column
    if "Id" in data.columns:
        data = data.drop("Id", axis=1)
    
    # Fill numeric columns with median
    data = data.fillna(data.median(numeric_only=True))
    
    # Fill categorical columns with "Missing"
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].fillna("Missing")
        
    if is_training and "SalePrice" in data.columns:
        y = data["SalePrice"]
        X = data.drop("SalePrice", axis=1)
        X = pd.get_dummies(X, drop_first=True)
        return X, y
    else:
        X = pd.get_dummies(data)
        return X

def align_features(df, model_columns):
    """
    Aligns the features of a dataframe with the model's training columns.
    
    Args:
        df (pd.DataFrame): The dataframe to align.
        model_columns (Index/list): The columns used during training.
        
    Returns:
        pd.DataFrame: Aligned dataframe.
    """
    return df.reindex(columns=model_columns, fill_value=0)
