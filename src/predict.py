import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Global variables to store the model and scalers
model = None
scaler = None
y_scaler = None

def get_model():
    """
    Lazy loads the model and scalers on the first request and returns them for subsequent requests.
    Raises RuntimeError if the model or scalers cannot be loaded.
    
    Returns:
        model: The pretrained model
        scaler: The feature scaler
        y_scaler: The target scaler
    """
    global model, scaler, y_scaler
    if model is None or scaler is None or y_scaler is None:
        try:
            # Load the dictionary containing model and scalers
            model_data = joblib.load("models/lgbm_all_meter.joblib")
            
            # Extract model, scaler, and y_scaler
            model = model_data["model"]
            scaler = model_data["scaler"]
            y_scaler = model_data["y_scaler"]
            
            logging.info("Model and scalers loaded successfully.")
        except Exception as e:
            logging.error("Failed to load the model or scalers: %s", e)
            raise RuntimeError("Model or scalers could not be loaded.")
    
    return model, scaler, y_scaler


# Validate the input data for model prediction
def validate_input(input_data):
    """
    Validate the input data for the predict endpoint.
    
    Args:
        input_data (dict): The JSON input data from the request.
        
    Returns:
        tuple: (bool, str) - A tuple where the first element indicates if the input is valid,
                             and the second element is a message explaining the validation result.
    """
    required_fields = ["building_id", "site_id", "meter", "date", "primaryspaceusage", "sqm", "airTemperature"]

    # Check if all required fields are present
    for field in required_fields:
        if field not in input_data:
            return False, f"'{field}' field is required in the request body."

    # Check types and constraints
    if not isinstance(input_data["site_id"], int):
        return False, "'site_id' must be an integer."

    if input_data["meter"] not in ['electricity', 'chilledwater', 'gas', 'hotwater', 'solar', 'water', 'steam', 'irrigation']:
        return False, "'meter' must be one of ['electricity', 'chilledwater', 'gas', 'hotwater', 'solar', 'water', 'steam', 'irrigation']."

    try:
        # Check if date is in the correct format YYYY-MM-DD
        pd.to_datetime(input_data["date"], format="%Y-%m-%d")
    except ValueError:
        return False, "'date' must be in the format YYYY-MM-DD."

    if input_data["primaryspaceusage"] not in ['Education', 'Office']:
        return False, "'primaryspaceusage' must be one of ['Education', 'Office']."

    if not isinstance(input_data["sqm"], (int, float)):
        return False, "'sqm' must be a numeric value."

    if not isinstance(input_data["airTemperature"], (int, float)):
        return False, "'airTemperature' must be a numeric value."

    # If all checks pass
    return True, "Input is valid."


# Define features and types based on dataset
numerical_features = ['sqm', 'airTemperature']
categorical_features = ['meter', 'primaryspaceusage', 'site_id', 'weekday', 'month']

# Preprocess data for model prediction
def preprocess_data(X, y=None, fit_scaler=False, scaler=None, y_scaler=None):
    """
    Preprocesses the input data by transforming categorical and numerical features.
    
    Args:
        X (pd.DataFrame): The input features DataFrame.
        y (pd.Series, optional): The target variable for training. Default is None.
        fit_scaler (bool): If True, fit the scalers; otherwise, use provided scalers. Default is False.
        scaler (StandardScaler, optional): The scaler for numerical features, fitted during training.
        y_scaler (StandardScaler, optional): The scaler for the target variable, fitted during training.

    Returns:
        If training (fit_scaler=True):
            Tuple[pd.DataFrame, np.ndarray, StandardScaler, StandardScaler]: Processed features, scaled target, feature scaler, target scaler.
        If testing (fit_scaler=False):
            Tuple[pd.DataFrame, np.ndarray]: Processed features and scaled target.
    """
    if 'date' in X.columns:
        X['date'] = pd.to_datetime(X['date'], format="%Y-%m-%d")
        X['month'] = X['date'].dt.month
        X['weekday'] = X['date'].dt.weekday
        X = X.drop(columns=['date', 'building_id'])  # Drop 'date' if it’s not needed for prediction
    
    # Convert categorical features to 'category' type to ensure consistency
    for cat_col in categorical_features:
        if cat_col in X.columns:
            X[cat_col] = X[cat_col].astype('category')
    
    # Log-transform 'sqm' to handle skewness
    if 'sqm' in X.columns:
        X['sqm'] = np.log1p(X['sqm'])

    # Scaling numerical features
    if fit_scaler:
        # Initialize the scaler for numerical features if fitting for the first time
        scaler = StandardScaler()
        X[numerical_features] = scaler.fit_transform(X[numerical_features])
        
        if y is not None:
            # Cap extreme values at the 99th percentile in the target
            cap_value = np.percentile(y, 99)
            y = np.clip(y, None, cap_value)  # Cap target variable at the 99th percentile
            
            # Log-transform and scale the target variable
            y_log = np.log1p(y)
            y_scaler = StandardScaler()
            y_scaled = y_scaler.fit_transform(y_log.values.reshape(-1, 1)).flatten()
            
            return X, y_scaled, scaler, y_scaler
        else:
            return X, scaler
    else:
        # Transform numerical features using the provided scaler
        X[numerical_features] = scaler.transform(X[numerical_features])
        
        if y is not None:
            # Log-transform and scale the target variable using the provided target scaler
            y_log = np.log1p(y)
            y_scaled = y_scaler.transform(y_log.values.reshape(-1, 1)).flatten()
            return X, y_scaled
        else:
            return X