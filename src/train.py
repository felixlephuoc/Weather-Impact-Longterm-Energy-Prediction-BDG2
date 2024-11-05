import time
import logging
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Define features based on dataset
numerical_features = ['sqm', 'airTemperature']
categorical_features = ['meter', 'primaryspaceusage', 'site_id', 'weekday', 'month']

# Metric functions
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def coefficient_of_variation_root_mean_squared_error(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    return rmse / np.mean(y_true) * 100

def mean_bias_error(y_true, y_pred):
    return np.mean(y_true - y_pred)

def normalized_mean_bias_error(y_true, y_pred):
    return (np.sum(y_true - y_pred) / (len(y_true) * np.mean(y_true))) * 100

def r_squared(y_true, y_pred):
    return r2_score(y_true, y_pred)

# Preprocess data
def preprocess_data(X, y=None, fit_scaler=False, scaler=None, y_scaler=None, is_training=True):
    """
    Preprocesses the input data by transforming categorical and numerical features and adding date-based features.

    Args:
        X (pd.DataFrame): The input features DataFrame.
        y (pd.Series, optional): The target variable for training. Default is None.
        fit_scaler (bool): If True, fit the scalers; otherwise, use provided scalers. Default is False.
        scaler (StandardScaler, optional): The scaler for numerical features, fitted during training.
        y_scaler (StandardScaler, optional): The scaler for the target variable, fitted during training.
        is_training (bool): If True, prepare data for training. If False, prepare for testing/inference.

    Returns:
        If training (fit_scaler=True):
            Tuple[pd.DataFrame, np.ndarray, StandardScaler, StandardScaler]: Processed features, scaled target, feature scaler, target scaler.
        If testing (fit_scaler=False):
            Tuple[pd.DataFrame, np.ndarray]: Processed features and scaled target.
    """
    
    # Parse and process the 'date' column if it exists
    if 'date' in X.columns:
        X['date'] = pd.to_datetime(X['date'], format="%Y-%m-%d")
        X['month'] = X['date'].dt.month
        X['weekday'] = X['date'].dt.weekday
        X = X.drop(columns=['date'])  # Drop 'date' column after processing
    
    # Drop additional columns in testing mode if necessary
    if not is_training and 'building_id' in X.columns:
        X = X.drop(columns=['building_id'])

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

# Time estimator callback for training progress
class TimeEstimatorCallback:
    def __init__(self, num_boost_round, display_interval=50):
        self.num_boost_round = num_boost_round
        self.display_interval = display_interval
        self.start_time = None

    def __call__(self, env):
        if env.iteration == 0:
            self.start_time = time.time()
        
        if env.iteration % self.display_interval == 0:
            elapsed_time = time.time() - self.start_time
            avg_time_per_iter = elapsed_time / (env.iteration + 1)
            remaining_iters = self.num_boost_round - env.iteration
            estimated_remaining_time = remaining_iters * avg_time_per_iter
            
            if env.evaluation_result_list:
                rmse = env.evaluation_result_list[0][2]
                print(f"[Iteration {env.iteration}] RMSE: {rmse:.4f} | Elapsed: {elapsed_time:.2f}s | Estimated Remaining: {estimated_remaining_time:.2f}s")

# Train model function
def train_final_model(X_train, y_train, best_params):
    final_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'device': 'cpu',
        'verbosity': -1,
    }
    final_params.update(best_params)

    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
    
    num_boost_round = best_params.get('num_boost_round', 1000)

    model = lgb.train(
        final_params,
        lgb_train,
        num_boost_round=num_boost_round,
        valid_sets=[lgb_train],
        callbacks=[lgb.log_evaluation(100), TimeEstimatorCallback(num_boost_round, 100)]
    )

    return model

# Main function to handle the training process
def train(best_params=None):
    # Set default values if not provided
    best_params = {
        'num_leaves': best_params.get('num_leaves', 1000),
        'learning_rate': best_params.get('learning_rate', 0.01),
        'max_bin': best_params.get('max_bin', 512),
        'lambda_l1': best_params.get('lambda_l1', 0.01),
        'lambda_l2': best_params.get('lambda_l2', 0.01),
        'min_child_samples': best_params.get('min_child_samples', 20),
        'max_depth': best_params.get('max_depth', 15),
        'num_boost_round': best_params.get('num_boost_round', 1000)
    }

    # Load and preprocess data
    temp_df = pd.read_csv("data/cleaned/train.csv", nrows=0)
    columns_to_use = temp_df.columns[1:]
    train_data = pd.read_csv("data/cleaned/train.csv", usecols=columns_to_use)
    
    train_data = train_data.drop(columns=['building_name', 'site_name', 'sqft', 'sub_primaryspaceusage', 'timezone', 'season'])
    train_data['date'] = pd.to_datetime(train_data['date'])
    train_data.set_index('building_id', inplace=True)

    X_meter = train_data.drop(columns=['meter_reading', 'date'])
    y_meter = train_data['meter_reading']

    X_train_full_processed, y_train_full_scaled, scaler, y_scaler = preprocess_data(
        X_meter.copy(), y_meter, fit_scaler=True
    )

    model = train_final_model(X_train_full_processed, y_train_full_scaled, best_params)
    
    # Make predictions on the training dataset
    y_train_pred_scaled = model.predict(X_train_full_processed)

    # Inverse transform the scaled predictions and target back to original scale
    y_train_pred_log = y_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
    y_train_pred = np.expm1(y_train_pred_log)  # Reverse log transformation
    
    y_train_true_log = y_scaler.inverse_transform(y_train_full_scaled.reshape(-1, 1)).flatten()
    y_train_true = np.expm1(y_train_true_log)  # Reverse log transformation

    # Calculate metrics and format for reporting
    metrics = {
        "root_mean_squared_error": round(root_mean_squared_error(y_train_true, y_train_pred), 4),
        "coefficient_of_variation_rmse": round(coefficient_of_variation_root_mean_squared_error(y_train_true, y_train_pred), 4),
        "mean_bias_error": round(mean_bias_error(y_train_true, y_train_pred), 4),
        "normalized_mean_bias_error": round(normalized_mean_bias_error(y_train_true, y_train_pred), 4),
        "r_squared": round(r_squared(y_train_true, y_train_pred), 4),
    }

    # Convert CV-RMSE and NMBE to percentage format
    metrics["coefficient_of_variation_rmse"] = f"{metrics['coefficient_of_variation_rmse']}%"
    metrics["normalized_mean_bias_error"] = f"{metrics['normalized_mean_bias_error']}%"
    
    # Save model and scalers
    model_data = {
        "model": model,
        "scaler": scaler,
        "y_scaler": y_scaler
    }
    joblib.dump(model_data, "./models/lgbm_all_meter_test.joblib")
    logging.info("Model and scalers saved successfully.")
    return metrics # Return the evaluation metrics for the training set