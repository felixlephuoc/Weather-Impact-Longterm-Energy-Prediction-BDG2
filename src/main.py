import os
import sys
import logging
import traceback
import joblib
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS  # Import CORS for cross-origin support
from train import train, preprocess_data
from predict import get_model, validate_input

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS) for the app
# app.config["DEBUG"] = True  # Enable debug mode


# Healthcheck endpoint
@app.route("/healthcheck", methods=['GET'])
def healthcheck():
    """
    Healthcheck endpoint to verify the status of the service.

    This endpoint can be used to check if the API is up and running.
    It returns a JSON response with the status and a message.
    """
    return jsonify({"status": "OK",
                    "message": "API is up and running"})


# Endpoint to train the model
@app.route("/train", methods=['POST'])
def train_endpoint():
    try:
        # Get best_params from the request body directly, or default to an empty dictionary
        best_params = request.get_json() or {}

        # Call the train function with best_params and get the metrics
        metrics = train(best_params=best_params)
        
        return jsonify({
            "status": "OK",
            "message": "Model training completed successfully and saved.",
            "metrics": metrics
        })
    except Exception as e:
        logging.error("Model training failed: %s", e)
        logging.error("Traceback: %s", traceback.format_exc())
        return jsonify({"status": "ERROR", "message": "Model training failed"}), 500


@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Get the model and scalers
        model, scaler, y_scaler = get_model()

        # Get input data from request
        input_data = request.get_json()

        # Validate input data
        is_valid, validation_message = validate_input(input_data)
        if not is_valid:
            return jsonify({"status": "INVALID", "message": validation_message}), 400

        # Convert input data to a DataFrame
        df_input = pd.DataFrame([input_data])  # Convert single input dict to DataFrame

        # Preprocess the input data with fit_scaler=False and is_training=False
        processed_data = preprocess_data(df_input, fit_scaler=False, scaler=scaler, y_scaler=y_scaler, is_training=False)

        # Make predictions
        predictions_scaled = model.predict(processed_data)

        # Convert predictions back to the original scale
        predictions_log = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        predictions = np.expm1(predictions_log)  # Reverse the log transformation

        # Construct a detailed response message
        response = {
            "status": "OK",
            "message": f"Energy consumption prediction completed successfully for building ID {input_data['building_id']}.",
            "predictions": predictions.tolist(),
            "units": "kWh",
            "details": {
                "building_id": input_data["building_id"],
                "site_id": input_data["site_id"],
                "meter_type": input_data["meter"],
                "prediction_date": input_data["date"],
                "primary_space_usage": input_data["primaryspaceusage"]
            }
        }
        
        return jsonify(response)
    
    except RuntimeError as e:
        logging.error("Failed to load the model or scalers: %s", e)
        logging.error("Traceback: %s", traceback.format_exc())
        return jsonify({"status": "ERROR", "message": "An error occured when prediction"}), 500
    
    except Exception as e:
        logging.error("Prediction failed: %s", e)
        logging.error("Traceback: %s", traceback.format_exc())
        return jsonify({"status": "ERROR", "message": "Prediction failed"}), 500


if __name__ == "__main__":
    # Determine port based on environment or configuration
    port = int(os.getenv("PORT", 8000))
    logging.info("Starting server on port %s", port)
    app.run(host="0.0.0.0", port=port)
