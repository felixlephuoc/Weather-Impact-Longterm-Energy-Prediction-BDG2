# Train Model API Documentation

## Endpoint

**`POST /train`**

## Description

This endpoint trains a machine learning model on the training dataset using specified hyperparameters (optional). After training, the model and associated scalers are saved for future use. The endpoint also returns evaluation metrics on the training dataset, rounded to 4 decimal places, with specific metrics displayed in percentage format.

## Request

### Headers

- **Content-Type**: `application/json`

### Request Body (JSON)

| Parameter             | Type    | Required | Default  | Description                                                                                                                                      |
| --------------------- | ------- | -------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `num_leaves`        | integer | No       | `1000` | Maximum number of leaves in one tree. Controls the complexity of the model. Higher values increase model complexity and the risk of overfitting. |
| `learning_rate`     | float   | No       | `0.01` | Step size at each iteration while moving towards a minimum of the loss function. Lower values lead to more precise but slower learning.          |
| `max_bin`           | integer | No       | `512`  | Maximum number of bins for feature discretization. Controls memory usage and precision of model.                                                 |
| `lambda_l1`         | float   | No       | `0.01` | L1 regularization term. Helps to prevent overfitting by penalizing the absolute values of leaf weights.                                          |
| `lambda_l2`         | float   | No       | `0.01` | L2 regularization term. Helps to prevent overfitting by penalizing the square of leaf weights.                                                   |
| `min_child_samples` | integer | No       | `20`   | Minimum number of samples in a leaf. Higher values prevent the model from learning overly specific patterns.                                     |
| `max_depth`         | integer | No       | `15`   | Maximum depth of each tree. Controls model complexity and risk of overfitting.                                                                   |
| `num_boost_round`   | integer | No       | `1000` | Number of boosting rounds or iterations for training. Higher values may yield better accuracy but increase training time.                        |

### Example Request

```bash
curl --location 'http://localhost:8000/train' \
--header 'Content-Type: application/json' \
--data '{
    "num_leaves": 500,
    "learning_rate": 0.05,
    "max_bin": 256,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "min_child_samples": 30,
    "max_depth": 10,
    "num_boost_round": 100
}'
```

### Example Successful Response
```json
{
    "status": "OK",
    "message": "Model training completed successfully and saved.",
    "metrics": {
        "root_mean_squared_error": 493470.6144,
        "coefficient_of_variation_rmse": "298.3810%",
        "mean_bias_error": 65988.3742,
        "normalized_mean_bias_error": "39.9004%",
        "r_squared": 0.6863
    }
}
```
