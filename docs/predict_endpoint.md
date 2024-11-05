# Predict Model API Documentation

## Endpoint

**`POST /predict`**

## Description

This endpoint predicts the energy consumption of a building based on provided input data. The model takes into account features such as the building's area, temperature, usage type, and more. The input data is validated and preprocessed, and the prediction is returned in kilowatt-hours (kWh).

## Request

### Headers

- **Content-Type**: `application/json`

### Request Body (JSON)

| Parameter             | Type    | Required | Description                                                                                                                                               |
|-----------------------|---------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `building_id`         | integer | Yes      | Unique identifier for the building.                                                                                                                       |
| `site_id`             | integer | Yes      | Identifier for the site where the building is located. Must be an integer.                                                                                |
| `meter`               | string  | Yes      | Type of meter used for measuring energy. Acceptable values are: `electricity`, `chilledwater`, `gas`, `hotwater`, `solar`, `water`, `steam`, `irrigation`.|
| `date`                | string  | Yes      | Date for which the prediction is required, in the format `YYYY-MM-DD`.                                                                                    |
| `primaryspaceusage`   | string  | Yes      | Main usage category of the building's space. Must be either `Education` or `Office`.                                                                      |
| `sqm`                 | float   | Yes      | Area of the building in square meters. Must be a numeric value.                                                                                           |
| `airTemperature`      | float   | Yes      | Outside air temperature in degrees Celsius. Must be a numeric value.                                                                                      |

### Response Body (JSON)

| Field                         | Type       | Description                                                                                                                     |
|-------------------------------|------------|---------------------------------------------------------------------------------------------------------------------------------|
| `status`                      | string     | Status of the request, either "OK" or "ERROR".                                                                                  |
| `message`                     | string     | A message indicating the outcome of the request.                                                                                |
| `predictions`                 | array      | An array containing the predicted energy consumption value(s) in kilowatt-hours (kWh).                                          |
| `units`                       | string     | Unit of the prediction, typically "kWh".                                                                                        |
| `details`                     | object     | An object containing detailed information about the input data used for the prediction.                                         |
| `details.building_id`         | integer    | Identifier for the building for which the prediction was made.                                                                  |
| `details.site_id`             | integer    | Identifier of the site associated with the building.                                                                            |
| `details.meter_type`          | string     | Type of meter used for the prediction (from the input).                                                                         |
| `details.prediction_date`     | string     | The date for which the prediction was made.                                                                                     |
| `details.primary_space_usage` | string     | Primary usage of the building's space, either `Education` or `Office`.                                                          |

### Validation Rules

- `site_id` must be an integer.
- `meter` must be one of the following: `electricity`, `chilledwater`, `gas`, `hotwater`, `solar`, `water`, `steam`, `irrigation`.
- `date` must be a valid date in the format `YYYY-MM-DD`.
- `primaryspaceusage` must be either `Education` or `Office`.
- `sqm` and `airTemperature` must be numeric values.

### Example Request

```bash
curl --location 'http://localhost:8000/predict' \
--header 'Content-Type: application/json' \
--data '{
    "building_id": 377,
    "site_id": 7,
    "meter": "chilledwater",
    "date": "2017-12-05",
    "primaryspaceusage": "Office",
    "sqm": 2766.5,
    "airTemperature": 9.66
}'
```

### Example Successful Response
```json
{
    "status": "OK",
    "message": "Energy consumption prediction completed successfully for building ID 377.",
    "predictions": [
        67.2328
    ],
    "units": "kWh",
    "details": {
        "building_id": 377,
        "site_id": 7,
        "meter_type": "chilledwater",
        "prediction_date": "2017-12-05",
        "primary_space_usage": "Office"
    }
}
```