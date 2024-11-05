# Impact of Weather Data in Long-Term Energy Prediction (BDG-2)

This project leverages weather data and other building characteristics to predict long-term energy consumption for buildings using the BDG-2 dataset. The project provides a RESTful API to train models and make energy consumption predictions based on various inputs, including weather data and building-specific information.

### Prerequisites

Ensure you have Python 3.9+ installed on your machine.

## Getting Started

```bash
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
python3 src/main.py
```

## Docker Deployment

```bash
sh run_docker.sh
```

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/healthcheck
```

### Train model

Example

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

### Predict Energy consumption for one building at specific day

Example

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
