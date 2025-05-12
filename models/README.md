# Models Directory

This directory contains trained machine learning models for electricity consumption forecasting.

## Model Types

- **Deep Learning Models**:
  - LSTM (Long Short-Term Memory) models (`*lstm_model.keras`)
  - GRU (Gated Recurrent Unit) models (`*gru_model.keras`)
  - CNN (Convolutional Neural Network) models (`*cnn_model.keras`)
  - MLP (Multi-Layer Perceptron) models (`*MLP_model.keras`)

- **Traditional ML Models**:
  - Random Forest (`*random_forest*.pkl`)
  - Gradient Boosting (`*gradient_boosting*.pkl`)

## File Naming Convention

Model files follow this naming convention:
- `model_v{version_number}.pkl` for traditional ML models
- `{model_type}_model_{timestamp}.keras` for deep learning models

## Model Data

Each model is accompanied by metadata files:
- `*_history.pkl`: Training history for deep learning models
- `*_params.json`: Hyperparameters used for model training
- `*_data_info.pkl`: Information about data preprocessing

## Generating Models

The models are generated when you run the training process. To generate the models:

```bash
# Using the run.py script
python run.py --train

# Or using the run.sh script
./run.sh --train

# Or directly using DVC
dvc repro train_models
```

All models are automatically saved to this directory with a timestamp in the filename.

## Using the Models

To load and use these models, refer to the utility functions in `src/models/evaluate.py`.

Example:
```python
from src.models.evaluate import load_trained_model, predict_and_evaluate

model, model_type = load_trained_model("models/lstm_model_20220415_123456.keras")
``` 