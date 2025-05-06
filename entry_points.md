\
# Entry Points for Running the Code

This document outlines the commands required to run the data preparation, model training, and prediction scripts. It assumes the presence of a `SETTINGS.json` file in the root directory to specify necessary paths.

## 1. Prepare Data

This script reads raw training data, performs preprocessing, and saves the cleaned data.

**Command:**
```bash
python prepare_data.py
```

**Functionality:**
- Reads training data from `RAW_DATA_DIR` (e.g., `./Input_Files/`) as specified in `SETTINGS.json`.
- Runs preprocessing steps, which may include:
    - Loading metadata, features, and outcomes.
    - Merging datasets.
    - Feature engineering (e.g., from Week 1 scores, future motor scores).
    - Manual and automated feature selection.
    - Data type conversions and encoding (categorical, ordinal).
    - Imputation of missing values.
- Saves the cleaned and processed training data (e.g., `X_train_processed.joblib`, `y_train.joblib`) and any necessary auxiliary files (e.g., `final_features_used.joblib`, `preprocessor.joblib`, `processed_column_names.joblib`) to `CLEAN_DATA_DIR` (e.g., `./Clean_Data/`) as specified in `SETTINGS.json`.
- Saves the cleaned and processed test data (e.g., `X_test_processed.joblib`, `test_PIDs.joblib`) to `CLEAN_DATA_DIR`.

**Assumed `SETTINGS.json` entries:**
```json
{
  "RAW_DATA_DIR": "./Input_Files/",
  "CLEAN_DATA_DIR": "./Clean_Data/"
}
```

## 2. Train Model

This script reads the cleaned training data, trains the model, and saves the trained model.

**Command:**
```bash
python train.py
```

**Functionality:**
- Reads cleaned training data (e.g., `X_train_processed.joblib`, `y_train.joblib`) from `TRAIN_DATA_CLEAN_PATH` (derived from `CLEAN_DATA_DIR` in `SETTINGS.json`).
- Loads any auxiliary files (e.g., `preprocessor.joblib`, `final_features_used.joblib`) if needed for model pipeline construction.
- Trains the model (e.g., TabPFN, AutoTabPFN). This may involve:
    - Defining the model pipeline (preprocessor + regressor).
    - Performing cross-validation (optional).
    - Training single or multiple models for averaging.
- If checkpoint files are used during a long training process, they might be saved to and loaded from `CHECKPOINT_DIR` (e.g., `./Checkpoints/`) specified in `SETTINGS.json`.
- Saves the trained model(s) (e.g., `model.joblib`, or multiple model files for an ensemble) to `MODEL_DIR` (e.g., `./Models/`) as specified in `SETTINGS.json`.
- Saves any other artifacts necessary for prediction, if not already saved by `prepare_data.py` (e.g., fitted preprocessor if it's part of the saved model).

**Assumed `SETTINGS.json` entries:**
```json
{
  "CLEAN_DATA_DIR": "./Clean_Data/",
  "TRAIN_DATA_CLEAN_PATH": "./Clean_Data/X_train_processed.joblib", // Example
  "MODEL_DIR": "./Models/",
  "CHECKPOINT_DIR": "./Checkpoints/" // Optional
}
```

## 3. Make Predictions

This script reads the cleaned test data, loads the trained model, makes predictions, and saves the predictions.

**Command:**
```bash
python predict.py
```

**Functionality:**
- Reads cleaned test data (e.g., `X_test_processed.joblib`) from `TEST_DATA_CLEAN_PATH` (derived from `CLEAN_DATA_DIR` in `SETTINGS.json`).
- Loads the trained model(s) from `MODEL_DIR` (e.g., `./Models/`) as specified in `SETTINGS.json`.
- Loads any auxiliary files needed for prediction (e.g., `final_features_used.joblib`, `processed_column_names.joblib`, `test_PIDs.joblib`).
- Uses the model to make predictions on the new samples.
- Performs any necessary post-processing on predictions (e.g., clipping, rounding).
- Saves the predictions in the required submission format (e.g., `submission.csv`) to `SUBMISSION_DIR` (e.g., `./Submissions/`) as specified in `SETTINGS.json`.

**Assumed `SETTINGS.json` entries:**
```json
{
  "CLEAN_DATA_DIR": "./Clean_Data/",
  "TEST_DATA_CLEAN_PATH": "./Clean_Data/X_test_processed.joblib", // Example
  "MODEL_DIR": "./Models/",
  "SUBMISSION_DIR": "./Submissions/"
}
```
