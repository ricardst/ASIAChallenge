# ASIA Spinal Cord Injury Challenge - Winning Model Documentation

This document describes how to reproduce the winning submission for the Spinal Cord Injury Challenge - Functional Track by SCAI Lab ETHZ.

## Hardware Requirements

The model was trained and tested using the following hardware configuration:

* **CPU:** Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz
* **CPU Cores:** 16
* **Memory (RAM):** 251GB
* **GPU:** NVIDIA GeForce RTX 3090
* **Number of GPUs:** 1

*Note: A CUDA-enabled GPU is highly recommended due to the use of TabPFN/PyTorch.*

## Software Requirements

* **OS/Platform:** Rocky Linux 9.4 (Blue Onyx)
* **Python Version:** Python 3.9.18
* **Required Packages:** All necessary Python packages and their exact versions are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
* **Special Installation:** The `tabpfn-extensions` library needs to be installed directly from GitHub:
    ```bash
    pip install git+[https://github.com/automl/TabPFN-extensions.git](https://github.com/automl/TabPFN-extensions.git)
    ```

## Data Setup

1.  Download the competition data files from Kaggle.
2.  Create a directory named `Input_Files` within your main project directory (where the scripts reside). This can be configured in `SETTINGS.json` via the `RAW_DATA_DIR` key (default: `./Input_Files/`).
3.  Place all downloaded Kaggle CSV files (`metadata.csv`, `train_features.csv`, `train_outcomes_functional.csv`, `test_features.csv`, `test_outcomes_Fun_template_update.csv`) into the `Input_Files` directory.
4.  Place the external data files (`submission_MS_test_outcomes.csv`, `train_outcomes_MS.csv` - these result from the predictions of the Impairment track) into the `Input_Files` directory as well.
5.  The scripts are configured to look for data in the directory specified by `RAW_DATA_DIR` in `SETTINGS.json`. Ensure the filenames in the script configuration sections (primarily in `prepare_data.py`) match the actual data filenames.

## Running the Code

The project provides three main scripts:

1.  **`prepare_data.py`**: Handles data loading, preprocessing, and feature engineering. Saves processed data, including `X_train_processed.joblib`, `y_train.joblib`, `X_test_processed.joblib`, `preprocessor.joblib`, and `final_features_used.joblib`, into the directory specified by `CLEAN_DATA_DIR` in `SETTINGS.json` (default: `./Clean_Data/`).
2.  **`train.py`**: Trains the ensemble of models (e.g., AutoTabPFNRegressor) using the processed data from `CLEAN_DATA_DIR`. Saves the trained model pipelines (e.g., `model_avg_run_*.joblib` or `model_single_run.joblib`) into the directory specified by `MODEL_DIR` in `SETTINGS.json` (default: `./Models/`).
3.  **`predict.py`**: Loads the previously saved models from `MODEL_DIR` and the corresponding preprocessor and feature list from `CLEAN_DATA_DIR`. Prepares the test data, generates predictions by averaging the loaded models (if applicable), and creates a submission file in the directory specified by `SUBMISSION_DIR` in `SETTINGS.json` (default: `./Submissions/`).

**Workflow:**

The typical workflow is to run the scripts in the following order:

1.  **Prepare Data:**
    ```bash
    python prepare_data.py
    ```
    Output:
    *   Log file in `Log_Files/` (default, configurable via `LOG_DIR` in `SETTINGS.json`).
    *   Processed data files in `Clean_Data/` (default, configurable via `CLEAN_DATA_DIR` in `SETTINGS.json`).

2.  **Train Models:**
    ```bash
    python train.py
    ```
    Output:
    *   Log file in `Log_Files/`.
    *   Trained models in `Models/` (default, configurable via `MODEL_DIR` in `SETTINGS.json`).
    *   Plots (e.g., SHAP plots) in `Plots/` (default, configurable via `PLOTS_DIR` in `SETTINGS.json`), if not skipped via settings.
    Runtime: Approximately 5 hours + overhead on the specified hardware, depending on the `AUTO_TABPFN_TIME_BUDGET_SECONDS` and `N_AVERAGING_RUNS` settings in `train.py` (or `SETTINGS.json` if moved there).

3.  **Generate Submission from Saved Models:**
    This script uses the models saved by `train.py` and data artifacts from `prepare_data.py`.
    ```bash
    python predict.py
    ```
    Output:
    *   Log file in `Log_Files/` (e.g., `predict_model_[timestamp].log`).
    *   Submission CSV in `Submissions/` (e.g., `submission_[timestamp].csv`, configurable via `SUBMISSION_DIR` in `SETTINGS.json`).

    *Note: Ensure that the configuration flags (e.g., `USE_FUTURE_MOTOR_FEATURES`, `SELECT_*` flags in `prepare_data.py`, and model/averaging settings in `train.py` and `predict.py`) are consistent across the data preparation, training, and prediction stages if you are running them separately or modifying configurations.*

4.  **Generate Submission on a CPU (if no GPU is available):**
    If you are working on a machine without a CUDA-enabled GPU, you can use `predict_cpu.py`. This script is a modified version of `predict.py` that attempts to load and run the models on the CPU.
    ```bash
    python predict_cpu.py
    ```
    Output:
    *   Log file in `Log_Files/` (e.g., `predict_model_cpu_[timestamp].log`).
    *   Submission CSV in `Submissions/` (e.g., `submission_[timestamp].csv`).

    *Note: Performance will be significantly slower compared to running on a GPU. The same consistency notes for configuration flags apply here as well.*

## Summary
Our winning solution predicts the modified Benzel functional score (target modben) 6-12 months post-spinal cord injury using patient data available within the first week, along with metadata. The core modeling technique is the AutoTabPFNRegressor from the tabpfn_extensions library, which leverages the TabPFN model optimized for tabular data [1]. To enhance robustness, predictions from 5 independent training runs (using different random seeds) of the AutoTabPFNRegressor pipeline were averaged. Key input features included patient metadata (age, sex, BMI category), select week 1 clinical assessments (AIS grade, motor/sensory scores), the target prediction time (26 vs 52 weeks), and extensively engineered features derived from both week 1 scores and external data representing future motor scores (predicted for test, actual for train). The model was developed in Python using scikit-learn for preprocessing/pipelines and tabpfn / torch for the regressor. Each of the 5 training runs was allocated a time budget of 3600 seconds (1 hour), leading to a total training time of approximately 5 hours plus overhead on a CUDA-enabled GPU.

[1] Hollmann, N., Müller, S., Purucker, L. et al. Accurate predictions on small data with a tabular foundation model. Nature 637, 319–326 (2025). https://doi.org/10.1038/s41586-024-08328-6
