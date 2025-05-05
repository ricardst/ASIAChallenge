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
2.  Create a directory named `Input_Files` within your main project directory (where the scripts reside).
3.  Place all downloaded Kaggle CSV files (`metadata.csv`, `train_features.csv`, `train_outcomes_functional.csv`, `test_features.csv`, `test_outcomes_Fun_template_update.csv`) into the `Input_Files` directory.
4.  Place the external data files (`submission_MS_test_outcomes.csv`, `train_outcomes_MS.csv` - these result from the predictions of the Impairment track) into the `Input_Files` directory as well.
5.  The scripts (`TabPFN_Fun.py` and `Predict_from_saved_model.py`) are configured to look for data in the `./Input_Files/` subdirectory by default. Ensure the filenames in the script configuration sections match the actual data filenames within `Input_Files`.

## Running the Code

The project provides two main scripts and one additional script where the predictions of the Impairment Track originate from:

1.  **`TabPFN_Fun.py`**: Handles data preprocessing, feature engineering, training the ensemble of models, saving the trained models and feature list, generating predictions from the *current* training run, and creating a submission file.
2.  **`Predict_from_saved_model.py`**: Loads previously saved models and the corresponding feature list from a specific training run directory, prepares the test data accordingly, generates predictions by averaging the loaded models, and creates a submission file.
3.  **`TabPFN_Fun.py`**: Handles the pipline to get predictions for the Impairment Track. The output file is already in the Input_Files folder, so it's rendundant to run this. It's added here as explanation for the origin of the input file.

**Workflow:**

You typically run `TabPFN_Fun.py` first to train the models and save the necessary artifacts. Then, you can use `Predict_from_saved_model.py` to generate predictions from those specific saved artifacts without retraining.

**Option A: Train Models and Generate Submission (using `TabPFN_Fun.py`)**

This script trains the ensemble of models (default: 5 runs), saves each trained pipeline (`.joblib`) and the list of selected features (`final_features_used.joblib`) into a timestamped directory within `Submission_Files/trained_models_[run_id]`, averages the predictions from the *current* run, and generates a submission CSV.

```bash
python TabPFN_Fun.py
```

Output:
Log file in `Log_Files`.
Trained models and feature list in `Submission_Files/trained_models_[run_id]/`.
Submission CSV in `Submission_Files/submission_[run_id].csv`.

Runtime: 
Approximately 5 hours + overhead on the specified hardware, depending on the AUTO_TABPFN_TIME_BUDGET_SECONDS and N_AVERAGING_RUNS settings.

**Option B: Generate Submission from Previously Saved Models (using `Predict_from_saved_model.py`)**

Use this script if you have already run `TabPFN_Fun.py` and want to regenerate predictions using the saved models from that specific run without retraining.

1. Identify the Model Directory: 
Locate the directory created by the training run you want to use (e.g., Submission_Files/trained_models_AutoTabPFN_..._Avg5_..._2025-05-05_XX-XX-XX).

2. Configure the Script: 
Open Predict_from_saved_model.py and edit the MODELS_DIR_TO_LOAD variable to the full path of the directory identified in step 1.

3. Verify Configuration: 
Ensure that the configuration flags (e.g., USE_FUTURE_MOTOR_FEATURES, SELECT_* flags) in Predict_from_saved_model.py match the settings used during the training run that produced the models you are loading. This is crucial for correct data preparation.

4. Run the Prediction Script:
```bash
python Predict_from_saved_model.py
```

Output:
Log file: `prediction_log.log` (overwritten on each run).
Submission CSV in `Submission_Files/predictions_from_[run_id].csv` (where [run_id] corresponds to the models loaded)

## Summary
Our winning solution predicts the modified Benzel functional score (target modben) 6-12 months post-spinal cord injury using patient data available within the first week, along with metadata. The core modeling technique is the AutoTabPFNRegressor from the tabpfn_extensions library, which leverages the TabPFN model optimized for tabular data [1]. To enhance robustness, predictions from 5 independent training runs (using different random seeds) of the AutoTabPFNRegressor pipeline were averaged. Key input features included patient metadata (age, sex, BMI category), select week 1 clinical assessments (AIS grade, motor/sensory scores), the target prediction time (26 vs 52 weeks), and extensively engineered features derived from both week 1 scores and external data representing future motor scores (predicted for test, actual for train). The model was developed in Python using scikit-learn for preprocessing/pipelines and tabpfn / torch for the regressor. Each of the 5 training runs was allocated a time budget of 3600 seconds (1 hour), leading to a total training time of approximately 5 hours plus overhead on a CUDA-enabled GPU.

[1] Hollmann, N., Müller, S., Purucker, L. et al. Accurate predictions on small data with a tabular foundation model. Nature 637, 319–326 (2025). https://doi.org/10.1038/s41586-024-08328-6
