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

1.  Download the competition data files from Kaggle into a single directory (referred to as `DATA_DIR` in the script).
2.  Place the external data files (`EXTERNAL_PREDS_FILE`, `TRAIN_OUTCOMES_MOTOR_FILE` - if provided separately, or confirm they are part of Kaggle data) into the same `DATA_DIR`. -> These files result from the predictions of the Impairment track.
3.  Modify the `DATA_DIR` variable at the top of the script (`TabPFN_Fun.py`) to point to the directory where you placed the data.
4.  Ensure the filenames in the script configuration section match the actual data filenames.

## Running the Code

The provided script (`TabPFN_Fun.py`) handles data preprocessing, feature engineering, training, prediction, and submission file generation in a single run.

**NOTE:** The original competition script trains 5 models and averages their predictions to generate the final submission CSV. It **does not** save the trained model objects by default. For reproduction purposes, you need to:
    a) **Retrain the models:** Run the script as is. This will retrain all 5 models (takes ~3-4 hours) and generate the submission file `submission_[timestamp].csv`.
    b) **Predict using saved models:** This is currently not implemented.

**To train and predict (generate submission CSV):**

```bash
python TabPFN_Fun.py
```

## Summary
Our winning solution predicts the modified Benzel functional score (target modben) 6-12 months post-spinal cord injury using patient data available within the first week, along with metadata. The core modeling technique is the AutoTabPFNRegressor from the tabpfn_extensions library, which leverages the TabPFN model optimized for tabular data [1]. To enhance robustness, predictions from 5 independent training runs (using different random seeds) of the AutoTabPFNRegressor pipeline were averaged. Key input features included patient metadata (age, sex, BMI category), select week 1 clinical assessments (AIS grade, motor/sensory scores), the target prediction time (26 vs 52 weeks), and extensively engineered features derived from both week 1 scores and external data representing future motor scores (predicted for test, actual for train). The model was developed in Python using scikit-learn for preprocessing/pipelines and tabpfn / torch for the regressor. Each of the 5 training runs was allocated a time budget of 3600 seconds (1 hour), leading to a total training time of approximately 5 hours plus overhead on a CUDA-enabled GPU.

[1] Hollmann, N., Müller, S., Purucker, L. et al. Accurate predictions on small data with a tabular foundation model. Nature 637, 319–326 (2025). https://doi.org/10.1038/s41586-024-08328-6
