# ASIA Spinal Cord Injury Challenge - Winning Model Documentation

This document describes how to reproduce the winning submission for the [Competition Name] by [Team Name].

## Hardware Requirements

The model was trained and tested using the following hardware configuration:

* **CPU:** [Specify CPU details, e.g., Intel Xeon E5-2698 v4 @ 2.20GHz]
* **CPU Cores:** [Specify number of cores used, e.g., 16]
* **Memory (RAM):** [Specify RAM, e.g., 128 GB]
* **GPU:** [Specify GPU details, e.g., NVIDIA Tesla V100 or NVIDIA GeForce RTX 3090]
* **Number of GPUs:** [Specify number of GPUs used, e.g., 1]

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
2.  Place the external data files (`EXTERNAL_PREDS_FILE`, `TRAIN_OUTCOMES_MOTOR_FILE` - if provided separately, or confirm they are part of Kaggle data) into the same `DATA_DIR`.
3.  Modify the `DATA_DIR` variable at the top of the script (`TabPFN_Fun.py`) to point to the directory where you placed the data.
4.  Ensure the filenames in the script configuration section match the actual data filenames.

## Running the Code

The provided script (`TabPFN_Fun.py`) handles data preprocessing, feature engineering, training, prediction, and submission file generation in a single run.

**NOTE:** The original competition script trains 5 models and averages their predictions to generate the final submission CSV. It **does not** save the trained model objects by default. For reproduction purposes, you might need to:
    a) **Retrain the models:** Run the script as is. This will retrain all 5 models (takes ~5-6 hours) and generate the submission file `submission_[timestamp].csv`.
    b) **Predict using saved models:** This is currently not implemented.

**To train and predict (generate submission CSV):**

```bash
python TabPFN_Fun.py