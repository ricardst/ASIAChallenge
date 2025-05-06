import os
import joblib
import logging
import datetime
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from scipy.stats import spearmanr

# Assuming train.py is in the same directory and contains necessary functions
# If not, you might need to copy/paste generate_and_save_plots and spearman_corr
# or ensure train.py can be imported as a module.
from train import generate_and_save_plots, spearman_corr 

# --- Settings & Configuration (copied from train.py) ---
def load_settings():
    try:
        with open('SETTINGS.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("CRITICAL ERROR: SETTINGS.json not found.")
        exit(1)
    except json.JSONDecodeError:
        print("CRITICAL ERROR: SETTINGS.json is not valid JSON.")
        exit(1)

SETTINGS = load_settings()
CLEAN_DATA_DIR = SETTINGS.get('CLEAN_DATA_DIR', './Clean_Data/')
MODEL_DIR = SETTINGS.get('MODEL_DIR', './Models/')
LOG_DIR = SETTINGS.get('LOG_DIR', './Log_Files/')
PLOTS_DIR = SETTINGS.get('PLOTS_DIR', './Plots/')
BASE_RANDOM_STATE = 42 # Assuming this was the BASE_RANDOM_STATE used during training
MODEL_TYPE = 'AutoTabPFN' # Or whatever was used
# Add new settings for SHAP feature reduction
REDUCE_FEATURES_FOR_SHAP_PLOTS = SETTINGS.get('REDUCE_FEATURES_FOR_SHAP_PLOTS', False)
SHAP_TOP_N_FEATURES = SETTINGS.get('SHAP_TOP_N_FEATURES', 20)
SHAP_EXPLAIN_SAMPLE_SIZE = SETTINGS.get('SHAP_EXPLAIN_SAMPLE_SIZE', None) # New setting for SHAP sample size

# --- Setup Logging (minimal version for this script) ---
run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_id = f"regenerate_plots_{MODEL_TYPE}_{run_timestamp}"
log_filename = f"{run_id}.log"
log_file = os.path.join(LOG_DIR, log_filename)

try:
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
except OSError as e:
    print(f"CRITICAL ERROR: Could not create directories: {e}")
    exit(1)

logger = logging.getLogger(run_id)
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

logger.info(f"--- Starting Plot Regeneration: {run_id} ---")

# Define spearman_scorer as it's used in generate_and_save_plots
spearman_scorer = make_scorer(spearman_corr, greater_is_better=True)

def main_regenerate_plots():
    logger.info("Loading preprocessed data and first saved model...")

    try:
        X_train_processed = joblib.load(os.path.join(CLEAN_DATA_DIR, 'X_train_processed.joblib'))
        y_train = joblib.load(os.path.join(CLEAN_DATA_DIR, 'y_train.joblib'))
        
        # Attempt to load feature names
        processed_feature_names = []
        try:
            # First try loading from preprocessor if it was saved separately and has get_feature_names_out
            preprocessor = joblib.load(os.path.join(CLEAN_DATA_DIR, 'preprocessor.joblib'))
            processed_feature_names = preprocessor.get_feature_names_out()
            logger.info(f"Retrieved {len(processed_feature_names)} feature names from saved preprocessor.")
        except Exception: # Broad exception as multiple things could go wrong
            logger.warning("Could not load feature names from preprocessor.joblib or it lacked get_feature_names_out.")
            try:
                processed_feature_names = joblib.load(os.path.join(CLEAN_DATA_DIR, 'processed_feature_names.joblib'))
                logger.info(f"Loaded {len(processed_feature_names)} processed feature names from file.")
            except FileNotFoundError:
                logger.error("processed_feature_names.joblib not found.")
                if isinstance(X_train_processed, pd.DataFrame):
                    processed_feature_names = X_train_processed.columns.tolist()
                    logger.info(f"Using column names from X_train_processed DataFrame ({len(processed_feature_names)} features).")
                else:
                    num_cols = X_train_processed.shape[1]
                    processed_feature_names = [f'feature_{i}' for i in range(num_cols)]
                    logger.warning(f"Using generic feature names for {num_cols} features.")
            except Exception as e_load_feat_names:
                logger.error(f"Error loading processed_feature_names.joblib: {e_load_feat_names}.")
                if isinstance(X_train_processed, pd.DataFrame):
                    processed_feature_names = X_train_processed.columns.tolist()
                else:
                    processed_feature_names = [f'feature_{i}' for i in range(X_train_processed.shape[1])]


        # Construct the filename of the first saved model pipeline
        # This assumes PERFORM_AVERAGING was True and N_AVERAGING_RUNS > 0
        # The first run (i=0) would use BASE_RANDOM_STATE as its seed.
        first_model_seed = BASE_RANDOM_STATE 
        first_model_filename = os.path.join(MODEL_DIR, f'model_avg_run_0_seed_{first_model_seed}.joblib')
        
        if not os.path.exists(first_model_filename):
            logger.error(f"Model file not found: {first_model_filename}")
            logger.error("Please ensure the model was saved with this naming convention or update the path.")
            # As a fallback, try the single run model if averaging was not performed or first model is missing
            logger.info("Attempting to load 'model_single_run.joblib' as a fallback.")
            first_model_filename = os.path.join(MODEL_DIR, 'model_single_run.joblib')
            if not os.path.exists(first_model_filename):
                logger.error(f"Fallback model file not found: {first_model_filename}. Exiting.")
                return


        logger.info(f"Loading model pipeline from: {first_model_filename}")
        loaded_pipeline = joblib.load(first_model_filename)
        
        # Extract the regressor part of the pipeline
        if hasattr(loaded_pipeline, 'named_steps') and 'regressor' in loaded_pipeline.named_steps:
            model_regressor_to_plot = loaded_pipeline.named_steps['regressor']
            logger.info("Successfully extracted regressor from loaded pipeline.")
        elif hasattr(loaded_pipeline, 'predict'): # Check if the loaded object itself is the regressor
            model_regressor_to_plot = loaded_pipeline
            logger.info("Loaded object appears to be a regressor itself.")
        else:
            logger.error("Could not extract regressor from the loaded model/pipeline. Ensure it's a scikit-learn pipeline with a 'regressor' step or a regressor model.")
            return

    except FileNotFoundError as e:
        logger.error(f"Error loading data or model file: {e}. Ensure paths are correct and files exist.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred during loading: {e}", exc_info=True)
        return

    logger.info(f"Loaded X_train_processed (Shape: {X_train_processed.shape}), y_train (Shape: {y_train.shape})")
    
    # Call the plotting function
    # The model_id_str can be adjusted as needed
    model_id_str_for_plots = f"{MODEL_TYPE}_loaded_first_avg_model" 
    if "single_run" in first_model_filename:
        model_id_str_for_plots = f"{MODEL_TYPE}_loaded_single_run_model"

    if model_regressor_to_plot:
        generate_and_save_plots(model_regressor_to_plot, X_train_processed, y_train,
                                processed_feature_names, PLOTS_DIR, model_id_str_for_plots, logger,
                                REDUCE_FEATURES_FOR_SHAP_PLOTS, SHAP_TOP_N_FEATURES,
                                shap_explain_sample_size=SHAP_EXPLAIN_SAMPLE_SIZE) # Pass new arg
    else:
        logger.error("Regressor for plotting was not available. Skipping plot generation.")

    logger.info("Plot regeneration process complete.")

if __name__ == '__main__':
    main_regenerate_plots()