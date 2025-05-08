\
import warnings
import json
import os
import datetime
import logging
import joblib # Assuming it's used in the actual main
import pandas as pd # Assuming it's used
import numpy as np # Assuming it's used
import glob

# Added for the patch
import torch
try:
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
    # We need AutoTabPFNRegressor in the global scope for the patch to find and modify it.
except ImportError:
    # If predict.py doesn't use AutoTabPFNRegressor directly, it might not be imported.
    # However, joblib.load will need its definition to unpickle.
    # The train.py script imports it, so it should be available in the environment.
    # We'll log an error within the patch if it's not found.
    pass # Let the patch handle the NameError if it occurs.


warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

# ==============================================================================
# Settings & Configuration (copied from predict.py)
# ==============================================================================
def load_settings():
    try:
        with open('SETTINGS.json', 'r') as f: # Assuming SETTINGS.json exists
            return json.load(f)
    except FileNotFoundError:
        print("Warning: SETTINGS.json not found. Using default paths.")
        return {} # Default empty settings
    except json.JSONDecodeError:
        print("Warning: SETTINGS.json is not valid JSON. Using default paths.")
        return {}

SETTINGS = load_settings()
CLEAN_DATA_DIR = SETTINGS.get('CLEAN_DATA_DIR', './Clean_Data/')
MODEL_DIR = SETTINGS.get('MODEL_DIR', './Models/')
SUBMISSION_DIR = SETTINGS.get('SUBMISSION_DIR', './Submissions/')
LOG_DIR = SETTINGS.get('LOG_DIR', './Log_Files/')
RAW_DATA_DIR = SETTINGS.get('RAW_DATA_DIR', './Input_Files/') # For submission template

# --- Prediction Configuration (can be moved to SETTINGS.json) ---
PERFORM_AVERAGING_PREDICT = True # Default, adjust as per your training
N_AVERAGING_RUNS_PREDICT = 5 # Default, adjust as per your training
TARGET_COL = 'modben' # Ensure this matches the target column name

# ==============================================================================
# Setup Logging (copied from predict.py)
# ==============================================================================
run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Modified run_id to reflect it's the CPU version
run_id = f"predict_model_cpu_{run_timestamp}" 
log_filename = f"{run_id}.log"
log_file = os.path.join(LOG_DIR, log_filename)

try:
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
except OSError as e:
    print(f"CRITICAL ERROR: Could not create directories: {e}")
    exit(1) # Or handle more gracefully

logger = logging.getLogger(run_id)
logger.setLevel(logging.INFO)
if not logger.handlers: # Ensure handlers are not added multiple times if re-run in same session
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh) # Add stream handler to also print to console

logger.info(f"--- Starting Model Prediction (CPU-Patched): {run_id} ---")
logger.info(f"Clean Data Directory (for X_test): {CLEAN_DATA_DIR}")
logger.info(f"Model Load Directory: {MODEL_DIR}")
logger.info(f"Submission Save Directory: {SUBMISSION_DIR}")
logger.info(f"Log File: {log_file}")

# ==============================================================================
# Main Prediction Logic
# ==============================================================================
def main():
    logger.info("Executing main function in predict_cpu.py")

    original_autotabpfn_setstate = None
    patch_applied_for_autotabpfn = False
    original_torch_load = None  # For torch.load patch
    patch_applied_for_torch_load = False # For torch.load patch

    # Attempt to patch AutoTabPFNRegressor for CPU loading if CUDA is not available
    if not torch.cuda.is_available():
        logger.info("CUDA not available. Applying patches for CPU loading.")
        
        # Patch 1: AutoTabPFNRegressor.__setstate__ (existing patch)
        try:
            # This line checks if AutoTabPFNRegressor is defined and accessible
            _ = AutoTabPFNRegressor
            
            original_autotabpfn_setstate = AutoTabPFNRegressor.__setstate__

            def patched_autotabpfn_setstate(self, state):
                logger.debug("Patched AutoTabPFNRegressor.__setstate__ called.")
                if 'device' in state and state['device'] == 'cuda':
                    logger.info("AutoTabPFNRegressor.__setstate__ patch: Original device was 'cuda', changing to 'cpu'.")
                    state['device'] = 'cpu'
                
                if 'model_config' in state and isinstance(state['model_config'], dict):
                    if state['model_config'].get('device') == 'cuda':
                        logger.info("AutoTabPFNRegressor.__setstate__ patch: model_config device was 'cuda', changing to 'cpu'.")
                        state['model_config']['device'] = 'cpu'
                
                original_autotabpfn_setstate(self, state)

            AutoTabPFNRegressor.__setstate__ = patched_autotabpfn_setstate
            patch_applied_for_autotabpfn = True
            logger.info("AutoTabPFNRegressor.__setstate__ has been patched for CPU loading.")
        
        except NameError:
            logger.error("AutoTabPFNRegressor is not defined in the global scope. Cannot apply __setstate__ patch. "
                         "Ensure 'from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor' "
                         "is present and working if this class is part of the pickled model.")
        except Exception as e_patch_setstate:
            logger.error(f"An unexpected error occurred during AutoTabPFNRegressor.__setstate__ patching: {e_patch_setstate}", exc_info=True)

        # Patch 2: torch.load
        try:
            logger.info("Attempting to patch torch.load for CPU-only environment.")
            original_torch_load = torch.load

            def patched_torch_load_on_cpu(*args, **kwargs):
                logger.debug(f"Patched torch.load_on_cpu invoked. Original args: {args}, kwargs: {kwargs}")
                
                forced_map_location = torch.device('cpu')
                
                if 'map_location' in kwargs:
                    original_map_loc = kwargs['map_location']
                    kwargs['map_location'] = forced_map_location
                    logger.info(f"Patched torch.load: Overriding kwarg 'map_location' from '{original_map_loc}' to '{forced_map_location}'.")
                elif len(args) > 1: # Assuming map_location is the second positional argument
                    original_map_loc = args[1]
                    new_args = list(args)
                    new_args[1] = forced_map_location
                    args = tuple(new_args)
                    logger.info(f"Patched torch.load: Overriding positional arg map_location from '{original_map_loc}' to '{forced_map_location}'.")
                else: # map_location not provided, add it
                    kwargs['map_location'] = forced_map_location
                    logger.info(f"Patched torch.load: Adding 'map_location={forced_map_location}' as it was not specified.")
                    
                return original_torch_load(*args, **kwargs)

            torch.load = patched_torch_load_on_cpu
            patch_applied_for_torch_load = True
            logger.info("torch.load has been globally patched to use map_location='cpu'.")
        
        except Exception as e_patch_torch_load:
            logger.error(f"An unexpected error occurred during torch.load patching: {e_patch_torch_load}", exc_info=True)
            # If torch.load patching fails, subsequent joblib.load might still fail on CUDA models.
    else:
        logger.info("CUDA is available. No patch needed for AutoTabPFNRegressor or torch.load.")

    logger.info("Loading preprocessed test data and PIDs...")
    try:
        X_test_processed = joblib.load(os.path.join(CLEAN_DATA_DIR, 'X_test_processed.joblib'))
        test_PIDs = joblib.load(os.path.join(CLEAN_DATA_DIR, 'test_PIDs.joblib'))
        # Load processed feature names if X_test_processed is a NumPy array and needs conversion
        # This logic is adapted from the original predict.py snippet
        if isinstance(X_test_processed, np.ndarray):
            logger.warning("Loaded X_test_processed is a NumPy array. Attempting to load processed feature names.")
            try:
                processed_feature_names = joblib.load(os.path.join(CLEAN_DATA_DIR, 'processed_feature_names.joblib'))
                X_test_processed = pd.DataFrame(X_test_processed, columns=processed_feature_names)
                logger.info(f"Converted X_test_processed to DataFrame using {len(processed_feature_names)} loaded feature names.")
            except FileNotFoundError:
                logger.error(f"Processed feature names file not found. Cannot convert NumPy X_test to DataFrame if column order is critical.")
                # Depending on the model, a NumPy array might be acceptable if column order was preserved.
            except Exception as e_feat_names:
                logger.error(f"Error loading processed feature names or converting X_test: {e_feat_names}", exc_info=True)
        
        # Load submission template for PIDs and time column, and for final formatting
        submission_template_file_path = os.path.join(RAW_DATA_DIR, 'test_outcomes_Fun_template_update.csv')
        submission_template_df = pd.read_csv(submission_template_file_path)

    except FileNotFoundError as e:
        logger.error(f"Error loading preprocessed data file: {e}. Run prepare_data.py first.")
        exit(1)
    logger.info(f"Loaded X_test_processed (Shape: {X_test_processed.shape}), test_PIDs (Count: {len(test_PIDs)})")

    # Sanity check row count from loaded PIDs vs X_test_processed
    if X_test_processed.shape[0] != len(test_PIDs):
        logger.error(f"FATAL: Row count mismatch! Preprocessed X_test has {X_test_processed.shape[0]} rows, but loaded test_PIDs has {len(test_PIDs)}.")
        exit(1)

    # --- Model Loading and Prediction ---
    model_files_pattern = os.path.join(MODEL_DIR, 'model_avg_run_*.joblib')
    avg_model_files = sorted(glob.glob(model_files_pattern))
    single_model_file = os.path.join(MODEL_DIR, 'model_single_run.joblib')

    all_test_predictions_from_models = []

    if avg_model_files: # Averaging models found
        logger.info(f"Found {len(avg_model_files)} averaged models. Proceeding with averaging predictions.")
        if len(avg_model_files) != N_AVERAGING_RUNS_PREDICT:
            logger.warning(f"Number of model files ({len(avg_model_files)}) does not match N_AVERAGING_RUNS_PREDICT ({N_AVERAGING_RUNS_PREDICT}). Using found models.")

        for i, model_file in enumerate(avg_model_files):
            try:
                logger.info(f"Loading model {i+1}/{len(avg_model_files)} from {model_file}...")
                # Load the entire pipeline (preprocessor + regressor)
                trained_pipeline = joblib.load(model_file)
                # Use only the regressor part of the pipeline for prediction
                regressor = trained_pipeline[-1] # Assuming the regressor is the last step
                current_preds = regressor.predict(X_test_processed)
                all_test_predictions_from_models.append(current_preds)
                logger.info(f"Predictions generated from model {i+1}.")
            except Exception as e_load_avg:
                logger.error(f"Error loading or predicting with averaged model {model_file}: {e_load_avg}", exc_info=True)
                # Decide if one failed model should stop all predictions

        if not all_test_predictions_from_models:
            logger.error("FATAL: No predictions generated from any averaged model files. Exiting.")
            exit(1)
        
        logger.info(f"Averaging predictions across {len(all_test_predictions_from_models)} successful model loads...")
        final_predictions_raw = np.mean(np.stack(all_test_predictions_from_models), axis=0)

    elif os.path.exists(single_model_file): # Single model found
        logger.info(f"Found single model file: {single_model_file}. Proceeding with single model prediction.")
        try:
            trained_pipeline = joblib.load(single_model_file)
            # Use only the regressor part of the pipeline for prediction
            regressor = trained_pipeline[-1] # Assuming the regressor is the last step
            final_predictions_raw = regressor.predict(X_test_processed)
            logger.info("Predictions generated from single model.")
        except Exception as e_load_single:
            logger.error(f"FATAL: Error loading or predicting with single model {single_model_file}: {e_load_single}", exc_info=True)
            exit(1)
    else:
        logger.error(f"FATAL: No model files found in {MODEL_DIR}. Searched for pattern 'model_avg_run_*.joblib' and file 'model_single_run.joblib'.")
        logger.error("Ensure train.py has run successfully and saved models to the correct directory specified in SETTINGS.json.")
        exit(1)

    # --- Post-Process Final Predictions ---
    logger.info("Post-processing final predictions...")
    try:
        MIN_SCORE = 1
        MAX_SCORE = 8
        final_predictions = final_predictions_raw.copy()
        logger.info(f"  Clipping predictions to range [{MIN_SCORE}, {MAX_SCORE}].")
        final_predictions = np.clip(final_predictions, MIN_SCORE, MAX_SCORE)
        logger.info("  Rounding predictions to nearest integer.")
        final_predictions = np.round(final_predictions).astype(int)
        logger.info("Post-processing complete.")
    except Exception as e_post:
        logger.error(f"ERROR during post-processing: {e_post}", exc_info=True)
        num_test_samples = X_test_processed.shape[0]
        final_predictions = np.full(num_test_samples, MIN_SCORE) # Fallback
        logger.error("Using dummy predictions due to post-processing error.")

    # --- Generate Submission File ---
    logger.info("Generating submission file...")
    try:
        # Use PIDs and time from the submission template for correct order and structure
        submission_df = submission_template_df[['PID', 'time']].copy()
        
        # Create a temporary DataFrame for predictions with PIDs to allow merging
        # This ensures that predictions are matched to the correct PIDs, even if X_test_processed order changed.
        # test_PIDs were saved by prepare_data.py in the same order as X_test_processed rows.
        predictions_with_pids_df = pd.DataFrame({
            'PID': test_PIDs, 
            TARGET_COL: final_predictions
        })

        # Merge predictions into the submission_df based on PID
        # This handles cases where submission_template_df might have different PIDs or order
        # than the test_PIDs used for training/prediction (though they should align for the challenge)
        submission_df = pd.merge(submission_df, predictions_with_pids_df, on='PID', how='left')

        # Fill any PIDs present in template but not in predictions with a default (e.g., MIN_SCORE)
        # This should ideally not happen if test_PIDs covers all submission PIDs.
        if submission_df[TARGET_COL].isnull().any():
            nan_count = submission_df[TARGET_COL].isnull().sum()
            logger.warning(f"Found {nan_count} NaNs in target column after merging predictions. Filling with {MIN_SCORE}.")
            submission_df[TARGET_COL].fillna(MIN_SCORE, inplace=True)
            submission_df[TARGET_COL] = submission_df[TARGET_COL].astype(int) # Ensure type consistency

        # Ensure final submission matches template columns exactly in order and content
        template_cols = submission_template_df.columns.tolist()
        final_submission_df = submission_df[template_cols]

        submission_filename = os.path.join(SUBMISSION_DIR, f"submission_{run_timestamp}.csv")
        final_submission_df.to_csv(submission_filename, index=False)
        logger.info(f"Submission file successfully saved to '{submission_filename}'")

    except KeyError as e_key:
        logger.error(f"ERROR generating submission file: Missing key column {e_key}. Check template matching or PID alignment.")
    except Exception as e_sub:
        logger.error(f"ERROR saving submission file: {e_sub}", exc_info=True)

    logger.info(f"--- Finished Prediction Run: {run_id} ---")

    # Restore original __setstate__ if it was patched
    if patch_applied_for_autotabpfn and original_autotabpfn_setstate is not None:
        try:
            _ = AutoTabPFNRegressor # Check again in case of issues
            AutoTabPFNRegressor.__setstate__ = original_autotabpfn_setstate
            logger.info("Restored original AutoTabPFNRegressor.__setstate__.")
        except NameError:
            logger.error("AutoTabPFNRegressor is not defined. Could not restore original __setstate__.")
        except Exception as e_restore_setstate: # Renamed variable to avoid conflict
            logger.error(f"An unexpected error occurred during __setstate__ restoration: {e_restore_setstate}", exc_info=True)

    # Restore original torch.load if it was patched
    if patch_applied_for_torch_load and original_torch_load is not None:
        try:
            torch.load = original_torch_load
            logger.info("Restored original torch.load.")
        except Exception as e_restore_torch_load:
            logger.error(f"An unexpected error occurred during torch.load restoration: {e_restore_torch_load}", exc_info=True)

    logger.info(f"--- Model Prediction (CPU-Patched) Finished: {run_id} ---")

if __name__ == '__main__':
    main()
