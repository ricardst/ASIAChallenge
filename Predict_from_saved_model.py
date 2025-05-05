# ==============================================================================
# Imports
# ==============================================================================
import pandas as pd
import numpy as np
import warnings
import re
import datetime
import logging
import torch # For checking CUDA availability (though model device is usually saved)
import gc # For garbage collection
import joblib # For loading models and feature lists
import os
import glob # For finding model files

# Scikit-learn imports (needed for loading pipelines containing these)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn import clone # Potentially needed if pipeline structure is complex

# Pandas type checking (needed for feature selection logic if re-run, but we load list)
import pandas.api.types as ptypes

# TabPFN specific import (needed to load pipelines containing these)
try:
    from tabpfn import TabPFNRegressor
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
except ImportError:
    print("Error: Required libraries not found.")
    print("Please install tabpfn and potentially tabpfn-extensions:")
    print("  pip install tabpfn")
    print("  pip install git+https://github.com/automl/TabPFN-extensions.git")
    exit(1)

# Import helper functions if they are in a separate file, otherwise copy them here.
# Assuming they are still in TabPFN_Fun.py for now, we need to redefine them or import.
# For simplicity, let's copy the necessary ones here.

# ==============================================================================
# Helper Functions (Copied from TabPFN_Fun.py - ensure consistency!)
# ==============================================================================

# --- Logging Setup (Simplified for Prediction) ---
log_predict_filename = 'prediction_log.log'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_predict_filename),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)


def engineer_features(df, week1_feature_cols):
    """
    Applies feature engineering based on WEEK 1 ISNCSCI scores.
    (Ensure this is identical to the function in the training script)
    """
    logger.debug(f"Starting Week 1 FE on df shape {df.shape}")
    eng_df = df.copy()
    wk1_cols = [c for c in week1_feature_cols if c in df.columns] # Ensure cols exist

    # Identify specific motor/sensory columns based on naming patterns
    motor_cols = [c for c in wk1_cols if re.match(r'(?:elbf|wrext|elbex|finfl|finab|hipfl|kneex|ankdo|greto|ankpl)[lr]01$', c)]
    lt_cols = [c for c in wk1_cols if re.search(r'[cts]\d+lt[lr]01$', c) or re.search(r's45lt[lr]01$', c)]
    pp_cols = [c for c in wk1_cols if re.search(r'[cts]\d+pp[lr]01$', c) or re.search(r's45pp[lr]01$', c)]

    motor_l_cols = [c for c in motor_cols if c.endswith('l01')]
    motor_r_cols = [c for c in motor_cols if c.endswith('r01')]
    lt_l_cols = [c for c in lt_cols if c.endswith('l01')]
    lt_r_cols = [c for c in lt_cols if c.endswith('r01')]
    pp_l_cols = [c for c in pp_cols if c.endswith('l01')]
    pp_r_cols = [c for c in pp_cols if c.endswith('r01')]

    # UEMS/LEMS components
    uems_muscle_codes = ['elbf','wrext','elbex','finfl','finab']
    lems_muscle_codes = ['hipfl','kneex','ankdo','greto','ankpl']
    uems_l_cols = [c for c in motor_l_cols if any(s in c for s in uems_muscle_codes)]
    uems_r_cols = [c for c in motor_r_cols if any(s in c for s in uems_muscle_codes)]
    lems_l_cols = [c for c in motor_l_cols if any(s in c for s in lems_muscle_codes)]
    lems_r_cols = [c for c in motor_r_cols if any(s in c for s in lems_muscle_codes)]

    # Calculate engineered features (handle missing cols gracefully)
    if motor_cols:
        eng_df['FE_TotalMotor_Wk1'] = eng_df[motor_cols].sum(axis=1, skipna=False)
        if uems_l_cols or uems_r_cols: eng_df['FE_UEMS_Wk1'] = eng_df[uems_l_cols + uems_r_cols].sum(axis=1, skipna=False)
        if lems_l_cols or lems_r_cols: eng_df['FE_LEMS_Wk1'] = eng_df[lems_l_cols + lems_r_cols].sum(axis=1, skipna=False)
        if motor_l_cols: eng_df['FE_MotorL_Wk1'] = eng_df[motor_l_cols].sum(axis=1, skipna=False)
        if motor_r_cols: eng_df['FE_MotorR_Wk1'] = eng_df[motor_r_cols].sum(axis=1, skipna=False)
        if 'FE_MotorL_Wk1' in eng_df.columns and 'FE_MotorR_Wk1' in eng_df.columns: eng_df['FE_MotorSymmAbsDiff_Wk1'] = (eng_df['FE_MotorL_Wk1'] - eng_df['FE_MotorR_Wk1']).abs()
        eng_df['FE_MotorMean_Wk1'] = eng_df[motor_cols].mean(axis=1, skipna=True)
        eng_df['FE_MotorStd_Wk1'] = eng_df[motor_cols].std(axis=1, skipna=True)
        eng_df['FE_MotorMin_Wk1'] = eng_df[motor_cols].min(axis=1, skipna=True)
        eng_df['FE_MotorMax_Wk1'] = eng_df[motor_cols].max(axis=1, skipna=True)
    if lt_cols:
        eng_df['FE_TotalLTS_Wk1'] = eng_df[lt_cols].sum(axis=1, skipna=False)
        if lt_l_cols: eng_df['FE_LTS_L_Wk1'] = eng_df[lt_l_cols].sum(axis=1, skipna=False);
        if lt_r_cols: eng_df['FE_LTS_R_Wk1'] = eng_df[lt_r_cols].sum(axis=1, skipna=False)
        if 'FE_LTS_L_Wk1' in eng_df.columns and 'FE_LTS_R_Wk1' in eng_df.columns: eng_df['FE_LTSSymmAbsDiff_Wk1'] = (eng_df['FE_LTS_L_Wk1'] - eng_df['FE_LTS_R_Wk1']).abs()
        eng_df['FE_LTSMean_Wk1'] = eng_df[lt_cols].mean(axis=1, skipna=True)
        eng_df['FE_LTSStd_Wk1'] = eng_df[lt_cols].std(axis=1, skipna=True)
    if pp_cols:
        eng_df['FE_TotalPPS_Wk1'] = eng_df[pp_cols].sum(axis=1, skipna=False)
        if pp_l_cols: eng_df['FE_PPS_L_Wk1'] = eng_df[pp_l_cols].sum(axis=1, skipna=False)
        if pp_r_cols: eng_df['FE_PPS_R_Wk1'] = eng_df[pp_r_cols].sum(axis=1, skipna=False)
        if 'FE_PPS_L_Wk1' in eng_df.columns and 'FE_PPS_R_Wk1' in eng_df.columns: eng_df['FE_PPSSymmAbsDiff_Wk1'] = (eng_df['FE_PPS_L_Wk1'] - eng_df['FE_PPS_R_Wk1']).abs()
        eng_df['FE_PPSMean_Wk1'] = eng_df[pp_cols].mean(axis=1, skipna=True)
        eng_df['FE_PPSStd_Wk1'] = eng_df[pp_cols].std(axis=1, skipna=True)

    # Fill NaNs for std dev columns (can happen if only one score is present)
    std_cols = [c for c in eng_df.columns if 'Std_Wk1' in c]
    eng_df[std_cols] = eng_df[std_cols].fillna(0)

    logger.info(f"Shape after Week 1 FE: {eng_df.shape}")
    return eng_df


def engineer_future_motor_features(df, motor_score_cols):
    """
    Applies feature engineering based on FUTURE motor scores (Wk26/52 actuals or predictions).
    (Ensure this is identical to the function in the training script)
    """
    SUFFIX = '_FutureMotor' # Consistent suffix for these features
    FEATURE_PREFIX = 'FM_'   # Prefix for these features

    if df is None or df.empty or not motor_score_cols:
        logger.warning("Input df is None/empty or no motor_score_cols provided to engineer_future_motor_features.")
        return pd.DataFrame({'PID': []}) # Return empty DataFrame with PID if possible

    # Ensure PID exists
    if 'PID' not in df.columns:
        logger.error("PID column missing in dataframe passed to engineer_future_motor_features.")
        return pd.DataFrame({'PID': []})

    # Select only relevant columns and copy to avoid modifying original df
    relevant_cols = ['PID'] + [col for col in motor_score_cols if col in df.columns]
    eng_df = df[relevant_cols].copy()

    motor_cols = [c for c in relevant_cols if c != 'PID'] # Get motor cols actually present
    if not motor_cols:
        logger.warning("No motor score columns found in provided dataframe.")
        return eng_df[['PID']] # Return just PID

    # Identify L/R columns and UEMS/LEMS components
    motor_l_cols = [c for c in motor_cols if c.endswith('l')]
    motor_r_cols = [c for c in motor_cols if c.endswith('r')]
    uems_muscle_codes_base = ['elbf','wrext','elbex','finfl','finab']
    lems_muscle_codes_base = ['hipfl','kneex','ankdo','greto','ankpl'] # Base names
    # Construct full L/R column names based on base codes and presence in data
    uems_l_cols = [f"{code}l" for code in uems_muscle_codes_base if f"{code}l" in motor_l_cols]
    uems_r_cols = [f"{code}r" for code in uems_muscle_codes_base if f"{code}r" in motor_r_cols]
    lems_l_cols = [f"{code}l" for code in lems_muscle_codes_base if f"{code}l" in motor_l_cols]
    lems_r_cols = [f"{code}r" for code in lems_muscle_codes_base if f"{code}r" in motor_r_cols]

    # Calculate engineered features with consistent naming
    eng_df[f'{FEATURE_PREFIX}TotalMotor{SUFFIX}'] = eng_df[motor_cols].sum(axis=1, skipna=False)
    if uems_l_cols or uems_r_cols: eng_df[f'{FEATURE_PREFIX}UEMS{SUFFIX}'] = eng_df[uems_l_cols + uems_r_cols].sum(axis=1, skipna=False)
    if lems_l_cols or lems_r_cols: eng_df[f'{FEATURE_PREFIX}LEMS{SUFFIX}'] = eng_df[lems_l_cols + lems_r_cols].sum(axis=1, skipna=False)
    if motor_l_cols: eng_df[f'{FEATURE_PREFIX}MotorL{SUFFIX}'] = eng_df[motor_l_cols].sum(axis=1, skipna=False)
    if motor_r_cols: eng_df[f'{FEATURE_PREFIX}MotorR{SUFFIX}'] = eng_df[motor_r_cols].sum(axis=1, skipna=False)
    # Calculate difference only if both L/R sums were calculable
    if f'{FEATURE_PREFIX}MotorL{SUFFIX}' in eng_df.columns and f'{FEATURE_PREFIX}MotorR{SUFFIX}' in eng_df.columns:
        eng_df[f'{FEATURE_PREFIX}MotorSymmAbsDiff{SUFFIX}'] = (eng_df[f'{FEATURE_PREFIX}MotorL{SUFFIX}'] - eng_df[f'{FEATURE_PREFIX}MotorR{SUFFIX}']).abs()

    engineered_cols = [col for col in eng_df.columns if col.startswith(FEATURE_PREFIX)]
    logger.info(f"Generated {len(engineered_cols)} features from future motor scores (suffix: {SUFFIX}).")

    # Return only PID and the newly created features
    return eng_df[['PID'] + engineered_cols]

# ==============================================================================
# Configuration (Should match the settings used during training)
# ==============================================================================

# --- Directory containing the saved models and feature list ---
# !!! IMPORTANT: SET THIS TO THE CORRECT DIRECTORY !!!
# Example: MODELS_DIR_TO_LOAD = './Submission_Files/trained_models_AutoTabPFN_SingleOutput_v6_FuncPred_M1_W1_F1_T1_E1_AutoFS1_Avg5_NoCV_2025-05-05_10-30-00'
MODELS_DIR_TO_LOAD = './Submission_Files/trained_models_REPLACE_WITH_YOUR_RUN_ID' # <<<--- CHANGE THIS

# --- File Paths (Input data paths should match training script) ---
DATA_DIR = './'
INPUT_DATA_DIR = f'{DATA_DIR}Input_Files/'
METADATA_FILE = f'{INPUT_DATA_DIR}metadata.csv'
# TRAIN_FEATURES_FILE = f'{INPUT_DATA_DIR}train_features.csv' # Not needed for prediction
# TRAIN_OUTCOMES_FILE = f'{INPUT_DATA_DIR}train_outcomes_functional.csv' # Not needed for prediction
TEST_FEATURES_FILE = f'{INPUT_DATA_DIR}test_features.csv' # Original Wk1 features for test
SUBMISSION_TEMPLATE_FILE = f'{INPUT_DATA_DIR}test_outcomes_Fun_template_update.csv' # Submission format

# --- External Future Motor Score Data (Settings must match training) ---
# These flags determine if future motor features were used during training
# and thus need to be recreated for prediction.
USE_FUTURE_MOTOR_FEATURES = True # <<<--- Must match training config
SELECT_FUTURE_MOTOR_FEATURES = True # <<<--- Must match training config (used to determine initial features)

EXTERNAL_PREDS_FILE = f'{INPUT_DATA_DIR}submission_MS_test_outcomes.csv' # <<<--- Must match training config
# TRAIN_OUTCOMES_MOTOR_FILE = f'{INPUT_DATA_DIR}train_outcomes_MS.csv' # Not needed for prediction

# --- Manual Feature Group Selection (Settings must match training) ---
# These flags determine the initial set of features before potential auto-selection
SELECT_METADATA = True           # <<<--- Must match training config
SELECT_WEEK1_ORIGINAL = True     # <<<--- Must match training config
SELECT_FE_FEATURES = True        # <<<--- Must match training config
SELECT_TARGET_TIME = True        # <<<--- Must match training config

# --- Automated Feature Selection (Only needed to know IF it was done) ---
# We load the final feature list, so thresholds aren't needed here,
# but knowing if selection happened helps verify the process.
DO_FEATURE_SELECTION = True      # <<<--- Must match training config

# --- Target Column ---
TARGET_COL = 'modben' # Should match training

# --- Post-processing Parameters ---
MIN_SCORE = 1
MAX_SCORE = 8

# --- Output File ---
# Generate a unique name based on the model directory being used
model_run_id = os.path.basename(MODELS_DIR_TO_LOAD).replace('trained_models_', '')
PREDICTION_OUTPUT_FILE = f'{DATA_DIR}Submission_Files/predictions_from_{model_run_id}.csv'

# ==============================================================================
# Data Loading and Preparation (Mirroring training script up to feature selection)
# ==============================================================================
logger.info(f"--- Starting Prediction using models from: {MODELS_DIR_TO_LOAD} ---")

# --- Load Base Data ---
logger.info("Loading base data files for prediction...")
try:
    metadata_df = pd.read_csv(METADATA_FILE)
    test_features_df = pd.read_csv(TEST_FEATURES_FILE)
    submission_template_df = pd.read_csv(SUBMISSION_TEMPLATE_FILE)

    # Pre-process: Replace 9 with NaN globally (Must match training)
    logger.warning("Replacing value 9 with NaN globally (matching training).")
    metadata_df.replace(9, np.nan, inplace=True)
    test_features_df.replace(9, np.nan, inplace=True)

except FileNotFoundError as e:
    logger.error(f"Error loading base data file: {e}. Check file paths.")
    exit(1)
logger.info("Base data loaded successfully.")

# --- Load External Future Motor Score Data (Test Predictions) ---
ext_preds_test_df = None
motor_score_cols_ext = []
if USE_FUTURE_MOTOR_FEATURES:
    logger.info("Loading external future motor score data (predicted for test)...")
    try:
        ext_preds_test_df = pd.read_csv(EXTERNAL_PREDS_FILE)
        logger.info(f"  Loaded test set predictions from: {EXTERNAL_PREDS_FILE} (Shape: {ext_preds_test_df.shape})")

        # Define expected motor score columns (Must match training)
        motor_score_cols_ext = ['elbfll','wrextl','elbexl','finfll','finabl','hipfll','kneexl','ankdol','gretol','ankpll',
                               'elbflr','wrextr','elbexr','finflr','finabr','hipflr','kneetr','ankdor','gretor','ankplr']

        missing_test_cols = [col for col in motor_score_cols_ext if col not in ext_preds_test_df.columns]
        if missing_test_cols:
            logger.error(f"Missing expected motor columns in test predictions file ({EXTERNAL_PREDS_FILE}): {missing_test_cols}")
            exit(1)
    except FileNotFoundError as e:
        logger.error(f"Error loading external motor score data file: {e}.")
        exit(1)
else:
    logger.info("Skipping loading of external future motor score data (USE_FUTURE_MOTOR_FEATURES=False).")


# --- Identify Initial Feature Groups (Mirroring training) ---
logger.info("Identifying initial feature groups...")
all_metadata_cols = [col for col in metadata_df.columns if col != 'PID']
all_test_week1_cols = [col for col in test_features_df.columns if col != 'PID']
# Note: We assume test_features only has Wk1, consistent with training script logic
common_week1_cols = sorted(all_test_week1_cols)

# --- Prepare Base Test Data (Merge Metadata + Common Week 1 Features) ---
logger.info("Merging metadata and common Week 1 features for test set...")
test_features_df_w1 = test_features_df[['PID'] + common_week1_cols].copy()
test_merged_df = pd.merge(metadata_df, test_features_df_w1, on='PID', how='inner')

# --- Feature Engineering Step 1: From Week 1 Data ---
logger.info("Performing Feature Engineering (from Week 1 data)...")
X_test_merged_fe_wk1 = engineer_features(test_merged_df, common_week1_cols)
engineered_features_wk1 = sorted([col for col in X_test_merged_fe_wk1.columns if col.startswith('FE_') and col.endswith('_Wk1')])

# --- Feature Engineering Step 2: From Future Motor Scores ---
engineered_features_future = []
X_test_merged_fe_all = X_test_merged_fe_wk1
if USE_FUTURE_MOTOR_FEATURES:
    logger.info("Performing Feature Engineering (from Future Motor Scores - Test Predictions)...")
    test_future_motor_features_df = engineer_future_motor_features(ext_preds_test_df, motor_score_cols_ext)
    engineered_features_future = sorted([col for col in test_future_motor_features_df.columns if col.startswith('FM_')])
    logger.info("Merging predicted future motor features into test data...")
    X_test_merged_fe_all = pd.merge(X_test_merged_fe_all, test_future_motor_features_df, on='PID', how='left')
    nan_check = X_test_merged_fe_all[[col for col in engineered_features_future if col in X_test_merged_fe_all.columns]].isnull().sum().sum()
    if nan_check > 0:
        logger.warning(f"Found {nan_check} NaNs in future motor features after merging to test data.")
    del test_future_motor_features_df
gc.collect()

# --- Select Initial Feature Set (Based on Manual Configuration used for training) ---
logger.info("Selecting initial feature set based on manual configuration flags (matching training)...")
initial_features_selected = []
if SELECT_METADATA: initial_features_selected.extend(all_metadata_cols)
if SELECT_WEEK1_ORIGINAL: initial_features_selected.extend(common_week1_cols)
if SELECT_FE_FEATURES: initial_features_selected.extend(engineered_features_wk1)
if USE_FUTURE_MOTOR_FEATURES and SELECT_FUTURE_MOTOR_FEATURES:
    initial_features_selected.extend(engineered_features_future)

available_cols_in_merged_df = X_test_merged_fe_all.columns.tolist()
selected_features_manual = sorted(list(set([f for f in initial_features_selected if f in available_cols_in_merged_df])))

# --- Prepare final X_test (using manually selected features + time) ---
logger.info("Aligning test data with submission template and adding time feature...")
submission_template_info = submission_template_df[['PID', 'time']].copy()
test_full_df = pd.merge(submission_template_info, X_test_merged_fe_all, on='PID', how='left')
test_PIDs = test_full_df['PID'] # Store PIDs for final submission
time_test = test_full_df['time'] # Store outcome times for test set

# Select the manually chosen features
X_test_pre_final = test_full_df[selected_features_manual].copy()

# Define the full list of features available *before* automated selection during training
FEATURES_BEFORE_AUTOFS = selected_features_manual.copy()
if SELECT_TARGET_TIME:
    X_test_pre_final['target_time'] = time_test
    FEATURES_BEFORE_AUTOFS.append('target_time')
    logger.info("Included 'target_time' in feature set.")
else:
    logger.info("Excluded 'target_time' from feature set.")

# --- Align Columns: Ensure test set has exactly same columns as FEATURES_BEFORE_AUTOFS ---
logger.info("Aligning columns before applying final feature selection list...")
missing_cols_test = set(FEATURES_BEFORE_AUTOFS) - set(X_test_pre_final.columns)
extra_cols_test = set(X_test_pre_final.columns) - set(FEATURES_BEFORE_AUTOFS)

if missing_cols_test:
    logger.warning(f"Columns missing in X_test_pre_final: {missing_cols_test}. Filling with NaN.")
    for col in missing_cols_test:
        X_test_pre_final[col] = np.nan

if extra_cols_test:
    logger.warning(f"Columns extra in X_test_pre_final: {extra_cols_test}. Dropping.")
    X_test_pre_final = X_test_pre_final.drop(columns=list(extra_cols_test))

# Ensure final feature set and order match FEATURES_BEFORE_AUTOFS
X_test_pre_final = X_test_pre_final[FEATURES_BEFORE_AUTOFS]
logger.info(f"Test data shape before applying final feature list: {X_test_pre_final.shape}")

# --- Load Final Feature List ---
logger.info("Loading the list of final features used during training...")
final_features_filename = os.path.join(MODELS_DIR_TO_LOAD, 'final_features_used.joblib')
try:
    FINAL_FEATURES_USED = joblib.load(final_features_filename)
    logger.info(f"Successfully loaded {len(FINAL_FEATURES_USED)} features from {final_features_filename}")
    # logger.debug(f"Features loaded: {FINAL_FEATURES_USED}")
except FileNotFoundError:
    logger.error(f"FATAL: Final features file not found: {final_features_filename}")
    logger.error("Cannot proceed without the list of features used for training.")
    exit(1)
except Exception as e:
    logger.error(f"FATAL: Error loading final features file: {e}", exc_info=True)
    exit(1)

# --- Select Final Features for Prediction ---
logger.info("Selecting the final features in the test set...")
try:
    # Ensure all required features are present before selection
    missing_final_feats = set(FINAL_FEATURES_USED) - set(X_test_pre_final.columns)
    if missing_final_feats:
        logger.error(f"FATAL: The prepared test data is missing columns required by the loaded feature list: {missing_final_feats}")
        exit(1)

    X_test = X_test_pre_final[FINAL_FEATURES_USED].copy()
    logger.info(f"Final test data shape for prediction: {X_test.shape}")
except KeyError as e:
    logger.error(f"FATAL: Error selecting final features from test data. Missing key: {e}")
    logger.error("This likely indicates a mismatch between the saved feature list and the prepared test data columns.")
    exit(1)

# Free memory
del X_test_pre_final, test_full_df, X_test_merged_fe_all, test_merged_df, test_features_df_w1
gc.collect()

# ==============================================================================
# Model Loading and Prediction
# ==============================================================================

# --- Find Saved Models ---
model_files = sorted(glob.glob(os.path.join(MODELS_DIR_TO_LOAD, 'model_avg_run_*.joblib')))

if not model_files:
    logger.error(f"FATAL: No model files (.joblib) found in the specified directory: {MODELS_DIR_TO_LOAD}")
    exit(1)

logger.info(f"Found {len(model_files)} model files to load for prediction.")

# --- Load Models and Predict ---
all_test_predictions = []
for i, model_path in enumerate(model_files):
    logger.info(f"--- Loading and Predicting with Model {i+1}/{len(model_files)} ---")
    logger.info(f"  Loading model: {model_path}")
    try:
        # Load the entire pipeline (preprocessor + model)
        current_pipeline = joblib.load(model_path)
        logger.info(f"  Model loaded successfully.")

        # Make predictions on the prepared test data
        logger.info(f"  Making predictions...")
        current_predictions = current_pipeline.predict(X_test)
        all_test_predictions.append(current_predictions)
        logger.info("  Predictions stored.")

        # Clean up memory for the loaded model
        del current_pipeline
        if 'current_predictions' in locals(): del current_predictions # Should be redundant but safe
        gc.collect()

    except FileNotFoundError:
        logger.error(f"  ERROR: Model file not found during loading (should not happen if glob worked): {model_path}")
        continue # Skip this model
    except Exception as e:
        logger.error(f"  ERROR loading or predicting with model {model_path}: {e}", exc_info=True)
        logger.warning(f"  Skipping predictions from this model.")
        continue # Skip this model

# --- Check if any predictions were successful ---
if not all_test_predictions:
    logger.error("FATAL: No predictions were successfully generated from any loaded model. Exiting.")
    exit(1)

# --- Calculate Average Predictions ---
logger.info(f"Averaging predictions across {len(all_test_predictions)} successful model loads...")
final_predictions_raw = np.mean(np.stack(all_test_predictions), axis=0)
logger.info("Averaging complete.")

# ==============================================================================
# Post-Processing and Submission
# ==============================================================================

# --- Post-Process Final Predictions ---
logger.info("Post-processing final predictions...")
try:
    # Clip predictions to the valid range
    final_predictions = final_predictions_raw.copy()
    logger.info(f"  Clipping predictions to range [{MIN_SCORE}, {MAX_SCORE}].")
    final_predictions = np.clip(final_predictions, MIN_SCORE, MAX_SCORE)

    # Round predictions to the nearest integer
    logger.info("  Rounding predictions to nearest integer.")
    final_predictions = np.round(final_predictions).astype(int)

    logger.info("Post-processing complete.")

except Exception as e:
    logger.error(f"ERROR during post-processing: {e}", exc_info=True)
    num_test_samples = len(X_test)
    final_predictions = np.full(num_test_samples, MIN_SCORE)
    logger.error("Using dummy predictions due to post-processing error.")


# --- Generate Submission File ---
logger.info("Generating submission file...")
try:
    predictions_df = pd.DataFrame({TARGET_COL: final_predictions})
    submission_base_df = pd.DataFrame({'PID': test_PIDs, 'time': time_test})

    submission_base_df.reset_index(drop=True, inplace=True)
    predictions_df.reset_index(drop=True, inplace=True)

    submission_df = pd.concat([submission_base_df, predictions_df], axis=1)

    template_cols = submission_template_df.columns.tolist()
    final_submission_df = pd.DataFrame(columns=template_cols)

    for col in template_cols:
        if col in submission_df.columns:
            final_submission_df[col] = submission_df[col]
        else:
            logger.warning(f"Column '{col}' from template not found. Filling with NaN.")
            final_submission_df[col] = np.nan

    extra_cols = set(submission_df.columns) - set(template_cols)
    if extra_cols:
         logger.warning(f"Extra columns found (will be ignored): {extra_cols}")

    if final_submission_df[TARGET_COL].isnull().any():
        nan_count_sub = final_submission_df[TARGET_COL].isnull().sum()
        logger.warning(f"Found {nan_count_sub} NaNs in submission target column '{TARGET_COL}'. Filling with {MIN_SCORE}.")
        final_submission_df[TARGET_COL].fillna(MIN_SCORE, inplace=True)
        final_submission_df[TARGET_COL] = final_submission_df[TARGET_COL].astype(int)

    # Save the submission file
    final_submission_df[template_cols].to_csv(PREDICTION_OUTPUT_FILE, index=False)
    logger.info(f"Submission file successfully saved to '{PREDICTION_OUTPUT_FILE}'")

except KeyError as e:
    logger.error(f"ERROR generating submission file: Missing key column {e}.")
except Exception as e:
    logger.error(f"ERROR saving submission file: {e}", exc_info=True)

# ==============================================================================
# End of Prediction Script
# ==============================================================================
logger.info(f"--- Finished Prediction using models from: {MODELS_DIR_TO_LOAD} ---")
print(f"\nPrediction log saved to {log_predict_filename}")
print(f"Submission file saved to {PREDICTION_OUTPUT_FILE}")
