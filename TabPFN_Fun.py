# ==============================================================================
# Imports
# ==============================================================================
import pandas as pd
import numpy as np
import warnings
import re
import datetime
import logging
import torch # For checking CUDA availability
import gc # For garbage collection
import joblib # For saving models
import os

# Scikit-learn imports
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn import clone # For cloning pipeline in averaging loop

# Pandas type checking
import pandas.api.types as ptypes

from scipy.stats import spearmanr # For Spearman correlation

# TabPFN specific import
try:
    # Using AutoTabPFNRegressor for automated ensembling/tuning within TabPFN
    from tabpfn import TabPFNRegressor # Base class for reference
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
except ImportError:
    print("Error: Required libraries not found.")
    print("Please install tabpfn and potentially tabpfn-extensions:")
    print("  pip install tabpfn")
    print("  pip install git+https://github.com/automl/TabPFN-extensions.git") # Needed for AutoTabPFNRegressor
    exit(1)


warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

# ==============================================================================
# Configuration
# ==============================================================================

# --- File Paths ---
DATA_DIR = './' # <<<--- Adapt this path to your data directory
INPUT_DATA_DIR = f'{DATA_DIR}Input_Files/'
METADATA_FILE = f'{INPUT_DATA_DIR}metadata.csv'
TRAIN_FEATURES_FILE = f'{INPUT_DATA_DIR}train_features.csv' # Original Wk1 features
TRAIN_OUTCOMES_FILE = f'{INPUT_DATA_DIR}train_outcomes_functional.csv' # Target (modben)
TEST_FEATURES_FILE = f'{INPUT_DATA_DIR}test_features.csv' # Original Wk1 features for test
SUBMISSION_TEMPLATE_FILE = f'{INPUT_DATA_DIR}test_outcomes_Fun_template_update.csv' # Submission format

# --- External Future Motor Score Data ---
# Set to True to load and use features derived from future motor scores
# (requires the files below)
USE_FUTURE_MOTOR_FEATURES = True
# Path to the file containing PREDICTED future motor scores for the TEST set
# (e.g., a submission file from the neurological recovery track)
EXTERNAL_PREDS_FILE = f'{INPUT_DATA_DIR}submission_MS_test_outcomes.csv'
# Path to the file containing ACTUAL future motor scores for the TRAIN set
TRAIN_OUTCOMES_MOTOR_FILE = f'{INPUT_DATA_DIR}train_outcomes_MS.csv'

# --- Manual Feature Group Selection ---
# Select which categories of features to *initially* consider before automated selection.
SELECT_METADATA = True           # Include patient metadata (age, sex, etc.)
SELECT_WEEK1_ORIGINAL = True     # Include original ISNCSCI scores from Week 1
SELECT_FE_FEATURES = True        # Include features engineered from Week 1 scores (LEMS, UEMS, etc.)
SELECT_TARGET_TIME = True        # Include outcome measurement time (26/52 weeks) as a feature
SELECT_FUTURE_MOTOR_FEATURES = True # Include features engineered from future motor scores (actual/predicted)
                                    # (Only effective if USE_FUTURE_MOTOR_FEATURES is also True)

# --- Automated Feature Selection ---
# Further reduce the feature set based on statistical properties.
DO_FEATURE_SELECTION = True      # Master switch: True to enable automated selection below.
# 1. Variance Threshold: Remove features with near-zero variance.
VAR_THRESH = 0.01                # Threshold for variance; features with variance <= this are dropped.
# 2. Correlation Threshold: Remove highly correlated features to reduce redundancy.
CORR_THRESH = 0.93               # Threshold for correlation; one feature from a pair with abs(corr) >= this is dropped.
# 3. Univariate Selection: Keep the features most correlated with the target.
UNIVARIATE_K = 75                # Number of top numeric features to keep based on f_regression score. Use 'all' to skip this step.

# --- Model Configuration ---
MODEL_TYPE = 'AutoTabPFN'          # Model to use ('AutoTabPFN')
PERFORM_CV = False                 # Perform cross-validation? (Usually False when averaging)
CV_FOLDS = 5                       # Number of CV folds if PERFORM_CV is True

# --- Averaging Configuration ---
# Train multiple models with different seeds and average their predictions.
PERFORM_AVERAGING = True           # Enable/disable averaging.
N_AVERAGING_RUNS = 5               # Number of models to train in the ensemble.
BASE_RANDOM_STATE = 42             # Starting seed for reproducibility.

# --- AutoTabPFN Specific Configuration ---
AUTO_TABPFN_TIME_BUDGET_SECONDS = 3600 # Time limit for *each* AutoTabPFN model fit (in seconds).
TABPFN_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if available
# Parameters passed to AutoTabPFNRegressor constructor (excluding random_state)
AUTOTABPFN_PARAMS = {
    'device': TABPFN_DEVICE,
    'max_time': AUTO_TABPFN_TIME_BUDGET_SECONDS,
    # Add other AutoTabPFN parameters here if needed
}

# ==============================================================================
# Setup Logging
# ==============================================================================

# --- Generate Run ID and Log Filename ---
LOG_DIR = f'{DATA_DIR}Log_Files'
log_filename = 'Functional_Metrics_AutoTabPFN_AvgFE_AutoFS_Ext.log'
log_file = os.path.join(LOG_DIR, log_filename) # Full path to the log file
# Components for Run ID based on configuration
cv_mode_str = f"{CV_FOLDS}FoldCV" if (PERFORM_CV and not PERFORM_AVERAGING) else "NoCV"
avg_mode_str = f"Avg{N_AVERAGING_RUNS}" if PERFORM_AVERAGING else "SingleRun"
ms_str = f"M{int(SELECT_METADATA)}_W{int(SELECT_WEEK1_ORIGINAL)}_F{int(SELECT_FE_FEATURES)}_T{int(SELECT_TARGET_TIME)}_E{int(SELECT_FUTURE_MOTOR_FEATURES and USE_FUTURE_MOTOR_FEATURES)}"
fs_str = f"AutoFS{int(DO_FEATURE_SELECTION)}"
base_model_name = f"{MODEL_TYPE}_SingleOutput_v6_FuncPred_{ms_str}_{fs_str}" # v6 includes AutoFS
model_name = f"{base_model_name}_{avg_mode_str}_{cv_mode_str}"
run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_id = f"{model_name}_{run_timestamp}"
# Define submission file path within the new folder
SUBMISSION_OUTPUT_FILE = f'{DATA_DIR}Submission_Files/submission_{run_id}.csv' # Final submission filename
# Define models directory path within the new folder
MODELS_DIR = f'{DATA_DIR}Submission_Files/trained_models_{run_id}'

# --- Create Log Directory ---
try:
    os.makedirs(LOG_DIR, exist_ok=True)
    # No need to log here yet, logger isn't fully configured
except OSError as e:
    # Print error to console if directory creation fails, as logger isn't ready
    print(f"CRITICAL ERROR: Could not create log directory {LOG_DIR}: {e}")
    print("Exiting.")
    exit(1) # Exit if we can't create the log directory

# --- Configure Logger ---
logger = logging.getLogger(run_id)
logger.setLevel(logging.INFO)
# Prevent duplicate handlers if script is run multiple times (e.g., in notebooks)
if not logger.handlers:
    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # Console Handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

# --- Log Initial Configuration ---
logger.info(f"--- Starting Run: {run_id} ---")
logger.info(f"Goal: Predicting Functional Outcome (Benzel Score)")
logger.info(f"Model: {MODEL_TYPE}")
logger.info(f"Averaging: {PERFORM_AVERAGING} (Runs: {N_AVERAGING_RUNS if PERFORM_AVERAGING else 1})")
logger.info(f"Cross-Validation: {PERFORM_CV and not PERFORM_AVERAGING}")
logger.info(f"AutoTabPFN Base Params: {AUTOTABPFN_PARAMS}")
logger.info(f"Manual Feature Selection: Metadata={SELECT_METADATA}, Week1={SELECT_WEEK1_ORIGINAL}, FE_Wk1={SELECT_FE_FEATURES}, TargetTime={SELECT_TARGET_TIME}, FutureMotor={SELECT_FUTURE_MOTOR_FEATURES and USE_FUTURE_MOTOR_FEATURES}")
logger.info(f"Automated Feature Selection: {DO_FEATURE_SELECTION} (VarThresh={VAR_THRESH}, CorrThresh={CORR_THRESH}, K={UNIVARIATE_K})")
if USE_FUTURE_MOTOR_FEATURES:
    logger.info(f"  External Preds File (Test): {EXTERNAL_PREDS_FILE}")
    logger.info(f"  External Outcomes File (Train): {TRAIN_OUTCOMES_MOTOR_FILE}")
logger.info(f"Data Directory: {DATA_DIR}")
logger.info(f"Log File: {log_file}")
logger.info(f"Submission File: {SUBMISSION_OUTPUT_FILE}")
logger.info(f"TabPFN Device: {TABPFN_DEVICE}")

# --- Create Model Save Directory ---
try:
    # Create the directory if it doesn't exist.
    # exist_ok=True prevents an error if the directory already exists.
    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info(f"Created/verified models directory: {MODELS_DIR}")
except OSError as e:
    logger.error(f"Error creating models directory {MODELS_DIR}: {e}", exc_info=True)
    logger.error("Models will not be saved. Exiting.")

# ==============================================================================
# Helper Functions
# ==============================================================================

def spearman_corr(y_true, y_pred):
    """
    Calculates Spearman Rank Correlation Coefficient.
    Handles NaNs and constant arrays gracefully for use in scoring.
    """
    y_true_arr = np.array(y_true).squeeze()
    y_pred_arr = np.array(y_pred).squeeze()
    # Return 0 if either array is all NaN
    if np.all(np.isnan(y_true_arr)) or np.all(np.isnan(y_pred_arr)):
        return 0.0
    # Check for constant arrays (std dev is 0)
    if y_true_arr.ndim == 0 or y_pred_arr.ndim == 0 or np.std(y_true_arr) == 0 or np.std(y_pred_arr) == 0:
        # Correlation is 1 if they are identical constants, 0 otherwise
        return 1.0 if np.all(y_true_arr == y_pred_arr) else 0.0
    # Calculate Spearman correlation
    corr, _ = spearmanr(y_true_arr, y_pred_arr)
    # Return 0 if calculation results in NaN (e.g., due to ties with very small N)
    return corr if not np.isnan(corr) else 0.0

# Create a scorer object for use with scikit-learn functions
spearman_scorer = make_scorer(spearman_corr, greater_is_better=True)


def select_features(X_train, y_train, X_test, features,
                    var_thresh=0.01,
                    corr_thresh=0.95,
                    k='all'):
    """
    Performs automated feature selection on numeric features using:
    1. Low Variance Filter (VarianceThreshold)
    2. High Correlation Filter (Pearson correlation)
    3. Univariate Selection (SelectKBest with f_regression)

    Non-numeric features are automatically kept.
    Requires SimpleImputer internally for handling NaNs during selection steps.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test features.
        features (list): List of feature names in X_train/X_test to consider.
        var_thresh (float): Variance threshold for step 1.
        corr_thresh (float): Absolute correlation threshold for step 2.
        k (int or 'all'): Number of top features to select in step 3.

    Returns:
        tuple: (X_train_selected, X_test_selected, final_feature_list)
               DataFrames containing only the selected features, and the list of selected feature names.
               Original NaNs are preserved in the returned DataFrames.
    """
    logger.info(f"Starting automated feature selection on {len(features)} candidate features...")
    start_time_fs = datetime.datetime.now()

    # Separate numeric and non-numeric features
    numeric_feats = [f for f in features if ptypes.is_numeric_dtype(X_train[f])]
    non_numeric_feats = [f for f in features if f not in numeric_feats]
    logger.info(f"Identified {len(numeric_feats)} numeric and {len(non_numeric_feats)} non-numeric features.")

    # Handle edge case where no numeric features exist
    if not numeric_feats:
        logger.warning("No numeric features found for selection. Returning original features.")
        return X_train.copy(), X_test.copy(), features

    # --- Perform selection only on numeric features ---
    X_train_num = X_train[numeric_feats].copy()
    X_test_num  = X_test[numeric_feats].copy()

    # Temporarily impute NaNs for selection calculations
    logger.debug("Imputing numeric features (median) temporarily for selection calculations...")
    imp = SimpleImputer(strategy='median')
    X_train_num_imp = pd.DataFrame(imp.fit_transform(X_train_num), columns=numeric_feats, index=X_train.index)
    # X_test_num_imp = pd.DataFrame(imp.transform(X_test_num), columns=numeric_feats, index=X_test.index) # Impute test later if needed

    selected_numeric_feats = numeric_feats # Start with all numeric features

    # 1. Low Variance Filter
    logger.info(f"Applying Variance Threshold (threshold={var_thresh})...")
    vt = VarianceThreshold(threshold=var_thresh)
    vt.fit(X_train_num_imp)
    variance_mask = vt.get_support()
    selected_numeric_feats = [f for f, keep in zip(selected_numeric_feats, variance_mask) if keep]
    X_train_num_imp = X_train_num_imp[selected_numeric_feats] # Keep only selected columns for next step
    logger.info(f"Variance Threshold kept {len(selected_numeric_feats)} numeric features.")

    # Check if any numeric features remain
    if not selected_numeric_feats:
        logger.warning("Variance Threshold removed all numeric features. Returning only non-numeric.")
        feats_final = non_numeric_feats
        return X_train[feats_final].copy(), X_test[feats_final].copy(), feats_final

    # 2. High Correlation Filter
    logger.info(f"Applying Correlation Threshold (threshold={corr_thresh})...")
    corr_matrix = X_train_num_imp.corr().abs()
    # Get upper triangle of correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Find index of feature columns with correlation greater than threshold
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_thresh)]
    selected_numeric_feats = [f for f in selected_numeric_feats if f not in to_drop]
    X_train_num_imp = X_train_num_imp[selected_numeric_feats] # Keep only selected columns
    logger.info(f"Correlation Threshold kept {len(selected_numeric_feats)} numeric features.")

    # Check if any numeric features remain
    if not selected_numeric_feats:
        logger.warning("Correlation Threshold removed all numeric features. Returning only non-numeric.")
        feats_final = non_numeric_feats
        return X_train[feats_final].copy(), X_test[feats_final].copy(), feats_final

    # 3. Univariate Selection
    if k != 'all' and isinstance(k, int) and k > 0 and len(selected_numeric_feats) > k:
        k_final = min(k, len(selected_numeric_feats)) # Adjust k if needed
        logger.info(f"Applying Univariate Selection (f_regression, k={k_final})...")
        skb = SelectKBest(score_func=f_regression, k=k_final)
        # Fit on the imputed training data
        skb.fit(X_train_num_imp, y_train)
        univariate_mask = skb.get_support()
        selected_numeric_feats = [f for f, keep in zip(selected_numeric_feats, univariate_mask) if keep]
        logger.info(f"Univariate Selection kept {len(selected_numeric_feats)} numeric features.")
    else:
        logger.info(f"Skipping Univariate Selection (k='{k}', available features={len(selected_numeric_feats)}).")

    # --- Combine selected numeric and original non-numeric features ---
    final_selected_features = sorted(selected_numeric_feats + non_numeric_feats)

    end_time_fs = datetime.datetime.now()
    logger.info(f"Automated Feature Selection complete. Selected {len(final_selected_features)} total features. Time: {end_time_fs - start_time_fs}")

    # Return the original DataFrames subsetted to the final feature list
    # This preserves the original NaNs for the main modeling pipeline
    X_train_selected = X_train[final_selected_features].copy()
    X_test_selected = X_test[final_selected_features].copy()

    return X_train_selected, X_test_selected, final_selected_features


def engineer_features(df, week1_feature_cols):
    """
    Applies feature engineering based on WEEK 1 ISNCSCI scores.
    Calculates sums, means, std devs, L/R differences for motor and sensory scores.

    Args:
        df (pd.DataFrame): Input dataframe containing Week 1 features.
        week1_feature_cols (list): List of column names for Week 1 features.

    Returns:
        pd.DataFrame: Dataframe with engineered features added.
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
    Uses a consistent suffix '_FutureMotor' for generated features.

    Args:
        df (pd.DataFrame): Input df containing future motor scores and 'PID'.
        motor_score_cols (list): List of column names for the motor scores.

    Returns:
        pd.DataFrame: Dataframe with 'PID' and engineered future motor features.
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
# Data Loading and Preparation
# ==============================================================================

# --- Load Base Data ---
logger.info("Loading base data files...")
try:
    metadata_df = pd.read_csv(METADATA_FILE)
    train_features_df = pd.read_csv(TRAIN_FEATURES_FILE) # Contains Wk1, Wk4, etc.
    train_outcomes_df = pd.read_csv(TRAIN_OUTCOMES_FILE) # Contains target 'modben'
    test_features_df = pd.read_csv(TEST_FEATURES_FILE)   # Contains Wk1 features for test set
    submission_template_df = pd.read_csv(SUBMISSION_TEMPLATE_FILE) # Format for submission

    # Pre-process: Replace 9 with NaN globally (common missing value indicator)
    # Warning: Verify '9' doesn't have another meaning in specific columns via data dictionary.
    logger.warning("Replacing value 9 with NaN globally. Verify appropriateness via data dictionary.")
    metadata_df.replace(9, np.nan, inplace=True)
    train_features_df.replace(9, np.nan, inplace=True)
    test_features_df.replace(9, np.nan, inplace=True)

except FileNotFoundError as e:
    logger.error(f"Error loading base data file: {e}. Check file paths.")
    exit(1)
logger.info("Base data loaded successfully.")


# --- Load External Future Motor Score Data ---
ext_preds_test_df = None
ext_actuals_train_df = None
motor_score_cols_ext = [] # List of motor score column names in external files
if USE_FUTURE_MOTOR_FEATURES:
    logger.info("Loading external future motor score data (predicted for test, actual/OOF for train)...")
    try:
        ext_preds_test_df = pd.read_csv(EXTERNAL_PREDS_FILE)
        logger.info(f"  Loaded test set predictions from: {EXTERNAL_PREDS_FILE} (Shape: {ext_preds_test_df.shape})")

        ext_actuals_train_df = pd.read_csv(TRAIN_OUTCOMES_MOTOR_FILE)
        logger.info(f"  Loaded train set actuals/OOF from: {TRAIN_OUTCOMES_MOTOR_FILE} (Shape: {ext_actuals_train_df.shape})")

        # Define expected motor score columns based on example header
        motor_score_cols_ext = ['elbfll','wrextl','elbexl','finfll','finabl','hipfll','kneexl','ankdol','gretol','ankpll',
                               'elbflr','wrextr','elbexr','finflr','finabr','hipflr','kneetr','ankdor','gretor','ankplr']

        # Verify columns exist in loaded dataframes
        missing_test_cols = [col for col in motor_score_cols_ext if col not in ext_preds_test_df.columns]
        missing_train_cols = [col for col in motor_score_cols_ext if col not in ext_actuals_train_df.columns]
        if missing_test_cols:
            logger.error(f"Missing expected motor columns in test predictions file ({EXTERNAL_PREDS_FILE}): {missing_test_cols}")
            exit(1)
        if missing_train_cols:
            logger.error(f"Missing expected motor columns in train actuals file ({TRAIN_OUTCOMES_MOTOR_FILE}): {missing_train_cols}")
            exit(1)

    except FileNotFoundError as e:
        logger.error(f"Error loading external motor score data file: {e}.")
        logger.error("Cannot proceed if USE_FUTURE_MOTOR_FEATURES is True but files are missing.")
        exit(1)
else:
    logger.info("Skipping loading of external future motor score data (USE_FUTURE_MOTOR_FEATURES=False).")


# --- Identify Initial Feature Groups ---
logger.info("Identifying initial feature groups...")
TARGET_COL = 'modben' # The target variable we want to predict

# Check if target exists in the outcomes file
if TARGET_COL not in train_outcomes_df.columns:
    logger.error(f"Target column '{TARGET_COL}' not found in {TRAIN_OUTCOMES_FILE}. Exiting.")
    exit(1)

# Define potential feature groups based on data source and naming conventions
all_metadata_cols = [col for col in metadata_df.columns if col != 'PID']
# Week 1 features have '01' suffix or are 'ais1'
all_train_week1_cols = [col for col in train_features_df.columns if re.search(r'01$', col) or col == 'ais1']
# Assume test features file only contains week 1 features (as per challenge description)
all_test_week1_cols = [col for col in test_features_df.columns if col != 'PID']
# Use only Week 1 features present in both train and test feature files
common_week1_cols = sorted(list(set(all_train_week1_cols).intersection(all_test_week1_cols)))

logger.info(f"Identified {len(all_metadata_cols)} potential metadata columns.")
logger.info(f"Identified {len(common_week1_cols)} common Week 1 ISNCSCI columns.")


# --- Prepare Base Training Data (Merge Metadata + Common Week 1 Features) ---
logger.info("Merging metadata and common Week 1 features for training set...")
# Select only common week 1 features + PID from train features
train_features_w1_df = train_features_df[['PID'] + common_week1_cols].copy()
# Merge with metadata
train_merged_df = pd.merge(metadata_df, train_features_w1_df, on='PID', how='inner')
logger.info(f"  Training data shape after initial merge: {train_merged_df.shape}")


# --- Feature Engineering Step 1: From Week 1 Data ---
logger.info("Performing Feature Engineering (from Week 1 data)...")
X_train_merged_fe_wk1 = engineer_features(train_merged_df, common_week1_cols)
engineered_features_wk1 = sorted([col for col in X_train_merged_fe_wk1.columns if col.startswith('FE_') and col.endswith('_Wk1')])
logger.info(f"  Identified {len(engineered_features_wk1)} Week 1 engineered features.")
# Free up memory
del train_merged_df, train_features_w1_df
gc.collect()


# --- Feature Engineering Step 2: From Future Motor Scores ---
engineered_features_future = [] # Initialize empty list
X_train_merged_fe_all = X_train_merged_fe_wk1 # Start with Wk1 FE features
if USE_FUTURE_MOTOR_FEATURES:
    logger.info("Performing Feature Engineering (from Future Motor Scores - Train Actuals/OOF)...")
    # Engineer features from the loaded actual/OOF training data
    train_future_motor_features_df = engineer_future_motor_features(ext_actuals_train_df, motor_score_cols_ext)
    engineered_features_future = sorted([col for col in train_future_motor_features_df.columns if col.startswith('FM_')]) # Get names of new features
    logger.info(f"  Identified {len(engineered_features_future)} future motor engineered features.")

    # Merge these features into the main training dataframe
    logger.info("Merging future motor features into training data...")
    X_train_merged_fe_all = pd.merge(X_train_merged_fe_all, train_future_motor_features_df, on='PID', how='left')
    logger.info(f"  Training data shape after merging future motor features: {X_train_merged_fe_all.shape}")
    # Check for NaNs introduced by merge (if PIDs didn't perfectly align)
    nan_check = X_train_merged_fe_all[engineered_features_future].isnull().sum().sum()
    if nan_check > 0:
        logger.warning(f"  Found {nan_check} NaNs in future motor features after merging to train data (check PID alignment).")
    del train_future_motor_features_df # Free memory
else:
    logger.info("Skipping future motor feature engineering (USE_FUTURE_MOTOR_FEATURES=False).")
gc.collect()


# --- Select Initial Feature Set (Based on Manual Configuration) ---
logger.info("Selecting initial feature set based on manual configuration flags...")
initial_features_selected = []
if SELECT_METADATA: initial_features_selected.extend(all_metadata_cols)
if SELECT_WEEK1_ORIGINAL: initial_features_selected.extend(common_week1_cols)
if SELECT_FE_FEATURES: initial_features_selected.extend(engineered_features_wk1)
# Add future motor features only if the master flag and the selection flag are True
if USE_FUTURE_MOTOR_FEATURES and SELECT_FUTURE_MOTOR_FEATURES:
    initial_features_selected.extend(engineered_features_future)

# Ensure we only keep features that actually exist in the merged dataframe, remove duplicates
available_cols_in_merged_df = X_train_merged_fe_all.columns.tolist()
selected_features_manual = sorted(list(set([f for f in initial_features_selected if f in available_cols_in_merged_df])))
logger.info(f"  Manually selected {len(selected_features_manual)} base features (before adding target_time/autoFS).")


# --- Prepare final X_train, y_train (using manually selected features) ---
logger.info("Merging functional outcomes and finalizing training data (pre-autoFS)...")
# Merge the functional outcomes (target variable)
train_full_df = pd.merge(X_train_merged_fe_all, train_outcomes_df[['PID', TARGET_COL, 'time']], on='PID', how='inner')

# Select only the manually chosen features
X_train_pre_fs = train_full_df[selected_features_manual].copy()
y_train_raw = train_full_df[TARGET_COL].copy()
time_train_raw = train_full_df['time'].copy()

# Drop rows where the target variable itself is NaN
valid_target_indices = y_train_raw.dropna().index
initial_rows = len(X_train_pre_fs)
X_train_pre_fs = X_train_pre_fs.loc[valid_target_indices].reset_index(drop=True)
y_train = y_train_raw.loc[valid_target_indices].reset_index(drop=True)
time_train = time_train_raw.loc[valid_target_indices].reset_index(drop=True)
final_rows = len(X_train_pre_fs)
logger.info(f"  Dropped {initial_rows - final_rows} rows with missing target values.")

# Define the full list of features available *before* automated selection
# (including target_time if selected)
FEATURES_BEFORE_AUTOFS = selected_features_manual.copy()
if SELECT_TARGET_TIME:
    X_train_pre_fs['target_time'] = time_train
    FEATURES_BEFORE_AUTOFS.append('target_time')
    logger.info("  Including 'target_time' in initial feature set.")
else:
    logger.info("  Excluding 'target_time' from initial feature set.")

logger.info(f"  Training data shape before autoFS: {X_train_pre_fs.shape}")
# Free memory
del X_train_merged_fe_all, train_full_df, y_train_raw, time_train_raw
gc.collect()


# --- Prepare Test Data (Apply FE, Merge External, Select Manual) ---
logger.info("Preparing test data (pre-autoFS)...")
# Merge metadata and common Wk1 features for test set
test_features_df_w1 = test_features_df[['PID'] + common_week1_cols].copy()
test_merged_df = pd.merge(metadata_df, test_features_df_w1, on='PID', how='inner')
# Apply Week 1 FE
X_test_merged_fe_wk1 = engineer_features(test_merged_df, common_week1_cols)

# Apply Future Motor FE using PREDICTED scores for test set
X_test_merged_fe_all = X_test_merged_fe_wk1 # Start with Wk1 FE features
if USE_FUTURE_MOTOR_FEATURES:
    logger.info("  Performing Feature Engineering (from Future Motor Scores - Test Predictions)...")
    test_future_motor_features_df = engineer_future_motor_features(ext_preds_test_df, motor_score_cols_ext)
    logger.info("  Merging predicted future motor features into test data...")
    X_test_merged_fe_all = pd.merge(X_test_merged_fe_all, test_future_motor_features_df, on='PID', how='left')
    # Check for NaNs (can happen if PIDs in prediction file don't match test set PIDs)
    nan_check = X_test_merged_fe_all[[col for col in engineered_features_future if col in X_test_merged_fe_all.columns]].isnull().sum().sum()
    if nan_check > 0:
        logger.warning(f"  Found {nan_check} NaNs in future motor features after merging to test data (check PID alignment).")
    del test_future_motor_features_df # Free memory

gc.collect()

# Align test data with submission template PIDs and outcome times
submission_template_info = submission_template_df[['PID', 'time']].copy()
test_full_df = pd.merge(submission_template_info, X_test_merged_fe_all, on='PID', how='left')
test_PIDs = test_full_df['PID'] # Store PIDs for final submission
time_test = test_full_df['time'] # Store outcome times for test set

# Select the same manually chosen base features as for training
logger.info(f"  Selecting {len(selected_features_manual)} manually selected base features for test set...")
X_test_pre_fs = test_full_df[selected_features_manual].copy()

# Add target time if it was selected for training
if SELECT_TARGET_TIME:
    X_test_pre_fs['target_time'] = time_test

# --- Align Columns: Ensure test set has exactly same columns as train set before auto FS ---
logger.info("  Aligning columns between train and test sets (pre-autoFS)...")
missing_cols_test = set(FEATURES_BEFORE_AUTOFS) - set(X_test_pre_fs.columns)
extra_cols_test = set(X_test_pre_fs.columns) - set(FEATURES_BEFORE_AUTOFS)

if missing_cols_test:
    logger.warning(f"  Columns missing in X_test_pre_fs: {missing_cols_test}. Filling with NaN.")
    for col in missing_cols_test:
        X_test_pre_fs[col] = np.nan # Add missing columns

if extra_cols_test:
    logger.warning(f"  Columns extra in X_test_pre_fs: {extra_cols_test}. Dropping.")
    X_test_pre_fs = X_test_pre_fs.drop(columns=list(extra_cols_test))

# Ensure final feature set and order match FEATURES_BEFORE_AUTOFS
X_test_pre_fs = X_test_pre_fs[FEATURES_BEFORE_AUTOFS]

logger.info(f"  Test data shape before autoFS: {X_test_pre_fs.shape}")
# Free memory
del X_test_merged_fe_all, test_full_df, X_test_merged_fe_wk1
gc.collect()


# --- Automated Feature Selection Step ---
if DO_FEATURE_SELECTION:
    logger.info("--- Running Automated Feature Selection ---")
    # Apply selection function to X_train_pre_fs and X_test_pre_fs
    X_train, X_test, FINAL_FEATURES_USED = select_features(
        X_train_pre_fs, y_train, X_test_pre_fs, FEATURES_BEFORE_AUTOFS,
        var_thresh=VAR_THRESH, corr_thresh=CORR_THRESH, k=UNIVARIATE_K
    )
    logger.info(f"--- Feature Selection Complete: {len(FINAL_FEATURES_USED)} features remaining ---")
    logger.info(f"Final features selected by automated process: {FINAL_FEATURES_USED}")

    # Log the selected features if needed (can be long)
    # logger.debug(f"Selected features: {FINAL_FEATURES_USED}")
else:
    logger.info("--- Skipping Automated Feature Selection ---")
    # Use the dataframes and feature list from before this step
    X_train = X_train_pre_fs.copy()
    X_test = X_test_pre_fs.copy()
    FINAL_FEATURES_USED = FEATURES_BEFORE_AUTOFS # This list contains all manually selected features
    logger.info(f"Using {len(FINAL_FEATURES_USED)} features selected manually (automated selection skipped): {FINAL_FEATURES_USED}")


# Log final shapes and NaN counts going into the model pipeline
logger.info(f"Final data shapes for modeling: X_train={X_train.shape}, X_test={X_test.shape}")
train_nan_count = X_train.isnull().sum().sum()
test_nan_count = X_test.isnull().sum().sum()
if train_nan_count > 0: logger.warning(f"Final X_train contains {train_nan_count} NaNs (will be passed to model).")
if test_nan_count > 0: logger.warning(f"Final X_test contains {test_nan_count} NaNs (will be passed to model).")

# --- Save the list of final features used ---
# This list is needed by the prediction script to select the correct columns
final_features_filename = os.path.join(MODELS_DIR, 'final_features_used.joblib')
try:
    joblib.dump(FINAL_FEATURES_USED, final_features_filename)
    logger.info(f"Saved list of {len(FINAL_FEATURES_USED)} final features used to: {final_features_filename}")
except Exception as e_save_feats:
    logger.error(f"ERROR saving final features list: {e_save_feats}", exc_info=True)
    # Decide if this is critical - maybe exit if prediction script relies on it?
    # For now, just log the error.
    
# Free memory
del X_train_pre_fs, X_test_pre_fs
gc.collect()


# --- Preprocessing Pipeline Definition ---
logger.info(f"Setting up preprocessing pipeline for {len(FINAL_FEATURES_USED)} selected features...")

# Define base categories - these might not all be present in FINAL_FEATURES_USED
meta_categorical_base = ['age_category', 'bmi_category', 'tx1_r', 'sexcd']
w1_ordinal_base = ['ais1']
ais_categories = ['A', 'B', 'C', 'D', 'E'] # Fixed order for AIS grade

# Identify feature types *within the final selected features*
categorical_features = sorted([f for f in meta_categorical_base if f in FINAL_FEATURES_USED])
ordinal_features = sorted([f for f in w1_ordinal_base if f in FINAL_FEATURES_USED])
processed_cols = set(categorical_features + ordinal_features)
# Numerical features are all remaining features in the final list
numerical_features = sorted([col for col in FINAL_FEATURES_USED if col not in processed_cols])

logger.info(f"  Encoding {len(categorical_features)} Categorical Features: {categorical_features}")
logger.info(f"  Encoding {len(ordinal_features)} Ordinal Features: {ordinal_features}")
logger.info(f"  Passing through {len(numerical_features)} Numerical Features.")

# Verify all selected features are categorized
check_final = set(FINAL_FEATURES_USED)
check_assigned = set(categorical_features + ordinal_features + numerical_features)
if check_final != check_assigned:
    logger.error("FATAL: Preprocessing column assignment mismatch! Some features in FINAL_FEATURES_USED were not categorized.")
    logger.error(f"  Features in FINAL_FEATURES_USED but not assigned: {check_final - check_assigned}")
    logger.error(f"  Features assigned but not in FINAL_FEATURES_USED: {check_assigned - check_final}")
    exit(1)
else:
    logger.info("  Preprocessing column assignment verification successful.")

# Define transformers for categorical and ordinal features
# Imputation is necessary here because encoders require non-missing values
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')), # Impute categorical NaNs
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))      # One-hot encode
])

# Handle ordinal features only if they exist after selection
if ordinal_features:
    # Use predefined categories for AIS grade
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Impute ordinal NaNs
        ('ordinal', OrdinalEncoder(categories=[ais_categories] * len(ordinal_features), # Use defined order
                                   handle_unknown='use_encoded_value', unknown_value=-1)) # Handle unseen values in test
    ])
else:
    ordinal_transformer = 'drop' # Drop if no ordinal features are selected

# Dynamically build the list of transformers for ColumnTransformer
transformers_list = []
if categorical_features:
    transformers_list.append(('cat', cat_transformer, categorical_features))
if ordinal_features and ordinal_transformer != 'drop':
    transformers_list.append(('ord', ordinal_transformer, ordinal_features))
# Pass through all remaining (numerical) features - TabPFN handles NaNs here
if numerical_features:
    transformers_list.append(('num', 'passthrough', numerical_features))

# Define the preprocessor
# It handles specified transformations and drops any columns not explicitly handled
# (which should be none if categorization logic is correct)
preprocessor = ColumnTransformer(
    transformers=transformers_list,
    remainder='drop',
    verbose_feature_names_out=False # Keep feature names simpler
)
# Configure pipeline steps to output pandas DataFrames where possible
preprocessor.set_output(transform="pandas")


# --- Define Model Pipeline ---
logger.info(f"Defining base {MODEL_TYPE} model pipeline...")
# Instantiate the AutoTabPFN regressor (random_state set later per run)
# Pass base parameters, excluding random_state
base_autotabpfn_estimator = AutoTabPFNRegressor(**{k: v for k, v in AUTOTABPFN_PARAMS.items() if k != 'random_state'})

# Combine preprocessor and regressor into a single pipeline
# This is cloned for each averaging run
base_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', base_autotabpfn_estimator)
])

# --- Save Preprocessed Data ---
logger.info("Preprocessing and saving data input to the model...")
try:
    # Fit the preprocessor (defined above) on X_train and transform both X_train and X_test.
    # This ensures the preprocessor is fitted correctly before transforming X_test.
    # The 'preprocessor' object here is the same one that will be used in the pipeline.
    logger.info(f"Fitting preprocessor and transforming X_train (shape: {X_train.shape})...")
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    logger.info(f"Transforming X_test (shape: {X_test.shape})...")
    X_test_processed = preprocessor.transform(X_test)

    # Save the processed dataframes
    X_train_processed_filename = os.path.join(MODELS_DIR, 'X_train_processed.joblib')
    X_test_processed_filename = os.path.join(MODELS_DIR, 'X_test_processed.joblib')

    joblib.dump(X_train_processed, X_train_processed_filename)
    logger.info(f"Saved preprocessed training data to: {X_train_processed_filename} (Shape: {X_train_processed.shape})")
    joblib.dump(X_test_processed, X_test_processed_filename)
    logger.info(f"Saved preprocessed test data to: {X_test_processed_filename} (Shape: {X_test_processed.shape})")

    # Save the column names from the processed data, as they might change (e.g., after one-hot encoding)
    # This is useful if you load the regressor part of the model separately later.
    if hasattr(X_train_processed, 'columns'):
        processed_feature_names = X_train_processed.columns.tolist()
        processed_features_filename = os.path.join(MODELS_DIR, 'processed_feature_names.joblib')
        joblib.dump(processed_feature_names, processed_features_filename)
        logger.info(f"Saved {len(processed_feature_names)} processed feature names to: {processed_features_filename}")
    else:
        logger.warning("Preprocessed X_train is not a DataFrame. Cannot save processed feature names.")

except Exception as e_preprocess_save:
    logger.error(f"ERROR saving preprocessed data: {e_preprocess_save}", exc_info=True)
    # Decide if this is critical. For now, just log and continue.

# ==============================================================================
# Model Training and Prediction
# ==============================================================================

all_test_predictions = [] # List to store predictions from each averaging run

# --- Averaging Loop ---
if PERFORM_AVERAGING:
    logger.info(f"--- Starting Model Training: Averaging {N_AVERAGING_RUNS} {MODEL_TYPE} runs ---")
    trained_models = []

    for i in range(N_AVERAGING_RUNS):
        current_run_seed = BASE_RANDOM_STATE + i
        logger.info(f"--- Averaging Run {i+1}/{N_AVERAGING_RUNS} (Seed: {current_run_seed}) ---")

        # Create a fresh copy of the pipeline for this run
        current_pipeline = clone(base_pipeline)
        # Set the random seed specifically for the AutoTabPFN regressor step
        current_pipeline.set_params(regressor__random_state=current_run_seed)
        logger.info(f"  Cloned pipeline and set regressor random_state to {current_run_seed}")

        # Train the pipeline on the (potentially feature-selected) training data
        logger.info(f"  Training model... (Max time: {AUTO_TABPFN_TIME_BUDGET_SECONDS}s)")
        start_time = datetime.datetime.now()
        try:
            current_pipeline.fit(X_train, y_train) # Fit on final X_train
            end_time = datetime.datetime.now()
            logger.info(f"  Training complete. Time: {end_time - start_time}")
            model_filename = f"{MODELS_DIR}/model_avg_run_{i+1}_seed_{current_run_seed}.joblib"
            try:
                joblib.dump(current_pipeline, model_filename)
                logger.info(f"  Saved trained model for run {i+1} to: {model_filename}")
                trained_models.append(model_filename) # Optional: keep track of saved models
            except Exception as e_save:
                logger.error(f"  ERROR saving model for run {i+1}: {e_save}", exc_info=True)
 
        except Exception as e:
            logger.error(f"  ERROR during training run {i+1} (Seed: {current_run_seed}): {e}", exc_info=True)
            logger.warning(f"  Skipping prediction storage for run {i+1} due to training error.")
            # Clean up failed pipeline before continuing
            del current_pipeline; gc.collect()
            continue # Proceed to the next averaging run

        # Make predictions on the (potentially feature-selected) test data
        logger.info(f"  Making predictions for run {i+1}...")
        try:
            current_predictions = current_pipeline.predict(X_test) # Predict on final X_test
            all_test_predictions.append(current_predictions)
            logger.info("  Predictions stored.")
        except Exception as e:
            logger.error(f"  ERROR during prediction run {i+1} (Seed: {current_run_seed}): {e}", exc_info=True)
            logger.warning(f"  Skipping prediction storage for run {i+1} due to prediction error.")
            # Prediction failed, don't store this run's output

        # Clean up memory after each run
        del current_pipeline
        if 'current_predictions' in locals(): del current_predictions
        gc.collect()


    # Check if any predictions were successfully generated
    if not all_test_predictions:
        logger.error("FATAL: No predictions were successfully generated from any averaging run. Exiting.")
        exit(1)

    # --- Calculate Average Predictions ---
    logger.info(f"Averaging predictions across {len(all_test_predictions)} successful runs...")
    # Stack predictions and calculate the mean across runs (axis=0)
    final_predictions_raw = np.mean(np.stack(all_test_predictions), axis=0)
    logger.info("Averaging complete.")

# --- Single Run (No Averaging) ---
else:
    logger.info(f"--- Starting Model Training: Single {MODEL_TYPE} run ---")

    # Clone the base pipeline for a single run
    final_pipeline_to_use = clone(base_pipeline)
    # Set the random seed for the single run
    final_pipeline_to_use.set_params(regressor__random_state=BASE_RANDOM_STATE)
    logger.info(f"Using single run configuration with random_state={BASE_RANDOM_STATE}")

    # --- Optional Cross-Validation ---
    if PERFORM_CV:
        logger.info(f"--- Performing {CV_FOLDS}-Fold Cross-Validation (Single Run Mode) ---")
        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=BASE_RANDOM_STATE)
        cv_n_jobs = 1 # Typically best for TabPFN to avoid resource conflicts

        # Calculate RMSE CV Score
        logger.info(f"  Calculating CV RMSE...")
        try:
            cv_scores_neg_rmse = cross_val_score(final_pipeline_to_use, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=cv_n_jobs)
            mean_cv_rmse = np.mean(-cv_scores_neg_rmse)
            std_cv_rmse = np.std(-cv_scores_neg_rmse)
            logger.info(f"  {CV_FOLDS}-Fold CV RMSE: {mean_cv_rmse:.4f} +/- {std_cv_rmse:.4f}")
        except Exception as e:
            logger.error(f"  ERROR during CV RMSE calculation: {e}", exc_info=True)

        # Calculate Spearman Rho CV Score
        logger.info(f"  Calculating CV Spearman Rho...")
        try:
            cv_spearman_scores = cross_val_score(final_pipeline_to_use, X_train, y_train, cv=kf, scoring=spearman_scorer, n_jobs=cv_n_jobs)
            mean_cv_spearman = np.mean(cv_spearman_scores)
            std_cv_spearman = np.std(cv_spearman_scores)
            logger.info(f"  {CV_FOLDS}-Fold CV Spearman Rho: {mean_cv_spearman:.4f} +/- {std_cv_spearman:.4f}")
        except Exception as e:
            logger.error(f"  ERROR during CV Spearman calculation: {e}", exc_info=True)
        gc.collect() # Clean up memory after CV

    # --- Train Final Single Model ---
    logger.info(f"Training final single model... (Max time: {AUTO_TABPFN_TIME_BUDGET_SECONDS}s)")
    try:
        start_time = datetime.datetime.now()
        final_pipeline_to_use.fit(X_train, y_train) # Fit on final X_train
        end_time = datetime.datetime.now()
        logger.info(f"  Final model training complete. Time: {end_time - start_time}")

        # Make predictions using the single trained model
        logger.info("Making predictions...")
        final_predictions_raw = final_pipeline_to_use.predict(X_test) # Predict on final X_test
        logger.info("Predictions generated for single run.")

    except Exception as e:
        logger.error(f"ERROR during final model training or prediction: {e}", exc_info=True)
        # Generate dummy predictions if training/prediction fails
        num_test_samples = len(X_test)
        final_predictions_raw = np.full(num_test_samples, 1) # Default prediction (min score)
        logger.error("Generated dummy predictions due to error.")

# ==============================================================================
# Post-Processing and Submission
# ==============================================================================

# --- Post-Process Final Predictions ---
logger.info("Post-processing final predictions...")
try:
    # Define valid score range for the target variable (Benzel Score)
    MIN_SCORE = 1
    MAX_SCORE = 8

    # Check against actual training target range (optional sanity check)
    try:
        actual_min = int(y_train.min())
        actual_max = int(y_train.max())
        logger.info(f"  Actual target range observed in training data: [{actual_min}, {actual_max}]")
        if actual_min < MIN_SCORE or actual_max > MAX_SCORE:
             logger.warning(f"  Actual training range [{actual_min}, {actual_max}] exceeds expected range [{MIN_SCORE}, {MAX_SCORE}].")
    except Exception :
        logger.error("  Could not determine actual min/max from training target.")

    # Clip predictions to the valid range
    final_predictions = final_predictions_raw.copy() # Work on a copy
    logger.info(f"  Clipping predictions to range [{MIN_SCORE}, {MAX_SCORE}].")
    final_predictions = np.clip(final_predictions, MIN_SCORE, MAX_SCORE)

    # Round predictions to the nearest integer for the ordinal target
    logger.info("  Rounding predictions to nearest integer.")
    final_predictions = np.round(final_predictions).astype(int)

    logger.info("Post-processing complete.")

except Exception as e:
    logger.error(f"ERROR during post-processing: {e}", exc_info=True)
    # Fallback to dummy predictions if post-processing fails
    num_test_samples = len(X_test)
    final_predictions = np.full(num_test_samples, MIN_SCORE) # Use min score as default
    logger.error("Using dummy predictions due to post-processing error.")


# --- Generate Submission File ---
logger.info("Generating submission file...")
try:
    # Create DataFrame with predictions
    predictions_df = pd.DataFrame({TARGET_COL: final_predictions})

    # Create base submission DataFrame with PID and time from template
    submission_base_df = pd.DataFrame({'PID': test_PIDs, 'time': time_test})

    # Ensure indices are aligned before concatenating
    submission_base_df.reset_index(drop=True, inplace=True)
    predictions_df.reset_index(drop=True, inplace=True)

    # Combine PID, time, and predictions
    submission_df = pd.concat([submission_base_df, predictions_df], axis=1)

    # Ensure final submission matches template columns exactly
    template_cols = submission_template_df.columns.tolist()
    final_submission_df = pd.DataFrame(columns=template_cols) # Create empty DF with correct columns

    # Populate the final DataFrame, handling potential missing columns gracefully
    for col in template_cols:
        if col in submission_df.columns:
            final_submission_df[col] = submission_df[col]
        else:
            logger.warning(f"Column '{col}' from template not found in generated submission. Filling with NaN.")
            final_submission_df[col] = np.nan

    # Check for extra columns (shouldn't happen with above logic, but as a safeguard)
    extra_cols = set(submission_df.columns) - set(template_cols)
    if extra_cols:
         logger.warning(f"Extra columns found in submission (will be ignored by column selection): {extra_cols}")

    # Final check for NaNs in target column (e.g., if a PID merge failed somewhere)
    if final_submission_df[TARGET_COL].isnull().any():
        nan_count_sub = final_submission_df[TARGET_COL].isnull().sum()
        logger.warning(f"Found {nan_count_sub} NaNs in submission target column '{TARGET_COL}'. Filling with default value {MIN_SCORE}.")
        final_submission_df[TARGET_COL].fillna(MIN_SCORE, inplace=True)
        final_submission_df[TARGET_COL] = final_submission_df[TARGET_COL].astype(int) # Ensure integer type

    # Save the submission file using the exact template columns order
    final_submission_df[template_cols].to_csv(SUBMISSION_OUTPUT_FILE, index=False)
    logger.info(f"Submission file successfully saved to '{SUBMISSION_OUTPUT_FILE}'")

except KeyError as e:
    logger.error(f"ERROR generating submission file: Missing key column {e}. Check template matching logic.")
except Exception as e:
    logger.error(f"ERROR saving submission file: {e}", exc_info=True)

# ==============================================================================
# End of Run
# ==============================================================================
logger.info(f"--- Finished Run: {run_id} ---")
print(f"\nRun details and metrics logged to {log_file}")

# --- Final print statement ---
print(f"\nPipeline {MODEL_TYPE} (ManualFS={ms_str}, AutoFS={fs_str}, Avg={PERFORM_AVERAGING}) finished.")
print("\nNext steps suggestions:")
print(f"1. Review automated feature selection results in log ({log_file}). Adjust thresholds/k if needed.")
print(f"2. Experiment with manual feature selection flags (SELECT_*) in the configuration.")
print("3. Carefully review feature engineering logic (Week 1 and Future Motor) against the data dictionary.")
print(f"4. Analyze performance (CV scores if run, leaderboard submission) and check log for warnings/errors.")
print(f"5. Adjust averaging runs ({N_AVERAGING_RUNS}) or AutoTabPFN time budget ({AUTO_TABPFN_TIME_BUDGET_SECONDS}s).")
print(f"6. Consider trying alternative models (e.g., Gradient Boosting Trees like LightGBM/XGBoost/CatBoost) for comparison.")