import os
import re
import gc
import joblib
import logging
import datetime
import json # For SETTINGS.json
import pandas as pd
import numpy as np
import pandas.api.types as ptypes # For is_numeric_dtype

# Scikit-learn imports
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression

# ==============================================================================
# Settings & Configuration
# ==============================================================================
def load_settings():
    """Loads settings from SETTINGS.json"""
    try:
        with open('SETTINGS.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("CRITICAL ERROR: SETTINGS.json not found. Please create it.")
        exit(1)
    except json.JSONDecodeError:
        print("CRITICAL ERROR: SETTINGS.json is not valid JSON.")
        exit(1)

SETTINGS = load_settings()
RAW_DATA_DIR = SETTINGS.get('RAW_DATA_DIR', './Input_Files/')
CLEAN_DATA_DIR = SETTINGS.get('CLEAN_DATA_DIR', './Clean_Data/')
LOG_DIR = SETTINGS.get('LOG_DIR', './Log_Files/')

# --- File Paths from RAW_DATA_DIR ---
METADATA_FILE = os.path.join(RAW_DATA_DIR, 'metadata.csv')
TRAIN_FEATURES_FILE = os.path.join(RAW_DATA_DIR, 'train_features.csv')
TRAIN_OUTCOMES_FILE = os.path.join(RAW_DATA_DIR, 'train_outcomes_functional.csv')
TEST_FEATURES_FILE = os.path.join(RAW_DATA_DIR, 'test_features.csv')
SUBMISSION_TEMPLATE_FILE = os.path.join(RAW_DATA_DIR, 'test_outcomes_Fun_template_update.csv')
EXTERNAL_PREDS_FILE = os.path.join(RAW_DATA_DIR, 'submission_MS_test_outcomes.csv') # For future motor scores on test
TRAIN_OUTCOMES_MOTOR_FILE = os.path.join(RAW_DATA_DIR, 'train_outcomes_MS.csv') # For future motor scores on train


# --- Configuration from TabPFN_Fun.py (can be moved to SETTINGS.json or kept here) ---
USE_FUTURE_MOTOR_FEATURES = True # Set to True to load and use features derived from future motor scores
SELECT_METADATA = True
SELECT_WEEK1_ORIGINAL = True
SELECT_FE_FEATURES = True
SELECT_TARGET_TIME = True
SELECT_FUTURE_MOTOR_FEATURES = True
DO_FEATURE_SELECTION = True
VAR_THRESH = 0.01
CORR_THRESH = 0.93
UNIVARIATE_K = 75

# ==============================================================================
# Setup Logging
# ==============================================================================
run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_id = f"prepare_data_{run_timestamp}"
log_filename = f"{run_id}.log"
log_file = os.path.join(LOG_DIR, log_filename)

try:
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CLEAN_DATA_DIR, exist_ok=True)
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

logger.info(f"--- Starting Data Preparation: {run_id} ---")
logger.info(f"Raw Data Directory: {RAW_DATA_DIR}")
logger.info(f"Clean Data Directory: {CLEAN_DATA_DIR}")
logger.info(f"Log File: {log_file}")

# ==============================================================================
# Helper Functions (Copied from TabPFN_Fun.py)
# ==============================================================================
def select_features(X_train, y_train, X_test, features,
                    var_thresh=0.01,
                    corr_thresh=0.95,
                    k='all'):
    logger.info(f"Starting automated feature selection on {len(features)} candidate features...")
    start_time_fs = datetime.datetime.now()

    numeric_feats = [f for f in features if ptypes.is_numeric_dtype(X_train[f])]
    non_numeric_feats = [f for f in features if f not in numeric_feats]
    logger.info(f"Identified {len(numeric_feats)} numeric and {len(non_numeric_feats)} non-numeric features.")

    if not numeric_feats:
        logger.warning("No numeric features found for selection. Returning original features.")
        final_selected_features = non_numeric_feats
        return X_train[final_selected_features].copy() if final_selected_features else X_train.copy(), \
               X_test[final_selected_features].copy() if final_selected_features else X_test.copy(), \
               final_selected_features

    X_train_num = X_train[numeric_feats].copy()
    # X_test_num  = X_test[numeric_feats].copy() # Not used directly in selection logic here

    logger.debug("Imputing numeric features (median) temporarily for selection calculations...")
    imp = SimpleImputer(strategy='median')
    X_train_num_imp = pd.DataFrame(imp.fit_transform(X_train_num), columns=numeric_feats, index=X_train.index)

    selected_numeric_feats = numeric_feats

    logger.info(f"Applying Variance Threshold (threshold={var_thresh})...")
    vt = VarianceThreshold(threshold=var_thresh)
    vt.fit(X_train_num_imp)
    variance_mask = vt.get_support()
    selected_numeric_feats = [f for f, keep in zip(selected_numeric_feats, variance_mask) if keep]
    X_train_num_imp = X_train_num_imp[selected_numeric_feats]
    logger.info(f"Variance Threshold kept {len(selected_numeric_feats)} numeric features.")

    if not selected_numeric_feats:
        logger.warning("Variance Threshold removed all numeric features. Returning only non-numeric.")
        final_selected_features = non_numeric_feats
        return X_train[final_selected_features].copy() if final_selected_features else X_train.copy(), \
               X_test[final_selected_features].copy() if final_selected_features else X_test.copy(), \
               final_selected_features

    logger.info(f"Applying Correlation Threshold (threshold={corr_thresh})...")
    corr_matrix = X_train_num_imp.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_thresh)]
    selected_numeric_feats = [f for f in selected_numeric_feats if f not in to_drop]
    X_train_num_imp = X_train_num_imp[selected_numeric_feats]
    logger.info(f"Correlation Threshold kept {len(selected_numeric_feats)} numeric features.")

    if not selected_numeric_feats:
        logger.warning("Correlation Threshold removed all numeric features. Returning only non-numeric.")
        final_selected_features = non_numeric_feats
        return X_train[final_selected_features].copy() if final_selected_features else X_train.copy(), \
               X_test[final_selected_features].copy() if final_selected_features else X_test.copy(), \
               final_selected_features

    if k != 'all' and isinstance(k, int) and k > 0 and len(selected_numeric_feats) > k:
        k_final = min(k, len(selected_numeric_feats))
        logger.info(f"Applying Univariate Selection (f_regression, k={k_final})...")
        skb = SelectKBest(score_func=f_regression, k=k_final)
        skb.fit(X_train_num_imp, y_train)
        univariate_mask = skb.get_support()
        selected_numeric_feats = [f for f, keep in zip(selected_numeric_feats, univariate_mask) if keep]
        logger.info(f"Univariate Selection kept {len(selected_numeric_feats)} numeric features.")
    else:
        logger.info(f"Skipping Univariate Selection (k='{k}', available features={len(selected_numeric_feats)}).")

    final_selected_features = sorted(selected_numeric_feats + non_numeric_feats)
    end_time_fs = datetime.datetime.now()
    logger.info(f"Automated Feature Selection complete. Selected {len(final_selected_features)} total features. Time: {end_time_fs - start_time_fs}")

    X_train_selected = X_train[final_selected_features].copy()
    X_test_selected = X_test[final_selected_features].copy()
    return X_train_selected, X_test_selected, final_selected_features

def engineer_features(df, week1_feature_cols):
    logger.debug(f"Starting Week 1 FE on df shape {df.shape}")
    eng_df = df.copy()
    wk1_cols = [c for c in week1_feature_cols if c in df.columns]

    motor_cols = [c for c in wk1_cols if re.match(r'(?:elbf|wrext|elbex|finfl|finab|hipfl|kneex|ankdo|greto|ankpl)[lr]01$', c)]
    lt_cols = [c for c in wk1_cols if re.search(r'[cts]\d+lt[lr]01$', c) or re.search(r's45lt[lr]01$', c)]
    pp_cols = [c for c in wk1_cols if re.search(r'[cts]\d+pp[lr]01$', c) or re.search(r's45pp[lr]01$', c)]

    motor_l_cols = [c for c in motor_cols if c.endswith('l01')]
    motor_r_cols = [c for c in motor_cols if c.endswith('r01')]
    lt_l_cols = [c for c in lt_cols if c.endswith('l01')]
    lt_r_cols = [c for c in lt_cols if c.endswith('r01')]
    pp_l_cols = [c for c in pp_cols if c.endswith('l01')]
    pp_r_cols = [c for c in pp_cols if c.endswith('r01')]

    uems_muscle_codes = ['elbf','wrext','elbex','finfl','finab']
    lems_muscle_codes = ['hipfl','kneex','ankdo','greto','ankpl']
    uems_l_cols = [c for c in motor_l_cols if any(s in c for s in uems_muscle_codes)]
    uems_r_cols = [c for c in motor_r_cols if any(s in c for s in uems_muscle_codes)]
    lems_l_cols = [c for c in motor_l_cols if any(s in c for s in lems_muscle_codes)]
    lems_r_cols = [c for c in motor_r_cols if any(s in c for s in lems_muscle_codes)]

    if motor_cols:
        eng_df['FE_TotalMotor_Wk1'] = eng_df[motor_cols].sum(axis=1, skipna=False)
        if uems_l_cols or uems_r_cols: eng_df['FE_UEMS_Wk1'] = eng_df[uems_l_cols + uems_r_cols].sum(axis=1, skipna=False)
        if lems_l_cols or lems_r_cols: eng_df['FE_LEMS_Wk1'] = eng_df[lems_l_cols + lems_r_cols].sum(axis=1, skipna=False)
        if motor_l_cols: eng_df['FE_MotorL_Wk1'] = eng_df[motor_l_cols].sum(axis=1, skipna=False)
        if motor_r_cols: eng_df['FE_MotorR_Wk1'] = eng_df[motor_r_cols].sum(axis=1, skipna=False)
        if 'FE_MotorL_Wk1' in eng_df.columns and 'FE_MotorR_Wk1' in eng_df.columns:
             eng_df['FE_MotorSymmAbsDiff_Wk1'] = (eng_df['FE_MotorL_Wk1'] - eng_df['FE_MotorR_Wk1']).abs()
        eng_df['FE_MotorMean_Wk1'] = eng_df[motor_cols].mean(axis=1, skipna=True)
        eng_df['FE_MotorStd_Wk1'] = eng_df[motor_cols].std(axis=1, skipna=True)
        eng_df['FE_MotorMin_Wk1'] = eng_df[motor_cols].min(axis=1, skipna=True)
        eng_df['FE_MotorMax_Wk1'] = eng_df[motor_cols].max(axis=1, skipna=True)
    if lt_cols:
        eng_df['FE_TotalLTS_Wk1'] = eng_df[lt_cols].sum(axis=1, skipna=False)
        if lt_l_cols: eng_df['FE_LTS_L_Wk1'] = eng_df[lt_l_cols].sum(axis=1, skipna=False)
        if lt_r_cols: eng_df['FE_LTS_R_Wk1'] = eng_df[lt_r_cols].sum(axis=1, skipna=False)
        if 'FE_LTS_L_Wk1' in eng_df.columns and 'FE_LTS_R_Wk1' in eng_df.columns:
            eng_df['FE_LTS_SymmAbsDiff_Wk1'] = (eng_df['FE_LTS_L_Wk1'] - eng_df['FE_LTS_R_Wk1']).abs()
        eng_df['FE_LTSMean_Wk1'] = eng_df[lt_cols].mean(axis=1, skipna=True)
        eng_df['FE_LTSStd_Wk1'] = eng_df[lt_cols].std(axis=1, skipna=True)
    if pp_cols:
        eng_df['FE_TotalPPS_Wk1'] = eng_df[pp_cols].sum(axis=1, skipna=False)
        if pp_l_cols: eng_df['FE_PPS_L_Wk1'] = eng_df[pp_l_cols].sum(axis=1, skipna=False)
        if pp_r_cols: eng_df['FE_PPS_R_Wk1'] = eng_df[pp_r_cols].sum(axis=1, skipna=False)
        if 'FE_PPS_L_Wk1' in eng_df.columns and 'FE_PPS_R_Wk1' in eng_df.columns:
            eng_df['FE_PPS_SymmAbsDiff_Wk1'] = (eng_df['FE_PPS_L_Wk1'] - eng_df['FE_PPS_R_Wk1']).abs()
        eng_df['FE_PPSMean_Wk1'] = eng_df[pp_cols].mean(axis=1, skipna=True)
        eng_df['FE_PPSStd_Wk1'] = eng_df[pp_cols].std(axis=1, skipna=True)

    std_cols = [c for c in eng_df.columns if 'Std_Wk1' in c]
    eng_df[std_cols] = eng_df[std_cols].fillna(0)
    logger.info(f"Shape after Week 1 FE: {eng_df.shape}")
    return eng_df

def engineer_future_motor_features(df, motor_score_cols):
    SUFFIX = '_FutureMotor'
    FEATURE_PREFIX = 'FM_'
    if df is None or df.empty or not motor_score_cols:
        logger.warning("Input df is None/empty or no motor_score_cols provided to engineer_future_motor_features.")
        return pd.DataFrame({'PID': []})
    if 'PID' not in df.columns:
        logger.error("PID column missing in dataframe passed to engineer_future_motor_features.")
        return pd.DataFrame({'PID': []})

    relevant_cols = ['PID'] + [col for col in motor_score_cols if col in df.columns]
    eng_df = df[relevant_cols].copy()
    motor_cols_present = [c for c in relevant_cols if c != 'PID']
    if not motor_cols_present:
        logger.warning("No motor score columns found in provided dataframe for future motor FE.")
        return eng_df[['PID']]

    motor_l_cols = [c for c in motor_cols_present if c.endswith('l')]
    motor_r_cols = [c for c in motor_cols_present if c.endswith('r')]
    uems_muscle_codes_base = ['elbf','wrext','elbex','finfl','finab']
    lems_muscle_codes_base = ['hipfl','kneex','ankdo','greto','ankpl']
    uems_l_cols = [f"{code}l" for code in uems_muscle_codes_base if f"{code}l" in motor_l_cols]
    uems_r_cols = [f"{code}r" for code in uems_muscle_codes_base if f"{code}r" in motor_r_cols]
    lems_l_cols = [f"{code}l" for code in lems_muscle_codes_base if f"{code}l" in motor_l_cols]
    lems_r_cols = [f"{code}r" for code in lems_muscle_codes_base if f"{code}r" in motor_r_cols]

    eng_df[f'{FEATURE_PREFIX}TotalMotor{SUFFIX}'] = eng_df[motor_cols_present].sum(axis=1, skipna=False)
    if uems_l_cols or uems_r_cols: eng_df[f'{FEATURE_PREFIX}UEMS{SUFFIX}'] = eng_df[uems_l_cols + uems_r_cols].sum(axis=1, skipna=False)
    if lems_l_cols or lems_r_cols: eng_df[f'{FEATURE_PREFIX}LEMS{SUFFIX}'] = eng_df[lems_l_cols + lems_r_cols].sum(axis=1, skipna=False)
    if motor_l_cols: eng_df[f'{FEATURE_PREFIX}MotorL{SUFFIX}'] = eng_df[motor_l_cols].sum(axis=1, skipna=False)
    if motor_r_cols: eng_df[f'{FEATURE_PREFIX}MotorR{SUFFIX}'] = eng_df[motor_r_cols].sum(axis=1, skipna=False)
    if f'{FEATURE_PREFIX}MotorL{SUFFIX}' in eng_df.columns and f'{FEATURE_PREFIX}MotorR{SUFFIX}' in eng_df.columns:
        eng_df[f'{FEATURE_PREFIX}MotorSymmAbsDiff{SUFFIX}'] = (eng_df[f'{FEATURE_PREFIX}MotorL{SUFFIX}'] - eng_df[f'{FEATURE_PREFIX}MotorR{SUFFIX}']).abs()

    engineered_cols = [col for col in eng_df.columns if col.startswith(FEATURE_PREFIX)]
    logger.info(f"Generated {len(engineered_cols)} features from future motor scores (suffix: {SUFFIX}).")
    return eng_df[['PID'] + engineered_cols]

# ==============================================================================
# Main Data Preparation Logic
# ==============================================================================
def main():
    logger.info("Loading base data files...")
    try:
        metadata_df = pd.read_csv(METADATA_FILE)
        train_features_df = pd.read_csv(TRAIN_FEATURES_FILE)
        train_outcomes_df = pd.read_csv(TRAIN_OUTCOMES_FILE)
        test_features_df = pd.read_csv(TEST_FEATURES_FILE)
        submission_template_df = pd.read_csv(SUBMISSION_TEMPLATE_FILE)

        logger.warning("Replacing value 9 with NaN globally. Verify appropriateness via data dictionary.")
        metadata_df.replace(9, np.nan, inplace=True)
        train_features_df.replace(9, np.nan, inplace=True)
        test_features_df.replace(9, np.nan, inplace=True)
    except FileNotFoundError as e:
        logger.error(f"Error loading base data file: {e}. Check file paths in SETTINGS.json and script.")
        exit(1)
    logger.info("Base data loaded successfully.")

    ext_preds_test_df = None
    ext_actuals_train_df = None
    motor_score_cols_ext = []
    if USE_FUTURE_MOTOR_FEATURES:
        logger.info("Loading external future motor score data...")
        try:
            ext_preds_test_df = pd.read_csv(EXTERNAL_PREDS_FILE)
            logger.info(f"  Loaded test set predictions from: {EXTERNAL_PREDS_FILE} (Shape: {ext_preds_test_df.shape})")
            ext_actuals_train_df = pd.read_csv(TRAIN_OUTCOMES_MOTOR_FILE)
            logger.info(f"  Loaded train set actuals/OOF from: {TRAIN_OUTCOMES_MOTOR_FILE} (Shape: {ext_actuals_train_df.shape})")
            motor_score_cols_ext = ['elbfll','wrextl','elbexl','finfll','finabl','hipfll','kneexl','ankdol','gretol','ankpll',
                                   'elbflr','wrextr','elbexr','finflr','finabr','hipflr','kneetr','ankdor','gretor','ankplr']
            missing_test_cols = [col for col in motor_score_cols_ext if col not in ext_preds_test_df.columns]
            missing_train_cols = [col for col in motor_score_cols_ext if col not in ext_actuals_train_df.columns]
            if missing_test_cols: logger.warning(f"Missing future motor score columns in test predictions file: {missing_test_cols}")
            if missing_train_cols: logger.warning(f"Missing future motor score columns in train actuals file: {missing_train_cols}")
        except FileNotFoundError as e:
            logger.error(f"Error loading external motor score data file: {e}. Cannot proceed if USE_FUTURE_MOTOR_FEATURES is True.")
            exit(1)
    else:
        logger.info("Skipping loading of external future motor score data.")

    TARGET_COL = 'modben'
    if TARGET_COL not in train_outcomes_df.columns:
        logger.error(f"Target column '{TARGET_COL}' not found in {TRAIN_OUTCOMES_FILE}. Exiting.")
        exit(1)

    all_metadata_cols = [col for col in metadata_df.columns if col != 'PID']
    all_train_week1_cols = [col for col in train_features_df.columns if re.search(r'01$', col) or col == 'ais1']
    all_test_week1_cols = [col for col in test_features_df.columns if col != 'PID'] # Assumes test_features only has Wk1
    common_week1_cols = sorted(list(set(all_train_week1_cols).intersection(all_test_week1_cols)))
    logger.info(f"Identified {len(common_week1_cols)} common Week 1 ISNCSCI columns.")

    train_features_w1_df = train_features_df[['PID'] + common_week1_cols].copy()
    train_merged_df = pd.merge(metadata_df, train_features_w1_df, on='PID', how='inner')
    X_train_merged_fe_wk1 = engineer_features(train_merged_df, common_week1_cols)
    engineered_features_wk1 = sorted([col for col in X_train_merged_fe_wk1.columns if col.startswith('FE_') and col.endswith('_Wk1')])
    del train_merged_df, train_features_w1_df; gc.collect()

    engineered_features_future = []
    X_train_merged_fe_all = X_train_merged_fe_wk1
    if USE_FUTURE_MOTOR_FEATURES and ext_actuals_train_df is not None:
        train_future_motor_features_df = engineer_future_motor_features(ext_actuals_train_df, motor_score_cols_ext)
        engineered_features_future = sorted([col for col in train_future_motor_features_df.columns if col.startswith('FM_')])
        X_train_merged_fe_all = pd.merge(X_train_merged_fe_all, train_future_motor_features_df, on='PID', how='left')
        nan_check = X_train_merged_fe_all[engineered_features_future].isnull().sum().sum()
        if nan_check > 0: logger.warning(f"Found {nan_check} NaNs in future motor features after merging to train data.")
        del train_future_motor_features_df; gc.collect()

    initial_features_selected = []
    if SELECT_METADATA: initial_features_selected.extend(all_metadata_cols)
    if SELECT_WEEK1_ORIGINAL: initial_features_selected.extend(common_week1_cols)
    if SELECT_FE_FEATURES: initial_features_selected.extend(engineered_features_wk1)
    if USE_FUTURE_MOTOR_FEATURES and SELECT_FUTURE_MOTOR_FEATURES: initial_features_selected.extend(engineered_features_future)
    
    available_cols_in_merged_df = X_train_merged_fe_all.columns.tolist()
    selected_features_manual = sorted(list(set([f for f in initial_features_selected if f in available_cols_in_merged_df])))
    logger.info(f"Manually selected {len(selected_features_manual)} base features.")

    train_full_df = pd.merge(X_train_merged_fe_all, train_outcomes_df[['PID', TARGET_COL, 'time']], on='PID', how='inner')
    X_train_pre_fs = train_full_df[selected_features_manual].copy()
    y_train_raw = train_full_df[TARGET_COL].copy()
    time_train_raw = train_full_df['time'].copy()
    
    valid_target_indices = y_train_raw.dropna().index
    X_train_pre_fs = X_train_pre_fs.loc[valid_target_indices].reset_index(drop=True)
    y_train = y_train_raw.loc[valid_target_indices].reset_index(drop=True)
    time_train = time_train_raw.loc[valid_target_indices].reset_index(drop=True)
    
    FEATURES_BEFORE_AUTOFS = selected_features_manual.copy()
    if SELECT_TARGET_TIME:
        X_train_pre_fs['target_time'] = time_train
        FEATURES_BEFORE_AUTOFS.append('target_time')
    del X_train_merged_fe_all, train_full_df, y_train_raw, time_train_raw; gc.collect()

    logger.info("Preparing test data...")
    test_features_df_w1 = test_features_df[['PID'] + common_week1_cols].copy()
    test_merged_df = pd.merge(metadata_df, test_features_df_w1, on='PID', how='inner')
    X_test_merged_fe_wk1 = engineer_features(test_merged_df, common_week1_cols)
    
    X_test_merged_fe_all = X_test_merged_fe_wk1
    if USE_FUTURE_MOTOR_FEATURES and ext_preds_test_df is not None:
        test_future_motor_features_df = engineer_future_motor_features(ext_preds_test_df, motor_score_cols_ext)
        X_test_merged_fe_all = pd.merge(X_test_merged_fe_all, test_future_motor_features_df, on='PID', how='left')
        # Ensure engineered_features_future columns exist before checking NaNs
        fm_cols_in_test = [col for col in engineered_features_future if col in X_test_merged_fe_all.columns]
        if fm_cols_in_test:
            nan_check = X_test_merged_fe_all[fm_cols_in_test].isnull().sum().sum()
            if nan_check > 0: logger.warning(f"Found {nan_check} NaNs in future motor features after merging to test data.")
        del test_future_motor_features_df; gc.collect()

    submission_template_info = submission_template_df[['PID', 'time']].copy()
    test_full_df = pd.merge(submission_template_info, X_test_merged_fe_all, on='PID', how='left')
    test_PIDs = test_full_df['PID'].copy() # Save PIDs for submission
    time_test = test_full_df['time'].copy()
    
    X_test_pre_fs = test_full_df[selected_features_manual].copy()
    if SELECT_TARGET_TIME:
        X_test_pre_fs['target_time'] = time_test

    missing_cols_test = set(FEATURES_BEFORE_AUTOFS) - set(X_test_pre_fs.columns)
    for col in missing_cols_test: X_test_pre_fs[col] = np.nan
    extra_cols_test = set(X_test_pre_fs.columns) - set(FEATURES_BEFORE_AUTOFS)
    X_test_pre_fs = X_test_pre_fs.drop(columns=list(extra_cols_test), errors='ignore')
    X_test_pre_fs = X_test_pre_fs[FEATURES_BEFORE_AUTOFS]
    del X_test_merged_fe_all, test_full_df, X_test_merged_fe_wk1; gc.collect()

    if DO_FEATURE_SELECTION:
        X_train, X_test, FINAL_FEATURES_USED = select_features(
            X_train_pre_fs, y_train, X_test_pre_fs, FEATURES_BEFORE_AUTOFS,
            var_thresh=VAR_THRESH, corr_thresh=CORR_THRESH, k=UNIVARIATE_K
        )
    else:
        X_train = X_train_pre_fs.copy()
        X_test = X_test_pre_fs.copy()
        FINAL_FEATURES_USED = FEATURES_BEFORE_AUTOFS
    del X_train_pre_fs, X_test_pre_fs; gc.collect()
    
    logger.info(f"Final features selected: {FINAL_FEATURES_USED}")
    joblib.dump(FINAL_FEATURES_USED, os.path.join(CLEAN_DATA_DIR, 'final_features_used.joblib'))

    meta_categorical_base = ['age_category', 'bmi_category', 'tx1_r', 'sexcd']
    w1_ordinal_base = ['ais1']
    ais_categories = ['A', 'B', 'C', 'D', 'E']
    categorical_features = sorted([f for f in meta_categorical_base if f in FINAL_FEATURES_USED])
    ordinal_features = sorted([f for f in w1_ordinal_base if f in FINAL_FEATURES_USED])
    processed_cols = set(categorical_features + ordinal_features)
    numerical_features = sorted([col for col in FINAL_FEATURES_USED if col not in processed_cols])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    ordinal_transformer_steps = []
    if ordinal_features: # Only define if there are ordinal features
        ordinal_transformer_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        # Ensure categories are correctly structured for multiple ordinal features
        ordinal_cats = [ais_categories] * len(ordinal_features) if 'ais1' in ordinal_features else [] # Example, adjust if more ordinals
        if ordinal_cats: # only add encoder if categories are defined
             ordinal_transformer_steps.append(('ordinal', OrdinalEncoder(categories=ordinal_cats, handle_unknown='use_encoded_value', unknown_value=-1)))
    
    ordinal_transformer = Pipeline(steps=ordinal_transformer_steps) if ordinal_transformer_steps else 'drop'

    transformers_list = []
    if categorical_features: transformers_list.append(('cat', cat_transformer, categorical_features))
    if ordinal_features and ordinal_transformer != 'drop': transformers_list.append(('ord', ordinal_transformer, ordinal_features))
    if numerical_features: transformers_list.append(('num', 'passthrough', numerical_features))

    preprocessor = ColumnTransformer(transformers=transformers_list, remainder='drop', verbose_feature_names_out=False)
    preprocessor.set_output(transform="pandas")

    logger.info(f"Fitting preprocessor and transforming X_train (shape: {X_train.shape})...")
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    logger.info(f"Transforming X_test (shape: {X_test.shape})...")
    X_test_processed = preprocessor.transform(X_test)

    joblib.dump(preprocessor, os.path.join(CLEAN_DATA_DIR, 'preprocessor.joblib'))
    joblib.dump(X_train_processed, os.path.join(CLEAN_DATA_DIR, 'X_train_processed.joblib'))
    joblib.dump(y_train, os.path.join(CLEAN_DATA_DIR, 'y_train.joblib'))
    joblib.dump(X_test_processed, os.path.join(CLEAN_DATA_DIR, 'X_test_processed.joblib'))
    joblib.dump(test_PIDs, os.path.join(CLEAN_DATA_DIR, 'test_PIDs.joblib'))
    
    if hasattr(X_train_processed, 'columns'):
        processed_feature_names = X_train_processed.columns.tolist()
        joblib.dump(processed_feature_names, os.path.join(CLEAN_DATA_DIR, 'processed_feature_names.joblib'))
        logger.info(f"Saved {len(processed_feature_names)} processed feature names.")

    logger.info("Data preparation complete. Processed files saved to Clean_Data directory.")

if __name__ == '__main__':
    main()
