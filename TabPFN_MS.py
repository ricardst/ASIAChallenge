import pandas as pd
import numpy as np
# Scikit-learn imports
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
# TabPFN specific import
try:
    from tabpfn import TabPFNRegressor
    # Or use AutoTabPFN if preferred and installed
    # from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
except ImportError:
    print("Please install tabpfn: pip install tabpfn")
    exit()
# Other necessary imports
import warnings
import re
import datetime
import logging
import torch # For checking CUDA
from sklearn import clone # For cloning pipeline in averaging loop

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = './' # <<<--- !!! ADAPT THIS PATH !!!
METADATA_FILE = f'{DATA_DIR}metadata.csv'
TRAIN_FEATURES_FILE = f'{DATA_DIR}train_features.csv'
TRAIN_OUTCOMES_FILE = f'{DATA_DIR}train_outcomes_MS.csv'
TEST_FEATURES_FILE = f'{DATA_DIR}test_features.csv'
SUBMISSION_TEMPLATE_FILE = f'{DATA_DIR}test_outcomes_MS_template.csv'

MIN_SCORE = 0
MAX_SCORE = 5

MODEL_TYPE = 'TabPFN' # Model being used in this script
PERFORM_CV = True # <<<--- Disable CV when averaging for speed
CV_FOLDS = 5 # Number of folds if PERFORM_CV is True

# --- NEW: Averaging Configuration ---
PERFORM_AVERAGING = False # Set to True to enable averaging
N_AVERAGING_RUNS = 5 # Number of models to train and average
BASE_RANDOM_STATE = 42 # Base seed for reproducibility

# --- TabPFN Configuration ---
TABPFN_PARAMS = {
    # 'random_state' will be set in the loop if averaging
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}
# Check for GPU availability
if torch.cuda.is_available():
    print("CUDA (GPU) available. TabPFN will likely use it.")
else:
    print("CUDA (GPU) not available. TabPFN will use CPU (might be slow).")


# --- Setup Logging ---
log_file = 'Metrics.log'
cv_mode_str = f"{CV_FOLDS}FoldCV" if (PERFORM_CV and not PERFORM_AVERAGING) else "NoCV" # CV disabled if averaging
avg_mode_str = f"Avg{N_AVERAGING_RUNS}" if PERFORM_AVERAGING else "SingleRun"
# --- UPDATE base_model_name to reflect feature engineering ---
base_model_name = f"{MODEL_TYPE}_MultiOutput_v6_FE_native_nan"
model_name = f"{base_model_name}_{avg_mode_str}_{cv_mode_str}" # Include averaging status
run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_id = f"{model_name}_{run_timestamp}"
SUBMISSION_OUTPUT_FILE = f'{DATA_DIR}submission_{run_id}.csv' # Set submission filename

logger = logging.getLogger(run_id)
logger.setLevel(logging.INFO)
# Prevent duplicate handlers if the script is run multiple times in the same session
if not logger.handlers:
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

logger.info(f"--- Starting Run: {run_id} ---")
logger.info(f"Selected Model Type: {MODEL_TYPE}")
logger.info(f"Manual Averaging Enabled: {PERFORM_AVERAGING} (Runs: {N_AVERAGING_RUNS if PERFORM_AVERAGING else 1})")
logger.info(f"Cross-Validation Enabled: {PERFORM_CV and not PERFORM_AVERAGING}") # Log effective CV status
logger.info(f"Base TabPFN Parameters: {TABPFN_PARAMS}")
logger.info(f"Data Directory: {DATA_DIR}")
logger.info(f"Output Submission File: {SUBMISSION_OUTPUT_FILE}")


# --- Define Helper Function ---
def get_week1_columns(all_columns):
    """Identifies feature columns corresponding to Week 1 (ending in '01' or '1')."""
    week1_cols = set()
    for col in all_columns:
        if re.search(r'01$', col): week1_cols.add(col)
        elif col in ['ais1']: week1_cols.add(col)
    week1_cols.add('PID')
    return list(dict.fromkeys(week1_cols)) # Ensure PID unique


# --- 1. Load Data ---
print("Loading data...")
try:
    metadata_df = pd.read_csv(METADATA_FILE)
    train_features_df = pd.read_csv(TRAIN_FEATURES_FILE)
    train_outcomes_df = pd.read_csv(TRAIN_OUTCOMES_FILE)
    test_features_df = pd.read_csv(TEST_FEATURES_FILE)
    submission_template_df = pd.read_csv(SUBMISSION_TEMPLATE_FILE)
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    logger.error(f"Error loading data: {e}")
    exit()
metadata_df.replace(9, np.nan, inplace=True)
print("Data loaded successfully.")

# --- 2. Identify Features and Targets ---
print("Identifying features and targets...")
TARGET_COLS = [col for col in train_outcomes_df.columns if col not in ['PID', 'time']]
train_week1_feature_cols = get_week1_columns(train_features_df.columns)
# W1 Features without PID for merging and processing
WEEK1_FEATURE_COLS = [col for col in train_week1_feature_cols if col != 'PID']

METADATA_COLS = [col for col in metadata_df.columns if col != 'PID']
print(f"Found {len(TARGET_COLS)} target columns.")
print(f"Using {len(WEEK1_FEATURE_COLS)} Week 1 feature columns.")
print(f"Using {len(METADATA_COLS)} metadata columns.")

# --- 3. Prepare Training Data (Initial Merge) ---
print("Preparing training data (initial merge)...")
train_features_w1_df = train_features_df[train_week1_feature_cols] # Select only W1 + PID
train_merged_df = pd.merge(metadata_df, train_features_w1_df, on='PID', how='inner')

# --- 3.5 Feature Engineering ---
print("Performing Feature Engineering...")
logger.info("Performing Feature Engineering...")

def engineer_features(df, week1_feature_cols):
    """Applies feature engineering to the dataframe."""
    eng_df = df.copy()
    wk1_cols = [c for c in week1_feature_cols if c in df.columns]

    motor_cols = [c for c in wk1_cols if re.match(r'(?:elbf|wrext|elbex|finfl|finab|hipfl|kneex|ankdo|greto|ankpl)[lr]01$', c)]
    lt_cols = [c for c in wk1_cols if re.search(r'lt[lr]01$', c)]
    pp_cols = [c for c in wk1_cols if re.search(r'pp[lr]01$', c)]
    motor_l_cols = [c for c in motor_cols if c.endswith('l01')]
    motor_r_cols = [c for c in motor_cols if c.endswith('r01')]
    lt_l_cols = [c for c in lt_cols if c.endswith('l01')]
    lt_r_cols = [c for c in lt_cols if c.endswith('r01')]
    pp_l_cols = [c for c in pp_cols if c.endswith('l01')]
    pp_r_cols = [c for c in pp_cols if c.endswith('r01')]
    uems_l_cols = [c for c in motor_l_cols if any(s in c for s in ['elbf','wrext','elbex','finfl','finab'])]
    uems_r_cols = [c for c in motor_r_cols if any(s in c for s in ['elbf','wrext','elbex','finfl','finab'])]
    lems_l_cols = [c for c in motor_l_cols if any(s in c for s in ['hipfl','kneex','ankdo','greto','ankpl'])]
    lems_r_cols = [c for c in motor_r_cols if any(s in c for s in ['hipfl','kneex','ankdo','greto','ankpl'])]

    if motor_cols:
        eng_df['FE_TotalMotor_Wk1'] = eng_df[motor_cols].sum(axis=1, skipna=False)
        eng_df['FE_UEMS_Wk1'] = eng_df[uems_l_cols + uems_r_cols].sum(axis=1, skipna=False)
        eng_df['FE_LEMS_Wk1'] = eng_df[lems_l_cols + lems_r_cols].sum(axis=1, skipna=False)
        eng_df['FE_MotorL_Wk1'] = eng_df[motor_l_cols].sum(axis=1, skipna=False)
        eng_df['FE_MotorR_Wk1'] = eng_df[motor_r_cols].sum(axis=1, skipna=False)
        eng_df['FE_MotorSymmAbsDiff_Wk1'] = (eng_df['FE_MotorL_Wk1'] - eng_df['FE_MotorR_Wk1']).abs()
        eng_df['FE_MotorMean_Wk1'] = eng_df[motor_cols].mean(axis=1, skipna=True)
        eng_df['FE_MotorStd_Wk1'] = eng_df[motor_cols].std(axis=1, skipna=True)
        eng_df['FE_MotorMin_Wk1'] = eng_df[motor_cols].min(axis=1, skipna=True)
        eng_df['FE_MotorMax_Wk1'] = eng_df[motor_cols].max(axis=1, skipna=True)

    if lt_cols:
        eng_df['FE_TotalLTS_Wk1'] = eng_df[lt_cols].sum(axis=1, skipna=False)
        eng_df['FE_LTS_L_Wk1'] = eng_df[lt_l_cols].sum(axis=1, skipna=False)
        eng_df['FE_LTS_R_Wk1'] = eng_df[lt_r_cols].sum(axis=1, skipna=False)
        eng_df['FE_LTSSymmAbsDiff_Wk1'] = (eng_df['FE_LTS_L_Wk1'] - eng_df['FE_LTS_R_Wk1']).abs()
        eng_df['FE_LTSMean_Wk1'] = eng_df[lt_cols].mean(axis=1, skipna=True)
        eng_df['FE_LTSStd_Wk1'] = eng_df[lt_cols].std(axis=1, skipna=True)

    if pp_cols:
        eng_df['FE_TotalPPS_Wk1'] = eng_df[pp_cols].sum(axis=1, skipna=False)
        eng_df['FE_PPS_L_Wk1'] = eng_df[pp_l_cols].sum(axis=1, skipna=False)
        eng_df['FE_PPS_R_Wk1'] = eng_df[pp_r_cols].sum(axis=1, skipna=False)
        eng_df['FE_PPSSymmAbsDiff_Wk1'] = (eng_df['FE_PPS_L_Wk1'] - eng_df['FE_PPS_R_Wk1']).abs()
        eng_df['FE_PPSMean_Wk1'] = eng_df[pp_cols].mean(axis=1, skipna=True)
        eng_df['FE_PPSStd_Wk1'] = eng_df[pp_cols].std(axis=1, skipna=True)

    std_cols = [c for c in eng_df.columns if 'Std_Wk1' in c]
    eng_df[std_cols] = eng_df[std_cols].fillna(0)

    print(f"Engineered features created. New shape: {eng_df.shape}")
    logger.info(f"Engineered features created. New shape: {eng_df.shape}")
    return eng_df

# Apply Feature Engineering to training data features
X_train_merged_fe = engineer_features(train_merged_df, WEEK1_FEATURE_COLS)

# --- Prepare final X_train, y_train ---
# Merge outcomes now
train_full_df = pd.merge(X_train_merged_fe, train_outcomes_df, on='PID', how='inner')

# Define lists of original and engineered feature names present *before* adding target_time
engineered_features = [col for col in X_train_merged_fe.columns if col.startswith('FE_')]
base_features_for_X = METADATA_COLS + WEEK1_FEATURE_COLS + engineered_features
base_features_for_X = sorted(list(set([f for f in base_features_for_X if f in X_train_merged_fe.columns])))

# Separate features (X) and target (y) RAW
X_train_raw_final = train_full_df[base_features_for_X].copy()
y_train_raw = train_full_df[TARGET_COLS].copy()
time_train_raw = train_full_df['time'].copy()

# Drop rows where the TARGET variable itself is NaN
valid_target_indices = y_train_raw.dropna().index
initial_rows = len(X_train_raw_final)
X_train = X_train_raw_final.loc[valid_target_indices].copy()
y_train = y_train_raw.loc[valid_target_indices].copy()
time_train = time_train_raw.loc[valid_target_indices].copy()
final_rows = len(X_train)

if initial_rows != final_rows: print(f"Dropped {initial_rows - final_rows} rows with missing values in TARGET columns.")
else: print("No rows dropped due to missing targets.")
logger.info(f"Dropped {initial_rows - final_rows} rows with missing targets" if initial_rows != final_rows else "No rows dropped due to missing targets")

# Add target time as a feature - X_train is now complete
X_train['target_time'] = time_train

# --- DEFINE FEATURES LIST BASED ON FINAL X_train ---
FEATURES = X_train.columns.tolist() # Use actual columns of the final X_train

logger.info(f"Final training data shapes (post FE, target NaNs dropped): X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Final training data shapes input to pipeline: X_train={X_train.shape}, y_train={y_train.shape}")
logger.info(f"Final FEATURES list includes {len(FEATURES)} columns.")

# --- 4. Prepare Test Data (Apply FE) ---
print("Preparing test data...")
# Merge metadata and Wk1 test features
test_features_df_w1 = test_features_df[['PID'] + WEEK1_FEATURE_COLS]
test_merged_df = pd.merge(metadata_df, test_features_df_w1, on='PID', how='inner')

# Apply Feature Engineering to test data features
X_test_merged_fe = engineer_features(test_merged_df, WEEK1_FEATURE_COLS)

# Align test data PIDs and target times with the submission template
submission_template_info = submission_template_df[['PID', 'time']]
test_full_df = pd.merge(submission_template_info, X_test_merged_fe, on='PID', how='left')

test_PIDs = test_full_df['PID']
time_test = test_full_df['time']

# Select features (original + engineered) - use base features first
# Re-calculate engineered features list based on actual columns created
engineered_features_test = [col for col in X_test_merged_fe.columns if col.startswith('FE_')]
base_features_for_X_test = METADATA_COLS + WEEK1_FEATURE_COLS + engineered_features_test
base_features_for_X_test = sorted(list(set([f for f in base_features_for_X_test if f in X_test_merged_fe.columns])))
X_test = test_full_df[base_features_for_X_test].copy()

# Add target time
X_test['target_time'] = time_test

# --- Ensure columns are in the same order and are the same set as X_train using final FEATURES list ---
missing_cols_test = set(FEATURES) - set(X_test.columns)
extra_cols_test = set(X_test.columns) - set(FEATURES)

if missing_cols_test:
    print(f"WARNING: Columns missing in X_test that are in FEATURES: {missing_cols_test}. Filling with NaN.")
    logger.warning(f"Columns missing in X_test: {missing_cols_test}. Filling with NaN.")
    for col in missing_cols_test:
         X_test[col] = np.nan # Add missing columns and fill with NaN

if extra_cols_test:
    print(f"WARNING: Columns extra in X_test not in FEATURES: {extra_cols_test}. Dropping.")
    logger.warning(f"Columns extra in X_test not in FEATURES: {extra_cols_test}. Dropping.")
    X_test = X_test.drop(columns=list(extra_cols_test))

# Ensure final feature set and order match FEATURES (derived from X_train)
X_test = X_test[FEATURES]

logger.info(f"Test data shape (post FE, aligned): X_test={X_test.shape}")
print(f"Test data shape (post FE, aligned): X_test={X_test.shape}")


# --- 5. Preprocessing Pipeline ---
print("Setting up preprocessing pipeline (Encoders + Explicit Passthrough, No Scaler)...")
logger.info("Setting up preprocessing pipeline (Encoders + Explicit Passthrough, No Scaler)...")

# Define feature types explicitly based on FINAL X_train columns (FEATURES list)
meta_categorical = ['age_category', 'bmi_category', 'tx1_r', 'sexcd']
w1_ordinal = ['ais1']
ais_categories = ['A', 'B', 'C', 'D', 'E']

# Use the final FEATURES list to determine categories
categorical_features = [f for f in meta_categorical if f in FEATURES]
ordinal_features = [f for f in w1_ordinal if f in FEATURES]

# Identify numerical passthrough features explicitly using FEATURES
processed_cols = set(categorical_features + ordinal_features)
numerical_passthrough_features = sorted([col for col in FEATURES if col not in processed_cols]) # All remaining columns in FEATURES

# Verification: Ensure all columns in X_train are covered by the lists
check_xtrain_cols = set(X_train.columns)
check_assigned_cols = set(categorical_features + ordinal_features + numerical_passthrough_features)
if check_xtrain_cols != check_assigned_cols:
     unassigned_in_check = check_xtrain_cols - check_assigned_cols
     extra_in_check = check_assigned_cols - check_xtrain_cols
     print(f"ERROR: Column assignment mismatch!")
     if unassigned_in_check: print(f"  Columns in X_train not assigned: {unassigned_in_check}")
     if extra_in_check: print(f"  Columns assigned but not in X_train: {extra_in_check}")
     logger.error(f"Column assignment mismatch! Unassigned: {unassigned_in_check}, Extra: {extra_in_check}")
     exit()
else:
    print("Column assignment verification successful.")
    logger.info("Column assignment verification successful.")


print(f"Encoding Categorical Features ({len(categorical_features)}): {categorical_features}")
print(f"Encoding Ordinal Features ({len(ordinal_features)}): {ordinal_features}")
print(f"Passing through Numerical Features ({len(numerical_passthrough_features)}): {numerical_passthrough_features[:5]}...")
logger.info(f"Encoding Categorical Features ({len(categorical_features)}): {categorical_features}")
logger.info(f"Encoding Ordinal Features ({len(ordinal_features)}): {ordinal_features}")
logger.info(f"Passing through Numerical Features ({len(numerical_passthrough_features)})")


# Define transformers (WITH IMPUTATION)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[ais_categories], handle_unknown='use_encoded_value', unknown_value=-1)),])

# Define preprocessor with EXPLICIT passthrough
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('ord', ordinal_transformer, ordinal_features),
        ('num_pass', 'passthrough', numerical_passthrough_features) # Use updated list
        ],
    remainder='drop' # Drop any columns not explicitly handled
    )

# --- 6. Define Model ---
print(f"Defining {MODEL_TYPE} model...")
# Using base TabPFN Regressor here, adjust if using AutoTabPFN
# random_state will be set in the loop
base_tabpfn_estimator = TabPFNRegressor(**{k: v for k, v in TABPFN_PARAMS.items() if k != 'random_state'})
model_wrapper = MultiOutputRegressor(base_tabpfn_estimator, n_jobs=1)

# Base pipeline structure used in loop
base_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', model_wrapper)])


# --- 7. Train Model (Averaging Loop or Single Run) ---

all_test_predictions = [] # Store predictions from each run

if PERFORM_AVERAGING:
    # ... (Averaging loop code remains the same) ...
    pass # No changes needed inside the averaging block for this request

else: # Single Run (No Averaging)
    print("--- Mode: Training single model ---")
    logger.info("Mode: Training single model")

    # Base pipeline already includes the estimator with default params (except seed)
    final_pipeline_to_use = clone(base_pipeline)
    final_pipeline_to_use.set_params(regressor__estimator__random_state=BASE_RANDOM_STATE)

    # Initialize CV score variable
    mean_cv_rmse = None
    std_cv_rmse = None

    # Optionally perform CV first if enabled
    if PERFORM_CV:
        print(f"Performing {CV_FOLDS}-fold cross-validation...")
        logger.info(f"Performing {CV_FOLDS}-fold cross-validation...")
        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=BASE_RANDOM_STATE)
        cv_n_jobs = 1
        try:
            cv_scores_neg_rmse = cross_val_score(
                final_pipeline_to_use, X_train, y_train, cv=kf,
                scoring='neg_root_mean_squared_error', n_jobs=cv_n_jobs
            )
            cv_rmse_scores = -cv_scores_neg_rmse
            mean_cv_rmse = np.mean(cv_rmse_scores) # Store the mean CV score
            std_cv_rmse = np.std(cv_rmse_scores)
            print(f"Cross-validation RMSE: {mean_cv_rmse:.4f} +/- {std_cv_rmse:.4f}")
            logger.info(f"{CV_FOLDS}-Fold CV RMSE: {mean_cv_rmse:.4f} +/- {std_cv_rmse:.4f}")
        except Exception as e:
            print(f"ERROR during cross-validation: {e}")
            logger.error(f"ERROR during cross-validation: {e}", exc_info=True)
            # Continue to final training, mean_cv_rmse remains None

    # Train the final model
    print("Training final model...")
    logger.info("Training final model...")
    try:
        start_time = datetime.datetime.now()
        # Fit the final pipeline instance (already created above)
        final_pipeline_to_use.fit(X_train, y_train)
        end_time = datetime.datetime.now()
        training_time = end_time - start_time
        print(f"Final model training complete. Time: {training_time}")
        logger.info(f"Final model training complete. Time: {training_time}")

        # --- NEW: Calculate Training Score and Compare (only if CV was done) ---
        if mean_cv_rmse is not None: # Check if CV was successfully performed
            print("Calculating score on the full training set...")
            logger.info("Calculating score on the full training set...")
            try:
                y_train_pred_raw = final_pipeline_to_use.predict(X_train)
                # Apply post-processing to training predictions for fair comparison
                y_train_pred = y_train_pred_raw.copy()
                y_train_pred[y_train_pred < MIN_SCORE] = MIN_SCORE
                y_train_pred[y_train_pred > MAX_SCORE] = MAX_SCORE
                y_train_pred = np.round(y_train_pred)

                training_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                print(f"Training RMSE: {training_rmse:.4f}")
                logger.info(f"Training RMSE: {training_rmse:.4f}")

                # Compare!
                overfitting_gap = mean_cv_rmse - training_rmse # CV - Train (Positive means CV error higher)
                print(f"Gap (CV RMSE - Train RMSE): {overfitting_gap:+.4f}")
                logger.info(f"Gap (CV RMSE - Train RMSE): {overfitting_gap:+.4f}")
                if overfitting_gap < -0.1: # Heuristic threshold for significant overfitting warning
                     print("WARNING: Training RMSE significantly lower than CV RMSE, potential overfitting.")
                     logger.warning("Potential Overfitting Detected (Train RMSE << CV RMSE)")

            except Exception as e:
                print(f"Error calculating training score: {e}")
                logger.error(f"Error calculating training score: {e}", exc_info=True)
        # --- END NEW BLOCK ---

        # Make predictions on test set using the final fitted pipeline
        print("Making predictions...")
        final_predictions_raw = final_pipeline_to_use.predict(X_test)
        print("Predictions generated.")
        logger.info("Predictions generated for single run.")

    except Exception as e:
         print(f"ERROR during final model training/prediction: {e}")
         logger.error(f"ERROR during final model training/prediction: {e}", exc_info=True)
         # Generate dummy predictions if training/prediction failed
         num_test_samples = len(X_test)
         num_targets = len(TARGET_COLS)
         final_predictions_raw = np.full((num_test_samples, num_targets), MIN_SCORE)
         print("Generated dummy predictions due to error.")
         logger.error("Generated dummy predictions due to training/prediction error.")


# --- 8. Post-Process Final Predictions ---
print("Post-processing final predictions...")
try:
    # Apply clamping and rounding to the final (potentially averaged) predictions
    final_predictions = final_predictions_raw.copy() # Work on a copy
    final_predictions[final_predictions < MIN_SCORE] = MIN_SCORE
    final_predictions[final_predictions > MAX_SCORE] = MAX_SCORE
    final_predictions = np.round(final_predictions)
    print("Clamping and rounding complete.")
    logger.info("Final predictions clamped and rounded.")
except Exception as e:
    print(f"ERROR during post-processing: {e}")
    logger.error(f"ERROR during post-processing: {e}", exc_info=True)
    # Use dummy values if post-processing failed
    num_test_samples = len(X_test)
    num_targets = len(TARGET_COLS)
    final_predictions = np.full((num_test_samples, num_targets), MIN_SCORE)
    print("Using dummy predictions due to post-processing error.")


# --- 9. Generate Submission File ---
print("Generating submission file...")
predictions_df = pd.DataFrame(final_predictions, columns=TARGET_COLS)
submission_df = pd.DataFrame({'PID': test_PIDs, 'time': time_test})
submission_df.reset_index(drop=True, inplace=True)
predictions_df.reset_index(drop=True, inplace=True)
submission_df = pd.concat([submission_df, predictions_df], axis=1)
template_cols = submission_template_df.columns.tolist()
missing_sub_cols = set(template_cols) - set(submission_df.columns)
if missing_sub_cols: print(f"Warning: Columns missing from submission file: {missing_sub_cols}")
extra_sub_cols = set(submission_df.columns) - set(template_cols)
if extra_sub_cols:
    print(f"Warning: Extra columns found in submission file: {extra_sub_cols}. Dropping them.")
    submission_df = submission_df.drop(columns=list(extra_sub_cols))
try:
    submission_df = submission_df[template_cols]
    submission_df.to_csv(SUBMISSION_OUTPUT_FILE, index=False)
    print(f"Submission file saved to '{SUBMISSION_OUTPUT_FILE}'")
    logger.info(f"Submission file saved to '{SUBMISSION_OUTPUT_FILE}'")
except KeyError as e:
    print(f"ERROR generating submission file: Missing column {e}")
    print("Template columns:", template_cols)
    print("Submission columns:", submission_df.columns.tolist())
    logger.error(f"ERROR generating submission file: Missing column {e}")
except Exception as e:
     print(f"ERROR saving submission file: {e}")
     logger.error(f"ERROR saving submission file: {e}", exc_info=True)


# --- Log Run End ---
logger.info(f"--- Finished Run: {run_id} ---")
print(f"\nRun details and metrics logged to {log_file}")

# --- Final print statement ---
print(f"\nPipeline {MODEL_TYPE} (MultiOutput, v6 FE + Native NaN Preproc, Averaging={PERFORM_AVERAGING}) finished.")
print("Next steps:")
print("1. Analyze performance (CV score if run, leaderboard submission).")
print("2. Adjust N_AVERAGING_RUNS or TabPFN parameters if needed.")
print("3. Compare with other models (GBDTs).")
print("4. Consider more advanced feature engineering or AutoTabPFN.")