import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
import warnings
import re # For finding week 1 columns
import datetime
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = './' # <<< --- !!! ADAPT THIS PATH to where your CSV files are located !!!
METADATA_FILE = f'{DATA_DIR}metadata.csv'
TRAIN_FEATURES_FILE = f'{DATA_DIR}train_features.csv'
TRAIN_OUTCOMES_FILE = f'{DATA_DIR}train_outcomes_MS.csv' # Motor score outcomes file
TEST_FEATURES_FILE = f'{DATA_DIR}test_features.csv'
SUBMISSION_TEMPLATE_FILE = f'{DATA_DIR}test_outcomes_MS_template.csv'
SUBMISSION_OUTPUT_FILE = f'{DATA_DIR}submission_baseline_v2_dropna.csv'

# --- NEW: Control Hyperparameter Search ---
PERFORM_HYPERPARAMETER_SEARCH = False # Set to False to use manual parameters

# --- NEW: Define Manual Parameters (if not tuning) ---
# These are example parameters, adjust them based on previous runs or defaults
MANUAL_LGBM_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'subsample_freq': 1,
    'random_state': 42,
    'verbose': -1,
    # Add device_type='gpu' here if using GPU and manual params
}

# --- Setup Logging ---
log_file = 'Metrics.log'
# --- ADJUST model_name based on mode ---
tuning_mode_str = "HPO_Search" if PERFORM_HYPERPARAMETER_SEARCH else "Manual_Params"
base_model_name = "LGBM_MultiOutput_v5_lgbm_native_nan" # Base name for this script version
model_name = f"{base_model_name}_{tuning_mode_str}" # Append mode
run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_id = f"{model_name}_{run_timestamp}"

# Create logger
logger = logging.getLogger(run_id)
logger.setLevel(logging.INFO)

# Prevent duplicate handlers if the script is run multiple times in the same session
if not logger.handlers:
    # Create file handler which logs even info messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(fh)

# --- Log Run Start ---
logger.info(f"--- Starting Run: {run_id} ---")
logger.info(f"Data Directory: {DATA_DIR}")
logger.info(f"Output Submission File: {SUBMISSION_OUTPUT_FILE}")

# --- Define Helper Function to Get Week 1 Columns ---
# (No changes needed in this helper function)
def get_week1_columns(all_columns):
    """Identifies feature columns corresponding to Week 1 (ending in '01' or '1')."""
    week1_cols = set()
    for col in all_columns:
        if re.search(r'01$', col):
             week1_cols.add(col)
        elif col in ['ais1']:
             week1_cols.add(col)
    week1_cols.add('PID')
    return list(week1_cols)

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
    print(f"Please ensure the following files are in the directory '{DATA_DIR}':")
    print(f"{METADATA_FILE}, {TRAIN_FEATURES_FILE}, {TRAIN_OUTCOMES_FILE}, {TEST_FEATURES_FILE}, {SUBMISSION_TEMPLATE_FILE}")
    exit()

# Replace potential "unknown" markers (like 9 for past history) with NaN
# Check data dictionary for all such markers - assuming 9 for now based on description
metadata_df.replace(9, np.nan, inplace=True)
# Add any other specific replacements if needed based on how missing data is coded

print("Data loaded successfully.")

# --- 2. Identify Features and Targets ---
print("Identifying features and targets...")
TARGET_COLS = [col for col in train_outcomes_df.columns if col not in ['PID', 'time']]
print(f"Found {len(TARGET_COLS)} target columns.")

train_week1_feature_cols = [col for col in train_features_df.columns if re.search(r'01$', col) or col == 'ais1']
test_feature_cols = [col for col in test_features_df.columns if col not in ['PID']]
WEEK1_FEATURE_COLS = test_feature_cols
print(f"Using {len(WEEK1_FEATURE_COLS)} Week 1 feature columns.")

METADATA_COLS = [col for col in metadata_df.columns if col != 'PID']
print(f"Using {len(METADATA_COLS)} metadata columns.")

# --- 3. Prepare Training Data ---
print("Preparing training data...")
train_features_w1 = train_features_df[['PID'] + WEEK1_FEATURE_COLS]
train_merged_df = pd.merge(metadata_df, train_features_w1, on='PID', how='inner')
train_full_df = pd.merge(train_merged_df, train_outcomes_df, on='PID', how='inner')

# Define features before separating X and y
FEATURES = METADATA_COLS + WEEK1_FEATURE_COLS + ['target_time'] # Add target_time here

# Separate features (X) and target (y) RAW
X_train_raw = train_full_df[METADATA_COLS + WEEK1_FEATURE_COLS].copy()
y_train_raw = train_full_df[TARGET_COLS].copy()
time_train_raw = train_full_df['time'].copy()

# --- Drop rows where the TARGET variable itself is NaN ---
valid_target_indices = y_train_raw.dropna().index
initial_rows = len(X_train_raw)

# --- Apply valid target indices to X, y, and time ---
X_train = X_train_raw.loc[valid_target_indices].copy() # Keep original NaNs in features
y_train = y_train_raw.loc[valid_target_indices].copy()
time_train = time_train_raw.loc[valid_target_indices].copy()
final_rows = len(X_train)

if initial_rows != final_rows:
    print(f"Dropped {initial_rows - final_rows} rows with missing values in TARGET columns.")
    logger.info(f"Dropped {initial_rows - final_rows} rows with missing values in TARGET columns.")
else:
    print("No rows dropped due to missing targets.")
    logger.info("No rows dropped due to missing targets.")

# Add target time as a feature
X_train['target_time'] = time_train

# --- Log data shape *before* any feature imputation/transformation ---
logger.info(f"Training data shape before preprocessing (target NaNs dropped): X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Training data shapes input to pipeline: X_train={X_train.shape}, y_train={y_train.shape}")

# --- 4. Prepare Test Data ---
print("Preparing test data...")
test_merged_df = pd.merge(metadata_df, test_features_df, on='PID', how='inner')
submission_template_info = submission_template_df[['PID', 'time']]
# Use left merge to keep all test PIDs from template
test_full_df = pd.merge(submission_template_info, test_merged_df, on='PID', how='left')

test_PIDs = test_full_df['PID']
time_test = test_full_df['time']

X_test = test_full_df[METADATA_COLS + WEEK1_FEATURE_COLS].copy()
X_test['target_time'] = time_test

# Ensure columns are in the same order as X_train
X_test = X_test[FEATURES] # FEATURES now includes 'target_time'

# --- ADD Logging for Data Shape ---
logger.info(f"Training data shape after NaN drop: X_train={X_train.shape}, y_train={y_train.shape}")
logger.info(f"Test data shape: X_test={X_test.shape}")
print(f"Test data shape: X_test={X_test.shape}")

if X_test.isnull().any().any():
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Warning: Missing values detected in the prepared test features.")
    print("The current pipeline does NOT include imputation (as requested).")
    print("The '.predict()' step might fail if these NaNs are not handled.")
    print("Consider adding imputation back into the pipeline later.")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


# --- 5. Preprocessing Pipeline ---
# --- Update Print statement ---
print("Setting up preprocessing pipeline (Encoders + Selective Impute, No Scaler)...")
logger.info("Setting up preprocessing pipeline (Encoders + Selective Impute, No Scaler)...")

# --- Define feature types explicitly ---
# Metadata features
meta_categorical = ['age_category', 'bmi_category', 'tx1_r', 'sexcd']
# Ordinal features from Week 1
w1_ordinal = ['ais1']
ais_categories = ['A', 'B', 'C', 'D', 'E']

# --- Identify columns for each transformer ---
categorical_features = [f for f in meta_categorical if f in X_train.columns]
ordinal_features = [f for f in w1_ordinal if f in X_train.columns]

# --- ALL OTHER features will be passed through (numerical + binary-like metadata) ---
processed_cols = set(categorical_features + ordinal_features)
numerical_passthrough_features = sorted([col for col in X_train.columns if col not in processed_cols])

print(f"Encoding Categorical Features ({len(categorical_features)}): {categorical_features}")
print(f"Encoding Ordinal Features ({len(ordinal_features)}): {ordinal_features}")
print(f"Passing through Numerical Features ({len(numerical_passthrough_features)}): {numerical_passthrough_features[:5]}...")
logger.info(f"Encoding Categorical Features ({len(categorical_features)}): {categorical_features}")
logger.info(f"Encoding Ordinal Features ({len(ordinal_features)}): {ordinal_features}")
logger.info(f"Passing through Numerical Features ({len(numerical_passthrough_features)})")


# --- Define transformers: Impute ONLY immediately before encoding ---
# Imputer is needed because encoders cannot handle NaN inputs.
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Impute before OHE
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Using dense for LGBM compatibility if needed, though it handles sparse too. Set back to True if memory is issue.
])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Impute before Ordinal
    ('ordinal', OrdinalEncoder(categories=[ais_categories], handle_unknown='use_encoded_value', unknown_value=-1)),
])


# --- Define preprocessor ---
# Apply specific transformers only to categorical/ordinal columns.
# Pass all other columns (numerical, binary) through without scaling or imputation.
preprocessor = ColumnTransformer(
    transformers=[
        # ('num', numerical_transformer, numerical_features), # REMOVED - using remainder='passthrough'
        ('cat', categorical_transformer, categorical_features),
        ('ord', ordinal_transformer, ordinal_features)
        ],
    remainder='passthrough' # <<<--- IMPORTANT: Pass numerical features directly to LightGBM
    )


# --- 6. Define Model ---
print("Defining model...")
lgbm = lgb.LGBMRegressor(random_state=42, device_type='gpu', verbose=-1)
model = MultiOutputRegressor(lgbm)

full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', model)])

# --- Section 7: Train Model (Conditional Tuning) ---

# Define the basic pipeline structure first
# (Preprocessor definition should be finalized before this point - Section 5)

# Note: Define lgbm instance here only if NOT tuning, otherwise defined inside search
#       Let's define the base pipeline structure without specific params first

base_lgbm = lgb.LGBMRegressor(random_state=42, verbose=-1) # Base instance for structure
base_model = MultiOutputRegressor(base_lgbm)
base_pipeline = Pipeline(steps=[('preprocessor', preprocessor), # Use preprocessor from Section 5
                                ('regressor', base_model)])


if PERFORM_HYPERPARAMETER_SEARCH:
    print("--- Mode: Performing Hyperparameter Search ---")
    logger.info("Mode: Performing Hyperparameter Search")

    # --- RandomizedSearchCV Setup ---
    # Define the parameter distribution to sample from
    param_distributions = {
        'regressor__estimator__n_estimators': randint(100, 800),
        'regressor__estimator__learning_rate': uniform(0.01, 0.2),
        'regressor__estimator__num_leaves': randint(20, 60),
        'regressor__estimator__max_depth': randint(3, 12),
        'regressor__estimator__reg_alpha': uniform(0.0, 1.0),
        'regressor__estimator__reg_lambda': uniform(0.0, 1.0),
        'regressor__estimator__colsample_bytree': uniform(0.6, 0.4),
        'regressor__estimator__subsample': uniform(0.6, 0.4),
        'regressor__estimator__subsample_freq': randint(0, 5)
    }

    n_iterations = 50 # <<<--- ADJUST AS NEEDED
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        estimator=base_pipeline, # Use the base pipeline structure
        param_distributions=param_distributions,
        n_iter=n_iterations,
        cv=kf,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1, # Use -1 for all cores, adjust if needed
        random_state=42,
        verbose=1,
        refit=True # Default, refits the best estimator on the whole data
    )

    print("Starting hyperparameter search...")
    logger.info(f"Starting RandomizedSearchCV with n_iter={n_iterations}")
    random_search.fit(X_train, y_train)

    print("\nHyperparameter search complete.")
    logger.info("Hyperparameter search complete.")

    best_params_loggable = {k: (int(v) if isinstance(v, np.integer) else
                                float(v) if isinstance(v, np.floating) else
                                v)
                           for k, v in random_search.best_params_.items()}
    best_rmse = -random_search.best_score_

    print(f"Best parameters found: {best_params_loggable}")
    print(f"Best cross-validation RMSE: {best_rmse:.4f}")
    logger.info(f"Best CV RMSE: {best_rmse:.4f}")
    logger.info(f"Best parameters: {best_params_loggable}")

    # The final pipeline IS the best estimator found by the search
    final_pipeline_to_use = random_search.best_estimator_


else: # Use Manual Parameters
    print("--- Mode: Using Manual Hyperparameters ---")
    logger.info("Mode: Using Manual Hyperparameters")
    logger.info(f"Manual LGBM Params: {MANUAL_LGBM_PARAMS}")

    # Create pipeline with specific manual parameters
    manual_lgbm = lgb.LGBMRegressor(**MANUAL_LGBM_PARAMS)
    manual_model = MultiOutputRegressor(manual_lgbm)
    manual_pipeline = Pipeline(steps=[('preprocessor', preprocessor), # Use preprocessor from Section 5
                                      ('regressor', manual_model)])

    # Optionally, perform CV on the manual pipeline to estimate performance
    print("Performing cross-validation on manual parameters...")
    logger.info("Performing cross-validation on manual parameters...")
    from sklearn.model_selection import cross_val_score
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_neg_rmse = cross_val_score(
        manual_pipeline,
        X_train, y_train,
        cv=kf,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1 # Use -1 for all cores, adjust if needed
    )
    cv_rmse_scores = -cv_scores_neg_rmse
    mean_cv_rmse = np.mean(cv_rmse_scores)
    std_cv_rmse = np.std(cv_rmse_scores)
    print(f"Cross-validation RMSE with manual parameters: {mean_cv_rmse:.4f} +/- {std_cv_rmse:.4f}")
    logger.info(f"CV RMSE with manual parameters: {mean_cv_rmse:.4f} +/- {std_cv_rmse:.4f}")

    # Train the final model on all training data
    print("Training final model with manual parameters...")
    logger.info("Training final model with manual parameters...")
    manual_pipeline.fit(X_train, y_train)
    print("Final model training complete.")
    logger.info("Final model training complete.")

    # The final pipeline IS the manually configured one
    final_pipeline_to_use = manual_pipeline

# --- 8. Make Predictions ---
# --- Use the selected pipeline (either from HPO search or manual fit) ---
print("Making predictions on test data using the final pipeline...")
try:
    predictions = final_pipeline_to_use.predict(X_test) # <<<--- Use common variable

    # Post-process predictions (clamping/rounding)
    MIN_SCORE = 0
    MAX_SCORE = 5 # ISNCSCI motor score range
    predictions[predictions < MIN_SCORE] = MIN_SCORE
    predictions[predictions > MAX_SCORE] = MAX_SCORE
    predictions = np.round(predictions) # Round predictions

    print("Predictions generated.")

except Exception as e:
    print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"ERROR during prediction: {e}")
    print(f"This likely occurred because the test set contains NaN values,")
    print(f"and imputation was removed from the preprocessing pipeline.")
    print(f"You will need to add imputation back to handle the test set.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    # Create dummy predictions array matching expected shape if prediction fails
    # So that submission file generation doesn't crash immediately
    num_test_samples = len(X_test)
    num_targets = len(TARGET_COLS)
    predictions = np.full((num_test_samples, num_targets), MIN_SCORE) # Fill with min score
    print("Generated dummy predictions (all zeros) due to error.")


# --- 9. Generate Submission File ---
print("Generating submission file...")
predictions_df = pd.DataFrame(predictions, columns=TARGET_COLS)

submission_df = pd.DataFrame({
    'PID': test_PIDs,
    'time': time_test
})

# Ensure index alignment before concatenation if test_PIDs index isn't standard range [0, n-1]
submission_df.reset_index(drop=True, inplace=True)
predictions_df.reset_index(drop=True, inplace=True)

submission_df = pd.concat([submission_df, predictions_df], axis=1)

template_cols = submission_template_df.columns.tolist()
# Check if all expected columns are present
missing_sub_cols = set(template_cols) - set(submission_df.columns)
if missing_sub_cols:
    print(f"Warning: Columns missing from submission file: {missing_sub_cols}")
extra_sub_cols = set(submission_df.columns) - set(template_cols)
if extra_sub_cols:
    print(f"Warning: Extra columns found in submission file: {extra_sub_cols}. Dropping them.")
    submission_df = submission_df.drop(columns=list(extra_sub_cols))

# Ensure column order matches the template
submission_df = submission_df[template_cols]

submission_df.to_csv(SUBMISSION_OUTPUT_FILE, index=False)
print(f"Submission file saved to '{SUBMISSION_OUTPUT_FILE}'")

# --- Log Run End ---
logger.info(f"--- Finished Run: {run_id} ---")
print(f"\nRun details and metrics logged to {log_file}")

# --- UPDATE final print statement and next steps ---
print("\nPipeline V5 (LGBM Native NaN, Ordinal/Cat Encode) finished.")
print("Next steps:")
print("1. Analyze the CV results from RandomizedSearchCV.")
print("2. Compare performance to the previous version with full imputation/scaling.")
print("3. Experiment with LightGBM's direct categorical feature handling (requires removing encoders and passing feature names/indices to LGBM).")
print("4. Try other models or further tune hyperparameters.")
print("5. Engineer more features.")