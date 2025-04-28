import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb # <<<--- ADD IMPORT
import catboost as cb # <<<--- ADD IMPORT
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score # Added cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
import warnings
import re
import datetime # Keep only one import
import logging
from scipy.stats import randint, uniform

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = './' # <<<--- !!! ADAPT THIS PATH !!!
METADATA_FILE = f'{DATA_DIR}metadata.csv'
TRAIN_FEATURES_FILE = f'{DATA_DIR}train_features.csv'
TRAIN_OUTCOMES_FILE = f'{DATA_DIR}train_outcomes_MS.csv'
TEST_FEATURES_FILE = f'{DATA_DIR}test_features.csv'
SUBMISSION_TEMPLATE_FILE = f'{DATA_DIR}test_outcomes_MS_template.csv'
# Output file name will be set later based on model choice

# --- User Choices ---
MODEL_TYPE = 'CatBoost' # Choose 'LightGBM', 'XGBoost', or 'CatBoost'
PERFORM_HYPERPARAMETER_SEARCH = True # Set to False to use manual parameters below

# --- Manual Parameters (Used only if PERFORM_HYPERPARAMETER_SEARCH = False) ---
# Adjust these based on defaults or previous tuning runs for the chosen model
MANUAL_LGBM_PARAMS = {
    'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': -1,
    'reg_alpha': 0.1, 'reg_lambda': 0.1, 'colsample_bytree': 0.8, 'subsample': 0.8,
    'subsample_freq': 1, 'random_state': 42, 'verbose': 1,
    # 'device_type': 'gpu', # Optional: Add if using GPU
}
MANUAL_XGB_PARAMS = {
    'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'random_state': 42, 'n_jobs': -1, # n_jobs for XGBoost itself
    # 'tree_method': 'hist', 'enable_categorical': True # Consider if using native cat features
    # 'tree_method': 'gpu_hist', # Optional: If using GPU
}
MANUAL_CATBOOST_PARAMS = {
    'iterations': 300, 'learning_rate': 0.05, 'depth': 6, 'l2_leaf_reg': 3,
    'loss_function': 'MultiRMSE', # CatBoost has MultiRMSE loss
    'random_state': 42, 'verbose': 1, # Suppress CatBoost verbosity during manual fit/CV
    # 'task_type': 'GPU', # Optional: If using GPU
    # 'cat_features': [...] # Optional: List of categorical feature indices/names if using native handling
}

# --- Setup Logging ---
log_file = 'Metrics.log'
tuning_mode_str = "HPO_Search" if PERFORM_HYPERPARAMETER_SEARCH else "Manual_Params"
base_model_name = f"{MODEL_TYPE}_MultiOutput_v5_native_nan" # Base name reflecting model and preprocessing
model_name = f"{base_model_name}_{tuning_mode_str}"
run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Use datetime.datetime
run_id = f"{model_name}_{run_timestamp}"

# Set submission file name based on run config
SUBMISSION_OUTPUT_FILE = f'{DATA_DIR}submission_{run_id}.csv'

logger = logging.getLogger(run_id)
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

logger.info(f"--- Starting Run: {run_id} ---")
logger.info(f"Selected Model Type: {MODEL_TYPE}")
logger.info(f"Hyperparameter Search Enabled: {PERFORM_HYPERPARAMETER_SEARCH}")
logger.info(f"Data Directory: {DATA_DIR}")
logger.info(f"Output Submission File: {SUBMISSION_OUTPUT_FILE}")

# --- Define Helper Function ---
def get_week1_columns(all_columns):
    week1_cols = set()
    for col in all_columns:
        if re.search(r'01$', col): week1_cols.add(col)
        elif col in ['ais1']: week1_cols.add(col)
    week1_cols.add('PID')
    return list(week1_cols)

# --- 1. Load Data ---
print("Loading data...")
# ... (Data loading code remains the same) ...
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
# ... (Code remains the same) ...
TARGET_COLS = [col for col in train_outcomes_df.columns if col not in ['PID', 'time']]
train_week1_feature_cols = [col for col in train_features_df.columns if re.search(r'01$', col) or col == 'ais1']
test_feature_cols = [col for col in test_features_df.columns if col not in ['PID']]
WEEK1_FEATURE_COLS = test_feature_cols
METADATA_COLS = [col for col in metadata_df.columns if col != 'PID']
print(f"Found {len(TARGET_COLS)} target columns.")
print(f"Using {len(WEEK1_FEATURE_COLS)} Week 1 feature columns.")
print(f"Using {len(METADATA_COLS)} metadata columns.")

# --- 3. Prepare Training Data ---
print("Preparing training data...")
# ... (Code remains the same - only drop rows with target NaNs) ...
train_features_w1 = train_features_df[['PID'] + WEEK1_FEATURE_COLS]
train_merged_df = pd.merge(metadata_df, train_features_w1, on='PID', how='inner')
train_full_df = pd.merge(train_merged_df, train_outcomes_df, on='PID', how='inner')
FEATURES = METADATA_COLS + WEEK1_FEATURE_COLS + ['target_time']
X_train_raw = train_full_df[METADATA_COLS + WEEK1_FEATURE_COLS].copy()
y_train_raw = train_full_df[TARGET_COLS].copy()
time_train_raw = train_full_df['time'].copy()
valid_target_indices = y_train_raw.dropna().index
initial_rows = len(X_train_raw)
X_train = X_train_raw.loc[valid_target_indices].copy()
y_train = y_train_raw.loc[valid_target_indices].copy()
time_train = time_train_raw.loc[valid_target_indices].copy()
final_rows = len(X_train)
if initial_rows != final_rows: print(f"Dropped {initial_rows - final_rows} rows with missing values in TARGET columns.")
else: print("No rows dropped due to missing targets.")
X_train['target_time'] = time_train
logger.info(f"Training data shape before preprocessing (target NaNs dropped): X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Training data shapes input to pipeline: X_train={X_train.shape}, y_train={y_train.shape}")

# --- 4. Prepare Test Data ---
print("Preparing test data...")
# ... (Code remains the same) ...
test_merged_df = pd.merge(metadata_df, test_features_df, on='PID', how='inner')
submission_template_info = submission_template_df[['PID', 'time']]
test_full_df = pd.merge(submission_template_info, test_merged_df, on='PID', how='left')
test_PIDs = test_full_df['PID']
time_test = test_full_df['time']
X_test = test_full_df[METADATA_COLS + WEEK1_FEATURE_COLS].copy()
X_test['target_time'] = time_test
X_test = X_test[FEATURES]
logger.info(f"Test data shape: X_test={X_test.shape}")
print(f"Test data shape: X_test={X_test.shape}")

# --- 5. Preprocessing Pipeline ---
print("Setting up preprocessing pipeline (Encoders + Selective Impute, No Scaler)...")
logger.info("Setting up preprocessing pipeline (Encoders + Selective Impute, No Scaler)...")
# ... (Feature type identification and transformer definitions remain the same) ...
meta_categorical = ['age_category', 'bmi_category', 'tx1_r', 'sexcd']
w1_ordinal = ['ais1']
ais_categories = ['A', 'B', 'C', 'D', 'E']
categorical_features = [f for f in meta_categorical if f in X_train.columns]
ordinal_features = [f for f in w1_ordinal if f in X_train.columns]
processed_cols = set(categorical_features + ordinal_features)
numerical_passthrough_features = sorted([col for col in X_train.columns if col not in processed_cols])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[ais_categories], handle_unknown='use_encoded_value', unknown_value=-1)),])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('ord', ordinal_transformer, ordinal_features)],
    remainder='passthrough') # Numerical features passed through

print(f"Encoding Categorical Features ({len(categorical_features)}): {categorical_features}")
print(f"Encoding Ordinal Features ({len(ordinal_features)}): {ordinal_features}")
print(f"Passing through Numerical Features ({len(numerical_passthrough_features)}): {numerical_passthrough_features[:5]}...")

# --- 6. Define Model Specific Components ---
# Define base estimator, manual parameters, and HPO distributions based on MODEL_TYPE

if MODEL_TYPE == 'LightGBM':
    base_estimator = lgb.LGBMRegressor(random_state=42, verbose=1)
    manual_params_dict = MANUAL_LGBM_PARAMS
    param_distributions = { # Prefixed for RandomizedSearchCV
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
elif MODEL_TYPE == 'XGBoost':
    base_estimator = xgb.XGBRegressor(random_state=42, n_jobs=1) # Use n_jobs=1 within RandomizedSearchCV if n_jobs=-1 is used there
    manual_params_dict = MANUAL_XGB_PARAMS
    param_distributions = { # Prefixed for RandomizedSearchCV
        'regressor__estimator__n_estimators': randint(100, 800),
        'regressor__estimator__learning_rate': uniform(0.01, 0.2),
        'regressor__estimator__max_depth': randint(3, 10), # XGBoost depth often smaller
        'regressor__estimator__subsample': uniform(0.6, 0.4),
        'regressor__estimator__colsample_bytree': uniform(0.6, 0.4),
        'regressor__estimator__gamma': uniform(0, 0.5), # Min loss reduction for split
        'regressor__estimator__reg_alpha': uniform(0.0, 1.0), # L1
        'regressor__estimator__reg_lambda': uniform(0.0, 1.0) # L2
    }
elif MODEL_TYPE == 'CatBoost':
    # Note: Using MultiRMSE assumes MultiOutputRegressor wrapper handles it.
    # Alternatively, CatBoostRegressor can handle multi-output directly if loss='MultiRMSE'.
    # Let's stick to the wrapper approach for consistency with others first.
    # Need to handle potential verbose conflicts with RandomizedSearchCV
    base_estimator = cb.CatBoostRegressor(random_state=42, verbose=1, loss_function='RMSE')
    manual_params_dict = MANUAL_CATBOOST_PARAMS
    param_distributions = { # Prefixed for RandomizedSearchCV
        'regressor__estimator__iterations': randint(100, 800),
        'regressor__estimator__learning_rate': uniform(0.01, 0.2),
        'regressor__estimator__depth': randint(2, 6),
        'regressor__estimator__l2_leaf_reg': uniform(1, 100), # L2 regularization
        'regressor__estimator__subsample': uniform(0.3, 0.7), # Similar to subsample
        # 'regressor__estimator__colsample_bylevel': uniform(0.6, 0.4), # CatBoost equivalent to colsample_bytree
        # 'regressor__estimator__border_count': randint(32, 255) # Controls splits for numerical features
    }
    # Note on CatBoost Preprocessing: For optimal performance, consider removing OHE/OrdinalEncoder
    # and passing original categorical feature names/indices via `cat_features` parameter
    # directly to CatBoostRegressor. This requires adjusting the `preprocessor`.
else:
    raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")


# --- 7. Train Model (Conditional Tuning) ---

# Define base pipeline structure using the chosen base_estimator
base_model_wrapper = MultiOutputRegressor(base_estimator)
base_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', base_model_wrapper)])

if PERFORM_HYPERPARAMETER_SEARCH:
    print(f"--- Mode: Performing Hyperparameter Search for {MODEL_TYPE} ---")
    logger.info(f"Mode: Performing Hyperparameter Search for {MODEL_TYPE}")

    n_iterations = 10 # <<<--- ADJUST AS NEEDED (Lower for faster testing)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        estimator=base_pipeline, # Use the pipeline with the correct base estimator
        param_distributions=param_distributions, # Use the distributions for the chosen model
        n_iter=n_iterations,
        cv=kf,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1, # Use all cores for CV folds; ensure base estimator uses 1 core if needed (e.g. XGBoost)
        random_state=42,
        verbose=1,
        refit=True
    )

    print("Starting hyperparameter search...")
    logger.info(f"Starting RandomizedSearchCV with n_iter={n_iterations}")
    random_search.fit(X_train, y_train) # Fit the search object

    print("\nHyperparameter search complete.")
    logger.info("Hyperparameter search complete.")

    best_params_loggable = {k: (int(v) if isinstance(v, (np.integer, int)) else
                                float(v) if isinstance(v, (np.floating, float)) else
                                v)
                           for k, v in random_search.best_params_.items()}
    best_rmse = -random_search.best_score_

    print(f"Best parameters found: {best_params_loggable}")
    print(f"Best cross-validation RMSE: {best_rmse:.4f}")
    logger.info(f"Best CV RMSE: {best_rmse:.4f}")
    logger.info(f"Best parameters: {best_params_loggable}")

    final_pipeline_to_use = random_search.best_estimator_ # Already refitted

else: # Use Manual Parameters
    print(f"--- Mode: Using Manual Hyperparameters for {MODEL_TYPE} ---")
    logger.info(f"Mode: Using Manual Hyperparameters for {MODEL_TYPE}")
    logger.info(f"Manual Params: {manual_params_dict}")

    # Create pipeline with specific manual parameters for the chosen model
    manual_estimator = None
    if MODEL_TYPE == 'LightGBM':
        manual_estimator = lgb.LGBMRegressor(**manual_params_dict)
    elif MODEL_TYPE == 'XGBoost':
        manual_estimator = xgb.XGBRegressor(**manual_params_dict)
    elif MODEL_TYPE == 'CatBoost':
        manual_estimator = cb.CatBoostRegressor(**manual_params_dict)

    manual_model_wrapper = MultiOutputRegressor(manual_estimator)
    manual_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('regressor', manual_model_wrapper)])

    # Optionally, perform CV on the manual pipeline
    print("Performing cross-validation on manual parameters...")
    logger.info("Performing cross-validation on manual parameters...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Handle potential n_jobs conflict if XGBoost/LGBM use multiple cores internally
    cv_n_jobs = 4 # Adjust based on your system cores vs model internal parallelization
    cv_scores_neg_rmse = cross_val_score(
        manual_pipeline, X_train, y_train, cv=kf,
        scoring='neg_root_mean_squared_error', n_jobs=cv_n_jobs
    )
    cv_rmse_scores = -cv_scores_neg_rmse
    mean_cv_rmse = np.mean(cv_rmse_scores)
    std_cv_rmse = np.std(cv_rmse_scores)
    print(f"Cross-validation RMSE with manual parameters: {mean_cv_rmse:.4f} +/- {std_cv_rmse:.4f}")
    logger.info(f"CV RMSE with manual parameters: {mean_cv_rmse:.4f} +/- {std_cv_rmse:.4f}")

    # Train the final model on all training data
    print("Training final model with manual parameters...")
    logger.info("Training final model with manual parameters...")
    manual_pipeline.fit(X_train, y_train) # Fit the manual pipeline
    print("Final model training complete.")
    logger.info("Final model training complete.")

    final_pipeline_to_use = manual_pipeline # Assign to common variable


# --- 8. Make Predictions ---
print(f"Making predictions on test data using the final {MODEL_TYPE} pipeline...")
# ... (Prediction code using final_pipeline_to_use remains the same) ...
try:
    predictions = final_pipeline_to_use.predict(X_test)
    MIN_SCORE = 0
    MAX_SCORE = 5
    predictions[predictions < MIN_SCORE] = MIN_SCORE
    predictions[predictions > MAX_SCORE] = MAX_SCORE
    predictions = np.round(predictions)
    print("Predictions generated.")
except Exception as e:
    print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"ERROR during prediction: {e}")
    print(f"Check input data and pipeline steps for {MODEL_TYPE}.")
    logger.error(f"ERROR during prediction for {MODEL_TYPE}: {e}", exc_info=True)
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    num_test_samples = len(X_test)
    num_targets = len(TARGET_COLS)
    predictions = np.full((num_test_samples, num_targets), MIN_SCORE)
    print("Generated dummy predictions due to error.")


# --- 9. Generate Submission File ---
print("Generating submission file...")
# ... (Submission file generation code remains the same) ...
predictions_df = pd.DataFrame(predictions, columns=TARGET_COLS)
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
submission_df = submission_df[template_cols]
submission_df.to_csv(SUBMISSION_OUTPUT_FILE, index=False)
print(f"Submission file saved to '{SUBMISSION_OUTPUT_FILE}'")

# --- Log Run End ---
logger.info(f"--- Finished Run: {run_id} ---")
print(f"\nRun details and metrics logged to {log_file}")

# --- Update final print statement ---
print(f"\nPipeline {MODEL_TYPE} (v5 Native NaN, Ordinal/Cat Encode) finished.")
print("Next steps:")
print("1. Analyze the CV results from RandomizedSearchCV (if run).")
print(f"2. Compare performance of {MODEL_TYPE} to LightGBM.")
print("3. If using CatBoost, consider modifying preprocessing to use native categorical handling for potentially better results.")
print("4. Try other models (Random Forest, Stacking) or further tune hyperparameters.")
print("5. Engineer more features.")