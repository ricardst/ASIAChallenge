import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
# SimpleImputer is removed as we drop NaNs for now
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

# --- Setup Logging ---
log_file = 'Metrics.log'
# Use a specific name for this run configuration
model_name = "LGBM_MultiOutput_RandomSearch_v3_dropna_rounded"
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

# Define features *before* dropping NaNs to ensure consistency
FEATURES = METADATA_COLS + WEEK1_FEATURE_COLS + ['target_time'] # Add target_time here
ALL_TRAIN_COLS = METADATA_COLS + WEEK1_FEATURE_COLS + TARGET_COLS + ['time'] # For checking NaNs

# Identify rows with missing values in relevant columns before separating X and y
cols_to_check_for_nan = list(set(METADATA_COLS + WEEK1_FEATURE_COLS + TARGET_COLS))
initial_rows = len(train_full_df)
train_full_df.dropna(subset=cols_to_check_for_nan, inplace=True)
final_rows = len(train_full_df)
print(f"Dropped {initial_rows - final_rows} rows with missing values from training data.")

# Separate features (X) and target (y) AFTER dropping NaNs
X_train = train_full_df[METADATA_COLS + WEEK1_FEATURE_COLS].copy()
y_train = train_full_df[TARGET_COLS].copy()
time_train = train_full_df['time'].copy()

# Add target time as a feature
X_train['target_time'] = time_train

print(f"Training data shapes after NaN drop: X_train={X_train.shape}, y_train={y_train.shape}")

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
print("Setting up preprocessing pipeline (No Imputation)...")

# Define feature types based on Data Dictionary and common sense
# Features ending in 'l'/'r' + number (e.g., elbfll01) or 'ltl'/'ltr'/'ppl'/'ppr' + number are ISNCSCI scores -> numerical
isnsci_pattern = r'(?:[a-z]{5,6}[lr]|[a-z]{1,5}[lr](?:[0-9]|1[0-2]|[tT][0-9]|t1[0-2]|[sS][1-5]|s45))[0-9]{2}$' # Motor/Sensory pattern
numerical_features_base = [col for col in X_train.columns if re.match(isnsci_pattern, col)]

# Other potential numerical (check dictionary)
numerical_features_other = ['target_time'] # Treat target_time (26/52) as numerical for scaling

# Identify binary/categorical from metadata based on dictionary
# sexcd: 1 female, 2 male -> needs encoding (treat as categorical)
# srdecc1, surgcd1, spcsuc1, scdecc1, hemccd1, mhpsyccd, mhneurcd, mhcardcd, mhmetacd: 0 no, 1 yes -> can be treated as numerical (0/1) or categorical. Let's treat as numerical 0/1 for simplicity here.
binary_like_metadata = ['srdecc1', 'surgcd1', 'spcsuc1', 'scdecc1', 'hemccd1', 'mhpsyccd', 'mhneurcd', 'mhcardcd', 'mhmetacd', 'sexcd']

# Remaining metadata: age_category, bmi_category, tx1_r -> categorical
categorical_features = ['age_category', 'bmi_category', 'tx1_r'] # Add 'sexcd' here if preferred over treating as 0/1/nan

# Ordinal features
ordinal_features = ['ais1'] # AIS grade at week 1
ais_categories = ['A', 'B', 'C', 'D', 'E'] # Define order

# Consolidate numerical features
numerical_features = list(set(numerical_features_base + numerical_features_other + binary_like_metadata) - set(ordinal_features) - set(categorical_features))

# Ensure all defined features are actually in X_train.columns
numerical_features = [f for f in numerical_features if f in X_train.columns]
categorical_features = [f for f in categorical_features if f in X_train.columns]
ordinal_features = [f for f in ordinal_features if f in X_train.columns]


print(f"Num Features ({len(numerical_features)}): {numerical_features[:5]}...")
print(f"Cat Features ({len(categorical_features)}): {categorical_features}")
print(f"Ord Features ({len(ordinal_features)}): {ordinal_features}")


# Create preprocessing pipelines (NO IMPUTATION)
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    # No Imputer
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

ordinal_transformer = Pipeline(steps=[
    # No Imputer - OrdinalEncoder needs careful thought on NaN handling if it were present
    ('ordinal', OrdinalEncoder(categories=[ais_categories], # List of lists
                               handle_unknown='use_encoded_value',
                               unknown_value=-1)) # Assign -1 to unknowns (if any appear during transform)
    # Optional: Add scaler *after* ordinal encoding if desired
    # ('scaler', StandardScaler())
])


# Use ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('ord', ordinal_transformer, ordinal_features)
        ],
    remainder='passthrough' # Keep any columns not explicitly handled (should be none ideally)
    )

# Check if all features intended for training are covered
all_processed_features = numerical_features + categorical_features + ordinal_features
unprocessed = set(X_train.columns) - set(all_processed_features)
if unprocessed:
     print(f"Warning: The following features are in X_train but not assigned a transformer: {unprocessed}")


# --- 6. Define Model ---
print("Defining model...")
lgbm = lgb.LGBMRegressor(random_state=42, device_type='gpu', verbose=-1)
model = MultiOutputRegressor(lgbm)

full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', model)])

# --- 7. Train Model & Cross-Validation --- REMOVED
"""
print("Training model and performing cross-validation...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
oof_predictions = np.zeros(y_train.shape) # Out-of-fold predictions for analysis
train_indices = X_train.index # Keep track of original indices

for fold, (train_idx_local, val_idx_local) in enumerate(kf.split(X_train, y_train)):
    # Map local fold indices back to original dataframe indices
    train_idx_orig = train_indices[train_idx_local]
    val_idx_orig = train_indices[val_idx_local]

    X_train_fold, X_val_fold = X_train.loc[train_idx_orig], X_train.loc[val_idx_orig]
    y_train_fold, y_val_fold = y_train.loc[train_idx_orig], y_train.loc[val_idx_orig]

    # Clone the pipeline to ensure fresh state for each fold
    from sklearn import clone
    
    fold_pipeline = clone(full_pipeline)
    fold_pipeline.fit(X_train_fold, y_train_fold)
    y_pred_fold = fold_pipeline.predict(X_val_fold)

    # Clamp predictions
    MIN_SCORE = 0
    MAX_SCORE = 5 # ISNCSCI motor score range
    y_pred_fold[y_pred_fold < MIN_SCORE] = MIN_SCORE
    y_pred_fold[y_pred_fold > MAX_SCORE] = MAX_SCORE
    # Optional: Round predictions to nearest integer
    y_pred_fold = np.round(y_pred_fold)

    oof_predictions[val_idx_local] = y_pred_fold # Store OOF predictions using local index

    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")
    cv_scores.append(rmse)

print(f"Average CV RMSE: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open("CV_logs.log", "a") as log_file:
    model_name = "MultiOutputRegressor(LGBMRegressor)"
    log_file.write(f"{timestamp} - Model: {model_name}\n")
    log_file.write(f"{timestamp} - CV Scores: {cv_scores}\n")
    log_file.write(f"{timestamp} - Average CV RMSE: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}\n")


# Train final model on all (non-NaN) training data
print("Training final model on all cleaned training data...")
full_pipeline.fit(X_train, y_train)
print("Final model training complete.")
"""
# --- ADD RandomizedSearchCV ---
print("Setting up RandomizedSearchCV for hyperparameter tuning...")

# Define the parameter distribution to sample from
# Prefix parameters with 'regressor__estimator__' because 'regressor' is the name of the
# MultiOutputRegressor step in the pipeline, and 'estimator' is the LGBM instance within it.
param_distributions = {
    'regressor__estimator__n_estimators': randint(100, 800),
    'regressor__estimator__learning_rate': uniform(0.01, 0.2),
    'regressor__estimator__num_leaves': randint(20, 60),
    'regressor__estimator__max_depth': randint(3, 12), # -1 means no limit, let's set a range
    'regressor__estimator__reg_alpha': uniform(0.0, 1.0), # L1 regularization
    'regressor__estimator__reg_lambda': uniform(0.0, 1.0), # L2 regularization
    'regressor__estimator__colsample_bytree': uniform(0.6, 0.4), # Sample 60% to 100% of features
    'regressor__estimator__subsample': uniform(0.6, 0.4), # Sample 60% to 100% of data rows (if subsample_freq > 0)
    'regressor__estimator__subsample_freq': randint(0, 5) # How often to subsample rows (0 = never)
    # Add other parameters if needed, e.g., boosting_type, min_child_samples etc.
}

# Configure the random search
# Using neg_root_mean_squared_error because RandomizedSearchCV maximizes the score
n_iterations = 2
kf = KFold(n_splits=5, shuffle=True, random_state=42) # Reuse the KFold definition
random_search = RandomizedSearchCV(
    estimator=full_pipeline,          # The pipeline object
    param_distributions=param_distributions,
    n_iter=n_iterations,                       # Number of parameter settings to sample <<< --- ADJUST AS NEEDED (more is better but slower)
    cv=kf,                           # Cross-validation strategy
    scoring='neg_root_mean_squared_error', # Lower RMSE is better, so optimize negative RMSE
    n_jobs=-1,                       # Use all available cores
    random_state=42,                 # For reproducibility
    verbose=1                        # Show progress
    # refit=True is the default, so it automatically refits the best model on the whole training set
)

print("Starting hyperparameter search...")
# --- ADD Logging before search ---
logger.info(f"Starting RandomizedSearchCV with n_iter={n_iterations}")

random_search.fit(X_train, y_train)

print("\nHyperparameter search complete.")
print(f"Best parameters found: {random_search.best_params_}")
best_rmse = -random_search.best_score_
print(f"Best cross-validation RMSE: {best_rmse:.4f}")

random_search.fit(X_train, y_train)

print("\nHyperparameter search complete.")
print(f"Best parameters found: {random_search.best_params_}")
# Score is negative RMSE, so negate it to get positive RMSE
best_rmse = -random_search.best_score_
print(f"Best cross-validation RMSE: {best_rmse:.4f}")

# --- ADD Logging after search ---
logger.info(f"Hyperparameter search complete.")
logger.info(f"Best CV RMSE: {best_rmse:.4f}")
# Convert numpy types in best_params_ to standard python types for cleaner logging if necessary
best_params_loggable = {k: (int(v) if isinstance(v, np.integer) else
                            float(v) if isinstance(v, np.floating) else
                            v)
                       for k, v in random_search.best_params_.items()}
logger.info(f"Best parameters: {best_params_loggable}")

# The best estimator found during the search, already refitted on the full training data
best_pipeline = random_search.best_estimator_

# --- 8. Make Predictions ---
print("Making predictions on test data using the best found pipeline...")
try:
    # Use the best estimator found by RandomizedSearchCV
    predictions = best_pipeline.predict(X_test) # <<< --- CHANGE: Use best_pipeline

    # Post-process predictions (clamping/rounding)
    MIN_SCORE = 0
    MAX_SCORE = 5 # ISNCSCI motor score range
    predictions[predictions < MIN_SCORE] = MIN_SCORE
    predictions[predictions > MAX_SCORE] = MAX_SCORE
    # Round predictions to nearest integer
    predictions = np.round(predictions)

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

print("\nPipeline V2 (drop NaN, Ordinal AIS) finished.")
print("Next steps:")
print("1. Analyze the CV results. Check OOF predictions if needed.")
print("2. Re-introduce imputation (e.g., SimpleImputer) into the preprocessing pipeline to handle potential NaNs in the test set robustly.")
print("3. Experiment with different imputation strategies.")
print("4. Try different models (RandomForest, XGBoost) or tune LightGBM hyperparameters.")
print("5. Engineer more features from Week 1 data + metadata.")