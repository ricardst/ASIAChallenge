import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import spearmanr, randint, uniform
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
import re
import datetime
import logging
import gc

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = './' # <<<--- !!! ADAPT THIS PATH !!! Ensure data files are here
METADATA_FILE = f'{DATA_DIR}metadata.csv'
TRAIN_FEATURES_FILE = f'{DATA_DIR}train_features.csv'
TRAIN_OUTCOMES_FILE = f'{DATA_DIR}train_outcomes_functional.csv'
TEST_FEATURES_FILE = f'{DATA_DIR}test_features.csv'
SUBMISSION_TEMPLATE_FILE = f'{DATA_DIR}test_outcomes_Fun_template_update.csv'

# --- User Choices ---
MODEL_TYPE = 'CatBoost' # Choose 'LightGBM', 'XGBoost', or 'CatBoost'
PERFORM_HYPERPARAMETER_SEARCH = True # Set to False to use manual parameters below
IMPUTE_NUMERICALS = False # <<<--- NEW FLAG: Set to True to impute numerical NaNs, False to let model handle them
N_CV_SPLITS = 5
N_HPO_ITERATIONS = 15

# --- Manual Parameters (Used only if PERFORM_HYPERPARAMETER_SEARCH = False) ---
# (Manual parameter dictionaries remain the same as before)
MANUAL_LGBM_PARAMS = {
    'n_estimators': 400, 'learning_rate': 0.04, 'num_leaves': 25, 'max_depth': 7,
    'reg_alpha': 0.15, 'reg_lambda': 0.1, 'colsample_bytree': 0.7, 'subsample': 0.75,
    'subsample_freq': 1, 'random_state': 42, 'n_jobs': -1, 'verbose': -1,
}
MANUAL_XGB_PARAMS = {
    'n_estimators': 400, 'learning_rate': 0.04, 'max_depth': 6, 'subsample': 0.75,
    'colsample_bytree': 0.7, 'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 0.15,
    'random_state': 42, 'n_jobs': -1,
}
MANUAL_CATBOOST_PARAMS = {
    'iterations': 500, 'learning_rate': 0.05, 'depth': 6, 'l2_leaf_reg': 4,
    'loss_function': 'RMSE', 'random_state': 42, 'verbose': 0,
}


# --- Setup Logging ---
log_file = 'Functional_Metrics.log'
tuning_mode_str = "HPO_Search" if PERFORM_HYPERPARAMETER_SEARCH else "Manual_Params"
impute_mode_str = "ImputeNum" if IMPUTE_NUMERICALS else "NativeNaN" # <<<--- Add impute status to name
# <<<--- Updated base model name to reflect functional track, single output, and imputation strategy
base_model_name = f"{MODEL_TYPE}_SingleOutput_v2_FuncPred_{impute_mode_str}"
model_name = f"{base_model_name}_{tuning_mode_str}"
run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_id = f"{model_name}_{run_timestamp}"

SUBMISSION_OUTPUT_FILE = f'{DATA_DIR}submission_{run_id}.csv'

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

logger.info(f"--- Starting Run: {run_id} ---")
logger.info(f"Predicting Functional Outcome (Benzel Score)")
logger.info(f"Selected Model Type: {MODEL_TYPE}")
logger.info(f"Hyperparameter Search Enabled: {PERFORM_HYPERPARAMETER_SEARCH}")
logger.info(f"Impute Numerical Features: {IMPUTE_NUMERICALS}") # <<<--- Log imputation choice
logger.info(f"Data Directory: {DATA_DIR}")
logger.info(f"Output Submission File: {SUBMISSION_OUTPUT_FILE}")


# --- Define Spearman Scorer ---
# (Spearman function remains the same)
def spearman_corr(y_true, y_pred):
    if np.all(np.isnan(y_pred)) or np.all(np.isnan(y_true)): return 0.0
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    if y_true_arr.ndim > 1 or y_pred_arr.ndim > 1: # Ensure 1D
        y_true_arr = y_true_arr.squeeze()
        y_pred_arr = y_pred_arr.squeeze()
    if np.std(y_pred_arr) == 0 or np.std(y_true_arr) == 0:
        return 1.0 if np.all(y_true_arr == y_pred_arr) else 0.0
    corr, _ = spearmanr(y_true_arr, y_pred_arr)
    return corr if not np.isnan(corr) else 0.0

spearman_scorer = make_scorer(spearman_corr, greater_is_better=True)


# --- Load Data ---
logger.info("Loading data...")
try:
    metadata_df = pd.read_csv(METADATA_FILE)
    train_features_df = pd.read_csv(TRAIN_FEATURES_FILE)
    train_outcomes_df = pd.read_csv(TRAIN_OUTCOMES_FILE)
    test_features_df = pd.read_csv(TEST_FEATURES_FILE)
    submission_template_df = pd.read_csv(SUBMISSION_TEMPLATE_FILE)

    # Replace 9 with NaN globally (adjust if 9 is valid data for some columns)
    # Be cautious if '9' has meaning in specific features other than 'unknown'
    logger.warning("Replacing value 9 with NaN globally. Verify this is appropriate for all features.")
    metadata_df.replace(9, np.nan, inplace=True)
    train_features_df.replace(9, np.nan, inplace=True)
    test_features_df.replace(9, np.nan, inplace=True)

except FileNotFoundError as e:
    logger.error(f"Error loading data: {e}. Please check DATA_DIR and file names.")
    exit()
logger.info("Data loaded successfully.")


# --- Identify Features and Target ---
logger.info("Identifying features and target...")
TARGET_COL = 'modben'
if TARGET_COL not in train_outcomes_df.columns:
     logger.error(f"Target column '{TARGET_COL}' not found in {TRAIN_OUTCOMES_FILE}. Check column names.")
     potential_targets = [c for c in train_outcomes_df.columns if 'ben' in c.lower() or 'walk' in c.lower()]
     if potential_targets:
         TARGET_COL = potential_targets[0]
         logger.warning(f"Using fallback target column: '{TARGET_COL}'")
     else:
         logger.error("No suitable target column found. Exiting.")
         exit()

train_week1_feature_cols = [col for col in train_features_df.columns if re.search(r'01$', col) or col == 'ais1']
test_feature_cols = [col for col in test_features_df.columns if col not in ['PID']]
common_feature_cols = list(set(train_week1_feature_cols).intersection(test_feature_cols))
logger.info(f"Using {len(common_feature_cols)} Week 1 feature columns common to train and test.")
WEEK1_FEATURE_COLS = sorted(common_feature_cols)
METADATA_COLS = [col for col in metadata_df.columns if col != 'PID']
logger.info(f"Using {len(METADATA_COLS)} metadata columns.")
logger.info(f"Target column: {TARGET_COL}")


# --- Prepare Training Data ---
logger.info("Preparing training data...")
train_features_w1 = train_features_df[['PID'] + WEEK1_FEATURE_COLS].copy()
train_merged_df = pd.merge(metadata_df, train_features_w1, on='PID', how='inner')
train_full_df = pd.merge(train_merged_df, train_outcomes_df[['PID', TARGET_COL, 'time']], on='PID', how='inner')

FEATURES = METADATA_COLS + WEEK1_FEATURE_COLS + ['target_time']

X_train_raw = train_full_df[METADATA_COLS + WEEK1_FEATURE_COLS].copy()
y_train_raw = train_full_df[TARGET_COL].copy()
time_train_raw = train_full_df['time'].copy()

valid_target_indices = y_train_raw.dropna().index
initial_rows = len(X_train_raw)
X_train = X_train_raw.loc[valid_target_indices].reset_index(drop=True)
y_train = y_train_raw.loc[valid_target_indices].reset_index(drop=True)
time_train = time_train_raw.loc[valid_target_indices].reset_index(drop=True)
final_rows = len(X_train)

logger.info(f"Dropped {initial_rows - final_rows} rows with missing values in TARGET column '{TARGET_COL}'.")
X_train['target_time'] = time_train
logger.info(f"Training data shapes ready for preprocessing: X_train={X_train.shape}, y_train={y_train.shape}")
if X_train.isnull().sum().sum() > 0:
    logger.warning(f"Training data (X_train) contains {X_train.isnull().sum().sum()} missing values BEFORE preprocessing.")


# --- Prepare Test Data ---
logger.info("Preparing test data...")
test_merged_df = pd.merge(metadata_df, test_features_df[['PID'] + WEEK1_FEATURE_COLS], on='PID', how='inner')
submission_template_info = submission_template_df[['PID', 'time']].copy()
test_full_df = pd.merge(submission_template_info, test_merged_df, on='PID', how='left')

test_PIDs = test_full_df['PID']
time_test = test_full_df['time']
X_test = test_full_df[METADATA_COLS + WEEK1_FEATURE_COLS].copy()
X_test['target_time'] = time_test
X_test = X_test[FEATURES] # Ensure column order matches X_train

logger.info(f"Test data shape: X_test={X_test.shape}")
if X_test.isnull().sum().sum() > 0:
    logger.warning(f"Test data (X_test) contains {X_test.isnull().sum().sum()} missing values BEFORE preprocessing.")


# --- Preprocessing Pipeline ---
logger.info("Setting up preprocessing pipeline...")

# Identify feature types based on X_train
meta_categorical = ['age_category', 'bmi_category', 'tx1_r', 'sexcd']
w1_ordinal = ['ais1']
ais_categories = ['A', 'B', 'C', 'D', 'E']

categorical_features = sorted([f for f in meta_categorical if f in X_train.columns])
ordinal_features = sorted([f for f in w1_ordinal if f in X_train.columns])
processed_cols = set(categorical_features + ordinal_features)
numerical_features = sorted([col for col in X_train.columns if col not in processed_cols])

logger.info(f"Identified {len(categorical_features)} categorical features: {categorical_features}")
logger.info(f"Identified {len(ordinal_features)} ordinal features: {ordinal_features}")
logger.info(f"Identified {len(numerical_features)} numerical features.")

# Define transformers for categorical and ordinal features (always applied)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Impute before encoding
    ('ordinal', OrdinalEncoder(categories=[ais_categories] * len(ordinal_features),
                               handle_unknown='use_encoded_value', unknown_value=-1))
    ])

# --- Conditionally Define Numerical Transformer ---
transformers_list = [
    ('cat', categorical_transformer, categorical_features),
    ('ord', ordinal_transformer, ordinal_features)
]

if IMPUTE_NUMERICALS:
    logger.info("Numerical Imputation: Enabled (Median)")
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
        ])
    transformers_list.append(('num', numerical_transformer, numerical_features))
    remainder_strategy = 'passthrough' # Pass through anything else unexpected
else:
    logger.info("Numerical Imputation: Disabled (Using Model's Native NaN Handling)")
    # Explicitly tell ColumnTransformer to pass these features through
    transformers_list.append(('num', 'passthrough', numerical_features))
    remainder_strategy = 'passthrough' # Pass through anything else unexpected

# Create the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=transformers_list,
    remainder=remainder_strategy,
    verbose_feature_names_out=False) # Keep original feature names easier to read

preprocessor.set_output(transform="pandas") # Keep pandas DataFrame output if possible


# --- Define Model Specific Components ---
# (Model definitions and parameter distributions remain the same)
if MODEL_TYPE == 'LightGBM':
    base_estimator = lgb.LGBMRegressor(random_state=42, verbose=-1)
    manual_params_dict = MANUAL_LGBM_PARAMS
    param_distributions = {
        'regressor__n_estimators': randint(100, 1000), 'regressor__learning_rate': uniform(0.01, 0.15),
        'regressor__num_leaves': randint(15, 60), 'regressor__max_depth': randint(3, 15),
        'regressor__reg_alpha': uniform(0.0, 1.0), 'regressor__reg_lambda': uniform(0.0, 1.0),
        'regressor__colsample_bytree': uniform(0.5, 0.5), 'regressor__subsample': uniform(0.5, 0.5),
        'regressor__subsample_freq': randint(0, 10)
    }
elif MODEL_TYPE == 'XGBoost':
    base_estimator = xgb.XGBRegressor(random_state=42, n_jobs=1)
    manual_params_dict = MANUAL_XGB_PARAMS
    param_distributions = {
        'regressor__n_estimators': randint(100, 1000), 'regressor__learning_rate': uniform(0.01, 0.15),
        'regressor__max_depth': randint(3, 12), 'regressor__subsample': uniform(0.5, 0.5),
        'regressor__colsample_bytree': uniform(0.5, 0.5), 'regressor__gamma': uniform(0, 0.5),
        'regressor__reg_alpha': uniform(0.0, 1.0), 'regressor__reg_lambda': uniform(0.0, 1.0)
    }
elif MODEL_TYPE == 'CatBoost':
    base_estimator = cb.CatBoostRegressor(random_state=42, verbose=0, loss_function='RMSE')
    manual_params_dict = MANUAL_CATBOOST_PARAMS
    param_distributions = {
        'regressor__iterations': randint(100, 1000), 'regressor__learning_rate': uniform(0.01, 0.15),
        'regressor__depth': randint(2, 6), 'regressor__l2_leaf_reg': uniform(1, 100),
        'regressor__subsample': uniform(0.3, 0.7),
    }
else:
    raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")


# --- Train Model (Conditional Tuning) ---
base_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', base_estimator)])
kf = KFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=42)

if PERFORM_HYPERPARAMETER_SEARCH:
    logger.info(f"--- Mode: Performing Hyperparameter Search for {MODEL_TYPE} ---")
    logger.info(f"Using {N_CV_SPLITS}-Fold CV and {N_HPO_ITERATIONS} HPO iterations.")
    random_search = RandomizedSearchCV(
        estimator=base_pipeline, param_distributions=param_distributions,
        n_iter=N_HPO_ITERATIONS, cv=kf, scoring='neg_root_mean_squared_error',
        n_jobs=-1, random_state=42, verbose=1, refit=True
    )
    logger.info("Starting hyperparameter search...")
    random_search.fit(X_train, y_train)
    logger.info("Hyperparameter search complete.")
    best_params_loggable = {k: (int(v) if isinstance(v, (np.integer, int)) else
                               float(v) if isinstance(v, (np.floating, float)) else v)
                           for k, v in random_search.best_params_.items()}
    best_rmse = -random_search.best_score_
    logger.info(f"Best CV RMSE: {best_rmse:.4f}")
    logger.info(f"Best parameters found: {best_params_loggable}")
    final_pipeline_to_use = random_search.best_estimator_

    logger.info("Calculating Spearman Rho on CV folds for the best HPO model...")
    try:
        cv_spearman_scores = cross_val_score(
            final_pipeline_to_use, X_train, y_train, cv=kf, scoring=spearman_scorer, n_jobs=-1
        )
        mean_cv_spearman = np.mean(cv_spearman_scores)
        std_cv_spearman = np.std(cv_spearman_scores)
        logger.info(f"Best HPO Model Cross-validation Spearman Rho: {mean_cv_spearman:.4f} +/- {std_cv_spearman:.4f}")
    except Exception as e:
        logger.error(f"Could not calculate CV Spearman Rho for best HPO model: {e}")

else: # Use Manual Parameters
    logger.info(f"--- Mode: Using Manual Hyperparameters for {MODEL_TYPE} ---")
    logger.info(f"Manual Params: {manual_params_dict}")
    manual_estimator = None
    if MODEL_TYPE == 'LightGBM': manual_estimator = lgb.LGBMRegressor(**manual_params_dict)
    elif MODEL_TYPE == 'XGBoost': manual_estimator = xgb.XGBRegressor(**manual_params_dict)
    elif MODEL_TYPE == 'CatBoost': manual_estimator = cb.CatBoostRegressor(**manual_params_dict)
    manual_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', manual_estimator)])

    logger.info(f"Performing {N_CV_SPLITS}-Fold cross-validation on manual parameters...")
    oof_predictions = np.zeros(len(X_train))
    cv_rmse_scores = []
    cv_spearman_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        logger.debug(f"Starting Fold {fold+1}/{N_CV_SPLITS}")
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        manual_pipeline.fit(X_train_fold, y_train_fold)
        val_preds = manual_pipeline.predict(X_val_fold)
        oof_predictions[val_idx] = val_preds
        fold_rmse = np.sqrt(mean_squared_error(y_val_fold, val_preds))
        fold_spearman = spearman_corr(y_val_fold, val_preds)
        cv_rmse_scores.append(fold_rmse)
        cv_spearman_scores.append(fold_spearman)
        logger.debug(f"Fold {fold+1} RMSE: {fold_rmse:.4f}, Spearman: {fold_spearman:.4f}")
        gc.collect()

    mean_cv_rmse = np.mean(cv_rmse_scores)
    std_cv_rmse = np.std(cv_rmse_scores)
    mean_cv_spearman = np.mean(cv_spearman_scores)
    std_cv_spearman = np.std(cv_spearman_scores)
    logger.info(f"Manual Params Cross-validation RMSE: {mean_cv_rmse:.4f} +/- {std_cv_rmse:.4f}")
    logger.info(f"Manual Params Cross-validation Spearman Rho: {mean_cv_spearman:.4f} +/- {std_cv_spearman:.4f}")

    logger.info("Training final model with manual parameters on all training data...")
    manual_pipeline.fit(X_train, y_train)
    logger.info("Final model training complete.")
    final_pipeline_to_use = manual_pipeline


# --- Make Predictions ---
logger.info(f"Making predictions on test data using the final {MODEL_TYPE} pipeline...")
try:
    # Check for NaNs before prediction if not imputing
    if not IMPUTE_NUMERICALS and X_test[numerical_features].isnull().sum().sum() > 0:
         logger.warning(f"Test data has numerical NaNs being passed to {MODEL_TYPE} for native handling.")
    elif IMPUTE_NUMERICALS and X_test.isnull().sum().sum() > 0:
         # If imputing, NaNs should only exist in cat/ord features if imputer failed (unlikely)
         logger.warning(f"Test data has NaNs before prediction even with imputation enabled. Check preprocessing steps.")

    predictions = final_pipeline_to_use.predict(X_test)

    # Post-processing Predictions
    MIN_SCORE = 1
    MAX_SCORE = 8
    try:
        actual_min = y_train.min()
        actual_max = y_train.max()
        logger.info(f"Actual target range in training data: [{actual_min}, {actual_max}]")
        if actual_min != MIN_SCORE or actual_max != MAX_SCORE:
            logger.warning(f"Defined score range [{MIN_SCORE}, {MAX_SCORE}] differs from actual training range [{actual_min}, {actual_max}]. Adjust if necessary.")
            # Optionally, adjust clipping range dynamically:
            # MIN_SCORE = actual_min
            # MAX_SCORE = actual_max
    except Exception as e:
        logger.error(f"Could not get min/max from training target: {e}")

    logger.info(f"Clipping predictions to range [{MIN_SCORE}, {MAX_SCORE}] and rounding.")
    predictions = np.clip(predictions, MIN_SCORE, MAX_SCORE)
    predictions = np.round(predictions).astype(int)
    logger.info("Predictions generated and post-processed.")

except Exception as e:
    logger.error(f"ERROR during prediction: {e}", exc_info=True)
    num_test_samples = len(X_test)
    predictions = np.full(num_test_samples, MIN_SCORE) # Default prediction
    logger.warning("Generated dummy predictions due to error.")


# --- Generate Submission File ---
logger.info("Generating submission file...")
predictions_df = pd.DataFrame({TARGET_COL: predictions})
submission_df = pd.DataFrame({'PID': test_PIDs, 'time': time_test})
submission_df.reset_index(drop=True, inplace=True)
predictions_df.reset_index(drop=True, inplace=True)
submission_df = pd.concat([submission_df, predictions_df], axis=1)

template_cols = submission_template_df.columns.tolist()
final_submission_df = pd.DataFrame(columns=template_cols)
final_submission_df[['PID', 'time']] = submission_df[['PID', 'time']]
final_submission_df[TARGET_COL] = submission_df[TARGET_COL]

missing_sub_cols = set(template_cols) - set(final_submission_df.columns)
if missing_sub_cols: logger.warning(f"Columns missing from submission file: {missing_sub_cols}")
extra_sub_cols = set(final_submission_df.columns) - set(template_cols)
if extra_sub_cols: logger.warning(f"Extra columns found in submission file: {extra_sub_cols}.")

if final_submission_df[TARGET_COL].isnull().any():
    logger.warning(f"NaNs found in submission target column '{TARGET_COL}'. Filling with default value {MIN_SCORE}.")
    final_submission_df[TARGET_COL].fillna(MIN_SCORE, inplace=True)
    final_submission_df[TARGET_COL] = final_submission_df[TARGET_COL].astype(int)

final_submission_df.to_csv(SUBMISSION_OUTPUT_FILE, index=False)
logger.info(f"Submission file saved to '{SUBMISSION_OUTPUT_FILE}'")


# --- Log Run End ---
logger.info(f"--- Finished Run: {run_id} ---")
print(f"\nRun details and metrics logged to {log_file}")

# --- Update final print statement ---
print(f"\nPipeline {MODEL_TYPE} (Functional Prediction - Benzel Score, ImputeNumericals={IMPUTE_NUMERICALS}) finished.")
print("Next steps:")
print(f"1. Review the log file ({log_file}) for CV scores (RMSE and Spearman Rho) and potential warnings about NaN handling.")
print(f"2. Compare results between IMPUTE_NUMERICALS=True and IMPUTE_NUMERICALS=False.")
print("3. Analyze feature importances.")
print("4. Experiment with Feature Engineering (LEMS, UEMS, etc.).")
print("5. Fine-tune hyperparameters further.")
print("6. If using CatBoost, consider implementing native categorical feature handling.")
print("7. Try different model types.")
print("8. Verify the global replacement of '9' with NaN was appropriate for your dataset.")
print("9. Double-check the prediction clipping range [1, 8] against the actual Benzel score definition and training data range.")