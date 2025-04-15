import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_regression
import re
import logging
from datetime import datetime

# --- Configuration ---
# Use the same paths as your main script
DATA_DIR = './' # <<< --- !!! ADAPT THIS PATH !!!
METADATA_FILE = f'{DATA_DIR}metadata.csv'
TRAIN_FEATURES_FILE = f'{DATA_DIR}train_features.csv'
TRAIN_OUTCOMES_FILE = f'{DATA_DIR}train_outcomes_MS.csv'
TEST_FEATURES_FILE = f'{DATA_DIR}test_features.csv' # Not strictly needed, but keeps config consistent

# --- Setup Logging (Optional but good practice) ---
log_file = 'Feature_Analysis.log'
run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
analysis_id = f"Mutual_Information_{run_timestamp}"

logger = logging.getLogger(analysis_id)
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

logger.info(f"--- Starting Mutual Information Analysis: {analysis_id} ---")

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
logger.info("Loading data...")
try:
    metadata_df = pd.read_csv(METADATA_FILE)
    train_features_df = pd.read_csv(TRAIN_FEATURES_FILE)
    train_outcomes_df = pd.read_csv(TRAIN_OUTCOMES_FILE)
    # test_features_df = pd.read_csv(TEST_FEATURES_FILE) # Test not needed here
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    logger.error(f"Error loading data: {e}")
    exit()

metadata_df.replace(9, np.nan, inplace=True)
print("Data loaded.")
logger.info("Data loaded.")

# --- 2. Identify Features and Targets ---
print("Identifying features and targets...")
TARGET_COLS = [col for col in train_outcomes_df.columns if col not in ['PID', 'time']]
train_week1_feature_cols = [col for col in train_features_df.columns if re.search(r'01$', col) or col == 'ais1']
METADATA_COLS = [col for col in metadata_df.columns if col != 'PID']
WEEK1_FEATURE_COLS = [col for col in train_week1_feature_cols if col != 'PID'] # Use week 1 features from train

# --- 3. Prepare Data (Similar to main script, including imputation setup) ---
print("Preparing data...")
logger.info("Preparing data...")
train_features_w1 = train_features_df[['PID'] + WEEK1_FEATURE_COLS]
train_merged_df = pd.merge(metadata_df, train_features_w1, on='PID', how='inner')
train_full_df = pd.merge(train_merged_df, train_outcomes_df, on='PID', how='inner')

FEATURES = METADATA_COLS + WEEK1_FEATURE_COLS + ['target_time']

X_train_raw = train_full_df[METADATA_COLS + WEEK1_FEATURE_COLS].copy()
y_train_raw = train_full_df[TARGET_COLS].copy()
time_train_raw = train_full_df['time'].copy()

# Drop rows where any target variable is NaN
valid_target_indices = y_train_raw.dropna().index
initial_rows = len(X_train_raw)
X_train_raw = X_train_raw.loc[valid_target_indices]
y_train = y_train_raw.loc[valid_target_indices] # Final y_train
time_train = time_train_raw.loc[valid_target_indices] # Final time_train
final_rows = len(X_train_raw)
if initial_rows != final_rows:
    print(f"Dropped {initial_rows - final_rows} rows with missing values in TARGET columns.")
    logger.info(f"Dropped {initial_rows - final_rows} rows with missing values in TARGET columns.")

# Add target time as a feature to X (as done in the modeling pipeline)
X_train = X_train_raw.copy() # Make a copy before adding time feature if needed elsewhere
X_train['target_time'] = time_train

print(f"Data prepared for preprocessing: X_train={X_train.shape}, y_train={y_train.shape}")
logger.info(f"Data prepared for preprocessing: X_train={X_train.shape}, y_train={y_train.shape}")

# --- 4. Define and Apply Preprocessing (Mirroring the main script) ---
print("Defining and applying preprocessing...")
logger.info("Defining and applying preprocessing...")

# Define feature types (mirroring previous script)
isnsci_pattern = r'(?:[a-z]{5,6}[lr]|[a-z]{1,5}[lr](?:[0-9]|1[0-2]|[tT][0-9]|t1[0-2]|[sS][1-5]|s45))[0-9]{2}$'
numerical_features_base = [col for col in X_train.columns if re.match(isnsci_pattern, col)]
numerical_features_other = ['target_time']
binary_like_metadata = ['srdecc1', 'surgcd1', 'spcsuc1', 'scdecc1', 'hemccd1', 'mhpsyccd', 'mhneurcd', 'mhcardcd', 'mhmetacd', 'sexcd']
categorical_features = ['age_category', 'bmi_category', 'tx1_r']
ordinal_features = ['ais1']
ais_categories = ['A', 'B', 'C', 'D', 'E']

numerical_features = list(set(numerical_features_base + numerical_features_other + binary_like_metadata) - set(ordinal_features) - set(categorical_features))
numerical_features = [f for f in numerical_features if f in X_train.columns]
categorical_features = [f for f in categorical_features if f in X_train.columns]
ordinal_features = [f for f in ordinal_features if f in X_train.columns]

# --- ADD CHECK FOR ZERO VARIANCE / ALL NAN COLUMNS ---
print("Checking for zero variance or all-NaN columns before preprocessing...")
logger.info("Checking for zero variance or all-NaN columns before preprocessing...")
cols_to_remove = []

# Check numerical features for zero variance (or undefined variance if all NaN)
zero_var_num_cols = []
for col in numerical_features:
    if col in X_train.columns:
        col_var = X_train[col].var(skipna=True)
        if pd.isna(col_var) or col_var == 0:
            zero_var_num_cols.append(col)
if zero_var_num_cols:
    print(f"Warning: Zero/undefined variance detected in numerical columns: {zero_var_num_cols}")
    logger.warning(f"Zero/undefined variance detected in numerical columns: {zero_var_num_cols}")
    cols_to_remove.extend(zero_var_num_cols)

# Check all columns for being entirely NaN
all_nan_cols = X_train.columns[X_train.isnull().all()].tolist()
if all_nan_cols:
     print(f"Warning: Columns with ALL NaN values detected: {all_nan_cols}")
     logger.warning(f"Columns with ALL NaN values detected: {all_nan_cols}")
     cols_to_remove.extend(all_nan_cols)

# Remove identified columns
cols_to_remove = sorted(list(set(cols_to_remove))) # Unique sorted list
if cols_to_remove:
    print(f"Removing problematic columns: {cols_to_remove}")
    logger.info(f"Removing problematic columns: {cols_to_remove}")
    X_train = X_train.drop(columns=cols_to_remove)
    # Update feature lists to exclude removed columns
    numerical_features = [f for f in numerical_features if f not in cols_to_remove]
    categorical_features = [f for f in categorical_features if f not in cols_to_remove]
    ordinal_features = [f for f in ordinal_features if f not in cols_to_remove]
else:
    print("No problematic (zero variance or all NaN) columns found.")
# --- END OF CHECK ---

# --- Now define transformers and preprocessor using the cleaned feature lists ---
print(f"Final feature counts for preprocessing: Numerical={len(numerical_features)}, Categorical={len(categorical_features)}, Ordinal={len(ordinal_features)}")
logger.info(f"Final feature counts for preprocessing: Numerical={len(numerical_features)}, Categorical={len(categorical_features)}, Ordinal={len(ordinal_features)}")

# Define transformers (WITH IMPUTATION)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[ais_categories], handle_unknown='use_encoded_value', unknown_value=-1)),
    ])

# Define preprocessor (using updated feature lists)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('ord', ordinal_transformer, ordinal_features)
        ],
    remainder='drop',
    sparse_threshold=0.3
    )

# Fit and transform the training data
X_processed = preprocessor.fit_transform(X_train)
# Get feature names AFTER transformation
feature_names_out = preprocessor.get_feature_names_out()

# --- ADD CHECK FOR FINITE VALUES AFTER PREPROCESSING ---
print("Checking for NaNs/Infs after preprocessing...")
logger.info("Checking for NaNs/Infs after preprocessing...")
is_sparse = hasattr(X_processed, "toarray")
X_check = X_processed.toarray() if is_sparse else X_processed
if not np.all(np.isfinite(X_check)):
    print("Error: NaNs or Infs found in X_processed AFTER preprocessing and column removal!")
    logger.error("NaNs or Infs found in X_processed AFTER preprocessing and column removal!")
    # Find and log problematic columns again if needed
    nan_inf_mask = ~np.isfinite(X_check)
    problem_cols_idx = np.where(np.any(nan_inf_mask, axis=0))[0]
    if len(problem_cols_idx) > 0 and len(problem_cols_idx) <= len(feature_names_out): # Ensure index is valid
        problem_cols_names = feature_names_out[problem_cols_idx]
        print(f"Problematic transformed columns detected: {problem_cols_names}")
        logger.error(f"Problematic transformed columns: {problem_cols_names}")
    else:
        print(f"Could not reliably identify problematic columns (Indices: {problem_cols_idx}, Total Features: {len(feature_names_out)})")
        logger.error(f"Could not reliably identify problematic columns (Indices: {problem_cols_idx}, Total Features: {len(feature_names_out)})")

    exit() # Exit before MI calculation
else:
    print("Preprocessing successful. No NaNs/Infs detected in final features.")
    logger.info("Preprocessing successful. No NaNs/Infs detected in final features.")
# --- END OF CHECK ---


print(f"Preprocessing applied. Processed feature shape: {X_processed.shape}")
logger.info(f"Preprocessing applied. Processed feature shape: {X_processed.shape}")



# --- 5. Calculate Mutual Information ---
print("Calculating Mutual Information...")
logger.info("Calculating Mutual Information for each target variable...")

# Dictionary to store MI scores for each feature
mi_scores_dict = {feature_name: [] for feature_name in feature_names_out}

# Calculate MI for each target column
# Convert X_processed to dense if it's sparse and MI function requires it (or check function docs)
# mutual_info_regression handles sparse matrices.
for target_col in TARGET_COLS:
    print(f"  Calculating for target: {target_col}")
    mi_vector = mutual_info_regression(
        X_processed,
        y_train[target_col],
        discrete_features='auto', # Detects discrete features based on dtype/values - might treat OHE as discrete
        n_neighbors=3, # Default, adjust if needed
        random_state=42
    )
    # Add scores to dictionary
    for feature_name, mi_score in zip(feature_names_out, mi_vector):
        mi_scores_dict[feature_name].append(mi_score)

print("MI calculation complete.")
logger.info("MI calculation complete.")

# --- 6. Aggregate and Rank Features ---
print("Aggregating and ranking features by MI...")
logger.info("Aggregating and ranking features by MI...")

# Calculate average MI across all targets for each feature
mi_aggregated = {feature: np.mean(scores) for feature, scores in mi_scores_dict.items()}

# Create DataFrame for ranking
mi_df = pd.DataFrame(list(mi_aggregated.items()), columns=['Feature', 'Average MI'])
mi_df = mi_df.sort_values(by='Average MI', ascending=False).reset_index(drop=True)

# --- 7. Display Results ---
print("\nTop 30 Features Ranked by Average Mutual Information across all Targets:")
print(mi_df.head(30).to_string())

# Save full ranked list to CSV
output_mi_file = f"mutual_information_ranking_{run_timestamp}.csv"
mi_df.to_csv(output_mi_file, index=False)
print(f"\nFull ranked list saved to: {output_mi_file}")
logger.info(f"Ranked MI list saved to: {output_mi_file}")
logger.info(f"Top 5 features: {mi_df['Feature'].tolist()[:5]}")

logger.info(f"--- Finished Mutual Information Analysis: {analysis_id} ---")
print("\nAnalysis finished.")