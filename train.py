import os
import gc
import joblib
import logging
import datetime
import json
import warnings
import pandas as pd
import numpy as np
import torch # For checking CUDA availability
import matplotlib.pyplot as plt # Added for plotting
import shap # Added for SHAP plots

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn import clone # For cloning pipeline
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression # Added for MI calculation

# TabPFN specific import
try:
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
    from tabpfn_extensions import interpretability # Added for SHAP
except ImportError:
    print("Error: Required libraries not found for AutoTabPFN.")
    print("Please install tabpfn and tabpfn-extensions:")
    print("  pip install tabpfn")
    print("  pip install git+https://github.com/automl/TabPFN-extensions.git")
    exit(1)

warnings.filterwarnings('ignore')

# ==============================================================================
# Settings & Configuration
# ==============================================================================
def load_settings():
    try:
        with open('SETTINGS.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("CRITICAL ERROR: SETTINGS.json not found.")
        exit(1)
    except json.JSONDecodeError:
        print("CRITICAL ERROR: SETTINGS.json is not valid JSON.")
        exit(1)

SETTINGS = load_settings()
CLEAN_DATA_DIR = SETTINGS.get('CLEAN_DATA_DIR', './Clean_Data/')
MODEL_DIR = SETTINGS.get('MODEL_DIR', './Models/')
LOG_DIR = SETTINGS.get('LOG_DIR', './Log_Files/')
# CHECKPOINT_DIR = SETTINGS.get('CHECKPOINT_DIR', './Checkpoints/') # If needed

# --- Model Configuration (from TabPFN_Fun.py, can be moved to SETTINGS.json) ---
MODEL_TYPE = 'AutoTabPFN'
PERFORM_CV = False
CV_FOLDS = 5
PERFORM_AVERAGING = True
N_AVERAGING_RUNS = 5
BASE_RANDOM_STATE = 42
AUTO_TABPFN_TIME_BUDGET_SECONDS = 3600
TABPFN_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AUTOTABPFN_PARAMS = {
    'device': TABPFN_DEVICE,
    'max_time': AUTO_TABPFN_TIME_BUDGET_SECONDS,
    # Add other AutoTabPFN parameters here if needed
}
TARGET_COL = 'modben' # Defined in prepare_data.py, ensure consistency
PLOTS_DIR = SETTINGS.get('PLOTS_DIR', './Plots/') # Added for saving plots
SKIP_PLOTS = SETTINGS.get('SKIP_PLOTS', False) # New setting to skip plots
REDUCE_FEATURES_FOR_SHAP_PLOTS = SETTINGS.get('REDUCE_FEATURES_FOR_SHAP_PLOTS', False)
SHAP_TOP_N_FEATURES = SETTINGS.get('SHAP_TOP_N_FEATURES', 20)
SHAP_EXPLAIN_SAMPLE_SIZE = SETTINGS.get('SHAP_EXPLAIN_SAMPLE_SIZE', None) # New setting for SHAP sample size

# ==============================================================================
# Setup Logging
# ==============================================================================
run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_id = f"train_model_{MODEL_TYPE}_{run_timestamp}"
log_filename = f"{run_id}.log"
log_file = os.path.join(LOG_DIR, log_filename)

try:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    # os.makedirs(CHECKPOINT_DIR, exist_ok=True) # If using checkpoints
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

logger.info(f"--- Starting Model Training: {run_id} ---")
logger.info(f"Clean Data Directory: {CLEAN_DATA_DIR}")
logger.info(f"Model Save Directory: {MODEL_DIR}")
logger.info(f"Log File: {log_file}")
logger.info(f"Model Type: {MODEL_TYPE}")
logger.info(f"Averaging: {PERFORM_AVERAGING}, Runs: {N_AVERAGING_RUNS if PERFORM_AVERAGING else 1}")
logger.info(f"CV: {PERFORM_CV}, Folds: {CV_FOLDS if PERFORM_CV else 'N/A'}")
logger.info(f"AutoTabPFN Params: {AUTOTABPFN_PARAMS}")
logger.info(f"Skip Plots: {SKIP_PLOTS}")
logger.info(f"Reduce Features for SHAP: {REDUCE_FEATURES_FOR_SHAP_PLOTS}, Top N: {SHAP_TOP_N_FEATURES if REDUCE_FEATURES_FOR_SHAP_PLOTS else 'N/A'}")
logger.info(f"SHAP Explain Sample Size: {SHAP_EXPLAIN_SAMPLE_SIZE if SHAP_EXPLAIN_SAMPLE_SIZE is not None else 'All'}")

# ==============================================================================
# Helper Functions (from TabPFN_Fun.py)
# ==============================================================================
def spearman_corr(y_true, y_pred):
    y_true_arr = np.array(y_true).squeeze()
    y_pred_arr = np.array(y_pred).squeeze()
    if np.all(np.isnan(y_true_arr)) or np.all(np.isnan(y_pred_arr)):
        return 0.0
    if y_true_arr.ndim == 0 or y_pred_arr.ndim == 0 or np.std(y_true_arr) == 0 or np.std(y_pred_arr) == 0:
        return 1.0 if np.all(y_true_arr == y_pred_arr) else 0.0
    corr, _ = spearmanr(y_true_arr, y_pred_arr)
    return corr if not np.isnan(corr) else 0.0

spearman_scorer = make_scorer(spearman_corr, greater_is_better=True)

# ==============================================================================
# Plotting Helper Function (Refactored)
# ==============================================================================
def generate_and_save_plots(model_regressor, X_train_processed, y_train, feature_names,
                              plots_dir, model_id_str, logger_instance,
                              reduce_features_shap=False, shap_top_n=20,
                              shap_explain_sample_size=None): # Added shap_explain_sample_size
    logger_instance.info(f"Generating SHAP plots for model: {model_id_str}...")
    os.makedirs(plots_dir, exist_ok=True)

    # --- SHAP Value based Importance ---
    logger_instance.info("Calculating SHAP values for feature importance...")
    try:
        # 1. Prepare data for SHAP computation
        X_shap_input_df = X_train_processed
        if not isinstance(X_shap_input_df, pd.DataFrame):
            if feature_names and hasattr(X_shap_input_df, 'shape') and X_shap_input_df.shape[1] == len(feature_names):
                X_shap_input_df = pd.DataFrame(X_shap_input_df, columns=feature_names)
                logger_instance.info("Converted X_train_processed to DataFrame with original feature names for SHAP computation.")
            else:
                logger_instance.error(
                    f"Cannot create DataFrame for SHAP: X_train_processed columns ({X_shap_input_df.shape[1] if hasattr(X_shap_input_df, 'shape') else 'N/A'}) "
                    f"mismatch/incompatible with provided feature_names (count: {len(feature_names if feature_names else [])}). Skipping SHAP plots."
                )
                return
        
        if X_shap_input_df.empty or X_shap_input_df.shape[1] == 0:
            logger_instance.warning("No features available in X_shap_input_df for SHAP calculation. Skipping SHAP plots.")
            return

        # current_all_feature_names are the columns of X_shap_input_df
        current_all_feature_names = X_shap_input_df.columns.tolist()

        # Potentially subsample the data for SHAP calculation
        X_for_shap_calculation = X_shap_input_df
        if shap_explain_sample_size is not None and shap_explain_sample_size > 0 and shap_explain_sample_size < len(X_shap_input_df):
            logger_instance.info(f"Subsampling X_shap_input_df from {len(X_shap_input_df)} to {shap_explain_sample_size} samples for SHAP calculation.")
            X_for_shap_calculation = X_shap_input_df.sample(n=shap_explain_sample_size, random_state=BASE_RANDOM_STATE)
            logger_instance.info(f"X_for_shap_calculation shape: {X_for_shap_calculation.shape}")
        
        # 2. Calculate SHAP values using (potentially sampled) data
        logger_instance.info(f"Calculating SHAP values using {X_for_shap_calculation.shape[0]} samples and {X_for_shap_calculation.shape[1]} features...")
        shap_values_all = interpretability.shap.get_shap_values(
            estimator=model_regressor,
            test_x=X_for_shap_calculation,  # Use the potentially sampled DataFrame
            attribute_names=X_for_shap_calculation.columns.tolist(), # Use columns from the actual data passed
        )
        
        # 3. Determine data and SHAP values for PLOTTING
        # X_data_for_plotting_source is the data for which shap_values_all were calculated
        X_data_for_plotting_source = X_for_shap_calculation
        
        # features_for_plot_source are the columns of X_data_for_plotting_source, matching shap_values_all's second dimension
        features_for_plot_source = X_data_for_plotting_source.columns.tolist()

        shap_values_for_actual_plot = shap_values_all
        X_df_for_actual_plot = X_data_for_plotting_source
        plot_feature_names_final = features_for_plot_source

        if reduce_features_shap:
            logger_instance.info(f"Reducing features for SHAP plots to top {shap_top_n} based on Mutual Information.")
            # MI calculation uses the original full X_train_processed and y_train to determine globally important features.
            # `feature_names` argument to this function is the original full list.
            try:
                X_np_for_mi = X_train_processed.to_numpy() if isinstance(X_train_processed, pd.DataFrame) else X_train_processed
                y_np_for_mi = y_train.to_numpy().ravel() if isinstance(y_train, (pd.Series, pd.DataFrame)) else np.array(y_train).ravel()

                if not feature_names or len(feature_names) != X_np_for_mi.shape[1]:
                    logger_instance.warning(
                        f"Original feature_names for MI (count: {len(feature_names if feature_names else 'N/A')}) "
                        f"are inconsistent with X_np_for_mi columns (count: {X_np_for_mi.shape[1]}). "
                        "Will use all (potentially sampled) data features for plotting."
                    )
                elif y_train is None:
                    logger_instance.warning("y_train is None. Cannot perform MI for plot reduction. Will use all (potentially sampled) data features for plotting.")
                else:
                    if np.isnan(X_np_for_mi).any():
                        logger_instance.warning("NaNs detected in X_np_for_mi for MI calculation. Imputing with column means.")
                        col_means = np.nanmean(X_np_for_mi, axis=0)
                        inds = np.where(np.isnan(X_np_for_mi))
                        X_np_for_mi[inds] = np.take(col_means, inds[1])
                        if np.isnan(X_np_for_mi).any():
                            logger_instance.error("NaNs still present after mean imputation for MI. Filling remaining NaNs with 0.")
                            X_np_for_mi = np.nan_to_num(X_np_for_mi, nan=0.0)

                    if X_np_for_mi.shape[0] != y_np_for_mi.shape[0]:
                        logger_instance.error(f"X_np_for_mi ({X_np_for_mi.shape[0]}) and y_np_for_mi ({y_np_for_mi.shape[0]}) differ in samples for MI. Using all (sampled) data features for plots.")
                    elif X_np_for_mi.shape[1] == 0:
                        logger_instance.warning("X_np_for_mi has 0 features for MI. Using all (sampled) data features for plots.")
                    else:
                        logger_instance.info("Calculating Mutual Information scores for plot feature selection (based on full data)...")
                        mi_scores = mutual_info_regression(X_np_for_mi, y_np_for_mi, random_state=BASE_RANDOM_STATE)
                        mi_series = pd.Series(mi_scores, index=feature_names) # Use original full feature_names for MI series
                        
                        top_mi_feature_names_for_plot = mi_series.nlargest(shap_top_n).index.tolist()
                        
                        if not top_mi_feature_names_for_plot:
                            logger_instance.warning("MI analysis for plotting yielded no top features. Using all (potentially sampled) data features for plots.")
                        else:
                            # Filter these top_mi_feature_names to those present in features_for_plot_source (columns of our (sampled) data)
                            valid_top_names_for_plot_from_mi = [name for name in top_mi_feature_names_for_plot if name in features_for_plot_source]
                            
                            if not valid_top_names_for_plot_from_mi:
                                logger_instance.warning(
                                    f"Top MI features ({top_mi_feature_names_for_plot[:5]}...) not found in the (sampled) data columns ({features_for_plot_source[:5]}...). "
                                    "Using all (potentially sampled) data features for plots."
                                )
                            else:
                                logger_instance.info(f"Top {len(valid_top_names_for_plot_from_mi)} features from MI for plotting: {valid_top_names_for_plot_from_mi[:5]}...")
                                # Get indices from features_for_plot_source (which align with shap_values_all columns)
                                name_to_idx_map = {name: i for i, name in enumerate(features_for_plot_source)}
                                top_indices_for_plot = [name_to_idx_map[name] for name in valid_top_names_for_plot_from_mi]

                                shap_values_for_actual_plot = shap_values_all[:, top_indices_for_plot]
                                X_df_for_actual_plot = X_data_for_plotting_source[valid_top_names_for_plot_from_mi]
                                plot_feature_names_final = valid_top_names_for_plot_from_mi
                                logger_instance.info(f"Reduced to {len(plot_feature_names_final)} features for SHAP plotting based on MI (applied to sampled data).")
            except Exception as e_reduce:
                logger_instance.error(f"Error during MI feature reduction for SHAP plotting: {e_reduce}. Using all (potentially sampled) data features for plots.", exc_info=True)
        
        # 4. Generate plots
        if X_df_for_actual_plot.empty or X_df_for_actual_plot.shape[1] == 0:
            logger_instance.warning("No features available for plotting after potential reduction/sampling. Skipping SHAP plots.")
            return

        logger_instance.info(f"Generating SHAP summary plot (beeswarm) with {len(plot_feature_names_final)} features using {X_df_for_actual_plot.shape[0]} samples.")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_for_actual_plot, X_df_for_actual_plot, feature_names=plot_feature_names_final, show=False, plot_size=None)
        plt.title(f"SHAP Summary Plot - {model_id_str}")
        plt.tight_layout()
        plot_filename_shap_summary = os.path.join(plots_dir, f"shap_summary_plot_{model_id_str}.png")
        plt.savefig(plot_filename_shap_summary)
        plt.close()
        logger_instance.info(f"Saved SHAP summary plot to {plot_filename_shap_summary}")

        logger_instance.info(f"Generating SHAP bar plot (global feature importance) with {len(plot_feature_names_final)} features using {X_df_for_actual_plot.shape[0]} samples.")
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values_for_actual_plot, X_df_for_actual_plot, feature_names=plot_feature_names_final, plot_type="bar", show=False, plot_size=None)
        plt.title(f"SHAP Global Feature Importance - {model_id_str}")
        plt.tight_layout()
        plot_filename_shap_bar = os.path.join(plots_dir, f"shap_bar_plot_{model_id_str}.png")
        plt.savefig(plot_filename_shap_bar)
        plt.close()
        logger_instance.info(f"Saved SHAP bar plot to {plot_filename_shap_bar}")

    except Exception as e_shap:
        logger_instance.error(f"Error generating SHAP plots: {e_shap}", exc_info=True)

# ==============================================================================
# Main Training Logic
# ==============================================================================
def main():
    logger.info("Loading preprocessed data...")
    try:
        X_train_processed = joblib.load(os.path.join(CLEAN_DATA_DIR, 'X_train_processed.joblib'))
        y_train = joblib.load(os.path.join(CLEAN_DATA_DIR, 'y_train.joblib'))
        preprocessor = joblib.load(os.path.join(CLEAN_DATA_DIR, 'preprocessor.joblib')) 
    except FileNotFoundError as e:
        logger.error(f"Error loading preprocessed data file: {e}. Run prepare_data.py first.")
        exit(1)
    logger.info(f"Loaded X_train_processed (Shape: {X_train_processed.shape}), y_train (Shape: {y_train.shape})")

    # --- Get Processed Feature Names for Plotting ---
    processed_feature_names = []
    try:
        processed_feature_names = preprocessor.get_feature_names_out()
        logger.info(f"Retrieved {len(processed_feature_names)} feature names from preprocessor.")
    except Exception as e_feat_names:
        logger.warning(f"Could not get feature names directly from preprocessor: {e_feat_names}")
        try:
            processed_feature_names = joblib.load(os.path.join(CLEAN_DATA_DIR, 'processed_feature_names.joblib'))
            logger.info(f"Loaded {len(processed_feature_names)} processed feature names from file.")
        except FileNotFoundError:
            logger.error("processed_feature_names.joblib not found. Plots might have generic feature labels.")
            if isinstance(X_train_processed, pd.DataFrame):
                processed_feature_names = X_train_processed.columns.tolist()
                logger.info(f"Using column names from X_train_processed DataFrame for plotting ({len(processed_feature_names)} features).")
            else:
                num_cols = X_train_processed.shape[1]
                processed_feature_names = [f'feature_{i}' for i in range(num_cols)]
                logger.warning(f"Using generic feature names for {num_cols} features for plotting.")
        except Exception as e_load_feat_names:
            logger.error(f"Error loading processed_feature_names.joblib: {e_load_feat_names}. Plots might have generic labels.")
            if isinstance(X_train_processed, pd.DataFrame):
                processed_feature_names = X_train_processed.columns.tolist()
            else:
                processed_feature_names = [f'feature_{i}' for i in range(X_train_processed.shape[1])]

    # Define base model estimator
    base_autotabpfn_estimator = AutoTabPFNRegressor(**{k: v for k, v in AUTOTABPFN_PARAMS.items() if k != 'random_state'})
    
    base_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', base_autotabpfn_estimator)
    ])

    if PERFORM_AVERAGING:
        logger.info(f"--- Starting Model Training: Averaging {N_AVERAGING_RUNS} {MODEL_TYPE} runs ---")
        trained_models_pipelines = []
        first_model_regressor_for_plots = None

        for i in range(N_AVERAGING_RUNS):
            run_seed = BASE_RANDOM_STATE + i
            logger.info(f"Starting Averaging Run {i+1}/{N_AVERAGING_RUNS} with seed {run_seed}...")
            
            current_pipeline = clone(base_pipeline)
            current_pipeline.set_params(regressor__random_state=run_seed)

            try:
                start_time_run = datetime.datetime.now()
                current_pipeline.named_steps['regressor'].fit(X_train_processed, y_train)
                end_time_run = datetime.datetime.now()
                logger.info(f"Run {i+1} trained successfully. Time: {end_time_run - start_time_run}")
                
                model_run_filename = os.path.join(MODEL_DIR, f'model_avg_run_{i}_seed_{run_seed}.joblib')
                joblib.dump(current_pipeline, model_run_filename)
                logger.info(f"Saved trained pipeline for run {i+1} to {model_run_filename}")
                trained_models_pipelines.append(current_pipeline)

                if i == 0 and processed_feature_names:
                    first_model_regressor_for_plots = current_pipeline.named_steps['regressor']

            except Exception as e_run:
                logger.error(f"Error during averaging run {i+1}: {e_run}", exc_info=True)

        if not trained_models_pipelines:
            logger.error("FATAL: No models were successfully trained during averaging. Exiting.")
            exit(1)
        logger.info(f"Completed {len(trained_models_pipelines)} averaging runs.")

        if not SKIP_PLOTS:
            if first_model_regressor_for_plots and processed_feature_names:
                generate_and_save_plots(first_model_regressor_for_plots, X_train_processed, y_train,
                                        processed_feature_names, PLOTS_DIR, f"{MODEL_TYPE}_avg_first_run", logger,
                                        REDUCE_FEATURES_FOR_SHAP_PLOTS, SHAP_TOP_N_FEATURES,
                                        shap_explain_sample_size=SHAP_EXPLAIN_SAMPLE_SIZE) # Pass new arg
            elif not first_model_regressor_for_plots:
                logger.warning("Skipping plots for averaged model as the first model regressor was not available.")
        else:
            logger.info("Skipping plot generation as per SKIP_PLOTS setting.")

    else:
        logger.info(f"--- Starting Model Training: Single {MODEL_TYPE} run ---")
        final_pipeline_to_use = clone(base_pipeline)
        final_pipeline_to_use.set_params(regressor__random_state=BASE_RANDOM_STATE)
        logger.info(f"Using single run configuration with random_state={BASE_RANDOM_STATE}")

        if PERFORM_CV:
            logger.info(f"Performing {CV_FOLDS}-Fold Cross-Validation...")
            cv_pipeline_for_scoring = clone(final_pipeline_to_use.named_steps['regressor'])
            try:
                cv_scores = cross_val_score(cv_pipeline_for_scoring, X_train_processed, y_train, 
                                            cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=BASE_RANDOM_STATE),
                                            scoring=spearman_scorer, error_score='raise')
                logger.info(f"CV Spearman Scores: {cv_scores}")
                logger.info(f"CV Mean Spearman Score: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
            except Exception as e_cv:
                logger.error(f"Error during Cross-Validation: {e_cv}", exc_info=True)

        logger.info(f"Training final single model... (Max time: {AUTO_TABPFN_TIME_BUDGET_SECONDS}s)")
        try:
            start_time_train = datetime.datetime.now()
            final_pipeline_to_use.named_steps['regressor'].fit(X_train_processed, y_train)
            end_time_train = datetime.datetime.now()
            logger.info(f"Final single model trained successfully. Time: {end_time_train - start_time_train}")

            model_filename = os.path.join(MODEL_DIR, 'model_single_run.joblib')
            joblib.dump(final_pipeline_to_use, model_filename)
            logger.info(f"Saved final single trained pipeline to {model_filename}")

            if not SKIP_PLOTS:
                if processed_feature_names:
                    single_model_regressor = final_pipeline_to_use.named_steps['regressor']
                    generate_and_save_plots(single_model_regressor, X_train_processed, y_train,
                                            processed_feature_names, PLOTS_DIR, f"{MODEL_TYPE}_single_run", logger,
                                            REDUCE_FEATURES_FOR_SHAP_PLOTS, SHAP_TOP_N_FEATURES,
                                            shap_explain_sample_size=SHAP_EXPLAIN_SAMPLE_SIZE) # Pass new arg
                else:
                    logger.warning("Skipping plots for single run model as processed feature names were not available.")
            else:
                logger.info("Skipping plot generation as per SKIP_PLOTS setting.")

        except Exception as e_train:
            logger.error(f"FATAL: Error training final single model: {e_train}", exc_info=True)
            exit(1)

    logger.info("Model training phase complete.")

if __name__ == '__main__':
    main()
