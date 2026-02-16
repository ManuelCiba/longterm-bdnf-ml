import glob
from multiprocessing import Pool
import multiprocessing
import os
import warnings
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
from numpy import matrix
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_validate
from collections import Counter
from lib.connectivity import matrix_handling
import time
# LightGBM
from lightgbm import LGBMClassifier # sklearn compatible version
# CatBoost
from catboost import CatBoostClassifier # sklearn compatible version


MAX_TRIAL_SECONDS = 30  # maximum seconds allowed per trial

def _preprocess_and_check_data(df_features, group_name='chip'):

    # print number of rows in df_features
    print("Number of rows in df_features: " + str(len(df_features)))
    # remove rows with NaN values
    df_features = df_features.dropna()
    # print number of rows in df_features
    print("Number of rows after removing rows with NaN values: " + str(len(df_features)))

    # split the df_features into y and X and groups
    y = df_features['y']  # 'y' is the column with class labels
    X = df_features.drop(columns=['y', group_name])  # Drop 'y' and 'chip' columns to get features
    groups = df_features[group_name].values

    # Remove columns where all values are zero
    X = X.loc[:, (X != 0).any(axis=0)]

    # 1) calculate class distribution per group
    df = pd.DataFrame({'y': y, 'group': groups})
    print("\nClass distribution per group:")
    print(df.groupby('group')['y'].value_counts())

    # 2) Count groups by their majority/only class
    group_class = df_features.groupby(group_name)['y'].agg(lambda x: x.mode().iloc[0])
    print("Groups per class:", Counter(group_class))

    # 3) Ensure enough groups per class for your CV folds:
    n_splits = _define_n_splits(y, groups)
    print("Number of splits for outer StratifiedGroupKFold:", n_splits)

    return df_features, X, y, groups, n_splits
 
def _define_n_splits(y, groups):
    # y: array of 0/1 labels
    # groups: array of group IDs (strings), same length as y

    # 1 Create a DataFrame for convenience
    df = pd.DataFrame({'y': y, 'group': groups})

    # 2 Count how many groups belong to each class
    #   - Take the majority class per group (since each group is single-class, this works)
    group_class = df.groupby('group')['y'].agg(lambda x: x.mode().iloc[0])

    # 3 Count groups per class
    groups_per_class = Counter(group_class)
    print("Groups per class:", groups_per_class)

    # 4 Compute n_splits = min number of groups per class
    n_splits = min(groups_per_class.values())

    return n_splits

def call(models, df_features):
    warnings.filterwarnings("ignore")

    # Step 1: preprocess and check data
    df_features, X, y, groups, n_splits = _preprocess_and_check_data(df_features, group_name='chip')

    # Step 2: Outer Cross-Validation using StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=n_splits)
    train_result_list = []  # To store the training metrics (inner CV results)
    test_result_list = []   # To store the test set performance (outer test results)
    X_train_list = [] # to store X_train for all splits, needed for SHAP
    y_train_list = []  # to store y_train for all splits, needed for SHAP
    X_test_list = []
    y_test_list = []

    cnt = 0
    for train_idx, test_idx in cv.split(X, y, groups):

        print("Split #" + str(cnt))
        cnt += 1

        # Step 3: Split the data into training and testing for this fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_test_list.append(X_test)
        y_test_list.append(y_test)

        # Step 4: Call the model evaluation process with the outer training data (inner CV results)
        train_result = _calling_the_models(models, X_train, y_train, groups_train)

        # Step 5: Loop through each model in train_result to evaluate them on the outer test set
        for idx, row in train_result.iterrows():
            model_name = row['Model']
            clf_best = row['clf_best']  # Retrieve the best classifier for this model

            # Predict on the outer test set
            y_test_pred = clf_best.predict(X_test)
            auc_test = roc_auc_score(y_test, y_test_pred)
            accuracy_test = accuracy_score(y_test, y_test_pred)
            recall_test = recall_score(y_test, y_test_pred)
            precision_test = precision_score(y_test, y_test_pred, zero_division=0)
            f1_test = f1_score(y_test, y_test_pred)

            # Collect test set metrics for this model
            test_result = {
                'Model': model_name,
                'AUC': auc_test,
                'Accuracy': accuracy_test,
                'Recall': recall_test,
                'Precision': precision_test,
                'F1': f1_test
            }

            # Append the current model's test results (for this outer fold) to the list
            test_result_list.append(test_result)

        # Append the inner cross-validation results (train) for this outer fold
        train_result_list.append(train_result)

    # Step 6: Combine the results across all outer folds
    df_train_results = pd.concat(train_result_list, ignore_index=True)  # Combine inner CV results into a DataFrame
    df_test_results = pd.DataFrame(test_result_list)  # Combine outer test results into a DataFrame

    return df_train_results, df_test_results, X_train_list, y_train_list, X_test_list, y_test_list


def _calling_the_models(models, X_train, y_train, groups_train):
    # Convert into a 1D array (Important for some ML models)
    y_train = y_train.values.ravel()

    # Initialize to an empty list
    train_results = []
    try:
        with Pool(processes=len(models)) as pool:
            train_results = pool.starmap(_parallel_evaluate,
                                         [(model, X_train, y_train, groups_train, len(models)) for model in models])
    except Exception as e:
        print(f"An error occurred during model evaluation: {e}")
        return None  # Return None or an empty list to indicate failure

    if not train_results:  # Check if results are empty
        print("No results were returned from the model evaluations.")
        return None  # or return []

    # Convert list to DataFrame; ensure the list is not empty
    try:
        df_train_results = pd.DataFrame(train_results)
    except Exception as e:
        print(f"Failed to convert train results to DataFrame: {e}")
        return None  # Handle the case where conversion fails

    return df_train_results


def _parallel_evaluate(model_name, X_train, y_train, groups_train, num_models):
    """
    This function now builds a model builder and an Optuna param-space function for each model name,
    calls the Optuna tuner, and returns the metrics dictionary compatible with your pipeline.
    """

    # define number of CPU cores used for model internals (not trials)
    num_cores = 1 #max(1, int(multiprocessing.cpu_count() / (num_models*2)))
    #print(f"Number of cores used for internal model training: {num_cores}")

    # Default trials by model (tune these for your compute)
    default_n_trials = {
        'RF': 40,
        'SVM': 60,
        'NB': 20,
        'MLP': 60,
        'KNN': 30,
        'LR': 30,
        'XGboost': 40,#100
        'LightGBM': 40,
        'CatBoost': 40,
    }

    # Build model_builder and param_space_func per model
    if model_name == 'RF':
        def model_builder(params):
            return RandomForestClassifier(random_state=1234, n_jobs=num_cores, **params)

        def param_space(trial):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            }

        n_trials = default_n_trials['RF']

    elif model_name == 'SVM':
        def model_builder(params):
            # SVC doesn't accept None for degree/kernel combos; pass only relevant params
            return SVC(probability=True, random_state=1234, **params)

        def param_space(trial):
            kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly", "sigmoid"])
            p = {}
            p["kernel"] = kernel
            p["C"] = trial.suggest_float("C", 1e-2, 1e3, log=True)
            if kernel in ["rbf", "poly", "sigmoid"]:
                p["gamma"] = trial.suggest_float("gamma", 1e-5, 1.0, log=True)
            if kernel == "poly":
                p["degree"] = trial.suggest_int("degree", 2, 5)
            return p

        n_trials = default_n_trials['SVM']

    elif model_name == 'NB':
        def model_builder(params):
            return GaussianNB(**params)

        def param_space(trial):
            return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-1, log=True)}

        n_trials = default_n_trials['NB']

    elif model_name == 'MLP':
        def model_builder(params):
            # map hidden_layer_sizes from tuple specification
            return MLPClassifier(random_state=1234,
                                 max_iter=500,
                                 tol=1e-4,
                                 early_stopping=True,
                                 validation_fraction=0.1,
                                 learning_rate_init=params.get("learning_rate_init", 1e-3),
                                 **{k: v for k, v in params.items() if k != "learning_rate_init"})

        def param_space(trial):
            n_layers = trial.suggest_int("n_layers", 1, 2)
            sizes = []
            for i in range(n_layers):
                sizes.append(trial.suggest_int(f"n_units_l{i}", 50, 300))
            hidden = tuple(sizes)
            return {
                "hidden_layer_sizes": hidden,
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "solver": trial.suggest_categorical("solver", ["adam", "sgd"]),
                "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "adaptive"]),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-5, 1e-2, log=True)
            }

        n_trials = default_n_trials['MLP']

    elif model_name == 'KNN':
        def model_builder(params):
            return KNeighborsClassifier(**params)

        def param_space(trial):
            return {
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
                "leaf_size": trial.suggest_int("leaf_size", 10, 50),
                "p": trial.suggest_categorical("p", [1, 2]),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "metric": trial.suggest_categorical("metric", ["minkowski", "chebyshev"])
            }

        n_trials = default_n_trials['KNN']

    elif model_name == 'LR':
        def model_builder(params):
            return LogisticRegression(random_state=1234, max_iter=500, **params)

        def param_space(trial):
            return {
                "C": trial.suggest_float("C", 1e-4, 1e4, log=True),
                "penalty": "l2",  # keep L2 for solver stability
                "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"])
            }

        n_trials = default_n_trials['LR']

    elif model_name == 'XGboost':
        def model_builder(params):
            # ensure compatibility: pass only allowed kwargs - user can extend
            return XGBClassifier(random_state=1234, eval_metric="logloss", n_jobs=num_cores, **params)

        def param_space(trial):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "max_depth": trial.suggest_int("max_depth", 2, 12),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "gamma": trial.suggest_float("gamma", 0.0, 10.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

        n_trials = default_n_trials['XGboost']

    elif model_name == 'LightGBM':
        def model_builder(params):
            return LGBMClassifier(random_state=1234, **params)

        def param_space(trial):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 512),
                "max_depth": trial.suggest_int("max_depth", -1, 16),  # -1 = no limit
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_split_gain": trial.suggest_float("min_split_gain", 0, 5),
                "n_jobs": 1,   # very important inside Optuna
                "objective": "binary",
                "boosting_type": "gbdt",
                "verbosity": -1, # suppress warnings
            }

        n_trials = default_n_trials['LightGBM']

    elif model_name == 'CatBoost':
        def model_builder(params):
            return CatBoostClassifier(random_state=1234, **params)

        def param_space(trial):
            return {
                "iterations": trial.suggest_int("iterations", 200, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                "depth": trial.suggest_int("depth", 3, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                "border_count": trial.suggest_int("border_count", 32, 255),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "verbose": 0,
                "thread_count": 1   # required to avoid Optuna deadlocks
            }

        n_trials = default_n_trials['CatBoost']

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # define scoring and internal CV 
    n_splits = _define_n_splits(y_train, groups_train)
    print("Number of splits for internal StratifiedGroupKFold:", n_splits)
    cv_internal = StratifiedGroupKFold(n_splits=n_splits)

    scoring = {
        'AUC': make_scorer(roc_auc_score),
        'Accuracy': make_scorer(accuracy_score),
        'Recall': make_scorer(recall_score),
        'Precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'F1': make_scorer(f1_score, average='weighted'),
        #'Balanced_Accuracy': make_scorer(sklearn.metrics.balanced_accuracy_score)
    }

    # run optuna-based tuning
    clf_best, cv_results = _perform_hyperparameter_tuning_optuna_with_cv(model_name=model_name,
        model_builder=model_builder,
        param_space_func=param_space,
        X=X_train,
        y=y_train,
        groups=groups_train,
        scoring=scoring,
        cv=cv_internal,
        n_trials=n_trials,
        refit_metric= "AUC" #"Balanced_Accuracy"
    )

    # collect metrics dictionary exactly as before
    metrics = {'Model': model_name, 'clf_best': clf_best}  # Add model and clf as initial keys

    # Compute statistics from cv_results to mimic original behavior.
    # cv_results provides arrays named like 'mean_test_AUC' etc.
    for metric in ['AUC', 'Accuracy', 'Recall', 'Precision', 'F1']:
        key = f'mean_test_{metric}'
        if key in cv_results:
            scores = np.array(cv_results[key])
        else:
            # fallback: if metric not present, set NaNs
            scores = np.array([np.nan])

        mean_score = np.nanmean(scores)
        std_score = np.nanstd(scores)

        # Calculate median
        median_score = np.nanmedian(scores)

        # Calculate IQR
        q75, q25 = np.nanpercentile(scores, [75, 25])
        iqr_score = q75 - q25

        # Calculate min and max
        min_score = np.nanmin(scores)
        max_score = np.nanmax(scores)

        metrics[metric + '_Mean'] = mean_score
        metrics[metric + '_Std'] = std_score
        metrics[metric + '_Median'] = median_score
        metrics[metric + '_IQR'] = iqr_score
        metrics[metric + '_Min'] = min_score
        metrics[metric + '_Max'] = max_score

    return metrics


def _perform_hyperparameter_tuning_optuna_with_cv(
    model_name,
    model_builder,       # function(params) -> model
    param_space_func,    # function(trial) -> dict of parameters
    X,
    y,
    groups,
    scoring,
    cv,
    n_trials=50,
    refit_metric="AUC",
    direction="maximize",
    seed=1234
):
    """
    Optuna-based hyperparameter tuning with internal cross-validation.
    Returns:
        clf_best: estimator fitted on full (X, y)
        cv_results: dict with keys:
            - 'params': list of param dicts tried
            - 'mean_test_{metric}': list of mean CV scores per trial for each metric in scoring
            - 'number': trial numbers
    """

    # Ensure y is 1D array (sklearn compatibility)
    y_arr = np.asarray(y).ravel()

    # convert X from DataFrame to numpy array for efficiency
    X_np = X.to_numpy()

    # We will store per-trial results in these lists
    trial_params = []
    trial_mean_scores = {f"mean_test_{name}": [] for name in scoring.keys()}
    trial_numbers = []

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)

    # objective function that computes cross-validated metrics and returns the refit metric
    def old_objective(trial):
        params = param_space_func(trial)
        model = model_builder(params)

        # cross-validate on provided CV splitter; cross_validate returns dict of test_* arrays
        try:
            cv_result = cross_validate(
                model,
                X,
                y_arr,
                groups=groups,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
                n_jobs=1  # avoid nested parallelization inside optuna trial (we parallelize models across processes)
            )
        except Exception as e:
            # if fitting fails for given params, return a very poor score
            print(f"Trial failed with error: {e}")
            # append NaNs to trial lists (filled later)
            for name in scoring.keys():
                trial_mean_scores[f"mean_test_{name}"].append(np.nan)
            trial_params.append(params)
            trial_numbers.append(trial.number)
            return float("-inf") if direction == "maximize" else float("inf")

        # record params and mean scores for each metric
        for name in scoring.keys():
            key = f"test_{name}"
            if key in cv_result:
                mean_val = np.mean(cv_result[key])
            else:
                mean_val = np.nan
            trial_mean_scores[f"mean_test_{name}"].append(mean_val)

        trial_params.append(params)
        trial_numbers.append(trial.number)

        # return the metric we want to optimize
        return float(np.mean(cv_result[f"test_{refit_metric}"]))

    def objective(trial):

        start = time.time()

        params = param_space_func(trial)
        model = model_builder(params)

        scores = {name: [] for name in scoring.keys()}

        # Manual CV loop instead of cross_validate
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X_np, y_arr, groups)):
            X_train_np, X_valid_np = X_np[train_idx], X_np[valid_idx]
            y_train, y_valid = y_arr[train_idx], y_arr[valid_idx]

            # LightGBM and CatBoost require feature names
            if model_name in ["LightGBM", "CatBoost"]:
                X_train = pd.DataFrame(X_train_np, columns=X.columns)
                X_valid = pd.DataFrame(X_valid_np, columns=X.columns)
            else:
                # XGBoost + RF accept numpy arrays, keep fast mode
                X_train = X_train_np
                X_valid = X_valid_np

            # DEBUG PRINTS
            #print(f"\n===== Trial {trial.number} | Fold {fold} =====")
            #print("Train labels:", np.unique(y_train, return_counts=True))
            #print("Valid labels:", np.unique(y_valid, return_counts=True))

            if time.time() - start > MAX_TRIAL_SECONDS:
                raise optuna.exceptions.TrialPruned()

            try:
                m = model_builder(params)
                m.fit(X_train, y_train)

                # compute all requested metrics
                y_pred_proba = m.predict_proba(X_valid)[:, 1]
                y_pred = m.predict(X_valid)

                for name, scorer in scoring.items():
                    if "roc_auc" in name:
                        score = scorer._score_func(y_valid, y_pred_proba, **scorer._kwargs)
                    else:
                        score = scorer._score_func(y_valid, y_pred, **scorer._kwargs)

                    scores[name].append(score)

            except Exception as e:
                print(f"Trial {trial.number} failed in fold {fold}: {e}")
                for name in scoring.keys():
                    scores[name].append(np.nan)
                break

        # average metrics
        for name in scoring.keys():
            trial_mean_scores[f"mean_test_{name}"].append(np.nanmean(scores[name]))

        trial_params.append(params)
        trial_numbers.append(trial.number)

        # return metric we optimize
        return float(np.nanmean(scores[refit_metric]))

    # run the study
    print(model_name)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=1)

    # construct cv_results similar to GridSearchCV.cv_results_
    cv_results = {"params": [], "number": []}
    for name in scoring.keys():
        cv_results[f"mean_test_{name}"] = []

    # fill from recorded lists (Optuna does not guarantee trial ordering in study.trials matches our append sequence; we use study.trials ordered by number)
    ordered_trials = sorted(study.trials, key=lambda t: t.number)
    # Build a map from trial.number to index in our recorded lists (trial_numbers)
    num_to_idx = {num: idx for idx, num in enumerate(trial_numbers)}

    for t in ordered_trials:
        idx = num_to_idx.get(t.number, None)
        if idx is None:
            # safety: if missing, append None/NaN placeholders
            cv_results["params"].append(t.params)
            for name in scoring.keys():
                cv_results[f"mean_test_{name}"].append(np.nan)
            cv_results["number"].append(t.number)
        else:
            cv_results["params"].append(trial_params[idx])
            for name in scoring.keys():
                cv_results[f"mean_test_{name}"].append(trial_mean_scores[f"mean_test_{name}"][idx])
            cv_results["number"].append(trial_numbers[idx])

    # Get best params according to study
    best_params = study.best_params if study.best_params is not None else (cv_results["params"][0] if cv_results["params"] else {})
    best_value = study.best_value if study.best_value is not None else np.nan

    # Refit best estimator on the full provided (X, y)
    try:
        clf_best = model_builder(best_params)
        clf_best.fit(X, y_arr)
    except Exception as e:
        print(f"Refit of best params failed with error: {e}. Returning un-fitted estimator.")
        # if refit fails, return an unfitted estimator object
        clf_best = model_builder(best_params)

    return clf_best, cv_results


def plot_test_results(df_test_results):
    # Group by Model and calculate median and min/max
    summary = df_test_results.groupby('Model').agg(['mean', 'std']).reset_index()
    summary.columns = ['Model'] + [f"{metric}_{stat}" for metric, stat in summary.columns[1:]]

    # Prepare data for plotting
    models = summary['Model']
    metrics = ['AUC', 'Accuracy', 'Recall', 'Precision', 'F1']

    # Initialize a bar plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define bar width and positions
    bar_width = 0.15
    x = np.arange(len(models))

    # Create a color array using the viridis colormap
    cmap = cm.get_cmap('viridis', len(metrics))

    # Plot each metric
    for i, metric in enumerate(metrics):
        mean_values = summary[f"{metric}_mean"]
        std_values = summary[f"{metric}_std"]

        # Define bar positions
        bar_positions = x + i * bar_width

        # Plot bars for the median values
        ax.bar(bar_positions, mean_values, width=bar_width,
               label=metric, color=cmap(i))

        # Add error bars for min/max
        ax.errorbar(bar_positions, mean_values,
                    yerr= std_values, #[median_values - min_values, max_values - median_values],
                    fmt='none', color='black', capsize=5)

    # Configure axes and labels
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Model Performance Metrics', fontsize=14)
    ax.set_xticks(x + bar_width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(models)
    ax.legend(title='Metrics')

    # Show plot
    plt.tight_layout()
    #plt.show()
    return plt.gcf()

def save_df_results_to_hd(target_path, df_train_results, df_test_results, X_train_list, y_train_list, X_test_list,
                          y_test_list):
    os.makedirs(target_path, exist_ok=True)

    # save lists as pickles format
    full_path = os.path.join(target_path, 'X_train_list.pkl')
    with open(full_path, 'wb') as f:
        pickle.dump(X_train_list, f)

    full_path = os.path.join(target_path, 'y_train_list.pkl')
    with open(full_path, 'wb') as f:
        pickle.dump(y_train_list, f)

    full_path = os.path.join(target_path, 'X_test_list.pkl')
    with open(full_path, 'wb') as f:
        pickle.dump(X_test_list, f)

    full_path = os.path.join(target_path, 'y_test_list.pkl')
    with open(full_path, 'wb') as f:
        pickle.dump(y_test_list, f)

    # save dataframe
    full_path = os.path.join(target_path, 'df_train_results.pkl')
    # df_results.to_csv(full_path, index=False)
    print("Saving: " + full_path)
    df_train_results.to_pickle(full_path)

    full_path = os.path.join(target_path, 'df_test_results.pkl')
    df_test_results.to_pickle(full_path)

def load_df_results_from_hd(ml_result_path):

    full_path = os.path.join(ml_result_path, 'X_train_list.pkl')
    with open(full_path, 'rb') as f:
        X_train_list = pickle.load(f)

    full_path = os.path.join(ml_result_path, 'y_train_list.pkl')
    with open(full_path, 'rb') as f:
        y_train_list = pickle.load(f)

    full_path = os.path.join(ml_result_path, 'X_test_list.pkl')
    with open(full_path, 'rb') as f:
        X_test_list = pickle.load(f)

    full_path = os.path.join(ml_result_path, 'y_test_list.pkl')
    with open(full_path, 'rb') as f:
        y_test_list = pickle.load(f)

    full_path_pkl = os.path.join(ml_result_path, 'df_train_results.pkl')
    df_train_results = pd.read_pickle(full_path_pkl)

    full_path_pkl = os.path.join(ml_result_path, 'df_test_results.pkl')
    df_test_results = pd.read_pickle(full_path_pkl)

    return df_train_results, df_test_results, X_train_list, y_train_list, X_test_list, y_test_list

###############################


def plot_and_analyze_df_results(df_results, X_test, y_test):
    fig_train = plot_results_train(df_results)
    fig_test = plot_results_test(df_results, X_test, y_test)
    return fig_train, fig_test


def print_result_in_terminal(df_results, X_test, y_test):
    # Print the results
    print('Train set performance')
    print(df_results)

    # Test set performance for the best model
    for model, best_model in zip(df_results['Model'], df_results['clf_best']):
        # best_model = df_results.loc[df_results['AUC'].idxmax()]['clf_best']
        # y_test1 = y_test.to_numpy()
        #  y_test1 = y_test1.reshape(len(y_test1),)
        n_classes = len(np.unique(y_test))
        y_test1 = label_binarize(y_test, classes=np.arange(n_classes))

        print(model)
        y_pred_test = best_model.predict(X_test)
        y_pred1 = label_binarize(y_pred_test, classes=np.arange(n_classes))
        print('Test set performance')
        print('AUC test:', roc_auc_score(y_test, y_pred_test))
        print('Accuracy test:', accuracy_score(y_test, y_pred_test))
        print('F1 score test:', f1_score(y_test, y_pred_test, average="macro", pos_label=0))
        print('Recall test:', recall_score(y_test, y_pred_test, average="macro", pos_label=0))
        print('Precision test:', precision_score(y_test, y_pred_test, average="macro", pos_label=0))

        print(classification_report(y_test, y_pred_test))


def plot_results_train(df_results):
    # Your existing code to load data and set variables...

    sns.set_palette("viridis")

    models = df_results['Model']
    metrics = ['AUC', 'Accuracy', 'Recall', 'Precision']
    error_metrics = ['AUC std', 'Accuracy std', 'Recall std', 'Precision std']

    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust the figure size as needed
    ax.set_xlabel("ML algorithms")
    ax.set_ylabel("Values")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    bar_width = 0.15
    positions = np.arange(len(models))

    for i, metric in enumerate(metrics):
        values = df_results[metric]
        errors = df_results[error_metrics[i]]
        # Adjusting x positions to center the bars for each group
        ax.bar(positions + i * bar_width - 0.3, values, bar_width, label=metric, yerr=errors, alpha=0.8, bottom=0)

    ax.set_xticks(np.arange(len(models)) + ((len(metrics) - 1) / 2 + 0.5) * bar_width)
    ax.set_xticklabels(models)
    ax.legend()

    # Increase font size for axis labels
    ax.set_xlabel("ML algorithms", fontsize=16)
    ax.set_ylabel("Values", fontsize=16)

    # Increase font size for tick labels on both axes
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # Increase font size for legend
    ax.legend(fontsize=12)

    # Adjust x-axis limits to include the entire graph
    ax.set_xlim(-0.5, len(models) + 0.5)
    # Align the minor tick label
    for label in ax.get_xticklabels(minor=True):
        label.set_horizontalalignment('center')
    # Set y-axis limits
    ax.set_ylim(0, 1.1)

    # Adjust legend position
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)

    plt.tight_layout()
    # plt.savefig('plot-train-sync.pdf', bbox_inches='tight', dpi=3000)
    # plt.show()

    return fig


def calculate_test_metrics(df_results, X_test, y_test):
    list_auc_test = []
    list_accuracy_test = []
    list_precision_test = []
    list_recall_test = []
    for model, best_model in zip(df_results['Model'], df_results['clf_best']):
        # best_model = df_results.loc[df_results['AUC'].idxmax()]['clf_best']
        # y_test1 = y_test.to_numpy()
        #  y_test1 = y_test1.reshape(len(y_test1),)
        n_classes = len(np.unique(y_test))
        # MC: y_test1 = label_binarize(y_test, classes=np.arange(n_classes))

        print(model)
        y_pred_test = best_model.predict(X_test)
        # MC: y_pred1 = label_binarize(y_pred_test, classes=np.arange(n_classes))
        print('Test set performance')
        print('AUC test:', roc_auc_score(y_test, y_pred_test))
        list_auc_test.append(roc_auc_score(y_test, y_pred_test))
        print('Accuracy test:', accuracy_score(y_test, y_pred_test))
        list_accuracy_test.append(accuracy_score(y_test, y_pred_test))

        print('F1 score test:', f1_score(y_test, y_pred_test, average="macro", pos_label=0))
        print('Recall test:', recall_score(y_test, y_pred_test, average="macro", pos_label=0))
        list_recall_test.append(recall_score(y_test, y_pred_test, average="macro", pos_label=0))

        print('Precision test:', precision_score(y_test, y_pred_test, average="macro", pos_label=0))
        list_precision_test.append(precision_score(y_test, y_pred_test, average="macro", pos_label=0))
        print(classification_report(y_test, y_pred_test))

    return list_auc_test, list_accuracy_test, list_recall_test, list_precision_test


def plot_results_test(df_results, X_test, y_test):
    list_auc_test, list_accuracy_test, list_recall_test, list_precision_test = calculate_test_metrics(df_results,
                                                                                                      X_test, y_test)

    fig = call_plot_test_metrics(df_results, list_auc_test, list_accuracy_test, list_recall_test, list_precision_test)
    return fig


def call_plot_test_metrics(df_results, list_auc_test, list_accuracy_test, list_recall_test, list_precision_test):
    # Plotting test metrics
    test_metrics = ['AUC', 'Accuracy', 'Recall', 'Precision']
    test_values = [list_auc_test, list_accuracy_test, list_recall_test, list_precision_test]

    fig, ax = plot_test_metrics(df_results, test_values, test_metrics)
    # plt.savefig('plot-test-sync.pdf', bbox_inches='tight', dpi=3000)
    # plt.show()
    return fig


def plot_test_metrics(df_results, test_values, test_metrics):
    models = df_results['Model']
    metrics = ['AUC', 'Accuracy', 'Recall', 'Precision']
    error_metrics = ['AUC std', 'Accuracy std', 'Recall std', 'Precision std']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("ML algorithms")
    ax.set_ylabel("Values")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    bar_width = 0.15
    positions = np.arange(len(models))

    for i, metric in enumerate(metrics):
        values = df_results[metric]
        errors = df_results[error_metrics[i]]
        ax.bar(positions + i * bar_width, values, bar_width, label=metric, alpha=0.8, bottom=0)
        #    ax.bar(positions + (len(metrics) + i) * bar_width, bar_width, label=metric, yerr=errors, alpha=0.8, bottom=0)
        # ax.set_xticks(positions + bar_width * (len(metrics) - 1) / 2)
        ax.set_xticks(np.arange(len(models)) + ((len(test_metrics) - 1) / 2 + 0.5) * bar_width)

    # Align the minor tick label
    for label in ax.get_xticklabels(minor=True):
        label.set_horizontalalignment('center')
    ax.set_xticklabels(models)
    ax.legend()
    # ax.set_xticks(positions + bar_width * ((len(metrics) - 1) / 2 + len(test_metrics)))
    # ax.set_xticklabels(models)
    # After setting x-axis ticks and labels

    # ax.legend()

    # Rest of the formatting - font sizes, limits, etc.
    ax.set_xlabel("ML algorithms", fontsize=16)
    ax.set_ylabel("Values", fontsize=16)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=12)
    # for label in ax.get_xticklabels(minor=True):
    #    label.set_horizontalalignment('center')
    ax.set_xticks(np.arange(len(models)) + ((len(test_metrics) - 1) / 2) * bar_width)
    ax.set_xticklabels(models)

    # ax.set_xlim(-bar_width, len(models) - 1 + (len(metrics) + len(test_metrics) - 1) * bar_width)
    ax.set_ylim(0, 1.1)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
    plt.tight_layout()

    return fig, ax
