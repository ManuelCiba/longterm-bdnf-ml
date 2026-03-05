import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
import multiprocessing
import os
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt

from lib.machine_learning import workflow_unpaired_optatune as workflow_unpaired_optatune
from lib.data_handler import folder_structure
from lib.data_handler import hd
import settings


def run_by_day(models, SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER):
    
    # check if unsupported models are in the list
    if 'RF' in models:
        print("ERROR: RF is now allowed in this workflow (NaN values were not filtered out)")  # remove RF model for day by day analysis due to long computation time
        return
    elif 'SVM' in models or 'NB' in models or 'KNN' in models or 'LR' in models or 'MLP' in models:
        print("ERROR: SVM, NB, KNN, LR, MLP are now allowed in this workflow (NaN values were not filtered out and features were not scaled.)")  # remove these models for day by day analysis due to long computation time
        return
    
    # get paths of all experiments
    path_experiment_list = _get_path_experiment_list()

    # for all experiments
    for path_experiment in path_experiment_list:

        # define source path
        full_path = os.path.join(path_experiment, 'feature_set_preprocessed.csv')

        # load the feature dataframes
        df_features = hd.load_csv_as_df(full_path, index_col=False)

        # drop columns that are not needed for the machine learning
        df_features = df_features.drop(columns=['group', 'recording', 'file']) 

        # get days after treatment
        days = df_features['days_after_treatment'].unique()

        # for each model:
        for model in models:
            print("Model: " + model)

            # for each day
            for day in days:

                print("Day: " + str(day))

                # check if already calculated
                source_path = path_experiment
                target_path = source_path.replace(SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER)
                target_path = os.path.join(target_path, model, 'day_' + str(day).zfill(2))
                full_path = os.path.join(target_path, 'df_train_results.pkl')
                if os.path.isfile(full_path):
                    print("Already calculated: " + full_path)
                    continue
                else:
                    print("Calculating: " + full_path)

                # only keep data of the specific day
                df_features_day = df_features[df_features['days_after_treatment'] == day]
                df_features_day = df_features_day.drop(columns=['days_after_treatment'])

                try:
                    df_train_results, df_test_results, X_train_list, y_train_list, X_test_list, y_test_list, chip_train_list, y_pred_all, y_true_all = workflow_unpaired_optatune.call(model, df_features_day)
                except Exception as e:
                    print("Error in ML workflow for day " + str(day) + ": " + str(e))
                    continue
                workflow_unpaired_optatune.save_df_results_to_hd(target_path,
                                        df_train_results,
                                        df_test_results,
                                        X_train_list,
                                        y_train_list,
                                        X_test_list,
                                        y_test_list,
                                        chip_train_list,
                                        y_pred_all,
                                        y_true_all)

            print("Calculation finished: " + full_path)


def calculate_ci_and_plot_results(models, SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER):
    # get paths of all experiments
    path_experiment_list = _get_path_experiment_list()

    # for all experiments
    for path_experiment in path_experiment_list:

        source_path = path_experiment.replace(SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER)
        # get all folder names in the source path
        models = [f for f in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, f))]
        
        for model in models:
            
            print("Model: " + model)

            model_path = os.path.join(source_path, model)
            days = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]
        
            for day in days:

                print("Day: " + str(day))

                target_path = os.path.join(model_path, day)

                df_train_results, df_test_results, X_train_list, y_train_list, X_test_list, y_test_list, chip_train_list, y_pred_all, y_true_all = workflow_unpaired_optatune.load_df_results_from_hd(target_path)
                
                # optionally recalculate test results (e.g. if code changed)
                if df_test_results is None:
                    df_test_results = workflow_unpaired_optatune.get_test_results(model, y_true_all, y_pred_all)

                # save figs
                fig_test_path = os.path.join(target_path, "fig_test.pdf")
                fig_train_path = os.path.join(target_path, "fig_train.pdf")
                fig_test = workflow_unpaired_optatune.plot_test_results(df_test_results)
                hd.save_figure(fig_test, fig_test_path, verbose=False)
                #hd.save_figure(fig_train, fig_train_path)

                # calculate ci
                mean, lower, upper = workflow_unpaired_optatune.calculate_confidence_interval_for_auc(y_true_all, y_pred_all, n_bootstraps=2000, alpha=0.95, seed=1234)
                df_test_results['AUC_CI_lower'] = lower
                df_test_results['AUC_CI_upper'] = upper
                df_test_results['AUC_CI_mean'] = mean
                workflow_unpaired_optatune.save_df_results_to_hd(target_path, df_train_results, df_test_results, X_train_list, y_train_list, X_test_list,
                          y_test_list, chip_train_list, y_pred_all, y_true_all, verbose=False)
                print(f"Confidence interval for AUC: mean={mean:.3f}, lower={lower:.3f}, upper={upper:.3f}")
                
def _get_path_experiment_list():
    # define parameter
    bin_sizes = settings.BIN_SIZES
    window_sizes = settings.WINDOW_SIZES
    window_overlaps = settings.WINDOW_OVERLAPS

    path_experiment_list = folder_structure.generate_paths(target_data_folder=SOURCE_DATA_FOLDER,
                                                           bin_sizes=bin_sizes,
                                                           window_sizes=window_sizes,
                                                           window_overlaps=window_overlaps,
                                                           chip_names=[],
                                                           groups=[])

    return path_experiment_list


if __name__ == '__main__':

    print("Starting machine learning...")

    warnings.filterwarnings("ignore")

    for SOURCE_DATA_FOLDER in settings.FEATURE_SET_LIST:

        # define folder name to save results
        TARGET_DATA_FOLDER = SOURCE_DATA_FOLDER.replace("0_feature_set", "1_ML")

        # do the machine learning
        models = settings.ML_MODELS
        run_by_day(models,SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER)
        calculate_ci_and_plot_results(models, SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER)