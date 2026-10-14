import os
from lib.data_handler import folder_structure
from lib.data_handler import hd
import settings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if settings.WELLS_SHAM == []:
    from lib.machine_learning import workflow_paired_optatune as workflow_ml
else:
    from lib.machine_learning import workflow_unpaired_optatune as workflow_ml


def find_best_result(SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER):

    base_folder = os.path.join(settings.PATH_RESULTS_FOLDER, SOURCE_DATA_FOLDER)

    # init empty list to fill with results
    list_results_test = []

    bin_sizes = settings.BIN_SIZES
    window_sizes = settings.WINDOW_SIZES
    window_overlaps = settings.WINDOW_OVERLAPS

    for bin_size in bin_sizes:
            for window_size in window_sizes:
                for window_overlap in window_overlaps:

                    # define path where df_results are saved
                    path_experiment = os.path.join(base_folder,
                                                    f"bin_{bin_size}",
                                                    f"window_{window_size}",
                                                    f"overlap_{window_overlap}")



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

                            df_train_results, df_test_results, X_train_list, y_train_list, X_test_list, y_test_list, chip_train_list, y_pred_all, y_true_all = workflow_ml.load_df_results_from_hd(target_path)

                            df_results = df_test_results

                            # fill in table
                            list_results_test.append({  'Day': day,
                                                        'Bin size': int(bin_size),
                                                        'Window size': int(window_size),
                                                        'Window overlap': window_overlap,
                                                        'ML model': model,
                                                        'AUC': df_results['AUC'].iloc[0],
                                                        'AUC_CI_lower': df_results['AUC_CI_lower'].iloc[0],
                                                        'AUC_CI_upper': df_results['AUC_CI_upper'].iloc[0],
                                                        'Accuracy': df_results['Accuracy'].iloc[0],
                                                        'Recall': df_results['Recall'].iloc[0],
                                                        'Precision': df_results['Precision'].iloc[0],
                                                        'F1': df_results['F1'].iloc[0],
                                                        })

    # make DataFrame and sort
    df_results_test = pd.DataFrame(list_results_test)
    df_sorted = df_results_test.sort_values(by='AUC_CI_lower', ascending=False)

    # save dataframe as csv
    full_path = os.path.join(base_folder, "results_test.csv")
    hd.save_df_as_csv(df_sorted, full_path)


if __name__ == '__main__':

    models = settings.ML_MODELS

     # for all machine learning folder:
    for FOLDER in settings.FEATURE_SET_LIST:

        print(FOLDER)

        SOURCE_DATA_FOLDER = FOLDER.replace("0_feature_set", "1_ML")

        find_best_result(SOURCE_DATA_FOLDER, SOURCE_DATA_FOLDER)