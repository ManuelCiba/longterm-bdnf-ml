# This is a script to preprocess feature sets to make them suitable for machine learning analysis.
# (scaling, normalization, one-hot encoding ect.)
import os
import settings
from lib.data_handler import folder_structure
from lib.data_handler import hd
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np


def preprocess_df_features_align_files(df_features):
    
    # get number of x-ticks (e.g. condition1: 0µM, condition2: 20µM ect.)
    n_ticks = len(settings.X_TICKS)
    
    # get number of windows per file (a recording of a sample was split into multiple time windows)
    n_windows = (
        df_features['file']
        .str.extract(r'window(\d+)\.csv')[0]
        .astype(int)
        .max()
        + 1
    )

    # get labels for the x-ticks
    labels = settings.X_TICK_LABELS
    
    # assign condition labels to each row. This only works if df_features is sorted!
    conditions = []
    condition_idx = 0
    for i, filename in enumerate(df_features['file']):
        # Advance to the next timepoint whenever a new window00 starts
        if i > 0 and filename == 'window00.csv':
            condition_idx += 1

        conditions.append(labels[condition_idx % len(labels)])
    
    df_features['condition'] = conditions

    n_per_x = (df_features.groupby("condition").size().reset_index(name="n_recordings"))

    if settings.FLAG_RELATIVE:
        metadata_cols = ["chip", "condition", "y", "file", "recording", "group"]
        feature_cols = [c for c in df_features.columns if c not in metadata_cols]
        df_features_norm = df_features.copy()

        baseline_means = (
            df_features[df_features["condition"] == "0"]
            .groupby("chip")[feature_cols]
            .mean()
        )

        for chip, baseline in baseline_means.iterrows():
            mask = df_features_norm["chip"] == chip
            df_features_norm.loc[mask, feature_cols] = (
                df_features_norm.loc[mask, feature_cols] / baseline
            )

        df_features = df_features_norm

    return df_features, n_per_x

def preprocess_df_features_align_timestamps(df_features):
    # calculate the difference between features after DRUG application and before DRUG application
    # this is done by subtracting the mean of the features before DRUG application from the raw features after DRUG application
    
    # `settings.DATE_TIME_DRUG_TREATMENT_START` contains the start date/time as a string
    DATE_TIME_DRUG_TREATMENT_STARTED = pd.to_datetime(settings.DATE_TIME_DRUG_TREATMENT_STARTED, format="%Y%m%d_%H%M%S")
    DATE_TIME_DRUG_TREATMENT_FINISHED = pd.to_datetime(settings.DATE_TIME_DRUG_TREATMENT_FINISHED, format="%Y%m%d_%H%M%S")
    
    # Ensure the 'recording' column is in datetime format
    df_features['recording'] = pd.to_datetime(df_features['recording'], format="%Y%m%d_%H%M%S")

    # (re)calculate the days after treatment for each recording (relative to the DRUG treatment finish date)
    #df_features["days_after_treatment"] = (
    #    df_features["recording"].dt.normalize()
    #    - DATE_TIME_DRUG_TREATMENT_FINISHED.normalize()).dt.days

    df_features["hours_after_washout"] = (
        df_features["recording"] - DATE_TIME_DRUG_TREATMENT_FINISHED).dt.total_seconds() / 3600

    df_features["days_after_treatment"] = np.floor(df_features["hours_after_washout"] / 24).astype(int)
    df_features = df_features.drop(columns=['hours_after_washout'])  # drop the intermediate column

    # If the baseline correction is needed, set to True
    if False:
        # drop the daytime column and insert later again after the calculation
        days_after_treatment = df_features.pop('days_after_treatment')

        # Filter rows before DRUG application
        baseline_rows = df_features[df_features['recording'] < DATE_TIME_DRUG_TREATMENT_STARTED]

        # Select columns containing "Spike-contrast" or "s_"
        spike_contrast_columns = [col for col in df_features.columns if "Spike-contrast" in col or "s_" in col]

        # Calculate the mean baseline curve (mean of all rows before DRUG application)
        mean_baseline_curve = baseline_rows[spike_contrast_columns].mean()

        # Subtract the baseline from all rows after the DRUG application
        df_features.loc[df_features['recording'] >= DATE_TIME_DRUG_TREATMENT_STARTED, spike_contrast_columns] -= mean_baseline_curve

        # Reinsert the days_after_treatment column
        df_features = pd.concat([days_after_treatment, df_features], axis=1)

    # drop the baseline rows to only keep post-treatment data
    #df_features = df_features[df_features['recording'] >= DATE_TIME_DRUG_TREATMENT_FINISHED]

    # get n per day
    n_per_day = (df_features.groupby("days_after_treatment").size().reset_index(name="n_recordings"))

    # remove excluded days
    if settings.DATE_TIME_LAST_RECORDING != "":
        print(f"Removing recordings after {settings.DATE_TIME_LAST_RECORDING} ...")
        df_features = _remove_excluded_days(df_features)

    
    return df_features, n_per_day

# currently not used, but can be used to only use acute data for machine learning
def preprocess_df_features_acute(df_features):
    # calculate the difference between features after DRUG application and before DRUG application
    # this is done by subtracting the mean of the features before DRUG application from the raw features after DRUG application
    
    # `settings.DATE_TIME_DRUG_TREATMENT_START` contains the start date/time as a string
    DATE_TIME_DRUG_TREATMENT_STARTED = pd.to_datetime(settings.DATE_TIME_DRUG_TREATMENT_STARTED, format="%Y%m%d_%H%M%S")
    DATE_TIME_DRUG_TREATMENT_FINISHED = pd.to_datetime(settings.DATE_TIME_DRUG_TREATMENT_FINISHED, format="%Y%m%d_%H%M%S")
    
    # Ensure the 'recording' column is in datetime format
    df_features['recording'] = pd.to_datetime(df_features['recording'], format="%Y%m%d_%H%M%S")

    # drop the daytime column and insert later again after the calculation
    days_after_treatment = df_features.pop('days_after_treatment')

    # Filter rows before DRUG application
    baseline_rows = df_features[df_features['recording'] < DATE_TIME_DRUG_TREATMENT_STARTED]

    # Select columns containing "Spike-contrast" or "s_"
    spike_contrast_columns = [col for col in df_features.columns if "Spike-contrast" in col or "s_" in col]

    # Calculate the mean baseline curve (mean of all rows before DRUG application)
    mean_baseline_curve = baseline_rows[spike_contrast_columns].mean()

    # Subtract the baseline from all rows after the DRUG application
    df_features.loc[df_features['recording'] >= DATE_TIME_DRUG_TREATMENT_STARTED, spike_contrast_columns] -= mean_baseline_curve

    # Reinsert the days_after_treatment column
    df_features = pd.concat([days_after_treatment, df_features], axis=1)

    # only keep acute data
    df_features = df_features[df_features['recording'] <= DATE_TIME_DRUG_TREATMENT_FINISHED]
    df_features = df_features[df_features['recording'] >= DATE_TIME_DRUG_TREATMENT_STARTED]

    # drop columns that are not needed for the machine learning
    df_features = df_features.drop(columns=['group', 'recording', 'file']) 
    # drop colums that contain any NaN values
    df_features = df_features.dropna(axis=1) 

    # Drop columns where all values are zeros
    df_features = df_features.loc[:, ~(df_features == 0).all()]

    return df_features

def _remove_excluded_days(df):
    # Convert 'recording' column to datetime
    rec = pd.to_datetime(df['recording'], format="%Y%m%d_%H%M%S")

    # Convert the last recording date from settings to datetime
    DATE_TIME_LAST_RECORDING = pd.to_datetime(settings.DATE_TIME_LAST_RECORDING, format="%Y%m%d_%H%M%S")

    # Filter the DataFrame to include only rows before or on the last recording date
    df_filtered = df[rec <= DATE_TIME_LAST_RECORDING]

    return df_filtered


if __name__ == '__main__':

    # define parameter
    bin_sizes = settings.BIN_SIZES
    window_sizes = settings.WINDOW_SIZES
    window_overlaps = settings.WINDOW_OVERLAPS
    chip_names = settings.WELLS_SHAM + settings.WELLS_DRUG  # Combine control and DRUG wells

    path_experiment_list = folder_structure.generate_paths(target_data_folder="PATH",
                                                     bin_sizes=bin_sizes,
                                                     window_sizes=window_sizes,
                                                     window_overlaps=window_overlaps,
                                                     chip_names=[],
                                                     groups=[])


    for path_experiment in path_experiment_list:

        for feature_set in settings.FEATURE_SET_LIST:

            source_path = path_experiment.replace("PATH", feature_set)
            target_path = source_path  # save to the same path but different filename   
            # load the feature set DataFrame
            full_source_path = os.path.join(source_path, 'feature_set_raw.csv')
            df_feature_set = hd.load_csv_as_df(full_source_path)
            # preprocess 

            if settings.FLAG_ACUTE_MODE == 0:
                df_feature_preprocessed, n_per_day = preprocess_df_features_align_timestamps(df_feature_set)
            else:
                df_feature_preprocessed, n_per_day = preprocess_df_features_align_files(df_feature_set)

            # save the preprocessed feature set
            full_target_path = os.path.join(target_path, 'feature_set_preprocessed.csv')
            hd.save_df_as_csv(df_feature_preprocessed, full_target_path)
            # save the number of samples per day
            full_n_per_day_path = os.path.join(target_path, 'n_per_day.csv')
            hd.save_df_as_csv(n_per_day,full_n_per_day_path)