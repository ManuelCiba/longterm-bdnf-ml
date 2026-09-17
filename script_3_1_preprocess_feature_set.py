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

    # get labels for the x-ticks
    labels = settings.X_TICK_LABELS
    ticks = settings.X_TICKS

    # Keep only the recordings specified in ticks for each chip. 
    # recording is in YYYYMMDD_HHMMSS format, so the existing sort order is chronological. 
    recording_idx = ( df_features[["chip", "recording"]] .drop_duplicates() .groupby("chip") .cumcount() ) 
    recordings_to_keep = ( df_features[["chip", "recording"]] .drop_duplicates() .loc[recording_idx.isin(ticks)] ) 
    df_features = df_features.merge( recordings_to_keep, on=["chip", "recording"], how="inner" )
    
    # assign condition labels to each row. This only works if df_features is sorted!
    x_tick_labels = []
    x_tick_labels_idx = 0
    x_ticks = []
    x_ticks_idx = 0
    for i, filename in enumerate(df_features['file']):
        # Advance to the next timepoint whenever a new window00 starts
        if i > 0 and filename == 'window00.csv':
            x_tick_labels_idx += 1
            x_ticks_idx += 1

        x_tick_labels.append(labels[x_tick_labels_idx % len(labels)])
        x_ticks.append(ticks[x_ticks_idx % len(ticks)])
    
    df_features["x_ticks"] = x_ticks
    df_features["x_tick_labels"] = x_tick_labels
    n_per_x = (df_features.groupby("x_ticks").size().reset_index(name="n_recordings"))

    if settings.FLAG_RELATIVE:
        metadata_cols = settings.METADATA_COLS
        feature_cols = [c for c in df_features.columns if c not in metadata_cols]
        df_features_norm = df_features.copy()

        baseline_means = (
            df_features[df_features["x_ticks"] == settings.RELATIVE_BASELINE_X_TICK]
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

    # calculate the days after treatment for each recording (relative to the DRUG treatment finish date)
    df_features["hours_after_washout"] = (
        df_features["recording"] - DATE_TIME_DRUG_TREATMENT_FINISHED).dt.total_seconds() / 3600

    
    #df_features["x_tick_labels"] = (
    #    df_features[settings.X] * settings.TIME_WINDOW_HOURS
    #).astype(str) + "-" + (
    #    (df_features[settings.X] + 1) * settings.TIME_WINDOW_HOURS
    #).astype(str) 
    df_features["x_tick_labels"] = np.floor(df_features["hours_after_washout"] / settings.TIME_WINDOW_HOURS).astype(int)
    # TODO Change x_ticks to be from 0 ... n
    df_features["x_ticks"] = np.floor(df_features["hours_after_washout"] / settings.TIME_WINDOW_HOURS).astype(int)
        

    df_features = df_features.drop(columns=['hours_after_washout'])  # drop the intermediate column
    

    if settings.FLAG_RELATIVE:
        df_features = _normalize_features(df_features)

    if False:  # OLD normalization code
        DATE_TIME_NORMALIZATION = pd.to_datetime(settings.DATE_TIME_NORMALIZATION, format="%Y%m%d_%H%M%S")
            
        metadata_cols = settings.METADATA_COLS
        feature_cols = [c for c in df_features.columns if c not in metadata_cols]
        df_features_norm = df_features.copy()

        baseline_means = (
            df_features[df_features["recording"] == DATE_TIME_NORMALIZATION]
            .groupby("chip")[feature_cols]
            .mean()
        )

        for chip, baseline in baseline_means.iterrows():
            mask = df_features_norm["chip"] == chip
            df_features_norm.loc[mask, feature_cols] = (
                df_features_norm.loc[mask, feature_cols] / baseline
            )

        df_features = df_features_norm

    # OLD normalization code
    if False:
        # drop the daytime column and insert later again after the calculation
        days_after_treatment = df_features.pop(settings.X)

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
    n_per_x = (df_features.groupby("x_ticks").size().reset_index(name="n_recordings"))

    # remove excluded days
    if settings.DATE_TIME_LAST_RECORDING != "":
        print(f"Removing recordings after {settings.DATE_TIME_LAST_RECORDING} ...")
        df_features = _remove_excluded_days(df_features)

    
    return df_features, n_per_x

def _normalize_features(df_features):
    normalization_time = settings.DATE_TIME_NORMALIZATION

    metadata_cols = settings.METADATA_COLS
    feature_cols = [c for c in df_features.columns if c not in metadata_cols]

    df_features_norm = df_features.copy()

    # Ensure recording column is datetime
    df_features["recording"] = pd.to_datetime(
        df_features["recording"],
        format="%Y%m%d_%H%M%S"
    )

    # Support either YYYYMMDD or YYYYMMDD_HHMMSS
    if len(normalization_time) == 8:
        # Use all recordings from that day
        normalization_date = pd.to_datetime(
            normalization_time,
            format="%Y%m%d"
        ).date()

        baseline_mask = (
            df_features["recording"].dt.date == normalization_date
        )

    else:
        # Use exact recording timestamp
        normalization_datetime = pd.to_datetime(
            normalization_time,
            format="%Y%m%d_%H%M%S"
        )

        baseline_mask = (
            df_features["recording"] == normalization_datetime
        )

    baseline_means = (
        df_features.loc[baseline_mask]
        .groupby("chip")[feature_cols]
        .mean()
    )

    if baseline_means.empty:
        raise ValueError(
            f"No recordings found for normalization time '{normalization_time}'"
        )

    for chip, baseline in baseline_means.iterrows():
        mask = df_features_norm["chip"] == chip

        df_features_norm.loc[mask, feature_cols] = (
            df_features_norm.loc[mask, feature_cols]
            .div(baseline)
        )

    df_features = df_features_norm

    return df_features

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
    days_after_treatment = df_features.pop("x_ticks")

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