# This is a script to preprocess feature sets to make them suitable for machine learning analysis.
# (scaling, normalization, one-hot encoding ect.)
import os
import re
import settings
from lib.data_handler import folder_structure
from lib.data_handler import hd
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
import glob





def preprocess_df_features_align_files(df_features):
    
    # get number of x-ticks (e.g. condition1: 0µM, condition2: 20µM ect.)
    n_ticks = len(settings.X_TICKS)

    # get labels for the x-ticks
    labels = settings.X_TICK_LABELS
    ticks = settings.X_TICKS

    # ---------------------------------------------------------
    # Add dummy recordings
    # ---------------------------------------------------------
    df_features = _add_dummy_recordings(df_features)

    # ---------------------------------------------------------
    # Keep only the recordings specified in ticks for each chip
    # ---------------------------------------------------------
    recordings = (
        df_features[["chip", "recording"]]
        .drop_duplicates()
    )

    recording_idx = (
        recordings
        .groupby("chip")
        .cumcount()
    )

    recordings_to_keep = recordings.loc[
        recording_idx.isin(ticks)
    ]

    df_features = df_features.merge(
        recordings_to_keep.assign(_keep=True),
        on=["chip", "recording"],
        how="inner"
    ).drop(columns="_keep")

    # ---------------------------------------------------------
    # Assign condition labels to each row
    # ---------------------------------------------------------
    x_tick_labels = []
    x_tick_labels_idx = 0

    x_ticks = []
    x_ticks_idx = 0

    for i, filename in enumerate(df_features["file"]):

        # Advance to the next timepoint whenever a new window00 starts
        if i > 0 and filename == "window00.csv":
            x_tick_labels_idx += 1
            x_ticks_idx += 1

        x_tick_labels.append(
            labels[x_tick_labels_idx % len(labels)]
        )

        x_ticks.append(
            ticks[x_ticks_idx % len(ticks)]
        )

    df_features["x_ticks"] = x_ticks
    df_features["x_tick_labels"] = x_tick_labels

    # ---------------------------------------------------------
    # Remove dummy recordings
    # ---------------------------------------------------------
    df_features = _remove_dummy_recordings(df_features)

    n_per_x = (df_features.groupby("x_ticks").size().reset_index(name="n_recordings"))

    #
    # Do normalization if requested
    #
    if settings.FLAG_RELATIVE:
        metadata_cols = settings.METADATA_COLS
        feature_cols = [c for c in df_features.columns if c not in metadata_cols]
        df_features_norm = df_features.copy()

        baseline_means = (
            df_features[df_features["x_ticks"] == settings.BASELINE_X_TICK]
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


def _add_dummy_recordings(df_features):
    """
    Add rows for dummy recordings to df_features.

    Dummy files must have the format:
        YYYYMMDD_HHMMSS_dummy.mat

    The dummy files themselves are not modified or deleted.
    Their rows are marked with '_is_dummy' so they can be removed later.
    """

    # Get all windows already present in the dataframe
    windows = np.unique(df_features["file"])

    # Mark existing rows as non-dummy
    df_features = df_features.copy()
    df_features["_is_dummy"] = False

    # Collect dummy rows
    dummy_rows = []

    chips = np.unique(df_features["chip"])

    for chip in chips:

        # Define full path of selected chip
        path_chip = os.path.join(
            os.getcwd(),
            settings.PATH_DATA_FOLDER,
            chip
        )

        # Get all dummy .mat files
        files = glob.glob(
            os.path.join(path_chip, "*dummy.mat")
        )

        for dummy_file in files:

            # Dummy filename:
            # YYYYMMDD_HHMMSS_dummy.mat
            recording = os.path.basename(dummy_file)

            # Create one row for every window
            for window in windows:
                dummy_rows.append({
                    "chip": chip,
                    "recording": recording,
                    "file": window,
                    "_is_dummy": True
                })

    # Add dummy rows to dataframe
    if dummy_rows:
        df_dummy = pd.DataFrame(dummy_rows)

        df_features = pd.concat(
            [df_features, df_dummy],
            ignore_index=True
        )

    # Sort chronologically within each chip
    df_features = (
        df_features
        .sort_values(["chip", "recording", "file"])
        .reset_index(drop=True)
    )

    return df_features


def _remove_dummy_recordings(df_features):
    """
    Remove dummy recording rows from df_features.

    The corresponding dummy files on the harddrive are NOT deleted.
    """

    df_features = (
        df_features[df_features["_is_dummy"] == False]
        .drop(columns="_is_dummy")
        .reset_index(drop=True)
    )

    return df_features


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

    # define x_tick_labels based on the hours after treatment, using the specified time window
    df_features["x_tick_labels"] = np.floor(df_features["hours_after_washout"] / settings.TIME_WINDOW_HOURS).astype(int)
    
    # Define corresponding x_ticks to be from 0 ... n
    df_features["x_ticks"] = df_features["x_tick_labels"].rank(method="dense").astype(int) - 1

    # remove the intermediate column 'hours_after_washout' as it is no longer needed
    df_features = df_features.drop(columns=['hours_after_washout'])  # drop the intermediate column
    

    if settings.FLAG_RELATIVE:
        df_features = _normalize_features(df_features)

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

    # round "days_after_treatment" (important especially if group1 and group2 were not recorded at same time but sequentially - in this case ML will learn the days!) 
    #df_feature_set["days_after_treatment"] = df_feature_set["days_after_treatment"].apply(_bin_time)    

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
    elif len(normalization_time) == 15:
        # Use exact recording timestamp
        normalization_datetime = pd.to_datetime(
            normalization_time,
            format="%Y%m%d_%H%M%S"
        )

        baseline_mask = (
            df_features["recording"] == normalization_datetime
        )
    else:
        raise ValueError(
            f"Invalid normalization time format: {normalization_time}. "
            "Expected either YYYYMMDD or YYYYMMDD_HHMMSS."
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


def _remove_excluded_days(df):
    # Convert 'recording' column to datetime
    rec = pd.to_datetime(df['recording'], format="%Y%m%d_%H%M%S")

    # Convert the last recording date from settings to datetime
    DATE_TIME_LAST_RECORDING = pd.to_datetime(settings.DATE_TIME_LAST_RECORDING, format="%Y%m%d_%H%M%S")

    # Filter the DataFrame to include only rows before or on the last recording date
    df_filtered = df[rec <= DATE_TIME_LAST_RECORDING]

    return df_filtered


# currently not used: TODO still necessary for special data sets? (e.g. if group1 and group2 were not recorded at same time but sequentially - in this case ML will learn the days!)
def _bin_time(x):
    # Define custom bins for decimals
    day = int(x)  # whole day
    return day
    #frac = x - day
    #if frac < 0.55:
    #    return day + 0.4 # morning
    #else:
    #    return day + 0.6  # noon


def _transform_to_paired_sample_design(df):
    """
    Transform the feature set to a paired sample design, by copying the baseline features to each condition
    """

    # --------------------------------------------------
    # 1. Keep the original data
    # --------------------------------------------------
    df_original = df.copy()

    # --------------------------------------------------
    # 2. Get all baseline rows
    # --------------------------------------------------
    baseline = df[df["x_ticks"] == settings.BASELINE_X_TICK].copy()

    # Make sure copied baseline rows are class 0
    baseline["y"] = 0
    baseline["group"] = "SHAM"

    
    # --------------------------------------------------
    # 3. Copy baseline for each x_tick (e.g. condition, day ect)
    # --------------------------------------------------
    x_ticks_unique = df["x_ticks"].unique()
    x_tick_labels_unique = df["x_tick_labels"].unique()
    for x_tick, x_tick_label in zip(x_ticks_unique, x_tick_labels_unique):

        baseline_df = baseline.copy()

        # Change baseline x_tick to destination x_tick
        baseline_df["x_ticks"] = x_tick
        baseline_df["x_tick_labels"] = x_tick_label

        # Add copied baseline rows to original dataframe
        df_original = pd.concat(
            [df_original, baseline_df],
            ignore_index=True
        )

    # --------------------------------------------------
    # 5. Sort
    # --------------------------------------------------
    df_paired = df_original.sort_values(
        by=["chip", "recording", "x_ticks"]
    ).reset_index(drop=True)

    return df_paired

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

            # if no SHAM, transform the feature table to a paired sample design
            if settings.WELLS_SHAM == []:
                df_feature_preprocessed = _transform_to_paired_sample_design(df_feature_preprocessed)

            # save the preprocessed feature set
            full_target_path = os.path.join(target_path, 'feature_set_preprocessed.csv')
            hd.save_df_as_csv(df_feature_preprocessed, full_target_path)
            # save the number of samples per day
            full_n_per_day_path = os.path.join(target_path, 'n_per_day.csv')
            hd.save_df_as_csv(n_per_day,full_n_per_day_path)