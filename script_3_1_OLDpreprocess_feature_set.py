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

def _one_hot_encode_days_after_treatment(df):

    # 1) Create a formatted string column with .nf decimal places
    df['day_str'] = df['days_after_treatment'].map(lambda x: f"{x:.0f}") 

    # 2) One-hot encode that string column
    df = pd.get_dummies(
        df,
        columns=['day_str'],
        prefix='day',
        dtype=int
    )

    # 3) (Optional) Drop the original floats if you no longer need them
    df.drop(columns=['days_after_treatment'], inplace=True)

    return df

def preprocess_df_features_longterm(df_features, target_path):
    # calculate the difference between features after DRUG application and before DRUG application
    # this is done by subtracting the mean of the features before DRUG application from the raw features after DRUG application
    
    # `settings.DATE_TIME_DRUG_TREATMENT_START` contains the start date/time as a string
    DATE_TIME_DRUG_TREATMENT_STARTED = pd.to_datetime(settings.DATE_TIME_DRUG_TREATMENT_STARTED, format="%Y%m%d_%H%M%S")
    DATE_TIME_DRUG_TREATMENT_FINISHED = pd.to_datetime(settings.DATE_TIME_DRUG_TREATMENT_FINISHED, format="%Y%m%d_%H%M%S")
    
    # Ensure the 'recording' column is in datetime format
    df_features['recording'] = pd.to_datetime(df_features['recording'], format="%Y%m%d_%H%M%S")

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
    df_features = df_features[df_features['recording'] >= DATE_TIME_DRUG_TREATMENT_FINISHED]

    # drop columns that are not needed for the machine learning
    df_features = df_features.drop(columns=['group', 'recording', 'file']) 
    # drop colums that contain any NaN values
    #df_features = df_features.dropna(axis=1) 
    # drop rows that contain any NaN values
    df_features = df_features.dropna(axis=0)

    # one hot encode the days after treatment
    df_features = _one_hot_encode_days_after_treatment(df_features)

    # scale column "days_after_treatment" between 0 and 1 
    # Initialize the scaler
    scaler = MinMaxScaler()
    # Scale all features except for "chip" and "y" column
    exclude_columns = ["chip", "y"]
    # ... and except for Spike-contrast columns
    exclude_columns += [col for col in df_features.columns if "Spike-contrast" in col]
    # ... and except for day columns
    # DO NOT EXCLUDE ANYMORE AS SCALING HAS NO EFFECT ON 0-1-VALUES AND SCALER COLUMNS SHOULD BE NOT EMPTY
    #exclude_columns += [col for col in df_features.columns if "day" in col]
    # Identify columns to scale
    columns_to_scale = df_features.columns.difference(exclude_columns)
    # Apply scaling to the selected columns
    df_features[columns_to_scale] = scaler.fit_transform(df_features[columns_to_scale])

    # save the data
    # Save the scaler object
    scaler_path = os.path.join(target_path, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump({"scaler": scaler, "columns": columns_to_scale}, f)

    # save the DataFrame
    feature_path = os.path.join(target_path, 'feature_set_processed.csv')
    hd.save_df_as_csv(df_features, feature_path)
    print(f"Saved feature set to {feature_path}")

def preprocess_df_features_longterm_days(df_features, target_path, no_corr=False):
    # calculate the difference between features after DRUG application and before DRUG application
    # this is done by subtracting the mean of the features before DRUG application from the raw features after DRUG application
    
    # `settings.DATE_TIME_DRUG_TREATMENT_START` contains the start date/time as a string
    DATE_TIME_DRUG_TREATMENT_STARTED = pd.to_datetime(settings.DATE_TIME_DRUG_TREATMENT_STARTED, format="%Y%m%d_%H%M%S")
    DATE_TIME_DRUG_TREATMENT_FINISHED = pd.to_datetime(settings.DATE_TIME_DRUG_TREATMENT_FINISHED, format="%Y%m%d_%H%M%S")
    
    # Ensure the 'recording' column is in datetime format
    df_features['recording'] = pd.to_datetime(df_features['recording'], format="%Y%m%d_%H%M%S")

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
    df_features = df_features[df_features['recording'] >= DATE_TIME_DRUG_TREATMENT_FINISHED]

    # drop columns that are not needed for the machine learning
    df_features = df_features.drop(columns=['group', 'recording', 'file']) 
    # drop colums that contain any NaN values
    #df_features = df_features.dropna(axis=1) 
    # drop rows that contain any NaN values
    df_features = df_features.dropna(axis=0)

    # scale column "days_after_treatment" between 0 and 1 
    # Initialize the scaler
    scaler = MinMaxScaler()
    # Scale all features except for "chip" and "y" column
    exclude_columns = ["chip", "y"]
    # ... and except for Spike-contrast columns
    exclude_columns += [col for col in df_features.columns if "Spike-contrast" in col]
    # Identify columns to scale
    columns_to_scale = df_features.columns.difference(exclude_columns)
    # Apply scaling to the selected columns
    df_features[columns_to_scale] = scaler.fit_transform(df_features[columns_to_scale])

    if no_corr:

        # Define the columns you don't want to include in correlation analysis
        exclude_columns = ["chip", "y"]
        feature_columns = df_features.columns.difference(exclude_columns)
        df_feature_only = df_features[feature_columns]

        # Compute correlation matrix
        corr_matrix = df_feature_only.corr().abs()

        # Track visited features
        visited = set()
        to_keep = []

        # Threshold for high correlation
        threshold = 0.95

        for col in corr_matrix.columns:
            if col in visited:
                continue

            # Find features highly correlated with this one (including itself)
            high_corr = corr_matrix.index[corr_matrix[col] > threshold].tolist()

            # Mark all as visited
            visited.update(high_corr)

            # Keep just one representative (e.g., the current col)
            to_keep.append(col)

        # Keep only the selected features + non-feature columns
        df_features = df_features[to_keep + exclude_columns]


    # save the data
    # Save the scaler object
    scaler_path = os.path.join(target_path, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump({"scaler": scaler, "columns": columns_to_scale}, f)

    # save the DataFrame
    feature_path = os.path.join(target_path, 'feature_set_processed.csv')
    hd.save_df_as_csv(df_features, feature_path)
    print(f"Saved feature set to {feature_path}")

def OLD_create_day_interaction_features(df_features):
    """
    Create interaction features between regular feature columns and one-hot encoded day columns.
    
    Parameters
    ----------
    df_features : pd.DataFrame
        DataFrame containing:
          - feature columns (numeric)
          - one-hot encoded day columns (e.g., 'day_1', 'day_30')
          - columns 'chip' and 'y' that should be ignored
    
    Returns
    -------
    df_interactions : pd.DataFrame
        DataFrame containing only the interaction features,
        with column names formatted as '<feature>_x_<day_col>'
    """
    
    # Identify the columns to exclude
    exclude_cols = ['chip', 'y']
    
    # Identify day columns (one-hot encoded)
    day_cols = [col for col in df_features.columns if col.startswith('day_')]
    
    # Identify feature columns (exclude chip, y, and day_ columns)
    feature_cols = [
        col for col in df_features.columns
        if col not in exclude_cols and col not in day_cols
    ]
    
    # Check there’s something to combine
    if not feature_cols or not day_cols:
        raise ValueError("No valid feature or day columns found.")
    
    # --- Create interaction features ---
    interaction_frames = []
    for day_col in day_cols:
        # multiply all feature columns by the current one-hot day column
        tmp = df_features[feature_cols].mul(df_features[day_col], axis=0)
        
        # rename columns for clarity
        tmp.columns = [f"{feat}_{day_col}" for feat in feature_cols]
        
        interaction_frames.append(tmp)
    
    # Combine all interaction DataFrames horizontally
    df_interactions = pd.concat(interaction_frames, axis=1)
    
    return df_interactions

def _create_day_interaction_features(df_features):
    """
    Creates interaction features between one-hot day columns and all other feature columns.
    Keeps 'chip' and 'y' columns unchanged.

    Example:
    If you have columns 'day_1', 'day_2' and 'spikerate', you will get
    new columns like 'spikerate_day_1', 'spikerate_day_2'.
    """

    # Identify relevant columns
    exclude_columns = ['chip', 'y']
    day_columns = [c for c in df_features.columns if c.startswith('day_')]
    feature_columns = [c for c in df_features.columns 
                       if c not in exclude_columns + day_columns]

    interaction_data = {}

    # Compute interaction features
    for f in feature_columns:
        for d in day_columns:
            new_col = f"{f}_{d}"
            interaction_data[new_col] = df_features[f] * df_features[d]

    # Build new DataFrame with interactions + chip + y
    df_interactions = pd.DataFrame(interaction_data, index=df_features.index)
    df_interactions['chip'] = df_features['chip']
    df_interactions['y'] = df_features['y']

    return df_interactions

def preprocess_df_features_longterm_per_day(df_features, target_path):
    # calculate the difference between features after DRUG application and before DRUG application
    # this is done by subtracting the mean of the features before DRUG application from the raw features after DRUG application
    
    # `settings.DATE_TIME_DRUG_TREATMENT_START` contains the start date/time as a string
    DATE_TIME_DRUG_TREATMENT_STARTED = pd.to_datetime(settings.DATE_TIME_DRUG_TREATMENT_STARTED, format="%Y%m%d_%H%M%S")
    DATE_TIME_DRUG_TREATMENT_FINISHED = pd.to_datetime(settings.DATE_TIME_DRUG_TREATMENT_FINISHED, format="%Y%m%d_%H%M%S")
    
    # Ensure the 'recording' column is in datetime format
    df_features['recording'] = pd.to_datetime(df_features['recording'], format="%Y%m%d_%H%M%S")

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
    df_features = df_features[df_features['recording'] >= DATE_TIME_DRUG_TREATMENT_FINISHED]

    # drop columns that are not needed for the machine learning
    df_features = df_features.drop(columns=['group', 'recording', 'file']) 
    # drop colums that contain any NaN values
    #df_features = df_features.dropna(axis=1) 
    # drop rows that contain any NaN values
    df_features = df_features.dropna(axis=0)

    # one hot encode the days after treatment
    df_features = _one_hot_encode_days_after_treatment(df_features)

    # scale column "days_after_treatment" between 0 and 1 
    # Initialize the scaler
    scaler = MinMaxScaler()
    # Scale all features except for "chip" and "y" column
    exclude_columns = ["chip", "y"]
    # ... and except for Spike-contrast columns
    exclude_columns += [col for col in df_features.columns if "Spike-contrast" in col]
    # ... and except for day columns
    # DO NOT EXCLUDE ANYMORE AS SCALING HAS NO EFFECT ON 0-1-VALUES AND SCALER COLUMNS SHOULD BE NOT EMPTY
    #exclude_columns += [col for col in df_features.columns if "day" in col]
    # Identify columns to scale
    columns_to_scale = df_features.columns.difference(exclude_columns)
    # Apply scaling to the selected columns
    df_features[columns_to_scale] = scaler.fit_transform(df_features[columns_to_scale])

    # save the data
    # Save the scaler object
    scaler_path = os.path.join(target_path, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump({"scaler": scaler, "columns": columns_to_scale}, f)


    # create interaction features between day columns and feature columns
    df_features2 = _create_day_interaction_features(df_features)

    # save the DataFrame
    feature_path = os.path.join(target_path, 'feature_set_processed.csv')
    hd.save_df_as_csv(df_features2, feature_path)
    print(f"Saved feature set to {feature_path}")
    



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

        if settings.FOLDER_NAME_feature_set_synchrony_curve in settings.FEATURE_SET_LIST:
            ############################################################
            # FEATURE-SET 1: only synchrony-curve
            source_path = path_experiment.replace("PATH", settings.FOLDER_NAME_feature_set_synchrony_curve)
            target_path = source_path  # save to the same path but different filename   
            # load the feature set DataFrame
            full_source_path = os.path.join(source_path, 'feature_set_raw.csv')
            df_feature_set = hd.load_csv_as_df(full_source_path)
            # preprocess 
            preprocess_df_features_longterm(df_feature_set, target_path)

        if settings.FOLDER_NAME_feature_set_synchrony_stats in settings.FEATURE_SET_LIST:
            ############################################################
            # FEATURE-SET 2: only synchrony stats
            source_path = path_experiment.replace("PATH", settings.FOLDER_NAME_feature_set_synchrony_stats)
            target_path = source_path  # save to the same path but different filename   
            # load the feature set DataFrame
            full_source_path = os.path.join(source_path, 'feature_set_raw.csv')
            df_feature_set = hd.load_csv_as_df(full_source_path)
            # preprocess 
            preprocess_df_features_longterm(df_feature_set, target_path)

        if settings.FOLDER_NAME_feature_set_synchrony_curve_days in settings.FEATURE_SET_LIST:
            ############################################################
            # FEATURE-SET 3: synchrony-curve and days (not-one-hot encoded)
            source_path = path_experiment.replace("PATH", settings.FOLDER_NAME_feature_set_synchrony_curve_days)
            target_path = source_path  # save to the same path but different filename   
            # load the feature set DataFrame
            full_source_path = os.path.join(source_path, 'feature_set_raw.csv')
            df_feature_set = hd.load_csv_as_df(full_source_path)
            # preprocess 
            preprocess_df_features_longterm_days(df_feature_set, target_path)

        if settings.FOLDER_NAME_feature_set_synchrony_stats_days in settings.FEATURE_SET_LIST:
            ############################################################
            # FEATURE-SET 4: synchrony stats and days (not-one-hot encoded)
            source_path = path_experiment.replace("PATH", settings.FOLDER_NAME_feature_set_synchrony_stats_days)
            target_path = source_path  # save to the same path but different filename   
            # load the feature set DataFrame
            full_source_path = os.path.join(source_path, 'feature_set_raw.csv')
            df_feature_set = hd.load_csv_as_df(full_source_path)
            # preprocess 
            preprocess_df_features_longterm_days(df_feature_set, target_path)

        if settings.FOLDER_NAME_feature_set_synchrony_curve_bursts in settings.FEATURE_SET_LIST:
            ############################################################
            # FEATURE-SET 5: synchrony curve and bursts (one-hot-encoded) 
            source_path = path_experiment.replace("PATH", settings.FOLDER_NAME_feature_set_synchrony_curve_bursts)
            target_path = source_path  # save to the same path but different filename   
            # load the feature set DataFrame
            full_source_path = os.path.join(source_path, 'feature_set_raw.csv')
            df_feature_set = hd.load_csv_as_df(full_source_path)
            # preprocess 
            preprocess_df_features_longterm(df_feature_set, target_path)

        if settings.FOLDER_NAME_feature_set_synchrony_stats_bursts in settings.FEATURE_SET_LIST:
            ############################################################
            # FEATURE-SET 5: synchrony curve and bursts (one-hot-encoded) 
            source_path = path_experiment.replace("PATH", settings.FOLDER_NAME_feature_set_synchrony_stats_bursts)
            target_path = source_path  # save to the same path but different filename   
            # load the feature set DataFrame
            full_source_path = os.path.join(source_path, 'feature_set_raw.csv')
            df_feature_set = hd.load_csv_as_df(full_source_path)
            # preprocess 
            preprocess_df_features_longterm(df_feature_set, target_path)

        if settings.FOLDER_NAME_feature_set_synchrony_curve_bursts_per_day in settings.FEATURE_SET_LIST:
            ############################################################
            # FEATURE-SET 5: synchrony stats and bursts and days (not-one-hot encoded)
            source_path = path_experiment.replace("PATH", settings.FOLDER_NAME_feature_set_synchrony_curve_bursts_per_day)
            target_path = source_path  # save to the same path but different filename   
            # load the feature set DataFrame
            full_source_path = os.path.join(source_path, 'feature_set_raw.csv')
            df_feature_set = hd.load_csv_as_df(full_source_path)
            # preprocess 
            preprocess_df_features_longterm_per_day(df_feature_set, target_path)

        if settings.FOLDER_NAME_feature_set_synchrony_stats_bursts_per_day in settings.FEATURE_SET_LIST:
            ############################################################
            # FEATURE-SET 5: synchrony stats and bursts and days (not-one-hot encoded)
            source_path = path_experiment.replace("PATH", settings.FOLDER_NAME_feature_set_synchrony_stats_bursts_per_day)
            target_path = source_path  # save to the same path but different filename   
            # load the feature set DataFrame
            full_source_path = os.path.join(source_path, 'feature_set_raw.csv')
            df_feature_set = hd.load_csv_as_df(full_source_path)
            # preprocess 
            preprocess_df_features_longterm_per_day(df_feature_set, target_path)

        if settings.FOLDER_NAME_feature_set_synchrony_curve_bursts_days in settings.FEATURE_SET_LIST:
            ############################################################
            # FEATURE-SET 5: synchrony curve and bursts and days (not-one-hot encoded)
            source_path = path_experiment.replace("PATH", settings.FOLDER_NAME_feature_set_synchrony_curve_bursts_days)
            target_path = source_path  # save to the same path but different filename   
            # load the feature set DataFrame
            full_source_path = os.path.join(source_path, 'feature_set_raw.csv')
            df_feature_set = hd.load_csv_as_df(full_source_path)
            # preprocess 
            preprocess_df_features_longterm_days(df_feature_set, target_path)
        
        if settings.FOLDER_NAME_feature_set_synchrony_stats_bursts_days in settings.FEATURE_SET_LIST:
            ############################################################
            # FEATURE-SET 6: synchrony stats and bursts and days (not-one-hot encoded)
            source_path = path_experiment.replace("PATH", settings.FOLDER_NAME_feature_set_synchrony_stats_bursts_days)
            target_path = source_path  # save to the same path but different filename   
            # load the feature set DataFrame
            full_source_path = os.path.join(source_path, 'feature_set_raw.csv')
            df_feature_set = hd.load_csv_as_df(full_source_path)
            # preprocess 
            preprocess_df_features_longterm_days(df_feature_set, target_path)

        if settings.FOLDER_NAME_feature_set_synchrony_curve_bursts_days_nocorr in settings.FEATURE_SET_LIST:
            ############################################################
            # FEATURE-SET 6: synchrony curve and bursts and days, without correlated features
            source_path = path_experiment.replace("PATH", settings.FOLDER_NAME_feature_set_synchrony_curve_bursts_days_nocorr)
            target_path = source_path  # save to the same path but different filename   
            # load the feature set DataFrame
            full_source_path = os.path.join(source_path, 'feature_set_raw.csv')
            df_feature_set = hd.load_csv_as_df(full_source_path)
            # preprocess 
            preprocess_df_features_longterm_days(df_feature_set, target_path, no_corr = True)
