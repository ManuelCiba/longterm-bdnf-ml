import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lib.data_handler import spiketrain_handler
from lib.data_handler import hd
import settings
import os
from datetime import datetime
import pandas as pd

def _get_day_after_treatment_from_filename(filename):
    # get the date from the folder name
    date_time_of_file = datetime.strptime(filename, "%Y%m%d_%H%M%S")
    date_time_DRUG_finished = datetime.strptime(settings.DATE_TIME_DRUG_TREATMENT_FINISHED, "%Y%m%d_%H%M%S")
    
    # Calculate the difference as a fraction of a day
    delta = date_time_of_file - date_time_DRUG_finished
    label_days_after_treatment = delta.total_seconds() / (24 * 3600)

    return label_days_after_treatment

def collect_spikerates(source_path, target_path):

    # Test if target file already exists
    if os.path.isfile(target_path):
        print(f"File {target_path} already exists. Skipping...")
        return

    # List all directories in the current path
    current_path = source_path
    chip_folders = [entry for entry in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, entry))]
    # sort the chip folders
    chip_folders.sort()

    # Initialize a list to hold results
    results_list = []

    # Loop through each folder (=Chip, e.g. A1) and process the files
    for chip_folder in chip_folders:

        print(f"Processing chip folder: {chip_folder}")

        # Define the full path to the folder
        chip_folder_path = os.path.join(current_path, chip_folder)

        
        # Define label DRUG or SHAM based on the folder name
        if chip_folder in settings.WELLS_DRUG:
            label_group = 'DRUG'
            label_y = 1
        elif chip_folder in settings.WELLS_SHAM:
            label_group = 'SHAM'
            label_y = 0
        else:
            print(f"Warning: {chip_folder} is not in the defined groups (DRUG or SHAM). Skipping this folder.")
            continue


        # List all _TS.mat files in the folder
        ts_files = [f for f in os.listdir(chip_folder_path) if f.endswith('.mat')]
        # sort the files
        ts_files.sort()

        # Loop through each TS file
        for ts_file in ts_files:

            print(f"Processing TS file: {ts_file}") 

            # define label that represents the days after DRUG treatment
            ts_file_no_suffix = ts_file[:-7]  # Remove the last 7 characters ("_TS.mat")
            label_days_after_treatment = _get_day_after_treatment_from_filename(ts_file_no_suffix)
            
            # get spikerate
            # Load the spike trains
            file_path = os.path.join(chip_folder_path, ts_file)
            TS, AMP, t, fs, time, date = spiketrain_handler.read_dr_cell_file(file_path)
            # count total number of non-nan values in TS
            spike_count = np.sum(~np.isnan(TS))
            # calculate spike rate in Hz for each electrode and then take the mean
            
            # To delete
            #t = t / 60  # convert to minutes
            t = float(np.nanmax(TS) - np.nanmin(TS))
            
            # spike rate over active electrodes only
            spikes_per_channel = np.sum(~np.isnan(TS), axis=0)
            active_spikes = spikes_per_channel[spikes_per_channel > 0]
            spike_rate = np.mean(active_spikes / t)
            
            # spike rate over all electrodes (including those with 0 spikes)
            spike_rate_with_zeros = np.mean(spikes_per_channel / t)
            
            # calculate number of active channels (at least one spike)
            # Active channels = channels with >=1 spike
            n_active_channels = np.sum(spikes_per_channel >= 1)

            # all spikes into one pot
            spike_rate_one_pot = spike_count / t 

            # make a dictionary with the results (so it can be easily converted to a DataFrame later)
            row = {
                'spikecount': spike_count,
                'spikerate': spike_rate,
                'spikerate_one_pot': spike_rate_one_pot,
                'spikerate_with_zeros': spike_rate_with_zeros,
                'n_active_channels': n_active_channels,
                'group': label_group,
                'y': label_y,
                'days_after_treatment': label_days_after_treatment,
                'chip': chip_folder,
                'recording': ts_file_no_suffix
            }
            results_list.append(row)



    # Convert the list of results to a DataFrame
    df = pd.DataFrame(results_list)

    # save the combined DataFrame to a CSV file
    hd.save_df_as_csv(df, target_path)
    print(f"Spikerates saved to {target_path}")

def plot_boxplots(df, target_path):
    """
    Preprocess the data and create violin plots with statistical significance testing.
    Args:
        df (pd.DataFrame): Input DataFrame containing the features and metadata.
    """
    # Identify feature columns (exclude non-feature columns)
    non_feature_cols = ["chip", "y", "group", "days_after_treatment", "file", "recording"]
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    # Group by 'chip', 'group', and 'days_after_treatment', calculate the mean
    df_mean = df.groupby(["chip", "group", "days_after_treatment"], as_index=False)[feature_cols].mean()

    # Group by 'group' and 'days_after_treatment' for statistical tests and plotting
    df_grouped = df_mean.groupby(["group", "days_after_treatment"])
    
    for feature in feature_cols:

        # Create the violin plot
        plt.figure(figsize=(30, 6))
        #sns.violinplot(
        #    x="days_after_treatment", y=feature, hue="group", 
        #    data=df_mean, split=True, inner="box", palette="muted"
        #)
        sns.boxplot(
            x="days_after_treatment", y=feature, hue="group", 
            data=df_mean, palette="muted",
                flierprops=dict(marker='.', markersize=4, linestyle='none', markerfacecolor='gray', alpha=0.7),  # small dot outliers
                boxprops=dict(edgecolor='gray'),   # box border color
                whiskerprops=dict(color='gray'),   # whisker color
                capprops=dict(color='gray'),       # cap (end whisker) color
                medianprops=dict(color='gray')    # median line color
        )

        # Add vertical lines to indicate integer day changes
        # Get sorted unique x positions
        x_values = sorted(df_mean['days_after_treatment'].unique())
        # Loop through and check where integer day changes
        for i in range(1, len(x_values)):
            if int(x_values[i]) != int(x_values[i-1]):
                plt.axvline(x=i-0.5, color='red', linestyle='--', alpha=0.3)


        # Increase font sizes
        plt.xlabel("Days After Treatment", fontsize=16)
        plt.ylabel(feature, fontsize=16)
        plt.title(f"{feature} by Group Over Time", fontsize=18)
        plt.xticks(fontsize=14, rotation=45)
        plt.yticks(fontsize=14)
        plt.legend(title="Group", fontsize=14, title_fontsize=15)

        # Save the plot
        plt.tight_layout()
        full_target_path = os.path.join(target_path, feature + '.pdf')
        hd.save_figure(plt.gcf(), full_target_path)
        print(f"Saved feature set to {full_target_path}")

# Define custom bins for decimals
def bin_time(x):
    day = int(x)  # whole day
    frac = x - day
    if frac < 0.49:
        return day + 0.4  # morning
    elif frac < 0.59:
        return day + 0.5  # noon
    else:
        return day + 0.6  # evening
    
if __name__ == '__main__':

    # define target path
    target_path = os.path.join(settings.PATH_RESULTS_FOLDER, settings.FOLDER_NAME_SPIKERATES, "spikerates.csv")

    # load TS files and collect all spikerates into one csv file
    collect_spikerates(source_path=settings.PATH_DATA_FOLDER, target_path=target_path)

    # load the csv file
    df = hd.load_csv_as_df(target_path)
    df["days_after_treatment"] = df["days_after_treatment"].apply(bin_time)

    # plot
    target_path = os.path.join(settings.PATH_RESULTS_FOLDER, settings.FOLDER_NAME_SPIKERATES)
    plot_boxplots(df, target_path=target_path)