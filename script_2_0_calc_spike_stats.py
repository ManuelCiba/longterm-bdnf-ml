import numpy as np
import os
import settings
import itertools 
import warnings
from lib.data_handler import spiketrain_handler
from lib.data_handler import hd
import quantities as pq
import pandas as pd
from lib.features.spikes import compute_spike_features

SOURCE_PATH = os.path.join(settings.PATH_RESULTS_FOLDER, settings.FOLDER_NAME_split_data)
TARGET_PATH = os.path.join(settings.PATH_RESULTS_FOLDER, settings.FOLDER_NAME_feature_spikes)


def calculate_spikes(source_path, target_path):
    # Load the spike trains
    spike_trains_elephant = hd.load_obj_from_pkl(source_path)

    # Check if spike_trains_elephant is empty
    all_empty = all((not element.size) if isinstance(element, np.ndarray) else not bool(element) for element in spike_trains_elephant)
    if all_empty:
        print(f"No spikes found in {source_path}. Skipping calculation.")
        return
    
    features = compute_spike_features(spiketrain_list = spike_trains_elephant, rec_dur_s = None, min_spike_rate = settings.MIN_FR)
    df = pd.DataFrame([features])

    # Save the results 
    hd.save_df_as_csv(df, target_path)

if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    # define parameter
    bin_sizes = settings.BIN_SIZES
    window_sizes = settings.WINDOW_SIZES
    window_overlaps = settings.WINDOW_OVERLAPS

    # get all chip names
    chip_names = settings.WELLS_SHAM + settings.WELLS_DRUG  # Combine control and DRUG wells

    # Generate all combinations of parameters
    parameter_combinations = list(itertools.product(bin_sizes, window_sizes, window_overlaps))

    # loop through each combination of parameters
    for combination in parameter_combinations:
        bin_size, window_size, window_overlap = combination

        # Create a folder structure based on the parameter combination
        current_path = os.path.join(SOURCE_PATH, f"bin_{bin_size}",
                                     f"window_{window_size}", f"overlap_{window_overlap}")
  
        # get all directories in the current path (=chip names)
        chip_folders = [entry for entry in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, entry))]    
        # sort the chip folders
        chip_folders.sort()

        # Loop through each chip folder
        for chip_folder in chip_folders:

            # Define the full path to the chip folder
            chip_folder_path = os.path.join(current_path, chip_folder)

            # List all directories in the current path
            rec_folders = [entry for entry in os.listdir(chip_folder_path) if os.path.isdir(os.path.join(chip_folder_path, entry))]

            # Loop through each folder (=recordings) and process the files
            for rec_folder in rec_folders:
                # Define the full path to the folder
                rec_folder_path = os.path.join(chip_folder_path, rec_folder)
                result_path = rec_folder_path.replace(SOURCE_PATH, TARGET_PATH)

                # List all .pkl files in the folder
                files = [f for f in os.listdir(rec_folder_path) if f.endswith('.pkl')]
                files.sort()

                # Process each .pkl file (= spike train file) and
                # calculate snychrony
                for file in files:
                    source_path = os.path.join(rec_folder_path, file)
                    target_path = os.path.join(result_path, file)
                    target_path = target_path.replace("pkl", "csv")

                    # Test if target file already exists
                    if os.path.exists(target_path):
                        print(f"Already processed: {target_path}")
                        continue
                    else:
                        # Calculate synchrony and save results
                        print(f"Calculating spikes: {target_path}")
                        calculate_spikes(source_path, target_path)