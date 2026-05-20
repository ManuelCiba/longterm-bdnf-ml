import elephant
import matplotlib.pyplot as plt
import itertools  # generate all parameter combinations  for parameter study
import numpy as np
from lib.data_handler import spiketrain_handler
from lib.data_handler import hd
import settings
import os
import quantities as pq
import warnings
from elephant.functional_connectivity import total_spiking_probability_edges as tspe
import viziphant  # visualization for elephant results
import pandas as pd
from lib.connectivity import matrix_handling 

SOURCE_PATH = os.path.join(settings.PATH_RESULTS_FOLDER, settings.FOLDER_NAME_split_data)
TARGET_PATH = os.path.join(settings.PATH_RESULTS_FOLDER, settings.FOLDER_NAME_feature_connectivity)

def calculate_and_plot_and_save_connectivity(source_path, target_path):
    # Load the spike trains
    spike_trains_elephant = hd.load_obj_from_pkl(source_path)
  
    # Check if spike_trains_elephant is empty
    all_empty = all((not element.size) if isinstance(element, np.ndarray) else not bool(element) for element in spike_trains_elephant)
    if all_empty:
        print(f"No spikes found in {source_path}. Skipping synchrony calculation.")
        return

    # bin spike trains
    bin_size = 1*pq.ms
    bst = elephant.conversion.BinnedSpikeTrain(spike_trains_elephant, bin_size=bin_size)
    bst_binary = bst.binarize()
    
    # calclate connectivity with tspe
    cm, dm = tspe(bst_binary, normalize = False)

    # save csv and jpg
    # save CM
    result_path = target_path
    matrix_handling.save_correlation_matrix(cm, result_path)
    if settings.FLAG_PLOT:
        figure_path = result_path.replace("csv", "jpg")
        matrix_handling.plot_correlation_matrix(cm, figure_path, vmin=-3, vmax=3)
    
    # save DM
    result_path = target_path.replace("CM", "DM")
    matrix_handling.save_correlation_matrix(dm, result_path)
    if settings.FLAG_PLOT:
        figure_path = result_path.replace("csv", "jpg")
        matrix_handling.plot_correlation_matrix(dm, figure_path, vmin=0, vmax=20)


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
            rec_folders.sort()

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
                    target_path = os.path.join(result_path, "CM", file)
                    target_path = target_path.replace(".pkl", ".csv")

                    # Test if target file already exists
                    if os.path.exists(target_path):
                        print(f"Already processed: {target_path}")
                        continue
                    else:
                        # Calculate connectivity and save results
                        print(f"Calculating connectivity: {target_path}")
                        calculate_and_plot_and_save_connectivity(source_path, target_path)