from importlib.resources import files

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
import pandas as pd
from lib.connectivity import matrix_handling 

SOURCE_PATH = os.path.join(settings.PATH_RESULTS_FOLDER, settings.FOLDER_NAME_feature_connectivity)
TARGET_PATH = os.path.join(settings.PATH_RESULTS_FOLDER, settings.FOLDER_NAME_feature_complex_network)

def calculate_and_plot_and_save_complex_network(source_path, target_path):
    df_cm = hd.load_csv_as_df(full_path=source_path, index_col=False)
                    
    #CM, CM_exc, CM_inh = _process_connectivity_matrix(df_cm, flag_tresholding=True)
    #result = _compute_connectivity_features_with_th(CM_exc, CM_inh)

    CM, CM_exc, CM_inh = _process_connectivity_matrix(df_cm, flag_tresholding=False)
    result_no_th = _compute_connectivity_features_no_th(CM_exc, CM_inh)

    # Save results as csv
    df = pd.DataFrame([result_no_th])
    hd.save_df_as_csv(df, target_path)


def _process_connectivity_matrix(pd_file, flag_tresholding = True):
    """
    Process a connectivity matrix by separating it into excitatory and inhibitory parts,
    and applying thresholds separately.

    Parameters:
        pd_file (pd.DataFrame): DataFrame containing the connectivity matrix with 
                                the first row/column as electrode names.
    
    Returns:
        pd.DataFrame: Updated CM_exc (excitatory connections).
        pd.DataFrame: Updated CM_inh (inhibitory connections).
    """
    # Exclude the first row and column for calculation
    CM_df = pd_file.iloc[:, 1:]  # Numerical part of the connectivity matrix
    CM = CM_df.to_numpy(dtype=float)

    # set diagonal to NaN to exclude self-connections from thresholding
    np.fill_diagonal(CM, np.nan)

    # Separate CM into CM_exc (positive values) and CM_inh (negative values)
    CM_exc = np.where(CM > 0, CM, np.nan)  # Keep only positive values
    CM_inh = np.where(CM < 0, CM, np.nan)  # Keep only negative values

    if flag_tresholding:
        # Threshold calculations for CM_exc
        mean_exc = np.nanmean(CM_exc)
        std_exc = np.nanstd(CM_exc)
        #threshold_exc = np.nanquantile(CM_exc, 0.1) # keep 10% biggest values
        threshold_exc = mean_exc + 2 * std_exc # too aggressive?
        CM_exc[CM_exc < threshold_exc] = np.nan

        # Threshold calculations for CM_inh
        mean_inh = np.nanmean(CM_inh)
        std_inh = np.nanstd(CM_inh)
        #threshold_inh = -1 * np.nanquantile(np.abs(CM_inh), 0.1)
        threshold_inh = mean_inh - 2 * std_inh  # Threshold for negative values is lower bound
        CM_inh[CM_inh > threshold_inh] = np.nan

    # Reconstruct DataFrames for CM_exc and CM_inh with original electrode names
    #CM_exc_df = pd.DataFrame(CM_exc, index=pd_file.index, columns=pd_file.columns[1:])
    #CM_inh_df = pd.DataFrame(CM_inh, index=pd_file.index, columns=pd_file.columns[1:])

    return CM, CM_exc, CM_inh

def _compute_connectivity_features_with_th(CM_exc, CM_inh):
    """
    Compute connectivity features for excitatory, inhibitory, and the entire connectivity matrix.

    Parameters:
        CM_exc (numpy.ndarray): Excitatory connectivity matrix (positive values).
        CM_inh (numpy.ndarray): Inhibitory connectivity matrix (negative values).
        
    Returns:
        dict: A dictionary with E/I connection strength ratio, E/I number ratio, 
              and second moment (for synaptic weights and number of connections) 
              for excitatory, inhibitory, and the entire matrix.
    """
    # Ensure matrices are numpy arrays
    CM_exc = np.nan_to_num(CM_exc, nan=0.0)
    CM_inh = np.nan_to_num(CM_inh, nan=0.0)

    # Combine excitatory and inhibitory matrices into a single connectivity matrix
    CM = CM_exc + CM_inh

    # 1. Connection Strength Ratios
    E_strength = np.sum(CM_exc)
    I_strength = np.sum(np.abs(CM_inh))  # Take absolute value for inhibitory
    connection_strength_ratio = E_strength / I_strength if I_strength != 0 else np.nan
    connection_strength_ratio_norm = (E_strength - I_strength) / (E_strength + I_strength) if (E_strength + I_strength) != 0 else 0.0   

    # 2. Number of Connections Ratios
    E_connections = np.count_nonzero(CM_exc)
    I_connections = np.count_nonzero(CM_inh)
    connection_number_ratio = E_connections / I_connections if I_connections != 0 else np.nan
    connection_number_ratio_norm = (E_connections - I_connections) / (E_connections + I_connections) if (E_connections + I_connections) != 0 else 0.0

    # 3. Second Moment of Synaptic Weights
    second_moment_weights_exc = np.mean(CM_exc[CM_exc != 0]**2)
    second_moment_weights_inh = np.mean(CM_inh[CM_inh != 0]**2)
    second_moment_weights_total = np.mean(CM[CM != 0]**2)

    # 4. Second Moment of Number of Connections
    # For excitatory
    num_connections_exc = np.count_nonzero(CM_exc, axis=1)  # Degree for excitatory
    k2_exc = np.mean(num_connections_exc**2)

    # For inhibitory
    num_connections_inh = np.count_nonzero(CM_inh, axis=1)  # Degree for inhibitory
    k2_inh = np.mean(num_connections_inh**2)

    # For entire matrix
    num_connections_total = np.count_nonzero(CM, axis=1)  # Degree for total matrix
    k2_total = np.mean(num_connections_total**2)

    # Compile results
    results = {
        "Connection Strength Ratio (E/I) Th": connection_strength_ratio,
        "Connection Strength (E-I)/(E+I) Th": connection_strength_ratio_norm,
        "Number of Connections Ratio (E/I) Th": connection_number_ratio,
        "Number of Connections (E-I)/(E+I) Th": connection_number_ratio_norm,
        "w2 (Excitatory) Th": second_moment_weights_exc,
        "w2 (Inhibitory) Th": second_moment_weights_inh,
        "w2 (Total) Th": second_moment_weights_total,
        "k2 (Excitatory) Th": k2_exc,
        "k2 (Inhibitory) Th": k2_inh,
        "k2 (Total) Th": k2_total,
    }

    return results

def _compute_connectivity_features_no_th(CM_exc, CM_inh, eps=1e-12):
    """
    Connectivity features for weighted asymmetric matrices.
    NaNs are preserved and ignored in computations.
    """

    # combine
    CM = np.nansum(np.stack([CM_exc, CM_inh]), axis=0)

    # ----------------------------------
    # 1️⃣ Strength ratios
    # ----------------------------------
    E_strength = np.nansum(CM_exc)
    I_strength = np.nansum(np.abs(CM_inh))

    connection_strength_ratio = (E_strength + eps) / (I_strength + eps)
    connection_strength_ratio_norm = (E_strength - I_strength) / (E_strength + I_strength + 1e-8)

    # ----------------------------------
    # 2️⃣ Number of connections (binary)
    # ----------------------------------
    E_connections = np.sum(~np.isnan(CM_exc) & (CM_exc > 0))
    I_connections = np.sum(~np.isnan(CM_inh) & (CM_inh < 0))

    connection_number_ratio = (E_connections + eps) / (I_connections + eps)
    connection_number_ratio_norm = (E_connections - I_connections) / (E_connections + I_connections + 1e-8)


    # ----------------------------------
    # 3️⃣ Second moment of weights (w2: )
    # ----------------------------------
    exc_vals = CM_exc[~np.isnan(CM_exc)]
    inh_vals = CM_inh[~np.isnan(CM_inh)]
    total_vals = CM[~np.isnan(CM)]

    second_moment_weights_exc = np.mean(exc_vals**2) if len(exc_vals) > 0 else np.nan
    second_moment_weights_inh = np.mean(inh_vals**2) if len(inh_vals) > 0 else np.nan
    second_moment_weights_total = np.mean(total_vals**2) if len(total_vals) > 0 else np.nan

    # ----------------------------------
    # 4️⃣ Weighted degree (node strength) (k2: how unevenly distributed are the connections? -> high k2 means few strong connections, low k2 means many weak connections)
    # ----------------------------------
    strength_exc = np.nansum(CM_exc, axis=1)
    strength_inh = np.nansum(np.abs(CM_inh), axis=1)
    strength_total = np.nansum(np.abs(CM), axis=1)

    k2_exc = np.mean(strength_exc**2)
    k2_inh = np.mean(strength_inh**2)
    k2_total = np.mean(strength_total**2)

    # ----------------------------------
    # Output
    # ----------------------------------

    # Compile results
    results = {
        "Connection Strength Ratio (E/I)": connection_strength_ratio,
        "Connection Strength (E-I)/(E+I)": connection_strength_ratio_norm,
        "Number of Connections Ratio (E/I)": connection_number_ratio,
        "Number of Connections (E-I)/(E+I)": connection_number_ratio_norm,
        "w2 (Excitatory)": second_moment_weights_exc,
        "w2 (Inhibitory)": second_moment_weights_inh,
        "w2 (Total)": second_moment_weights_total,
        "k2 (Excitatory)": k2_exc,
        "k2 (Inhibitory)": k2_inh,
        "k2 (Total)": k2_total,
    }

    return results

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
                rec_folder_path = os.path.join(chip_folder_path, rec_folder, "CM")
                result_path = rec_folder_path.replace(SOURCE_PATH, TARGET_PATH)

                # List all .pkl files in the folder
                #files = [f for f in os.listdir(rec_folder_path) if f.endswith('_CM.csv')]
                files = [f for f in os.listdir(rec_folder_path)]
                files.sort()

                # Process each .pkl file (= spike train file)
                for file in files:
                    source_path = os.path.join(rec_folder_path, file)
                    target_path = os.path.join(result_path, file)
                    target_path = target_path.replace(".pkl", ".csv")
                    target_path = target_path.replace("CM/", "")

                    # Test if target file already exists
                    if os.path.exists(target_path):
                        print(f"Already processed: {target_path}")
                        continue
                    else:
                        # Calculate complex network metrics and save results
                        print(f"Calculating complex network metrics: {target_path}")
                        calculate_and_plot_and_save_complex_network(source_path, target_path)