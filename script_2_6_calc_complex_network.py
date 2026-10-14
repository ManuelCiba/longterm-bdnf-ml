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
import bct

eps = 1e-12 # small value to avoid division by zero

SOURCE_PATH = os.path.join(settings.PATH_RESULTS_FOLDER, settings.FOLDER_NAME_feature_connectivity)
TARGET_PATH = os.path.join(settings.PATH_RESULTS_FOLDER, settings.FOLDER_NAME_feature_complex_network)

def calculate_and_plot_and_save_complex_network(source_path, target_path):
    df_cm = hd.load_csv_as_df(full_path=source_path, index_col=False)
    df_dm = hd.load_csv_as_df(full_path=source_path.replace("CM", "DM"), index_col=False)
               

    CM_all, CM_exc, CM_inh = _process_connectivity_matrix(df_cm, flag_tresholding=False)

    results_wd_exc =_compute_weighted_directed_features(CM_exc, CM_all, prefix="CM_exc")
    results_wd_inh =_compute_weighted_directed_features(CM_inh, CM_all, prefix="CM_inh")

    results_ei = _compute_EI_balance_features(CM_exc, CM_inh)
    results_bct_exc = _compute_bct_features(CM_exc, prefix="CM_exc")
    results_bct_inh = _compute_bct_features(CM_inh, prefix="CM_inh")

    results_all = {
        **results_wd_exc,
        **results_wd_inh,
        **results_ei,
        **results_bct_exc,
        **results_bct_inh
    }

    # Save results as csv
    df = pd.DataFrame([results_all])
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
    CM_all = CM_df.to_numpy(dtype=float)

    # set diagonal to NaN to exclude self-connections from thresholding
    np.fill_diagonal(CM_all, np.nan)

    # Separate CM into CM_exc (positive values) and CM_inh (negative values)
    CM_exc = np.where(CM_all > 0, CM_all, np.nan)  # Keep only positive values
    CM_inh = np.where(CM_all < 0, CM_all, np.nan)  # Keep only negative values

    # remove sign of CM_inh
    CM_inh = np.abs(CM_inh)

    if flag_tresholding:
        # Threshold calculations for CM_exc
        CM_exc = _apply_threshold(CM_exc)

        # Threshold calculations for CM_inh
        CM_inh = _apply_threshold(CM_inh)

    # Reconstruct DataFrames for CM_exc and CM_inh with original electrode names
    #CM_exc_df = pd.DataFrame(CM_exc, index=pd_file.index, columns=pd_file.columns[1:])
    #CM_inh_df = pd.DataFrame(CM_inh, index=pd_file.index, columns=pd_file.columns[1:])

    # remove inactive nodes (all zero connections, while keeping diagonal at zero - required for following feature calculation)
    CM_all = _remove_inactive_nodes(CM_all)
    CM_exc = _remove_inactive_nodes(CM_exc)
    CM_inh = _remove_inactive_nodes(CM_inh)

    return CM_all, CM_exc, CM_inh


def _apply_threshold(CM):
    mean_exc = np.nanmean(CM)
    std_exc = np.nanstd(CM)
    threshold_exc = np.nanquantile(CM, 0.3) # keep 30% biggest values
    #threshold_exc = mean_exc + 2 * std_exc # better keep number of connections constant
    CM[CM < threshold_exc] = np.nan
    return CM


def _remove_inactive_nodes(CM):
    """
    Prepare connectivity matrix for network analysis.

    Steps:
    1. Replace NaNs by zero
    2. Remove inactive nodes
    3. Remove self-connections

    Returns:
        CM_clean: cleaned matrix
        active_nodes: boolean mask of retained nodes
    """

    # Replace missing values
    CM = np.nan_to_num(CM, nan=0)


    # Identify active nodes
    # A node is active if it has at least one connection
    activity = (
        np.sum(np.abs(CM), axis=0)
        +
        np.sum(np.abs(CM), axis=1)
    )

    active_nodes = activity > 0


    # Remove inactive nodes
    CM = CM[np.ix_(active_nodes, active_nodes)]

    # Remove self connections
    np.fill_diagonal(CM, 0)


    return CM

def safe_stat(func, x):
    """
    Apply numpy statistic safely.
    Returns NaN for empty arrays.
    """
    if len(x) == 0:
        return np.nan
    return func(x)

def _compute_weighted_directed_features(CM, CM_all, prefix="CM_exc"):

    features = {}

    N = CM_all.shape[0]


    # ========================================================
    # 1) Basic network properties
    # ========================================================

    possible_edges = N * (N - 1)


    number_edges = np.count_nonzero(CM)

    features[f"{prefix}_n_nodes"] = N

    features[f"{prefix}_n_edges"] = number_edges

    features[f"{prefix}_density"] = (
        number_edges / possible_edges
        if possible_edges > 0 else np.nan
    )



    # ========================================================
    # 2) Edge weight statistics
    # ========================================================

    weights = CM[CM != 0]


    features[f"{prefix}_mean_edge_weight"] = (
        np.mean(weights)
        if len(weights) > 0 else np.nan
    )


    features[f"{prefix}_weight_second_moment"] = (
        np.mean(weights**2)
        if len(weights) > 0 else np.nan
    )



    # ========================================================
    # 3) Node strength (weighted degree)
    # ========================================================

    # strength:
    out_strength = np.sum(CM, axis=1)
    in_strength  = np.sum(CM, axis=0)

    features[f"{prefix}_mean_out_strength"] = np.mean(out_strength)

    features[f"{prefix}_mean_in_strength"] = np.mean(in_strength)

    features[f"{prefix}_max_out_strength"] = safe_stat(np.max, out_strength)

    features[f"{prefix}_max_in_strength"] = safe_stat(np.max, in_strength)



    # ========================================================
    # 4) k2 = second moment of strength distribution
    # ========================================================

    features[f"{prefix}_k2_out"] = np.mean(out_strength**2)

    features[f"{prefix}_k2_in"] = np.mean(in_strength**2)



    # normalized k2:
    # distinguishes hubs from globally stronger networks

    features[f"{prefix}_k2_norm_out"] = (
        np.mean(out_strength**2)
        /
        (np.mean(out_strength)**2 + eps)
    )


    features[f"{prefix}_k2_norm_in"] = (
        np.mean(in_strength**2)
        /
        (np.mean(in_strength)**2 + eps)
    )



    # ========================================================
    # 5) Strength heterogeneity
    # ========================================================

    features[f"{prefix}_strength_cv_out"] = (
        np.std(out_strength)
        /
        (np.mean(out_strength)+eps)
    )


    features[f"{prefix}_strength_cv_in"] = (
        np.std(in_strength)
        /
        (np.mean(in_strength)+eps)
    )


    return features


def _compute_EI_balance_features(CM_exc, CM_inh):

    features = {}

    # --------------------------------------------------------
    # Mean strength (normalized for electrode count)
    # --------------------------------------------------------

    exc_out = np.mean(
        bct.strengths_dir(CM_exc)
    )

    inh_out = np.mean(
        bct.strengths_dir(CM_inh)
    )


    exc_in = np.mean(
        bct.strengths_dir(CM_exc.T)
    )

    inh_in = np.mean(
        bct.strengths_dir(CM_inh.T)
    )



    features["EI_out_strength_ratio"] = (
        exc_out + eps
    ) / (
        inh_out + eps
    )


    features["EI_in_strength_ratio"] = (
        exc_in + eps
    ) / (
        inh_in + eps
    )



    features["EI_out_strength_balance"] = (
        exc_out - inh_out
    ) / (
        exc_out + inh_out + eps
    )


    features["EI_in_strength_balance"] = (
        exc_in - inh_in
    ) / (
        exc_in + inh_in + eps
    )



    # --------------------------------------------------------
    # Connection number balance
    # --------------------------------------------------------

    exc_edges = np.count_nonzero(CM_exc)

    inh_edges = np.count_nonzero(CM_inh)



    features["E_edge_number"] = exc_edges

    features["I_edge_number"] = inh_edges



    features["EI_edge_number_ratio"] = (
        exc_edges + eps
    ) / (
        inh_edges + eps
    )


    features["EI_edge_number_balance"] = (
        exc_edges - inh_edges
    ) / (
        exc_edges + inh_edges + eps
    )

    return features


def _compute_bct_features(CM, prefix="CM_exc"):

    features={}

    # ======================================================
    # Clustering coefficient
    # ======================================================

    clustering = bct.clustering_coef_wd(CM)

    features[prefix+"_clustering_mean"] = np.mean(clustering)



    # ======================================================
    # Shortest paths
    # ======================================================

    distance = bct.distance_wei(CM)


    D = distance[0]


    features[prefix+"_path_length"] = (
        np.mean(D[np.isfinite(D)])
    )


    # ======================================================
    # Efficiency
    # ======================================================

    efficiency = bct.efficiency_wei(CM)

    features[prefix+"_global_efficiency"] = efficiency



    # ======================================================
    # Betweenness
    # ======================================================

    BC = bct.betweenness_wei(CM)

    features[prefix+"_betweenness_mean"] = np.mean(BC)



    # ======================================================
    # Modularity
    # ======================================================

    try:
        ci, q = bct.community_louvain(CM)

        features[prefix+"_modularity"] = q
        features[prefix+"_number_modules"] = len(np.unique(ci))
        pc = bct.participation_coef(
                CM,
                ci
            )
        
        features[prefix+"_participation_mean"] = np.mean(pc)

    except Exception as e:
        features[prefix+"_modularity"] = np.nan
        features[prefix+"_number_modules"] = np.nan
        features[prefix+"_participation_mean"] = np.nan
        print(f"Error occurred while computing modularity for {prefix}: {e}")


    return features



def OLD_compute_connectivity_features_with_th(CM_exc, CM_inh):
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

def OLD_compute_connectivity_features_no_th(CM, CM_exc, CM_inh, eps=1e-12):
    """
    Connectivity features for weighted asymmetric matrices.
    NaNs are preserved and ignored in computations.
    """
    # ----------------------------------
    # 1️⃣ Strength ratios
    # ----------------------------------
    E_strength = np.nansum(CM_exc)
    I_strength = np.nansum(CM_inh)

    connection_strength_ratio = (E_strength + eps) / (I_strength + eps)
    connection_strength_ratio_norm = (E_strength - I_strength) / (E_strength + I_strength + 1e-8)

    # ----------------------------------
    # 2️⃣ Number of connections
    # ----------------------------------
    E_connections = np.nansum(CM_exc)
    I_connections = np.nansum(CM_inh)

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

def OLD_compute_EI_features(CM_exc, CM_inh, eps=1e-12):

    features = {}

    # ==========================================================
    # 1) Total connection strength (sum of weights)
    # ==========================================================

    E_out_total = np.nansum(CM_exc, axis=1)
    E_in_total  = np.nansum(CM_exc, axis=0)

    I_out_total = np.nansum(CM_inh, axis=1)
    I_in_total  = np.nansum(CM_inh, axis=0)


    features["E_out_mean_strength"] = np.mean(E_out_total)
    features["E_in_mean_strength"]  = np.mean(E_in_total)

    features["I_out_mean_strength"] = np.mean(I_out_total)
    features["I_in_mean_strength"]  = np.mean(I_in_total)



    # ==========================================================
    # 2) E/I strength ratios
    # ==========================================================

    features["EI_out_strength_ratio"] = (
        np.sum(E_out_total)+eps
    ) / (
        np.sum(I_out_total)+eps
    )


    features["EI_in_strength_ratio"] = (
        np.sum(E_in_total)+eps
    ) / (
        np.sum(I_in_total)+eps
    )


    features["EI_out_strength_balance"] = (
        np.sum(E_out_total)-np.sum(I_out_total)
    ) / (
        np.sum(E_out_total)+np.sum(I_out_total)+eps
    )


    features["EI_in_strength_balance"] = (
        np.sum(E_in_total)-np.sum(I_in_total)
    ) / (
        np.sum(E_in_total)+np.sum(I_in_total)+eps
    )



    # ==========================================================
    # 3) Number of connections
    # ==========================================================

    # count non-zero edges, NOT sum of weights
    E_connections = np.count_nonzero(CM_exc)
    I_connections = np.count_nonzero(CM_inh)


    features["E_connection_number"] = E_connections
    features["I_connection_number"] = I_connections


    features["EI_connection_number_ratio"] = (
        E_connections+eps
    ) / (
        I_connections+eps
    )


    features["EI_connection_number_balance"] = (
        E_connections-I_connections
    ) / (
        E_connections+I_connections+eps
    )



    # ==========================================================
    # 4) Second moment of edge weights
    # ==========================================================

    exc_vals = CM_exc[~np.isnan(CM_exc)]
    inh_vals = CM_inh[~np.isnan(CM_inh)]


    features["w2_exc"] = np.mean(exc_vals**2)
    features["w2_inh"] = np.mean(inh_vals**2)



    # ==========================================================
    # 5) k2 = second moment of node strength distribution
    # ==========================================================

    E_out = np.nansum(CM_exc, axis=1)
    E_in  = np.nansum(CM_exc, axis=0)

    I_out = np.nansum(CM_inh, axis=1)
    I_in  = np.nansum(CM_inh, axis=0)


    features["k2_E_out"] = np.mean(E_out**2)
    features["k2_E_in"]  = np.mean(E_in**2)

    features["k2_I_out"] = np.mean(I_out**2)
    features["k2_I_in"]  = np.mean(I_in**2)


    return features


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