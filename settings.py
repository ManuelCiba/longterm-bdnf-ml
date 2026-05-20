import os
import quantities as pq

FLAG_DATA_DMT_DIV8 = 1
FLAG_DATA_LSD_ISMAEL = 0   # Set to True if you want to use Ismael's data
FLAG_DATA_LSD_WENUS = 0    # Set to True if you want to use WENUS data
FLAG_DATA_BDNF = 0    # Set to True if you want to use BDNF data

FLAG_ESTRANGE_DATA1_CION = 0
FLAG_ESTRANGE_DATA1_XRAY = 0
FLAG_ESTRANGE_DATA2_XRAY = 0 # XRAY vs SHAM
FLAG_ESTRANGE_DATA2_TC_XRAY = 0 # Tumor cells + XRAY vs Sham
FLAG_ESTRANGE_DATA2_TC = 0  # Tumor cells only vs Sham (it uses data from DATA2_XRAY as sham and DATA2_TC_XRAY as "drug")

if FLAG_DATA_DMT_DIV8:
    # local path to the data (laptop)
    #PATH_DATA_FOLDER = "/home/mc/Documents/Data/DRUG/IsmaelLozanoFlores/LongTerm/TS/"

    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/DMT/Longterm_DIV8/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", "DMT_DIV8")

    WELLS_SHAM = ['A1_Plate01', 'A1_Plate02', 'A1_Plate03', 'A1_Plate04',
                 'A2_Plate01', 'A2_Plate02', 'A2_Plate03', 'A2_Plate04',
                 'A3_Plate01', 'A3_Plate02', 'A3_Plate03', 'A3_Plate04']
    WELLS_DRUG = ['B1_Plate01', 'B1_Plate02', 'B1_Plate03', 'B1_Plate04',
                  'B2_Plate01', 'B2_Plate02', 'B2_Plate03', 'B2_Plate04',
                  'B3_Plate01', 'B3_Plate02', 'B3_Plate03', 'B3_Plate04',]
    
    DATE_TIME_DRUG_TREATMENT_STARTED = "20260424_160000"  # ?
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20260427_160000"  # ?
    DATE_TIME_LAST_RECORDING = "" # set to "" for all data
    LABEL_SHAM = "Ctrl"
    LABEL_DRUG = "DMT"

if FLAG_DATA_LSD_ISMAEL:
    # local path to the data (laptop)
    #PATH_DATA_FOLDER = "/home/mc/Documents/Data/DRUG/IsmaelLozanoFlores/LongTerm/TS/"

    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/LSD/Ismael/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", "LSD_Ismael")

    # Ismael's data:
    # DRUG (10µM LSD) and SHAM (1% DMSO) well definitions
    WELLS_SHAM = ['A2', 'A3', 'A6', 'B1', 'B3', 'B4', 'B5', 'C5', 'D1', 'D2', 'D5'] # A1 excluded, because no baseline data available
    WELLS_DRUG = ['A4', 'A5', 'B2', 'B6', 'C1', 'C2', 'C3', 'C4', 'C6', 'D3', 'D4', 'D6']
    DATE_TIME_DRUG_TREATMENT_STARTED = "20231114_143512"  # At this time DRUG was present. 20231114_143512 = 14.11.2023, 14:35:12
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20231115_140043"  # At this time DRUG was already washed out. 20231115_140043 = 15.11.2023, 14:00:43
    DATE_TIME_LAST_RECORDING = "20231202_140043" # TODO: just for testing, set to "" for all data
    LABEL_SHAM = "Ctrl"
    LABEL_DRUG = "LSD"

if FLAG_DATA_LSD_WENUS:
    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/LSD/Wenus/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", "LSD_Wenus")

    # Wenus's data:
    # DRUG (10µM) and SHAM (1% DMSO) well definitions
    WELLS_SHAM = ['B1w', 'C6w', 'D3w', 'D4w', 'D5w', 'D6w']
    WELLS_DRUG = ['C1w', 'C2w', 'C3w', 'C4w', 'C5w', 'D1w', 'D2w'] # maybe exclud: C1, D2
    DATE_TIME_DRUG_TREATMENT_STARTED = "20230731_120800"  # LSD Application at 31.07.2023
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20230801_120800"  # LSD removed
    DATE_TIME_LAST_RECORDING = "20230814_120800"
    LABEL_SHAM = "Ctrl"
    LABEL_DRUG = "LSD"

if FLAG_DATA_BDNF:
    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/BDNF/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", "BDNF")

    # BDNF data:
    # Day of Plating: 05.06.2025
    # Application of BDNF: 8 div, 50 ng/ml (dissolved in DH2O), 
    # 13.06.-16.06.2025 (72h)
    WELLS_SHAM = ['B1_Plate1', 'B1_Plate2', 'B1_Plate3', 
                  'B2_Plate1', 'B2_Plate2', 'B2_Plate3',
                  'B3_Plate1', 'B3_Plate2', 'B3_Plate3']
    WELLS_DRUG = ['A1_Plate1', 'A1_Plate2', 'A1_Plate3', 
                 'A2_Plate1', 'A2_Plate2', 'A2_Plate3',
                 'A3_Plate1', 'A3_Plate2', 'A3_Plate3']
    DATE_TIME_DRUG_TREATMENT_STARTED = "20250613_124000"  # TODO: check the exact time when DRUG was present.
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20250616_150000"  # "Last medium change at 3 pm on 16.06.2025, so BDNF should be present until then."
    DATE_TIME_LAST_RECORDING = "20250702_150000"  # Last recording taken on 25.07.2025, but only one recording session. 
    LABEL_SHAM = "Ctrl"
    LABEL_DRUG = "BDNF 50ng/ml"

if FLAG_ESTRANGE_DATA1_CION:
    # DATA1 = Experiment by Margot Mayer, DATA2 = Experiment by Letizia
    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/ESTRANGE/data1/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", "ESTRANGE_DATA1_CION")

    WELLS_SHAM = ['A1_Plate1', 'A2_Plate1', 'A1_Plate2', 'A2_Plate2', 'A1_Plate3', 'A2_Plate3'] 
    WELLS_DRUG = ['A3_Plate1', 'B3_Plate1', 'A3_Plate2', 'B3_Plate2', 'A3_Plate3', 'B3_Plate3']
    DATE_TIME_DRUG_TREATMENT_STARTED = "20250527_000000"  
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20250527_000000" 
    DATE_TIME_LAST_RECORDING = "" # set to "" for all data
    LABEL_SHAM = "Ctrl"
    LABEL_DRUG = "CION"

    # Start or Organoid cultivation:06.03.2025 TB Irradiation on 80 div at GSI: 
    # 0 Gy, 15 Gy X-rays, 15 Gy C-Ions on 27.05.2025 TB
    # Slices from 30.05.2025 (3d a.R) TB
    # day of plating: 30.05.2025 

if FLAG_ESTRANGE_DATA1_XRAY:
    # DATA1 = Experiment by Margot Mayer, DATA2 = Experiment by Letizia
    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/ESTRANGE/data1/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", "ESTRANGE_DATA1_XRAY")

    #WELLS_SHAM = ['A1_Plate1', 'A2_Plate1', 'A1_Plate2', 'A2_Plate2', 'A1_Plate3', 'A2_Plate3'] 
    #WELLS_DRUG = ['B1_Plate1', 'B2_Plate1', 'B1_Plate2', 'B2_Plate2', 'B1_Plate3', 'B2_Plate3']
    WELLS_SHAM = ['A1_Plate1', 'A1_Plate2', 'A2_Plate2', 'A1_Plate3', 'A2_Plate3'] # Selected by MM
    WELLS_DRUG = ['B2_Plate1', 'B1_Plate2', 'B1_Plate3', 'B2_Plate3'] # Selected by MM
    DATE_TIME_DRUG_TREATMENT_STARTED = "20250527_000000"  
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20250527_000000" 
    DATE_TIME_LAST_RECORDING = "" # set to "" for all data
    LABEL_SHAM = "Ctrl"
    LABEL_DRUG = "X-Rays"

    # Start or Organoid cultivation:06.03.2025 TB Irradiation on 80 div at GSI: 
    # 0 Gy, 15 Gy X-rays, 15 Gy C-Ions on 27.05.2025 TB
    # Slices from 30.05.2025 (3d a.R) TB
    # day of plating: 30.05.2025 

if FLAG_ESTRANGE_DATA2_XRAY:
    # DATA1 = Experiment by Margot Mayer, DATA2 = Experiment by Letizia
    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/ESTRANGE/data2/XRAY/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", "ESTRANGE_DATA2_XRAY")

    #WELLS_SHAM = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'B1', 'B2', 'B3', 'B4', 'B5'] # 'B6' was empty
    #WELLS_DRUG = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6']
    WELLS_SHAM = ['A1', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3'] # Selected by MM
    WELLS_DRUG = ['C1', 'C2', 'C4', 'C6'] # Selected by MM
    DATE_TIME_DRUG_TREATMENT_STARTED = "20240705_000000" # exact time unknown, but before first recording  
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20240705_000000" # exact time unknown, but before first recording 
    DATE_TIME_LAST_RECORDING = "" # set to "" for all data
    LABEL_SHAM = "Ctrl"
    LABEL_DRUG = "X-Rays"

    # CO Slices (d60) were recorded for 25 days, (n_SHAM = 11, n_XRAY = 12). First recording is 3 DAP

if FLAG_ESTRANGE_DATA2_TC_XRAY:
    # DATA1 = Experiment by Margot Mayer, DATA2 = Experiment by Letizia
    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/ESTRANGE/data2/TC_XRAY/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", "ESTRANGE_DATA2_TC_XRAY")

    WELLS_SHAM = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'B1', 'B2', 'B3', 'B4'] # 'B5 and 'B6' were empty
    WELLS_DRUG = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6']
    # removed by MC: B1, A2 (only 1 el), A1   
    #WELLS_SHAM = ['A3', 'A4', 'A5', 'A6', 'B2', 'B3', 'B4'] # 'B5 and 'B6' were empty
    #WELLS_DRUG = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6']
    DATE_TIME_DRUG_TREATMENT_STARTED = "20240705_000000" # exact time unknown, but before first recording  
    # 20240705 bestrahlt und aussaat, 0708 is 3DAP
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20240705_000000" # exact time unknown, but before first recording 
    DATE_TIME_LAST_RECORDING = "" # set to "" for all data
    LABEL_SHAM = "TC"
    LABEL_DRUG = "TC + X-Rays"

    # CO Slices (d60) were recorded for 13 days, (n_TC_SHAM = 10, n_TC_XRAY = 12). XRAY 14 Gy.

if FLAG_ESTRANGE_DATA2_TC:
    # DATA1 = Experiment by Margot Mayer, DATA2 = Experiment by Letizia
    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/ESTRANGE/data2/TC/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", "ESTRANGE_DATA2_TC")

    # Here, "drug" is TC only, so it is the sham of the TC+XRAY data.
    WELLS_DRUG = ['A1_TC', 'A2_TC', 'A3_TC', 'A4_TC', 'A5_TC', 'A6_TC', 'B1_TC', 'B2_TC', 'B3_TC', 'B4_TC'] # 'B5 and 'B6' were empty
    # Here, "sham" is the data without TC and without XRAY. So it is the sham of the XRAY data.
    WELLS_SHAM = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'B1', 'B2', 'B3', 'B4', 'B5'] # 'B6' was empty
    DATE_TIME_DRUG_TREATMENT_STARTED = "20240705_000000" # exact time unknown, but before first recording  
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20240705_000000" # exact time unknown, but before first recording 
    DATE_TIME_LAST_RECORDING = "" # set to "" for all data
    LABEL_SHAM = "Ctrl"
    LABEL_DRUG = "TC"


# Flags to control the analysis
FLAG_PLOT = False
FLAG_LONGTERM = True # Set to True if you want to analyze the long-term data
FLAG_ACUTE = False # Set to True if you want to analyze the acute data

# Folder names for the results
FOLDER_NAME_RASTERPLOTS = "0_1_rasterplots"
FOLDER_NAME_SPIKERATES = "0_2_spikerates"
FOLDER_NAME_split_data = "1_0_split_data"
FOLDER_NAME_feature_spikes = "2_0_feature_spikes"
FOLDER_NAME_feature_bursts = "2_1_feature_bursts"
FOLDER_NAME_feature_networkbursts = "2_2_feature_networkbursts"
FOLDER_NAME_feature_synchrony_curve = "2_3_feature_synchrony_curve"
FOLDER_NAME_feature_synchrony_stats = "2_4_feature_synchrony_stats"
FOLDER_NAME_feature_connectivity = "2_5_feature_connectivity"
FOLDER_NAME_feature_complex_network = "2_6_feature_complex_network"


FOLDER_NAME_feature_set_synchrony_curve_bursts = "3_0_feature_set_synchrony_curve_bursts"
FOLDER_NAME_feature_set_synchrony_stats_bursts = "3_0_feature_set_synchrony_stats_bursts"
FOLDER_NAME_feature_set_all = "3_0_feature_set_all"


FOLDER_NAME_statistics = "6_0_statistics"

# Parameter which will be calculated
MIN_FR = 0.1 * pq.Hz  # Minimum firing rate in Hz, 0.1 Hz = 6 spikes per minute
BIN_SIZES = [1 * pq.ms] # not relevant anymore #,10 * pq.ms, 100 * pq.ms] 
WINDOW_SIZES = [240 * pq.s]#[60 * pq.s, 120 * pq.s, 240 * pq.s]  # 30 s excluded, took too long.  600 * pq.s excluded, not enough data points for 10 fold cross validation
if FLAG_DATA_LSD_WENUS:
    WINDOW_SIZES = [60 * pq.s]  # for Wenus's data, only 60s window size, because recordings are shorter
WINDOW_OVERLAPS = [0.75] # 0.5 = 50% overlap
FEATURE_SET_LIST = [
                    #FOLDER_NAME_feature_set_synchrony_curve_bursts,
                    #FOLDER_NAME_feature_set_synchrony_stats_bursts,
                    FOLDER_NAME_feature_set_all
    ]

#ML_MODELS = ['XGboost', 'LightGBM', 'CatBoost']
#ML_MODELS = ['XGboost', 'CatBoost']
ML_MODELS = ['CatBoost']
#ML_MODELS = ['XGboost']
#ML_MODELS = ['RF', 'XGboost'] 
#ML_MODELS = ['RF', 'XGboost', 'SVM', 'NB', 'LR', 'KNN', 'MLP']
