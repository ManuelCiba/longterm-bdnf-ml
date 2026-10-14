# Todo: threshold for shap?
# shap interdependance plot of x most important features

# TODO: change "FLAG_ACUTE_MODE" to "ANALYSIS_MODE=time or file"

import os
import quantities as pq

DATA = str(os.environ.get("DATA_tmp", 1)) # defined in auto_call_main.py
FLAG_RELATIVE = 1

#DATA = "PsyNet_20260805_LSD"
#DATA = "PsyNet_20260805_DMT"
#DATA = "PsyNet_20260805_DMT_VS_LSD" 
#DATA = "PsyNet_20260805_BDNF" # LSD, DMT, BDNF
#DATA = "PsyNet_20260820_DMT_24h" 

#DATA = "DMT_DIV64_HipCor_Hip"
#DATA = "DMT_DIV64_HipCor_Cor" # Cultures from ESTRANGE (radiated), hip and cor, dose response
#DATA = "DMT_DIV30_ACUTE" # Same cells from first DMT experiment
#DATA = "DMT_DIV8_LONGTERM" # First DMT experiment
#DATA = "DMT_DIV8_ACUTE" # First DMT experiment
#DATA = "LSD_Ismael"   # data frsom Ismael Lozano Flores, LSD 10uM longterm
#DATA = "LSD_Wenus"    # data from WENUS NAFEZ, LSD 10uM longterm
#DATA = "BDNF"    # BDNF data post washout, 50ng/ml, 8 div, 72h, 3 plates, 3 wells per plate

#DATA = "ESTRANGE_DATA1_CION"
#DATA = "ESTRANGE_DATA1_XRAY"
#DATA = "ESTRANGE_DATA2_XRAY" # XRAY vs SHAM
#DATA = "ESTRANGE_DATA2_TC_XRAY" # Tumor cells + XRAY vs Sham
#DATA = "ESTRANGE_DATA2_TC"  # Tumor cells only vs Sham (it uses data from DATA2_XRAY as sham and DATA2_TC_XRAY as "drug")
#DATA = "ESTRANGE_DATA3_SHAM_HIP_VS_COR" # Cor vs Hip (not radiated, only sham) 
#DATA = "ESTRANGE_DATA3_XRAY_COR_VS_SHAM"
#DATA = "ESTRANGE_DATA3_XRAY_HIP_VS_SHAM"
#DATA = "ESTRANGE_DATA3_XRAY_COR_VS_HIP"

FLAG_ACUTE_MODE = 0 # Set to True if you want to analyze data in acute mode (in each chip folder, each file corresponds to a drug concentration - time stamps are not used so also experiments from different days can be pooled)

print("DATA: " + DATA)




if DATA == "PsyNet_20260820_DMT_24h":

    FLAG_ACUTE_MODE = 0 # Set to True if you want to analyze data in acute mode (in each chip folder, each file corresponds to a drug concentration - time stamps are not used so also experiments from different days can be pooled)
    BASELINE_X_TICK = 1 # x-tick that is used as baseline (for relative change and paired sample design)
  
    X_LABEL = "Time after DMT application in hours"
    
    # settings for file mode or time mode
    if FLAG_ACUTE_MODE: 
        X_TICKS = [] 
        X_TICK_LABELS = []
    else:
        TIME_WINDOW_HOURS = 1 # average recordings over x hours (24h = days after treatment)
        DATE_TIME_DRUG_TREATMENT_STARTED = "20260821_165000"  # ? "20260424_160000"
        DATE_TIME_DRUG_TREATMENT_FINISHED = "20260821_165000"  # ? "20260427_160000"
        DATE_TIME_LAST_RECORDING = "" # set to "" for all data
        if FLAG_RELATIVE:
            DATE_TIME_NORMALIZATION = "20260821_155700" # set only date, or datetime
    
    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/PsyNet/2026-08-26_DMT_24h/included/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)
    PATH_RESULTS_FOLDER += "_" + str(TIME_WINDOW_HOURS) + "h"
    if FLAG_RELATIVE:
        PATH_RESULTS_FOLDER += "_relative"

    WELLS_SHAM = [] # if empty, then baseline of WELLS_DRUG is compared with all other time points (=paired samples)
    WELLS_DRUG = ['A1_132-1844', 'A2_132-1844', 'A3_132-1844', 'B1_132-1844', 'B2_132-1844', 'B3_132-1844'] 
    
    LABEL_SHAM = "Baseline (before DMT application)"
    LABEL_DRUG = "DMT (90 µM)"

if DATA == "PsyNet_20260805_DMT_VS_LSD":
    # Day of plating: 22.07.2026
    # Application of BDNF: 15 div, 50 ng/ml (dissolved in water) for 24h
    # Acute mode settings
    FLAG_ACUTE_MODE = 1 # Set to True if you want to analyze data in acute mode (in each chip folder, each file corresponds to a drug concentration - time stamps are not used so also experiments from different days can be pooled)
    #FLAG_RELATIVE = 0 # Set to True if you want to calculate the relative change to baseline (baseline is the first time point, which is before drug application, so it is the same for all wells). Only used in acute mode. In long-term mode, we cannot use relative change, because the time points are different for each well, so we have to use absolute values.
    BASELINE_X_TICK = 1
    X_TICKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] 
    X_TICK_LABELS = ["-1d", "-20min", "20min", "1h", "2h", "24h", "0d", "1d", "2d", "3d", "4d", "5d", "6d", "7d", "8d", "9d", "10d", "11d", "12d", "13d", "14d"] # for acute data, only 0h and 1.5h time points, because only these two time points are available for all wells. For long-term data, we can use all time points, because they are the same for all wells.
    X_LABEL = "Time after Treatment"

    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/PsyNet/2026-08-05_full/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)
    if FLAG_RELATIVE:
        PATH_RESULTS_FOLDER += "_relative"

    # RCx01: 141-2532: A1 DMSO, A2 DMT, A3 LSD, B1 BDNF, B2 DMSO, B3 DMT 
    # RCx02: 141-2511: A1 LSD, A2 BDNF, A3 DMSO, B1 DMT, B2 LSD, B3 BDNF
    # RCx03: 141-2608: A1 DMSO, A2 DMT, A3 LSD, B1 BDNF, B2 DMSO, B3 DMT 
    # RCx04: 147-8942: A1 LSD, A2 BDNF, A3 DMSO, B1 DMT, B2 LSD, B3 BDNF
    # RCx05: 147-8943: A1 DMSO, A2 DMT, A3 LSD, B1 BDNF, B2 DMSO, B3 DMT 
    # RCx06: 147-8921: A1 LSD, A2 BDNF, A3 DMSO, B1 DMT, B2 LSD, B3 BDNF
    # RCx07: 147-8902: A1 DMSO, A2 DMT, A3 LSD, B1 BDNF, B2 DMSO, B3 DMT 
    # RCx08: 147-8952: A1 LSD, A2 BDNF, A3 DMSO, B1 DMT, B2 LSD, B3 BDNF 
    WELLS_DMSO = [
        'A1_141-2532', 'B2_141-2532',
        'A3_141-2511',
        'A1_141-2608', 'B2_141-2608',
        'A3_147-8942',
        'A1_147-8943', 'B2_147-8943',
        'A3_147-8921',
        'A1_147-8902', 'B2_147-8902',
        'A3_147-8952'
    ]
    WELLS_DMT = [
        'A2_141-2532', 'B3_141-2532',
        'B1_141-2511',
        'A2_141-2608', 'B3_141-2608',
        'B1_147-8942',
        'A2_147-8943', 'B3_147-8943',
        'B1_147-8921',
        'A2_147-8902', 'B3_147-8902',
        'B1_147-8952'
    ]
    WELLS_LSD = [
        'A3_141-2532',
        'A1_141-2511', 'B2_141-2511',
        'A3_141-2608',
        'A1_147-8942', 'B2_147-8942',
        'A3_147-8943',
        'A1_147-8921', 'B2_147-8921',
        'A3_147-8902',
        'A1_147-8952', 'B2_147-8952'
    ]
    WELLS_BDNF = [
        'B1_141-2532',
        'A2_141-2511', 'B3_141-2511',
        'B1_141-2608',
        'A2_147-8942', 'B3_147-8942',
        'B1_147-8943',
        'A2_147-8921', 'B3_147-8921',
        'B1_147-8902',
        'A2_147-8952', 'B3_147-8952'
    ]

    WELLS_SHAM = WELLS_LSD
    WELLS_DRUG = WELLS_DMT
    
    
    DATE_TIME_DRUG_TREATMENT_STARTED = "20260424_160000"  # not used since acute mode
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20260427_160000"  # not used since acute mode
    DATE_TIME_LAST_RECORDING = "" # set to "" for all data
    LABEL_SHAM = "LSD (10 µM)"
    LABEL_DRUG = "DMT (90 µM)"

if DATA == "PsyNet_20260805_LSD":
    # Day of plating: 22.07.2026
    # Application of BDNF: 15 div, 50 ng/ml (dissolved in water) for 24h
    # Acute mode settings
    FLAG_ACUTE_MODE = 1 # Set to True if you want to analyze data in acute mode (in each chip folder, each file corresponds to a drug concentration - time stamps are not used so also experiments from different days can be pooled)
    #FLAG_RELATIVE = 0 # Set to True if you want to calculate the relative change to baseline (baseline is the first time point, which is before drug application, so it is the same for all wells). Only used in acute mode. In long-term mode, we cannot use relative change, because the time points are different for each well, so we have to use absolute values.
    BASELINE_X_TICK = 1
    X_TICKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] 
    X_TICK_LABELS = ["-1d", "-20min", "20min", "1h", "2h", "24h", "0d", "1d", "2d", "3d", "4d", "5d", "6d", "7d", "8d", "9d", "10d", "11d", "12d", "13d", "14d"] # for acute data, only 0h and 1.5h time points, because only these two time points are available for all wells. For long-term data, we can use all time points, because they are the same for all wells.
    X_LABEL = "Time after Treatment"

    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/PsyNet/2026-08-05_full/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)
    if FLAG_RELATIVE:
        PATH_RESULTS_FOLDER += "_relative"

    # RCx01: 141-2532: A1 DMSO, A2 DMT, A3 LSD, B1 BDNF, B2 DMSO, B3 DMT 
    # RCx02: 141-2511: A1 LSD, A2 BDNF, A3 DMSO, B1 DMT, B2 LSD, B3 BDNF
    # RCx03: 141-2608: A1 DMSO, A2 DMT, A3 LSD, B1 BDNF, B2 DMSO, B3 DMT 
    # RCx04: 147-8942: A1 LSD, A2 BDNF, A3 DMSO, B1 DMT, B2 LSD, B3 BDNF
    # RCx05: 147-8943: A1 DMSO, A2 DMT, A3 LSD, B1 BDNF, B2 DMSO, B3 DMT 
    # RCx06: 147-8921: A1 LSD, A2 BDNF, A3 DMSO, B1 DMT, B2 LSD, B3 BDNF
    # RCx07: 147-8902: A1 DMSO, A2 DMT, A3 LSD, B1 BDNF, B2 DMSO, B3 DMT 
    # RCx08: 147-8952: A1 LSD, A2 BDNF, A3 DMSO, B1 DMT, B2 LSD, B3 BDNF 
    WELLS_DMSO = [
        'A1_141-2532', 'B2_141-2532',
        'A3_141-2511',
        'A1_141-2608', 'B2_141-2608',
        'A3_147-8942',
        'A1_147-8943', 'B2_147-8943',
        'A3_147-8921',
        'A1_147-8902', 'B2_147-8902',
        'A3_147-8952'
    ]
    WELLS_DMT = [
        'A2_141-2532', 'B3_141-2532',
        'B1_141-2511',
        'A2_141-2608', 'B3_141-2608',
        'B1_147-8942',
        'A2_147-8943', 'B3_147-8943',
        'B1_147-8921',
        'A2_147-8902', 'B3_147-8902',
        'B1_147-8952'
    ]
    WELLS_LSD = [
        'A3_141-2532',
        'A1_141-2511', 'B2_141-2511',
        'A3_141-2608',
        'A1_147-8942', 'B2_147-8942',
        'A3_147-8943',
        'A1_147-8921', 'B2_147-8921',
        'A3_147-8902',
        'A1_147-8952', 'B2_147-8952'
    ]
    WELLS_BDNF = [
        'B1_141-2532',
        'A2_141-2511', 'B3_141-2511',
        'B1_141-2608',
        'A2_147-8942', 'B3_147-8942',
        'B1_147-8943',
        'A2_147-8921', 'B3_147-8921',
        'B1_147-8902',
        'A2_147-8952', 'B3_147-8952'
    ]

    WELLS_SHAM = WELLS_DMSO
    WELLS_DRUG = WELLS_LSD
    
    
    DATE_TIME_DRUG_TREATMENT_STARTED = "20260424_160000"  # not used since acute mode
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20260427_160000"  # not used since acute mode
    DATE_TIME_LAST_RECORDING = "" # set to "" for all data
    LABEL_SHAM = "DMSO (0.1%)"
    LABEL_DRUG = "LSD (10 µM)"

if DATA == "PsyNet_20260805_DMT":
    # Day of plating: 22.07.2026
    # Application of BDNF: 15 div, 50 ng/ml (dissolved in water) for 24h
    # Acute mode settings
    FLAG_ACUTE_MODE = 1 # Set to True if you want to analyze data in acute mode (in each chip folder, each file corresponds to a drug concentration - time stamps are not used so also experiments from different days can be pooled)
    #FLAG_RELATIVE = 0 # Set to True if you want to calculate the relative change to baseline (baseline is the first time point, which is before drug application, so it is the same for all wells). Only used in acute mode. In long-term mode, we cannot use relative change, because the time points are different for each well, so we have to use absolute values.
    BASELINE_X_TICK = 1
    X_TICKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] 
    X_TICK_LABELS = ["-1d", "-20min", "20min", "1h", "2h", "24h", "0d", "1d", "2d", "3d", "4d", "5d", "6d", "7d", "8d", "9d", "10d", "11d", "12d", "13d", "14d"] # for acute data, only 0h and 1.5h time points, because only these two time points are available for all wells. For long-term data, we can use all time points, because they are the same for all wells.
    X_LABEL = "Time after Treatment"

    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/PsyNet/2026-08-05_full/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)
    if FLAG_RELATIVE:
        PATH_RESULTS_FOLDER += "_relative"

    # RCx01: 141-2532: A1 DMSO, A2 DMT, A3 LSD, B1 BDNF, B2 DMSO, B3 DMT 
    # RCx02: 141-2511: A1 LSD, A2 BDNF, A3 DMSO, B1 DMT, B2 LSD, B3 BDNF
    # RCx03: 141-2608: A1 DMSO, A2 DMT, A3 LSD, B1 BDNF, B2 DMSO, B3 DMT 
    # RCx04: 147-8942: A1 LSD, A2 BDNF, A3 DMSO, B1 DMT, B2 LSD, B3 BDNF
    # RCx05: 147-8943: A1 DMSO, A2 DMT, A3 LSD, B1 BDNF, B2 DMSO, B3 DMT 
    # RCx06: 147-8921: A1 LSD, A2 BDNF, A3 DMSO, B1 DMT, B2 LSD, B3 BDNF
    # RCx07: 147-8902: A1 DMSO, A2 DMT, A3 LSD, B1 BDNF, B2 DMSO, B3 DMT 
    # RCx08: 147-8952: A1 LSD, A2 BDNF, A3 DMSO, B1 DMT, B2 LSD, B3 BDNF 
    WELLS_DMSO = [
        'A1_141-2532', 'B2_141-2532',
        'A3_141-2511',
        'A1_141-2608', 'B2_141-2608',
        'A3_147-8942',
        'A1_147-8943', 'B2_147-8943',
        'A3_147-8921',
        'A1_147-8902', 'B2_147-8902',
        'A3_147-8952'
    ]
    WELLS_DMT = [
        'A2_141-2532', 'B3_141-2532',
        'B1_141-2511',
        'A2_141-2608', 'B3_141-2608',
        'B1_147-8942',
        'A2_147-8943', 'B3_147-8943',
        'B1_147-8921',
        'A2_147-8902', 'B3_147-8902',
        'B1_147-8952'
    ]
    WELLS_LSD = [
        'A3_141-2532',
        'A1_141-2511', 'B2_141-2511',
        'A3_141-2608',
        'A1_147-8942', 'B2_147-8942',
        'A3_147-8943',
        'A1_147-8921', 'B2_147-8921',
        'A3_147-8902',
        'A1_147-8952', 'B2_147-8952'
    ]
    WELLS_BDNF = [
        'B1_141-2532',
        'A2_141-2511', 'B3_141-2511',
        'B1_141-2608',
        'A2_147-8942', 'B3_147-8942',
        'B1_147-8943',
        'A2_147-8921', 'B3_147-8921',
        'B1_147-8902',
        'A2_147-8952', 'B3_147-8952'
    ]

    WELLS_SHAM = WELLS_DMSO
    WELLS_DRUG = WELLS_DMT
    
    
    DATE_TIME_DRUG_TREATMENT_STARTED = "20260424_160000"  # not used since acute mode
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20260427_160000"  # not used since acute mode
    DATE_TIME_LAST_RECORDING = "" # set to "" for all data
    LABEL_SHAM = "DMSO (0.1%)"
    LABEL_DRUG = "DMT (90 µM)"

if DATA == "PsyNet_20260805_BDNF":
    # Day of plating: 22.07.2026
    # Application of BDNF: 15 div, 50 ng/ml (dissolved in water) for 24h
    # Acute mode settings
    FLAG_ACUTE_MODE = 1 # Set to True if you want to analyze data in acute mode (in each chip folder, each file corresponds to a drug concentration - time stamps are not used so also experiments from different days can be pooled)
    #FLAG_RELATIVE = 0 # Set to True if you want to calculate the relative change to baseline (baseline is the first time point, which is before drug application, so it is the same for all wells). Only used in acute mode. In long-term mode, we cannot use relative change, because the time points are different for each well, so we have to use absolute values.
    BASELINE_X_TICK = 1
    X_TICKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] 
    X_TICK_LABELS = ["-1d", "-20min", "20min", "1h", "2h", "24h", "0d", "1d", "2d", "3d", "4d", "5d", "6d", "7d", "8d", "9d", "10d", "11d", "12d", "13d", "14d"] # for acute data, only 0h and 1.5h time points, because only these two time points are available for all wells. For long-term data, we can use all time points, because they are the same for all wells.
    X_LABEL = "Time after Treatment"

    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/PsyNet/2026-08-05_full/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)
    if FLAG_RELATIVE:
        PATH_RESULTS_FOLDER += "_relative"

    # RCx01: 141-2532: A1 DMSO, A2 DMT, A3 LSD, B1 BDNF, B2 DMSO, B3 DMT 
    # RCx02: 141-2511: A1 LSD, A2 BDNF, A3 DMSO, B1 DMT, B2 LSD, B3 BDNF
    # RCx03: 141-2608: A1 DMSO, A2 DMT, A3 LSD, B1 BDNF, B2 DMSO, B3 DMT 
    # RCx04: 147-8942: A1 LSD, A2 BDNF, A3 DMSO, B1 DMT, B2 LSD, B3 BDNF
    # RCx05: 147-8943: A1 DMSO, A2 DMT, A3 LSD, B1 BDNF, B2 DMSO, B3 DMT 
    # RCx06: 147-8921: A1 LSD, A2 BDNF, A3 DMSO, B1 DMT, B2 LSD, B3 BDNF
    # RCx07: 147-8902: A1 DMSO, A2 DMT, A3 LSD, B1 BDNF, B2 DMSO, B3 DMT 
    # RCx08: 147-8952: A1 LSD, A2 BDNF, A3 DMSO, B1 DMT, B2 LSD, B3 BDNF 
    WELLS_DMSO = [
        'A1_141-2532', 'B2_141-2532',
        'A3_141-2511',
        'A1_141-2608', 'B2_141-2608',
        'A3_147-8942',
        'A1_147-8943', 'B2_147-8943',
        'A3_147-8921',
        'A1_147-8902', 'B2_147-8902',
        'A3_147-8952'
    ]
    WELLS_DMT = [
        'A2_141-2532', 'B3_141-2532',
        'B1_141-2511',
        'A2_141-2608', 'B3_141-2608',
        'B1_147-8942',
        'A2_147-8943', 'B3_147-8943',
        'B1_147-8921',
        'A2_147-8902', 'B3_147-8902',
        'B1_147-8952'
    ]
    WELLS_LSD = [
        'A3_141-2532',
        'A1_141-2511', 'B2_141-2511',
        'A3_141-2608',
        'A1_147-8942', 'B2_147-8942',
        'A3_147-8943',
        'A1_147-8921', 'B2_147-8921',
        'A3_147-8902',
        'A1_147-8952', 'B2_147-8952'
    ]
    WELLS_BDNF = [
        'B1_141-2532',
        'A2_141-2511', 'B3_141-2511',
        'B1_141-2608',
        'A2_147-8942', 'B3_147-8942',
        'B1_147-8943',
        'A2_147-8921', 'B3_147-8921',
        'B1_147-8902',
        'A2_147-8952', 'B3_147-8952'
    ]

    WELLS_SHAM = WELLS_DMSO
    WELLS_DRUG = WELLS_BDNF
    
    
    DATE_TIME_DRUG_TREATMENT_STARTED = "20260424_160000"  # not used since acute mode
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20260427_160000"  # not used since acute mode
    DATE_TIME_LAST_RECORDING = "" # set to "" for all data
    LABEL_SHAM = "DMSO (0.1%)"
    LABEL_DRUG = "BDNF (50 ng/ml in H2O)"

if DATA == "DMT_DIV64_HipCor_Hip":
    # Acute mode settings
    FLAG_ACUTE_MODE = 1 # Set to True if you want to analyze data in acute mode (in each chip folder, each file corresponds to a drug concentration - time stamps are not used so also experiments from different days can be pooled)
    FLAG_RELATIVE = 1 # Set to True if you want to calculate the relative change to baseline (baseline is the first time point, which is before drug application, so it is the same for all wells). Only used in acute mode. In long-term mode, we cannot use relative change, because the time points are different for each well, so we have to use absolute values.
    BASELINE_X_TICK = 0
    X_TICKS = [0, 1, 2, 3, 4, 5, 6] 
    X_TICK_LABELS = ["30 (20min)", "30 (1h)", "90 (20min)", "90 (1h)", "150 (20min)", "150 (1h)", "150 (24h)"] 
    X_LABEL = "DMT in µM"

    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/DMT/20260623/Acute_Hip_Cor/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)
    if FLAG_RELATIVE:
        PATH_RESULTS_FOLDER += "_relative"

    # hip:
    # Plate1: 132-1844 (DMT: A3)
    # Plate2: 141-2641 (DMSO: A1, A3)*
    # Plate3: 132-1848 (DMSO: A2, A3)
    # Plate4: 141-2618 (DMT: A1, A2, A3)*
    # * radiated from ESTRANGE project

    # cor
    # Plate1: 132-1844 (DMT: B1, B2, B3)
    # Plate2: 141-2641 
    # Plate3: 132-1848 (DMSO: B1, B2, B3)
    # Plate4: 141-2618 
    # * radiated from ESTRANGE project

    WELLS_SHAM = ['A1_141-2641', 'A3_141-2641', 'A2_132-1848', 'A3_132-1848']
    WELLS_DRUG = ['A3_132-1844', 'A1_141-2618', 'A2_141-2618', 'A3_141-2618']
    
    DATE_TIME_DRUG_TREATMENT_STARTED = ""  # ?
    DATE_TIME_DRUG_TREATMENT_FINISHED = ""  # ?
    DATE_TIME_LAST_RECORDING = "" # set to "" for all data
    LABEL_SHAM = "DMSO (hip)"
    LABEL_DRUG = "DMT (hip)"

if DATA == "DMT_DIV64_HipCor_Cor":
    # Acute mode settings
    FLAG_ACUTE_MODE = 1 # Set to True if you want to analyze data in acute mode (in each chip folder, each file corresponds to a drug concentration - time stamps are not used so also experiments from different days can be pooled)
    FLAG_RELATIVE = 1 # Set to True if you want to calculate the relative change to baseline (baseline is the first time point, which is before drug application, so it is the same for all wells). Only used in acute mode. In long-term mode, we cannot use relative change, because the time points are different for each well, so we have to use absolute values.
    BASELINE_X_TICK = 0
    X_TICKS = [0, 1, 2, 3, 4, 5, 6] 
    X_TICK_LABELS = ["30 (20min)", "30 (1h)", "90 (20min)", "90 (1h)", "150 (20min)", "150 (1h)", "150 (24h)"] 
    X_LABEL = "DMT in µM"

    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/DMT/20260623/Acute_Hip_Cor/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)
    if FLAG_RELATIVE:
        PATH_RESULTS_FOLDER += "_relative"

    # hip:
    # Plate1: 132-1844 (DMT: A3)
    # Plate2: 141-2641 (DMSO: A1, A3)*
    # Plate3: 132-1848 (DMSO: A2, A3)
    # Plate4: 141-2618 (DMT: A1, A2, A3)*
    # * radiated from ESTRANGE project

    # cor
    # Plate1: 132-1844 (DMT: B1, B2, B3)
    # Plate2: 141-2641 
    # Plate3: 132-1848 (DMSO: B1, B2, B3)
    # Plate4: 141-2618 
    # * radiated from ESTRANGE project

    WELLS_SHAM = ['B1_132-1848', 'B2_132-1848', 'B3_132-1848']
    WELLS_DRUG = ['B1_132-1844', 'B2_132-1844', 'B3_132-1844']
    
    DATE_TIME_DRUG_TREATMENT_STARTED = ""  # ?
    DATE_TIME_DRUG_TREATMENT_FINISHED = ""  # ?
    DATE_TIME_LAST_RECORDING = "" # set to "" for all data
    LABEL_SHAM = "DMSO (cor)"
    LABEL_DRUG = "DMT (cor)"

if DATA == "DMT_DIV30_ACUTE":

    # Acute mode settings
    FLAG_ACUTE_MODE = 1 # Set to True if you want to analyze data in acute mode (in each chip folder, each file corresponds to a drug concentration - time stamps are not used so also experiments from different days can be pooled)
    FLAG_RELATIVE = 1 # Set to True if you want to calculate the relative change to baseline (baseline is the first time point, which is before drug application, so it is the same for all wells). Only used in acute mode. In long-term mode, we cannot use relative change, because the time points are different for each well, so we have to use absolute values.
    BASELINE_X_TICK = 0
    X_TICKS = [0, 1, 2, 3] 
    X_TICK_LABELS = ["0", "90 (20min)", "90 (90min)", "0 (24h)"] # for acute data, only 0h and 1.5h time points, because only these two time points are available for all wells. For long-term data, we can use all time points, because they are the same for all wells.
    X_LABEL = "DMT in µM"

    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/DMT/20260424/Acute_DIV30/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)
    if FLAG_RELATIVE:
        PATH_RESULTS_FOLDER += "_relative"

    WELLS_SHAM = ['A1_132-1807', 'A1_141-2608', 'A1_141-2511', 'A1_141-2532',
                 'A2_132-1807', 'A2_141-2608', 'A2_141-2511', 'A2_141-2532',
                 'A3_132-1807', 'A3_141-2608', 'A3_141-2511', 'A3_141-2532']
    WELLS_DRUG = ['B1_132-1807', 'B1_141-2608', 'B1_141-2511', 'B1_141-2532',
                  'B2_132-1807', 'B2_141-2608', 'B2_141-2511', 'B2_141-2532',
                  'B3_132-1807', 'B3_141-2608', 'B3_141-2511', 'B3_141-2532',]
    
    # 19.05.2026: DMT application: 132-1807, 141-2608
    # 21.05.2026: DMT application: 141-2511, 141-2532
    DATE_TIME_DRUG_TREATMENT_STARTED = "20260424_160000"  # ?
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20260427_160000"  # ?
    DATE_TIME_LAST_RECORDING = "" # set to "" for all data
    LABEL_SHAM = "Sham"
    LABEL_DRUG = "DMT"

if DATA == "DMT_DIV8_LONGTERM":

    FLAG_RELATIVE = 1
    TIME_WINDOW_HOURS = 24 # for calculating the time after treatment (24h = days after treatment)
    X = "Time after treatment" # Column name for X-axis
    X_LABEL = "Time after DMT application in days"


    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/DMT/20260424/Longterm_DIV8/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)
    PATH_RESULTS_FOLDER += "_" + str(TIME_WINDOW_HOURS) + "h"
    if FLAG_RELATIVE:
        PATH_RESULTS_FOLDER += "_relative"


    WELLS_SHAM = ['A1_132-1807', 'A1_141-2608', 'A1_141-2511', 'A1_141-2532',
                 'A2_132-1807', 'A2_141-2608', 'A2_141-2511', 'A2_141-2532',
                 'A3_132-1807', 'A3_141-2608', 'A3_141-2511', 'A3_141-2532']
    WELLS_DRUG = ['B1_132-1807', 'B1_141-2608', 'B1_141-2511', 'B1_141-2532',
                  'B2_132-1807', 'B2_141-2608', 'B2_141-2511', 'B2_141-2532',
                  'B3_132-1807', 'B3_141-2608', 'B3_141-2511', 'B3_141-2532']
    
    DATE_TIME_DRUG_TREATMENT_STARTED = "20260424_000000"  # ? "20260424_160000"
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20260427_000000"  # ? "20260427_160000"
    DATE_TIME_LAST_RECORDING = "" # set to "" for all data
    DATE_TIME_NORMALIZATION = "20260428" 
    LABEL_SHAM = "Sham"
    LABEL_DRUG = "DMT"

if DATA == "DMT_DIV8_ACUTE":

    FLAG_RELATIVE = 1
    TIME_WINDOW_HOURS = 1 # for calculating the time after treatment (24h = days after treatment)
    X = "Time after treatment" # Column name for X-axis
    X_LABEL = "Time after DMT application in hours"

    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/DMT/20260424/Acute_DIV8/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)
    PATH_RESULTS_FOLDER += "_" + str(TIME_WINDOW_HOURS) + "h"
    if FLAG_RELATIVE:
        PATH_RESULTS_FOLDER += "_relative"

    # Plate2: 141-2608
    WELLS_SHAM = ['A1_141-2608','A2_141-2608','A3_141-2608']
    WELLS_DRUG = ['B1_141-2608','B2_141-2608','B3_141-2608']
    
    DATE_TIME_DRUG_TREATMENT_STARTED = "20260424_162000" 
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20260424_162000" # set same as started since I want the drug treatment start to be my t=0  #"20260427_160000"
    DATE_TIME_LAST_RECORDING = "" # set to "" for all data
    DATE_TIME_NORMALIZATION = "20260428_112500" # only effective if FLAG_RELATIVE = 1
    LABEL_SHAM = "Ctrl"
    LABEL_DRUG = "DMT"

if DATA == "LSD_Ismael":
    # local path to the data (laptop)
    #PATH_DATA_FOLDER = "/home/mc/Documents/Data/DRUG/IsmaelLozanoFlores/LongTerm/TS/"

    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/LSD/Ismael/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)

    # Ismael's data:
    # DRUG (10µM LSD) and SHAM (1% DMSO) well definitions
    WELLS_SHAM = ['A2', 'A3', 'A6', 'B1', 'B3', 'B4', 'B5', 'C5', 'D1', 'D2', 'D5'] # A1 excluded, because no baseline data available
    WELLS_DRUG = ['A4', 'A5', 'B2', 'B6', 'C1', 'C2', 'C3', 'C4', 'C6', 'D3', 'D4', 'D6']
    DATE_TIME_DRUG_TREATMENT_STARTED = "20231114_143512"  # At this time DRUG was present. 20231114_143512 = 14.11.2023, 14:35:12
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20231115_140043"  # At this time DRUG was already washed out. 20231115_140043 = 15.11.2023, 14:00:43
    DATE_TIME_LAST_RECORDING = "20231202_140043" # TODO: just for testing, set to "" for all data
    LABEL_SHAM = "Ctrl"
    LABEL_DRUG = "LSD"

if DATA == "LSD_Wenus":
    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/LSD/Wenus/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)

    # Wenus's data:
    # DRUG (10µM) and SHAM (1% DMSO) well definitions
    WELLS_SHAM = ['B1w', 'C6w', 'D3w', 'D4w', 'D5w', 'D6w']
    WELLS_DRUG = ['C1w', 'C2w', 'C3w', 'C4w', 'C5w', 'D1w', 'D2w'] # maybe exclud: C1, D2
    DATE_TIME_DRUG_TREATMENT_STARTED = "20230731_120800"  # LSD Application at 31.07.2023
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20230801_120800"  # LSD removed
    DATE_TIME_LAST_RECORDING = "20230814_120800"
    LABEL_SHAM = "Ctrl"
    LABEL_DRUG = "LSD"

if DATA == "BDNF":
    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/BDNF/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)

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

if DATA == "ESTRANGE_DATA1_CION":
    # DATA1 = Experiment by Margot Mayer, DATA2 = Experiment by Letizia
    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/ESTRANGE/data1/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)

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

if DATA == "ESTRANGE_DATA1_XRAY":
    # DATA1 = Experiment by Margot Mayer, DATA2 = Experiment by Letizia
    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/ESTRANGE/data1/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)

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

if DATA == "ESTRANGE_DATA2_XRAY":
    # DATA1 = Experiment by Margot Mayer, DATA2 = Experiment by Letizia
    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/ESTRANGE/data2/XRAY/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)

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

if DATA == "ESTRANGE_DATA2_TC_XRAY":
    # DATA1 = Experiment by Margot Mayer, DATA2 = Experiment by Letizia
    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/ESTRANGE/data2/TC_XRAY/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)

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

if DATA == "ESTRANGE_DATA2_TC":
    # DATA1 = Experiment by Margot Mayer, DATA2 = Experiment by Letizia
    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/ESTRANGE/data2/TC/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)

    # Here, "drug" is TC only, so it is the sham of the TC+XRAY data.
    WELLS_DRUG = ['A1_TC', 'A2_TC', 'A3_TC', 'A4_TC', 'A5_TC', 'A6_TC', 'B1_TC', 'B2_TC', 'B3_TC', 'B4_TC'] # 'B5 and 'B6' were empty
    # Here, "sham" is the data without TC and without XRAY. So it is the sham of the XRAY data.
    WELLS_SHAM = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'B1', 'B2', 'B3', 'B4', 'B5'] # 'B6' was empty
    DATE_TIME_DRUG_TREATMENT_STARTED = "20240705_000000" # exact time unknown, but before first recording  
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20240705_000000" # exact time unknown, but before first recording 
    DATE_TIME_LAST_RECORDING = "" # set to "" for all data
    LABEL_SHAM = "Ctrl"
    LABEL_DRUG = "TC"

if DATA == "ESTRANGE_DATA3_XRAY_COR_VS_SHAM":
    FLAG_ACUTE_MODE = 0 # Set to True if you want to analyze data in acute mode (in each chip folder, each file corresponds to a drug concentration - time stamps are not used so also experiments from different days can be pooled)
    #FLAG_RELATIVE = 0 # Set to True if you want to calculate the relative change to baseline (baseline is the first time point, which is before drug application, so it is the same for all wells). Only used in acute mode. In long-term mode, we cannot use relative change, because the time points are different for each well, so we have to use absolute values.

    X_LABEL = "Days after Treatment"
    
    # settings for file mode or time mode
    if FLAG_ACUTE_MODE:
        BASELINE_X_TICK = 0
        X_TICKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] 
        X_TICK_LABELS = ["-1", "1", "2", "3", "4", "5", "6", "7", "8", "9", "12", "13", "14"]
    else:
        TIME_WINDOW_HOURS = 24 # average recordings over x hours (24h = days after treatment)
        DATE_TIME_DRUG_TREATMENT_STARTED = "20260812_000000"  
        DATE_TIME_DRUG_TREATMENT_FINISHED = "20260812_180000"  
        DATE_TIME_LAST_RECORDING = "20260826_180000" # set to "" for all data
        DATE_TIME_NORMALIZATION = "20260812" # only effective if FLAG_RELATIVE = 1
    
    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/ESTRANGE/2026-08-12_RHip_RCx_2D_Axion_Radiation/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)
    if FLAG_RELATIVE:
        PATH_RESULTS_FOLDER += "_relative"

    # RCx01: 148-2339: 15Gy: A1-A3: Cor, B1-B3: Hip
    # RCx02: 148-2350: 15Gy: "
    # RCx03: 148-2330: 15Gy: "
    # RCx04: 148-2320: 15Gy: "
    # RCx05: 132-1848: Sham: "
    # RCx06: 132-1807: Sham: "
    # RCx07: 148-2319: Sham: "
    WELLS_SHAM_COR = [
        'A1_132-1848', 'A2_132-1848', 'A3_132-1848',
        'A1_132-1807', 'A2_132-1807', 'A3_132-1807',
        'A1_148-2319', 'A2_148-2319', 'A3_148-2319'
    ]
    WELLS_SHAM_HIP = [
        'B1_132-1848', 'B2_132-1848', 'B3_132-1848',
        'B1_132-1807', 'B2_132-1807', 'B3_132-1807',
        'B1_148-2319', 'B2_148-2319', 'B3_148-2319'
    ]
    WELLS_XRAY_COR = [
        'A1_148-2339', 'A2_148-2339', 'A3_148-2339',
        'A1_148-2350', 'A2_148-2350', 'A3_148-2350',
        'A1_148-2330', 'A2_148-2330', 'A3_148-2330',
        'A1_148-2320', 'A2_148-2320', 'A3_148-2320'
    ]
    WELLS_XRAY_HIP = [
        'B1_148-2339', 'B2_148-2339', 'B3_148-2339',
        'B1_148-2350', 'B2_148-2350', 'B3_148-2350',
        'B1_148-2330', 'B2_148-2330', 'B3_148-2330',
        'B1_148-2320', 'B2_148-2320', 'B3_148-2320'
    ]
    
    WELLS_SHAM = WELLS_SHAM_COR
    WELLS_DRUG = WELLS_XRAY_COR
    
    LABEL_SHAM = "SHAM (Cx)"
    LABEL_DRUG = "XRAY (Cx)"

if DATA == "ESTRANGE_DATA3_XRAY_HIP_VS_SHAM":
    FLAG_ACUTE_MODE = 0 # Set to True if you want to analyze data in acute mode (in each chip folder, each file corresponds to a drug concentration - time stamps are not used so also experiments from different days can be pooled)
    #FLAG_RELATIVE = 0 # Set to True if you want to calculate the relative change to baseline (baseline is the first time point, which is before drug application, so it is the same for all wells). Only used in acute mode. In long-term mode, we cannot use relative change, because the time points are different for each well, so we have to use absolute values.

    X_LABEL = "Days after Treatment"
    
    # settings for file mode or time mode
    if FLAG_ACUTE_MODE:
        BASELINE_X_TICK = 0
        X_TICKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] 
        X_TICK_LABELS = ["-1", "1", "2", "3", "4", "5", "6", "7", "8", "9", "12", "13", "14"]
    else:
        TIME_WINDOW_HOURS = 24 # average recordings over x hours (24h = days after treatment)
        DATE_TIME_DRUG_TREATMENT_STARTED = "20260812_000000"  
        DATE_TIME_DRUG_TREATMENT_FINISHED = "20260812_180000"  
        DATE_TIME_LAST_RECORDING = "20260826_180000" # set to "" for all data
        DATE_TIME_NORMALIZATION = "20260812" # only effective if FLAG_RELATIVE = 1
    
    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/ESTRANGE/2026-08-12_RHip_RCx_2D_Axion_Radiation/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)
    if FLAG_RELATIVE:
        PATH_RESULTS_FOLDER += "_relative"

    # RCx01: 148-2339: 15Gy: A1-A3: Cor, B1-B3: Hip
    # RCx02: 148-2350: 15Gy: "
    # RCx03: 148-2330: 15Gy: "
    # RCx04: 148-2320: 15Gy: "
    # RCx05: 132-1848: Sham: "
    # RCx06: 132-1807: Sham: "
    # RCx07: 148-2319: Sham: "
    WELLS_SHAM_COR = [
        'A1_132-1848', 'A2_132-1848', 'A3_132-1848',
        'A1_132-1807', 'A2_132-1807', 'A3_132-1807',
        'A1_148-2319', 'A2_148-2319', 'A3_148-2319'
    ]
    WELLS_SHAM_HIP = [
        'B1_132-1848', 'B2_132-1848', 'B3_132-1848',
        'B1_132-1807', 'B2_132-1807', 'B3_132-1807',
        'B1_148-2319', 'B2_148-2319', 'B3_148-2319'
    ]
    WELLS_XRAY_COR = [
        'A1_148-2339', 'A2_148-2339', 'A3_148-2339',
        'A1_148-2350', 'A2_148-2350', 'A3_148-2350',
        'A1_148-2330', 'A2_148-2330', 'A3_148-2330',
        'A1_148-2320', 'A2_148-2320', 'A3_148-2320'
    ]
    WELLS_XRAY_HIP = [
        'B1_148-2339', 'B2_148-2339', 'B3_148-2339',
        'B1_148-2350', 'B2_148-2350', 'B3_148-2350',
        'B1_148-2330', 'B2_148-2330', 'B3_148-2330',
        'B1_148-2320', 'B2_148-2320', 'B3_148-2320'
    ]
    
    WELLS_SHAM = WELLS_SHAM_HIP
    WELLS_DRUG = WELLS_XRAY_HIP
    
    LABEL_SHAM = "SHAM (Hip)"
    LABEL_DRUG = "XRAY (Hip)"

if DATA == "ESTRANGE_DATA3_XRAY_COR_VS_HIP":
    FLAG_ACUTE_MODE = 0 # Set to True if you want to analyze data in acute mode (in each chip folder, each file corresponds to a drug concentration - time stamps are not used so also experiments from different days can be pooled)
    #FLAG_RELATIVE = 0 # Set to True if you want to calculate the relative change to baseline (baseline is the first time point, which is before drug application, so it is the same for all wells). Only used in acute mode. In long-term mode, we cannot use relative change, because the time points are different for each well, so we have to use absolute values.

    X_LABEL = "Days after Treatment"
    
    # settings for file mode or time mode
    if FLAG_ACUTE_MODE:
        BASELINE_X_TICK = 0
        X_TICKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] 
        X_TICK_LABELS = ["-1", "1", "2", "3", "4", "5", "6", "7", "8", "9", "12", "13", "14"]
    else:
        TIME_WINDOW_HOURS = 24 # average recordings over x hours (24h = days after treatment)
        DATE_TIME_DRUG_TREATMENT_STARTED = "20260812_000000"  
        DATE_TIME_DRUG_TREATMENT_FINISHED = "20260812_180000"  
        DATE_TIME_LAST_RECORDING = "20260826_180000" # set to "" for all data
        DATE_TIME_NORMALIZATION = "20260812" # only effective if FLAG_RELATIVE = 1
    
    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/ESTRANGE/2026-08-12_RHip_RCx_2D_Axion_Radiation/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)
    if FLAG_RELATIVE:
        PATH_RESULTS_FOLDER += "_relative"

    # RCx01: 148-2339: 15Gy: A1-A3: Cor, B1-B3: Hip
    # RCx02: 148-2350: 15Gy: "
    # RCx03: 148-2330: 15Gy: "
    # RCx04: 148-2320: 15Gy: "
    # RCx05: 132-1848: Sham: "
    # RCx06: 132-1807: Sham: "
    # RCx07: 148-2319: Sham: "
    WELLS_SHAM_COR = [
        'A1_132-1848', 'A2_132-1848', 'A3_132-1848',
        'A1_132-1807', 'A2_132-1807', 'A3_132-1807',
        'A1_148-2319', 'A2_148-2319', 'A3_148-2319'
    ]
    WELLS_SHAM_HIP = [
        'B1_132-1848', 'B2_132-1848', 'B3_132-1848',
        'B1_132-1807', 'B2_132-1807', 'B3_132-1807',
        'B1_148-2319', 'B2_148-2319', 'B3_148-2319'
    ]
    WELLS_XRAY_COR = [
        'A1_148-2339', 'A2_148-2339', 'A3_148-2339',
        'A1_148-2350', 'A2_148-2350', 'A3_148-2350',
        'A1_148-2330', 'A2_148-2330', 'A3_148-2330',
        'A1_148-2320', 'A2_148-2320', 'A3_148-2320'
    ]
    WELLS_XRAY_HIP = [
        'B1_148-2339', 'B2_148-2339', 'B3_148-2339',
        'B1_148-2350', 'B2_148-2350', 'B3_148-2350',
        'B1_148-2330', 'B2_148-2330', 'B3_148-2330',
        'B1_148-2320', 'B2_148-2320', 'B3_148-2320'
    ]
    
    WELLS_SHAM = WELLS_XRAY_COR
    WELLS_DRUG = WELLS_XRAY_HIP
    
    LABEL_SHAM = "XRAY (Cx)"
    LABEL_DRUG = "XRAY (Hip)"

if DATA == "ESTRANGE_DATA3_SHAM_HIP_VS_COR":
    FLAG_ACUTE_MODE = 0 # Set to True if you want to analyze data in acute mode (in each chip folder, each file corresponds to a drug concentration - time stamps are not used so also experiments from different days can be pooled)
    #FLAG_RELATIVE = 0 # Set to True if you want to calculate the relative change to baseline (baseline is the first time point, which is before drug application, so it is the same for all wells). Only used in acute mode. In long-term mode, we cannot use relative change, because the time points are different for each well, so we have to use absolute values.

    X_LABEL = "Days after Treatment"
    
    # settings for file mode or time mode
    if FLAG_ACUTE_MODE:
        BASELINE_X_TICK = 0
        X_TICKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] 
        X_TICK_LABELS = ["-1", "1", "2", "3", "4", "5", "6", "7", "8", "9", "12", "13", "14"]
    else:
        TIME_WINDOW_HOURS = 24 # average recordings over x hours (24h = days after treatment)
        DATE_TIME_DRUG_TREATMENT_STARTED = "20260812_000000"  
        DATE_TIME_DRUG_TREATMENT_FINISHED = "20260812_180000"  
        DATE_TIME_LAST_RECORDING = "20260826_180000" # set to "" for all data
        DATE_TIME_NORMALIZATION = "20260812" # only effective if FLAG_RELATIVE = 1
    
    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-07_BDNF/data/ESTRANGE/2026-08-12_RHip_RCx_2D_Axion_Radiation/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", DATA)
    if FLAG_RELATIVE:
        PATH_RESULTS_FOLDER += "_relative"

    # RCx01: 148-2339: 15Gy: A1-A3: Cor, B1-B3: Hip
    # RCx02: 148-2350: 15Gy: "
    # RCx03: 148-2330: 15Gy: "
    # RCx04: 148-2320: 15Gy: "
    # RCx05: 132-1848: Sham: "
    # RCx06: 132-1807: Sham: "
    # RCx07: 148-2319: Sham: "
    WELLS_SHAM_COR = [
        'A1_132-1848', 'A2_132-1848', 'A3_132-1848',
        'A1_132-1807', 'A2_132-1807', 'A3_132-1807',
        'A1_148-2319', 'A2_148-2319', 'A3_148-2319'
    ]
    WELLS_SHAM_HIP = [
        'B1_132-1848', 'B2_132-1848', 'B3_132-1848',
        'B1_132-1807', 'B2_132-1807', 'B3_132-1807',
        'B1_148-2319', 'B2_148-2319', 'B3_148-2319'
    ]
    WELLS_XRAY_COR = [
        'A1_148-2339', 'A2_148-2339', 'A3_148-2339',
        'A1_148-2350', 'A2_148-2350', 'A3_148-2350',
        'A1_148-2330', 'A2_148-2330', 'A3_148-2330',
        'A1_148-2320', 'A2_148-2320', 'A3_148-2320'
    ]
    WELLS_XRAY_HIP = [
        'B1_148-2339', 'B2_148-2339', 'B3_148-2339',
        'B1_148-2350', 'B2_148-2350', 'B3_148-2350',
        'B1_148-2330', 'B2_148-2330', 'B3_148-2330',
        'B1_148-2320', 'B2_148-2320', 'B3_148-2320'
    ]
    
    WELLS_SHAM = WELLS_SHAM_COR
    WELLS_DRUG = WELLS_SHAM_HIP
    

    
    LABEL_SHAM = "SHAM (Cx)"
    LABEL_DRUG = "SHAM (Hip)"


# Global variables for easier code maintenance
METADATA_COLS = ["chip", "y", "group", "x_ticks", "file", "recording", "x_tick_labels"]
NOT_NEEDED_FOR_ML_COLS = ["group", "file", "recording", "x_ticks"]
NOT_NEEDED_FOR_SHAP_CORRELATION = ["y", "chip", "x_ticks", "x_tick_labels"]

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
if DATA == "LSD_Wenus":
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
