# TODO:
# For SHAP, mean over all windows for each chip? (currently each window is one data point in the shap plot)
# if yes, use chip_train_list for this (already stored in the shap values pkl file)
#
# repeated cross validation for best model (currently only one run, which can be very random)
#
# after shap, shuffle features (permutation importance) to confirm the results
#
# add null-model test (= shuffle labels and check if AUC is around 0.5) to validate the pipeline
#
# drop one feature if |rho| > 0.95 -> stabilzes SHAP
# or do the clustering before ML and use the mean over all features in the cluster
#
# !!! report std(SHAP) over folds !!!!

# How to run in background on server, so that calculation continues when vscode is closed:
# in vscode terminal enter:
#
# nohup .venv/bin/python main.py > output.log 2>&1 &
#
# Check output.log with:
# tail -f output.log
#
# To stop the process, find the PID with (the line that runs main.py):
# ps aux | grep python
# and then kill it with:
# kill <PID>
# or if it does not stop:   
# kill -9 <PID>


import os
import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    #exec(open(os.path.join(os.getcwd(), "script_0_1_plot_spiketrains.py")).read())
    if 0:
        exec(open(os.path.join(os.getcwd(), "script_1_0_split_data.py")).read())
        exec(open(os.path.join(os.getcwd(), "script_2_0_calc_spike_stats.py")).read())
        exec(open(os.path.join(os.getcwd(), "script_2_1_calc_bursts.py")).read())
        exec(open(os.path.join(os.getcwd(), "script_2_2_calc_networkbursts.py")).read())
        exec(open(os.path.join(os.getcwd(), "script_2_3_calc_synchrony_curve.py")).read())
        exec(open(os.path.join(os.getcwd(), "script_2_4_calc_synchrony_stats.py")).read())
        exec(open(os.path.join(os.getcwd(), "script_2_5_calc_connectivity.py")).read())
        exec(open(os.path.join(os.getcwd(), "script_2_6_calc_complex_network.py")).read())
    
    if 0:    
        exec(open(os.path.join(os.getcwd(), "script_3_0_make_feature_set.py")).read())
        exec(open(os.path.join(os.getcwd(), "script_3_1_preprocess_feature_set.py")).read())
        exec(open(os.path.join(os.getcwd(), "script_6_0_statistics.py")).read())
    
    if 1:
        exec(open(os.path.join(os.getcwd(), "script_4_0_machine_learning.py")).read())
        exec(open(os.path.join(os.getcwd(), "script_4_1_find_best_result.py")).read())
        exec(open(os.path.join(os.getcwd(), "script_4_2_plot_ml.py")).read())
        exec(open(os.path.join(os.getcwd(), "script_5_0_shap.py")).read())
    
    
    

    print("Finished all scripts.")