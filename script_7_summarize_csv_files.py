
from pathlib import Path
import pandas as pd
import settings
import os
from lib.data_handler import folder_structure, hd
import glob

###############################
# Helper
###############################

def _get_path_experiment_list(source_folder):
    # define parameter
    bin_sizes = settings.BIN_SIZES
    window_sizes = settings.WINDOW_SIZES
    window_overlaps = settings.WINDOW_OVERLAPS

    path_experiment_list = folder_structure.generate_paths(target_data_folder=source_folder,
                                                           bin_sizes=bin_sizes,
                                                           window_sizes=window_sizes,
                                                           window_overlaps=window_overlaps,
                                                           chip_names=[],
                                                           groups=[])

    return path_experiment_list

if __name__ == '__main__':

    for FOLDER in settings.FEATURE_SET_LIST:

        # Statistics
        source_path = os.path.join("6_0_statistics", FOLDER)
        path_experiment_list = _get_path_experiment_list(source_path)

        for path_experiment in path_experiment_list:

            # p-values
            full_path = os.path.join(path_experiment, "violinplots", "#p_values.csv")
            df = hd.load_csv_as_df(full_path, index_col=False)
            # preprocess
            df = df.drop("x", axis=1)
            df.rename(columns={'x_tick_labels': 'x'}, inplace=True)
            df_p_values = df


            # statistics
            full_path = os.path.join(path_experiment, "violinplots", "#boxplot_statistics.csv")
            df = hd.load_csv_as_df(full_path, index_col=False)
            # preprocess
            df = df.drop("x_ticks", axis=1)
            df.rename(columns={'x_tick_labels': 'x'}, inplace=True)
            df_boxplot = df

            # Shap
            list_shap_values = []
            list_shap_interaction = []
            shap_folder = os.path.join(path_experiment.replace("6_0_statistics/3_0_feature_set_all", "3_2_SHAP_all"), "CatBoost")
            folders = [f for f in os.listdir(shap_folder) if os.path.isdir(os.path.join(shap_folder, f))]
            for folder in folders:
                full_path = os.path.join(shap_folder, folder, "SHAP_all_grouped_CatBoost.csv")
                df = hd.load_csv_as_df(full_path, index_col=False)
                df["x"] = folder
                df_shap_values = df

                full_path = os.path.join(shap_folder, folder, "SHAP_all_interaction_CatBoost.csv")
                df = hd.load_csv_as_df(full_path, index_col=False)
                df["x"] = folder
                df_shap_interaction = df

                list_shap_values.append(df_shap_values)
                list_shap_interaction.append(df_shap_interaction)

    # ML
    source_path = os.path.join(settings.PATH_RESULTS_FOLDER, "3_1_ML_all")
    full_path = os.path.join(source_path, "AUC_summary_CatBoost.csv")
    df = hd.load_csv_as_df(full_path, index_col=False)
    # preprocess
    df = df.drop("Day", axis=1)
    df = df.drop("Bin size", axis=1)
    df = df.drop("Window size", axis=1)
    df = df.drop("Window overlap", axis=1)
    df_ml = df

    # SHAP
    df_shap_values = pd.concat(
        list_shap_values,
        ignore_index=True,
        sort=False,
    )
    df_shap_interaction = pd.concat(
        list_shap_interaction,
        ignore_index=True,
        sort=False,
    )

    metadata = {
        "Research questions": "Can we see any drug effect? Is there any post-washout effect or only acute effect? What is the effect and how does it compare to the literature? How does the effect of LSD, BDNF, DMT compare (each substance has its own experiment=this file structure)? Are any effects comparable with the drug effects known in the human brain (any transfer from in vitro to in vivo)? Is there any implication about damaging or therapeutic effects on the human brain? Is there any novelty in the results compared to most current state of the art science?",
        "ExperimentName_Date_Drug": settings.DATA,
        "Substance used in Class 0": settings.LABEL_SHAM,
        "Substance used in Class 1": settings.LABEL_DRUG,
        "x label": settings.X_LABEL,
        "Experiment description (Cell culture)": "Primary rat cortical neurons (if not stated otherwise) were grown on microelectrode array (MEA) chips. Each chip had 8x8 electrodes",
        "Experiment description (Drug application)": "Drugs were solved in DMSO (final concentration in the cell-medium 0.1%). Either DMT (90µM), LSD (10µM), or BDNF(50ng/ml) was used (see class 1 label). Drugs were applied at around 14..21 DIV (=days in vitro =cell culture age)",
        "Analysis description": "For each x value an independent comparison was performed (a comparison was done by a statistical test (p-values) for each feature and a ML-classification (AUC+SHAP)). The x tick labels only make sense in combination with the x label). For binary ML classification, a nested cross-validation was performed where each subject (=in vitro neuronal network on MEA chip) was used once as the test subject and all other subjects for training. For the SHAP value calculation, the features were first grouped (all features that correlate more than 0.7 end up in one group) and then the sum over all shap values for all features within one group defines the group-shap value. IMPORTANT: grouped features are not the same as GROUP column name. GROUP column defines the belonging of class0 or class1 (aka SHAM or DRUG).",
        "metadata.csv": "this file. It contains the metadata", 
        "ml.csv": "contains the AUCs of the ML classification result",
        "shap_values.csv": "contains the shap values. Look up boxplot.csv in order to get more details about the feature values.", 
        "shap_interaction.csv": "contains the shap interaction between all (grouped)feature combinations", 
        "boxplot.csv": "contains the median+-iqr values for all features per GROUP (class0 or class1).",
        "Interpretation of the results": "If AUC_min >= 0.7 (AUC_min aka AUC_CI_lower) the classification was successfull and we can assume a drug effect (note: if you think the threshold of 0.7 does not make sense, give me another suggestion). p-value < 0.05: small significant drug effect, p value < 0.01: significant drug effect, p value < 0.001: highly significant drug effect. Note: p-values were not corrected for multiple testing, since this experiment has an explorative nature. But if you find effects even with a correction, feel free to report them as well.",
        "x label may use this abbreviations and should be interpret in the context of the x-label meaning": "s: seconds, min: minutes, h: hours, d: days, w: weeks",
        "For the analysis all feature values were processed either as raw values or as relative values. Here (0: Only absolute, raw feature values were used; 1: Only feature values relative to baseline (=first time before drug addition, e.g. -20min) were used.)": str(settings.FLAG_RELATIVE),
        }
    

    df_metadata = pd.DataFrame([metadata])

    # save
    save_path = os.path.join(os.getcwd(), "Summary")
    full_path = os.path.join(save_path, settings.DATA + "_p_values.csv")
    hd.save_df_as_csv(df_p_values, full_path)
    full_path = os.path.join(save_path, settings.DATA + "_boxplot.csv")
    hd.save_df_as_csv(df_boxplot, full_path)
    full_path = os.path.join(save_path, settings.DATA + "_ml.csv")
    hd.save_df_as_csv(df_ml, full_path)
    full_path = os.path.join(save_path, settings.DATA + "_shap_values.csv")
    hd.save_df_as_csv(df_shap_values, full_path)
    full_path = os.path.join(save_path, settings.DATA + "_shap_interaction.csv")
    hd.save_df_as_csv(df_shap_interaction, full_path)
    full_path = os.path.join(save_path, settings.DATA + "_metadata.csv")
    hd.save_df_as_csv(df_metadata, full_path)