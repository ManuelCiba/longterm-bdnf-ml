#
# Preprocessed data from script_3_1_preprocess_feature_set.py is used to create violin plots and line plots, and to perform statistical testing (Mann-Whitney U test and LMM) between DRUG and SHAM groups at each day after treatment.
#  

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np
import os
import settings
from lib.data_handler import folder_structure
from lib.data_handler import hd
from datetime import datetime
from scipy.stats import mannwhitneyu, wilcoxon # statistical tests
from scipy.stats import false_discovery_control # FDR correction for multiple testing
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf # LMM
import patsy

FLAG_OVERWRITE = False 
correction_method = "p" # "p": no correction, "p_fdr": fdr correction, "p_bonf": bonferroni correction


def preprocess_and_plot(df, target_path):
    """
    Preprocess the data and create violin plots with statistical significance testing.
    Args:
        df (pd.DataFrame): Input DataFrame containing the features and metadata.
    """
    # Identify feature columns (exclude non-feature columns)
    non_feature_cols = settings.METADATA_COLS
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    # Group by 'chip', 'group', and '"x_ticks"', calculate the mean
    df_mean = df.groupby(["chip", "group", "x_ticks"], as_index=False)[feature_cols].mean()
    df_mean = df_mean.sort_values(by=["chip", "x_ticks"]).reset_index(drop=True)

    full_target_path_all_values = os.path.join(target_path, "#all_values.csv")
    hd.save_df_as_csv(df_mean, full_target_path_all_values)

    # simple pairwise test
    if 1:
        all_results = []
        for feature in feature_cols:

            # Perform statistical testing
            for day in df_mean["x_ticks"].unique():
                drug_vals = df_mean[(df_mean["x_ticks"] == day) & (df_mean["group"] == "DRUG")][feature]
                sham_vals = df_mean[(df_mean["x_ticks"] == day) & (df_mean["group"] == "SHAM")][feature]
                
                drug_vals = drug_vals.dropna().to_list()
                sham_vals = sham_vals.dropna().to_list()

                if settings.WELLS_SHAM == []:
                    # Wilcoxon signed-rank test (paired)
                    try:
                        _, p_val = wilcoxon(sham_vals, drug_vals)
                    except ValueError:
                        p_val = np.nan
                    all_results.append({
                        "feature": feature,
                        "x": day,
                        "p": p_val
                    })
                else:
                    # Mann-Whitney U test (unpaired)
                    if len(drug_vals) > 1 and len(sham_vals) > 1:
                        _, p_val = mannwhitneyu(sham_vals, drug_vals, alternative="two-sided")
                        all_results.append({
                            "feature": feature,
                            "x": day,
                            "p": p_val
                        })

    # LMM test - does not work yet - too unstable (better: use permutation test?)
    if 0:
        all_results = []
        for feature in feature_cols:
            df_lmm = lmm_daywise_group_pvalues(df_mean, feature, subject_col="chip")
            all_results.append(df_lmm)

        df_p = pd.concat(all_results, ignore_index=True)

    # transform all p-value results into dataframe
    df_p = pd.DataFrame(all_results)

    # replace nan values with 1 (p=1)
    df_p = df_p.fillna(1)

    # calculate FDR (Benjamini-Hochberg) to correct p-values
    df_p["p_fdr"] = multipletests(
        df_p["p"], method="fdr_bh"
    )[1]

    # calcualte Bonferroni correction
    df_p["p_bonf"] = multipletests(
        df_p["p"], method="bonferroni"
    )[1]

    # Save p values
    full_target_path_p = os.path.join(target_path, '#p_values' + '.csv')
    hd.save_df_as_csv(df_p, full_target_path_p)

    # plot results
    group_order = ["SHAM", "DRUG"]
    for feature in feature_cols:

        feature_formatted = feature.replace("/", "_per_").replace(" ", "_")
        full_target_path = os.path.join(target_path, feature_formatted + '.pdf')
        # skip loop if file already exists
        if os.path.isfile(full_target_path) and not FLAG_OVERWRITE:
            print("Already calculated: " + full_target_path)
            continue

        df_p_feature = df_p[df_p["feature"] == feature]
        p_values = df_p_feature.set_index("x")[correction_method].to_dict()

        # Create the plot
        plt.figure(figsize=(10, 6))
        # tab10 is default palette
        palette = "rocket" #"magma" #"viridis"
        if 0:
            ax = sns.boxplot(
                x="x_ticks", y=feature, hue="group", 
                data=df_mean, 
                palette=palette,
                flierprops=dict(marker='.', markersize=4, linestyle='none', markerfacecolor='gray', alpha=0.7),  # small dot outliers
                boxprops=dict(edgecolor='gray'),   # box border color
                whiskerprops=dict(color='gray'),   # whisker color
                capprops=dict(color='gray'),       # cap (end whisker) color
                medianprops=dict(color='gray')    # median line color
            )

        # --- Clean violin without that tiny embedded box ---
        if 1:
            ax = sns.violinplot(
                x="x_ticks",
                y=feature,
                hue="group",
                hue_order=group_order,
                data=df_mean,
                split=True,
                inner=None,#"point",          # None, "point", "stick", "box"
                palette=palette,
                linewidth=0.8,
                alpha=0.5,
                cut=0,
            )

        # --- Add proper full-size boxplots on top of the violin ---
        #sns.boxplot(
        #    x="x_ticks",
        #    y=feature,
        #    hue="group",
        #    data=df_mean,
        #    showcaps=False,
        #    showfliers=False,
        #    #width=0.2,
        #    boxprops={'facecolor':'none', 'linewidth':1.5, 'edgecolor':'black'},
        #    whiskerprops={'linewidth':1.2, 'color':'black'},
        #    medianprops={'linewidth':1.5, 'color':'black'},
        #    dodge=True,
        #    ax=ax
        #)

        

        # --- Add raw data points (jittered) ---
        if 1:
            sns.stripplot(
                x="x_ticks",
                y=feature,
                hue="group",
                hue_order=group_order,
                palette=palette,
                data=df_mean,
                dodge=True,
                jitter=True,
                size=6,
                alpha=0.5,
                marker='o',
                edgecolor='none',
                ax=ax,
                legend=False
            )

        # get the unique x_tick_labels
        tick_df = (
            df[["x_ticks", "x_tick_labels"]]
            .drop_duplicates()
            .sort_values("x_ticks")
        )

        # Labels & styling
        ax.set_title(f"{feature} by Group Over Time", fontsize=18)
        ax.set_xticks(range(len(tick_df)))
        ax.set_xticklabels(
            tick_df["x_tick_labels"],
            rotation=90,
            fontsize=14
        )
        #if settings.FLAG_ACUTE_MODE:
        #    ax.set_xticklabels(settings.X_TICK_LABELS, rotation=90, fontsize=14)
        ax.set_xlabel(settings.X_LABEL, fontsize=16)
        ax.set_ylabel(feature, fontsize=16)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14, title_fontsize=15)

        # Remove duplicate legends due to triple-plotting, rename labels
        ax.legend_.remove()
        handles, labels = ax.get_legend_handles_labels()
        labels[0] = labels[0].replace("SHAM", settings.LABEL_SHAM)
        labels[1] = labels[1].replace("SHAM", settings.LABEL_SHAM)
        labels[0] = labels[0].replace("DRUG", settings.LABEL_DRUG)
        labels[1] = labels[1].replace("DRUG", settings.LABEL_DRUG)
        ax.legend(handles[:2], labels[:2], fontsize=14)  
        
        if 0:
            # ------------------------------------------------------------
            # Add n-labels per box (SHAM and DRUG for each day)
            # ------------------------------------------------------------
            days = sorted(df_mean['"x_ticks"'].unique())
            groups = df_mean['group'].unique().tolist()  # e.g., ['SHAM','DRUG']
            num_groups = len(groups)

            # Loop over each BoxPlotContainer
            for i, container in enumerate(ax.containers):

                # Determine day + group index
                day_index = i // num_groups
                group_index = i % num_groups

                day = days[day_index]
                group = groups[group_index]

                # Count n
                n = df_mean[
                    (df_mean["x_ticks"] == day) &
                    (df_mean["group"] == group)
                ].shape[0]

                # Get the center of the box
                median_line = container.medians[0]              # Line2D
                xdata = median_line.get_xdata()
                x = float(np.mean(xdata))
                
                # Place text slightly below the plot
                y_min, y_max = ax.get_ylim()
                y_offset = (y_max - y_min) * 0.03

                ax.text(
                    x,
                    y_min - y_offset,
                    f"n={n}",
                    ha='center',
                    va='top',
                    fontsize=10,
                    color='gray'
                )

        #####################################################

        if 0:
            # Add vertical lines to indicate integer day changes
            # Get sorted unique x positions
            x_values = sorted(df_mean['"x_ticks"'].unique())
            # Loop through and check where integer day changes
            for i in range(1, len(x_values)):
                if int(x_values[i]) != int(x_values[i-1]):
                    plt.axvline(x=i-0.5, color='red', linestyle='--', alpha=0.3)


        # Add significance stars
        _add_significance_stars_grouped(p_values, df_mean, "x_ticks", feature)

        # Save the plot
        plt.tight_layout()
        hd.save_figure(plt.gcf(), full_target_path)
        print(f"Saved feature set to {full_target_path}")

def plot_lineplot(df, target_path):   
    # Identify feature columns (exclude non-feature columns)
    non_feature_cols = settings.METADATA_COLS
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    # Group by 'chip', 'group', and '"x_ticks"', calculate the mean
    df_mean = df.groupby(["chip", "x_ticks"], as_index=False)[feature_cols].mean()

    # plot results
    for feature in feature_cols:

        # Create the plot
        plt.figure(figsize=(10, 6))
        # tab10 is default palette
        palette = "rocket" #"magma" #"viridis"

        ax = sns.lineplot(
            data=df_mean,
            x="x_ticks",
            y=feature,
            hue="chip",
            style="chip",        # marker style per chip
            markers=True,        # automatically choose different markers
            dashes=False,        # keep lines solid
        )

        # get the unique x_tick_labels
        tick_df = (
            df[["x_ticks", "x_tick_labels"]]
            .drop_duplicates()
            .sort_values("x_ticks")
        )

        # Labels & styling
        ax.set_title(f"{feature} by Group Over Time", fontsize=18)
        ax.set_xticks(tick_df["x_ticks"].values)
        if settings.FLAG_ACUTE_MODE:
            ax.set_xticklabels(settings.X_TICK_LABELS, rotation=90, fontsize=14)
        #ax.set_xticklabels(
        #    tick_df["x_tick_labels"].values,
        #    rotation=90,
        #    fontsize=14
        #)

        ax.set_xlabel(settings.X_LABEL, fontsize=16)
        ax.set_ylabel(feature, fontsize=16)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14, title_fontsize=15)

        # Save the plot
        plt.tight_layout()
        feature = feature.replace("/", "_per_").replace(" ", "_")
        full_target_path = os.path.join(target_path, feature + '.pdf')
        hd.save_figure(plt.gcf(), full_target_path)
        print(f"Saved feature set to {full_target_path}")     

def lmm_daywise_group_pvalues(df_mean, feature, subject_col="chip"):
    """
    Robust per-day DRUG vs SHAM contrasts using LMM
    """

    data = df_mean.copy().reset_index(drop=True)
    data["day_f"] = data["x_ticks"].astype("category")

    # Quote the feature name safely
    formula = f"Q('{feature}') ~ C(group) * C(day_f)"

    model = smf.mixedlm(formula, data, groups=data[subject_col])
    result = model.fit(reml=False)

    # Build the design matrix once using patsy
    y, X = patsy.dmatrices(formula, data, return_type="dataframe")

    coef_names = X.columns
    params = result.params
    cov = result.cov_params()

    days = data["day_f"].cat.categories
    rows = []

    for day in days:
        # Two hypothetical rows: DRUG vs SHAM at this day
        row_drug = data.iloc[[0]].copy()
        row_sham = data.iloc[[0]].copy()

        row_drug["group"] = "DRUG"
        row_sham["group"] = "SHAM"
        row_drug["day_f"] = day
        row_sham["day_f"] = day

        # Build their design rows via patsy
        _, X_drug = patsy.dmatrices(formula, row_drug, return_type="dataframe")
        _, X_sham = patsy.dmatrices(formula, row_sham, return_type="dataframe")

        # Align columns
        X_drug = X_drug.reindex(columns=coef_names, fill_value=0)
        X_sham = X_sham.reindex(columns=coef_names, fill_value=0)

        contrast = (X_drug.iloc[0] - X_sham.iloc[0]).values

        # Safe t-test
        try:
            ttest = result.t_test(contrast)
            pval = float(ttest.pvalue)
        except Exception:
            pval = np.nan

        rows.append({
            "feature": feature,
            "day": day,
            "p": pval
        })

    return pd.DataFrame(rows)

def p_to_stars(p):

    if np.isnan(p):
        return "NaN"

    if p <= 0.00001:
        return "*****"
    if p <= 0.0001:
        return "****"
    if p <= 0.001:
        return "***"
    elif p <= 0.01:
        return "**"
    elif p <= 0.05:
        return "*"
    else:
        return "ns"
    
def _add_significance_stars_grouped(p_values, df, day_col, feature_col):
    """
    Adds bracket + stars between the two groups at each day.

    p_values: dict like {day_value: p_value}
    """

    ax = plt.gca()

    # Get the group order from the plot
    groups = df["group"].unique()
    n_groups = len(groups)

    # Get unique days in the same order as the plot
    days = sorted(df[day_col].unique())

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    for i, day in enumerate(days):
        if day not in p_values:
            continue

        p = p_values[day]
        stars = p_to_stars(p)

        # x positions for the two boxes
        x1 = i - 0.2
        x2 = i + 0.2

        # Find max y at this day (for placing bracket above data)
        day_data = df[df[day_col] == day][feature_col]
        ymax = day_data.max()

        # Vertical position of the bracket
        y = ymax + 0.05 * y_range
        h = 0.02 * y_range

        if stars == "ns":
            color = "gray"
        else:
            color = "black"

        # Draw bracket
        ax.plot([x1, x1, x2, x2],
                [y, y + h, y + h, y],
                lw=1.5, c=color)

        # Draw stars
        ax.text((x1 + x2) / 2, y + h, stars,
                ha='center', va='bottom',
                color=color, fontsize=16, zorder=6)


if __name__ == '__main__':

    print("Script started: script_6_0_statistics.py")

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

        for feature_set in settings.FEATURE_SET_LIST:

            source_path = path_experiment.replace("PATH", feature_set)
            target_path = path_experiment.replace("PATH", os.path.join(settings.FOLDER_NAME_statistics, feature_set))   
            # load the feature set DataFrame
            #full_source_path = os.path.join(source_path, 'feature_set_raw.csv')
            full_source_path = os.path.join(source_path, 'feature_set_preprocessed.csv')
            df = hd.load_csv_as_df(full_source_path)

            # Drop rows where "x_ticks" < 0
            #df = df.loc[df[""x_ticks""] >= 0].reset_index(drop=True)

            
            preprocess_and_plot(df, target_path + "/violinplots")
            #plot_lineplot(df, target_path + "/lineplots")
            

    print("Script finished: script_6_0_statistics.py")

    