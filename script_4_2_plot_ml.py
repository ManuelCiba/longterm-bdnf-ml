import os
from lib.data_handler import hd
import settings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


def LINEplot_auc_over_time(df, window_overlap_value=75, bin_size_value=1, base_folder="."):
    """
    Plot AUC over time with a transparent area for CI.
    
    Args:
        df: DataFrame with columns ['Day', 'ML model', 'AUC', 'AUC_CI_lower', 'AUC_CI_upper', ...]
        window_overlap_value: value to filter 'Window overlap'
        bin_size_value: value to filter 'Bin size'
        base_folder: folder to save the plot
    """
   

    # Filter the DataFrame
    filtered_df = df[
        (df['Window overlap'] == window_overlap_value) &
        (df['Bin size'] == bin_size_value)
    ].copy()

    # Extract numeric day from string, e.g., "day_31" -> 31
    filtered_df['x'] = filtered_df['Day'].apply(lambda s: float(s[4:]))

    # Sort by day
    filtered_df = filtered_df.sort_values(by='x')

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot AUC over time for each model
    models = filtered_df['ML model'].unique()
    palette = sns.color_palette("tab10", n_colors=len(models))

    for i, model in enumerate(models):
        df_model = filtered_df[filtered_df['ML model'] == model]
        x = df_model['x']
        y = df_model['AUC']
        y_lower = df_model['AUC_CI_lower']
        y_upper = df_model['AUC_CI_upper']

        # Transparent CI band
        plt.fill_between(x, y_lower, y_upper, alpha=0.2, color=palette[i])

        # Line for mean AUC
        plt.plot(x, y, label=model, color=palette[i], linewidth=2, marker='o')

    # Reference line
    plt.axhline(y=0.7, linestyle='--', color='gray', label='Chance Level')

    # Styling
    plt.title("AUC over Time", fontsize=18)
    plt.xlabel("Day", fontsize=16)
    plt.ylabel("AUC", fontsize=16)
    plt.ylim(0.0, 1.05)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, loc='lower right')
    plt.tight_layout()

    # Save figure
    full_path = os.path.join(base_folder, "AUC_over_time.pdf")
    plt.savefig(full_path)
    plt.close()
    print(f"Saved AUC-over-time plot to: {full_path}")

def plot_auc_per_model(df, window_overlap_value=75, bin_size_value=1, base_folder=".", xlabel="", xticks="", xtick_labels=""):
    """
    Plot AUC over time as bar plots with error bars, generating one plot per ML model.

    Args:
        df: DataFrame with columns ['Day', 'ML model', 'AUC', 'AUC_CI_lower', 'AUC_CI_upper', ...]
        window_overlap_value: value to filter 'Window overlap'
        bin_size_value: value to filter 'Bin size'
        base_folder: folder to save the plots
    """

    # Filter the DataFrame
    filtered_df = df[
        (df['Window overlap'] == window_overlap_value) &
        (df['Bin size'] == bin_size_value)
    ].copy()

    # Extract numeric day from string, e.g., "day_31" -> 31
    if xticks == "":
        filtered_df['x'] = filtered_df['Day'].apply(lambda s: float(s[4:]))
        # Sort by day
        filtered_df = filtered_df.sort_values(by='x')
    else:
        # If xticks are provided, we assume that the 'Day' column contains the same values as xticks, so we can directly use them as numeric values
        filtered_df['x'] = filtered_df['Day']
        order = {label: i for i, label in enumerate(xtick_labels)}
        filtered_df = filtered_df.sort_values(
            by="x",
            key=lambda s: s.map(order)
        )
    

    models = filtered_df['ML model'].unique()

    for model in models:
        df_model = filtered_df[filtered_df['ML model'] == model]

        # Calculate error bars
        y = df_model['AUC']
        yerr_lower = y - df_model['AUC_CI_lower']
        yerr_upper = df_model['AUC_CI_upper'] - y
        yerr = [yerr_lower.values, yerr_upper.values]

        x = df_model['x']

        # create xticklabels if not provided
        if xticks == "":
            xticks = x
            xtick_labels = [f"Day {int(d)}" for d in x]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Bar plot with error bars
        bars = ax.bar(x, y, yerr=yerr, capsize=5, alpha=0.7, color='teal', edgecolor='black')

        # Reference line
        ax.axhline(y=0.7, linestyle='--', color='gray')

        # Add AUC_CI_lower inside bars, rotated 90 degrees
        for bar, ci_lower in zip(bars, df_model['AUC_CI_lower']):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,   # X center of bar
                height * 0.05,                       # small padding from bottom
                f"AUCmin={ci_lower:.2f}",                   # label text
                ha='center',
                va='bottom',
                rotation=90,
                fontsize=16,
                color='white',
                weight='bold'
            )

        # Labels & styling
        ax.set_title(f"AUC - {model}", fontsize=18)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel("AUC", fontsize=16)
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=45, fontsize=16)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.tick_params(axis='y', labelsize=14)

        # Remove legend
        ax.get_legend().remove() if ax.get_legend() else None

        plt.tight_layout()

        # Save figure
        full_path = os.path.join(base_folder, f"AUC_summary_{model}.pdf")
        plt.savefig(full_path)
        plt.close()
        print(f"Saved AUC plot for {model} to: {full_path}")

if __name__ == '__main__':
    # for all machine learning folder:
    for FOLDER in settings.FEATURE_SET_LIST:

        #########################################################
        # define paths and load data
        print(FOLDER)

        SOURCE_DATA_FOLDER = FOLDER.replace("0_feature_set", "1_ML")

        base_folder = os.path.join(settings.PATH_RESULTS_FOLDER, SOURCE_DATA_FOLDER)

        # Test results (final AUC values)
        full_path = os.path.join(base_folder, 'results_test.csv')
        df = hd.load_csv_as_df(full_path)
        # relative to %
        df["Window overlap"] = df["Window overlap"] * 100

        if settings.FLAG_ACUTE_MODE == 0:
            condition_col = "days_after_treatment"
            xlabel = "Days After Treatment"
            xticks = ""
            xtick_labels = ""
        else:
            condition_col = "condition"
            xlabel = settings.X_LABEL
            xticks = settings.X_TICKS
            xtick_labels = settings.X_TICK_LABELS

        #plot_auc_over_time(df, base_folder=base_folder)
        plot_auc_per_model(df, base_folder=base_folder, xlabel=xlabel, xticks=xticks, xtick_labels=xtick_labels)



