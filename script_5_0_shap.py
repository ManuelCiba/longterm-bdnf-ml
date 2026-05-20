import os
import quantities as pq
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lib.machine_learning import workflow_unpaired_optatune as workflow_unpaired
from lib.data_handler import folder_structure
from lib.data_handler import hd
import settings
import pandas as pd
import matplotlib.pyplot as plt
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.cm as cm
from scipy.cluster.hierarchy import linkage, fcluster
import re
import textwrap
import math




def plot_feature_correlation(corr, title="Feature Correlation Matrix", figsize=(12, 10), save_path=None):

    n_features = len(corr.columns)

    # Scale figure size with number of features
    fig_size = max(12, n_features * 0.25)

    plt.figure(figsize=(fig_size, fig_size))

    ax = sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        center=0,
        square=True,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cbar_kws={"shrink": 0.8}
    )

    # Smaller font
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    # Increase padding
    plt.subplots_adjust(left=0.3, bottom=0.3)

    # Colorbar formatting
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(["-1", "-0.5", "0", "0.5", "1"])

    if save_path:
        hd.save_figure(plt.gcf(), save_path)

    return corr

def _collapse_shap_by_group(shap_expl, feature_groups):
    values = shap_expl.values
    base_values = shap_expl.base_values
    data = shap_expl.data
    feature_names = list(shap_expl.feature_names)

    shap_df = pd.DataFrame(values, columns=feature_names)
    data_df = pd.DataFrame(data, columns=feature_names)

    grouped_shap = {}
    grouped_data = {}

    for group_name, group_features in feature_groups.items():
        valid_feats = [f for f in group_features if f in shap_df.columns]
        if not valid_feats:
            continue
        # Sum SHAP values per sample
        grouped_shap[group_name] = shap_df[valid_feats].sum(axis=1)
        # Optional: mean (or first) raw feature value for plotting (color bar)
        grouped_data[group_name] = data_df[valid_feats].mean(axis=1)

    grouped_shap_df = pd.DataFrame(grouped_shap)
    grouped_data_df = pd.DataFrame(grouped_data)

    return shap.Explanation(
        values=grouped_shap_df.values,
        base_values=base_values,
        data=grouped_data_df.values,
        feature_names=list(grouped_shap_df.columns)
    )

def _generate_group_names(feature_groups):
    """
    feature_groups: dict {group_id: [feature1, feature2, ...]}
    returns: dict {group_id: readable_group_name}
    """
    group_names = {}

    for group_id, features in feature_groups.items():
        features = sorted(features)

        if len(features) == 1:
            group_names[group_id] = features[0]
            continue

        # Try to detect same base feature with different numeric values in parentheses
        base_names = []
        values = []

        pattern = re.compile(r"^(.*?)(?:\s*\(([\d\.]+)\s*s\))?$")

        for f in features:
            match = pattern.match(f)
            if match:
                base_names.append(match.group(1).strip())
                if match.group(2) is not None:
                    values.append(float(match.group(2)))
            else:
                base_names.append(f)

        # Check if all base names are identical and we extracted numbers
        if len(set(base_names)) == 1 and len(values) == len(features):
            base = base_names[0]
            min_v = min(values)
            max_v = max(values)
            group_names[group_id] = f"{base} ({min_v:.2f} – {max_v:.2f} s)"
        else:
            # Fallback: first feature + count
            group_names[group_id] = f"{features[0]} (+{len(features)-1})"

    return group_names

def calculate_correlation_and_feature_groups(df_features):
    
    # remove columns with zero variance
    df_features = df_features.loc[:, df_features.nunique() > 1]

    # drop non-feature columns and calculate correlation matrix
    corr = df_features.drop(columns=['y', 'chip', 'days_after_treatment']).corr(method='spearman')

    # cluster features based on correlation
    distance = 1 - np.abs(corr)
    
    Z = linkage(distance, method='average')
    clusters = fcluster(Z, t=0.3, criterion='distance')  # t=0.3 -> >0.7 correlation belongs together

    # Build feature groups
    feature_groups = {}
    for feature, cluster_id in zip(corr.columns, clusters):
        feature_groups.setdefault(cluster_id, []).append(feature)

    # Generate readable group names 
    group_names = _generate_group_names(feature_groups)

    # replace the group ids with the genenerated group names
    feature_groups_named = {
        group_names[group_id]: features
        for group_id, features in feature_groups.items()
        }
    return feature_groups_named, corr

def _combine_shap_explanations(shap_values_list, class_index=1):
    """
    Combine SHAP values from multiple Explanation objects manually.
    """
    all_values = []
    all_base_values = []
    all_data = []

    for expl in shap_values_list:
        values = expl.values
        if values.ndim == 3:  # multiclass
            values = values[:, :, class_index]

        base_values = expl.base_values
        if np.ndim(base_values) > 1:
            base_values = base_values[:, class_index]
        
        all_values.append(values)
        all_base_values.append(base_values)
        all_data.append(expl.data)

    combined = shap.Explanation(
        values=np.concatenate(all_values, axis=0),
        base_values=np.concatenate(all_base_values, axis=0),
        data=np.concatenate(all_data, axis=0),
        feature_names=shap_values_list[0].feature_names
    )
    return combined

def _combine_all_shap_values(shap_values_list, class_index=1):
    """
    Combine SHAP values across splits and return full matrix of per-sample SHAP values.
    """
    clean_shap_values_list = []
    for shap_values in shap_values_list:
        values = shap_values.values
        if values.ndim == 3:  # multi-class
            values = values[:, :, class_index]
        clean_shap_values_list.append(values)

    combined_shap = np.concatenate(clean_shap_values_list, axis=0)
    return combined_shap

def _combine_shap_values_across_splits(shap_values_list):

    combined_shap = _combine_all_shap_values(shap_values_list)

    # Calculate mean over all samples per split
    mean_abs_shap = np.abs(combined_shap).mean(axis=0)

    # Create a DataFrame
    shap_values_df = pd.DataFrame(mean_abs_shap, columns=X_train_list[0].columns)

    # Calculate median and min/max across all splits
    median_shap_values = shap_values_df.median()
    min_shap_values = shap_values_df.min()
    max_shap_values = shap_values_df.max()

    return median_shap_values, min_shap_values, max_shap_values

def plot_all_shap_values(base_folder, model_name, feature_groups=None):

    # TODO Here mean over all windows? (currently each window is one data point in the shap plot)
    # if yes, use chip_train_list for this

    # Define filename to load
    full_path = os.path.join(base_folder, "shap_values_" + model_name + ".pkl")

    data = hd.load_obj_from_pkl(full_path)
    shap_values_list = data['shap_values_list']
    #X_train_sampled_list = data['X_train_sampled_list']
    combined_shap = _combine_shap_explanations(shap_values_list, class_index=1)

    # Aggregate SHAP values by feature groups if provided
    if feature_groups is not None:
        grouped_shap_values = _collapse_shap_by_group(combined_shap, feature_groups)
        print(grouped_shap_values)
        combined_shap = grouped_shap_values
        output_path = os.path.join(base_folder, "SHAP_all_grouped_" + model_name + ".jpg")
    else:
        output_path = os.path.join(base_folder, "SHAP_all_" + model_name + ".jpg")
    
    _plot_shap_values(combined_shap, output_path)
    _plot_shap_values(combined_shap, output_path.replace("jpg", "pdf"))

def _plot_shap_values(shap_values, output_path):
    # Plot the SHAP values
    # Create subplots
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    # shap.summary_plot(shap_values, X_train, plot_type="bar", class_names=["00bic", "10bic"])

    # if dimension is 3, it is a multiclass format
    if len(shap_values.shape) == 3:
        shap.plots.beeswarm(shap_values[:, :, 1])  # only plot class1
    else:
        shap.plots.beeswarm(shap_values)


    hd.save_figure(fig, output_path)
    #hd.save_figure(fig, output_path.replace("jpg", "pdf"))

def OLDplot_shap_correlation(base_folder, models):
    shap_model_list = []

    # Loop through all models to load and compute median SHAP values
    for model_name in models:
        print(f"Processing model: {model_name}")

        # Define filename to load
        full_path = os.path.join(base_folder, f"shap_values_list_{model_name}.pkl")
        shap_values_list = hd.load_obj_from_pkl(full_path)

        # Combine SHAP values across splits to get median values
        median_shap_values, _, _ = _combine_shap_values_across_splits(shap_values_list)

        # Append the median SHAP values (1D array) for this model
        shap_model_list.append(median_shap_values)

    # Stack the list into a DataFrame to create a model-feature matrix
    shap_df = pd.DataFrame(shap_model_list, index=models).T

    # Calculate the correlation matrix between models based on SHAP values
    correlation_matrix = shap_df.corr()

    # Plot the correlation matrix as a heatmap
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', vmin=-1, vmax=1,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title("Correlation of SHAP Values Between Models")
    plt.xlabel("Models")
    plt.ylabel("Models")

    full_path = os.path.join(base_folder, "SHAP_correlation.pdf")
    hd.save_figure(fig, full_path)

def _plot_shap_dependence(shap_values, X_train, feature_name, scaler, scaled_columns, output_path, interaction_index=None):
    """
    Generate a SHAP dependence plot for a specific feature.

    Args:
        shap_values: Array-like SHAP values for the dataset.
        X_train: DataFrame of training features.
        feature_name: The name of the feature to plot.
        output_path: Path to save the dependence plot.
        interaction_index: Optional, name of the feature to show interaction effects.
    """
    # Multi-class case: pick SHAP values for the positive class (e.g., class 1)
    if len(shap_values.shape) == 3:
        if shap_values.shape[2] == 2:  # Multi-class SHAP values
            shap_values = shap_values[:, :, 1]  # Use SHAP values for class 1

    # Calculate the mean absolute SHAP values for each feature
    mean_abs_shap = np.median(np.abs(shap_values.values), axis=0)
    
    # Get the most important feature
    most_important_feature = X_train.columns[np.argmax(mean_abs_shap)]
    print(f"Most important feature: {most_important_feature}")

    # Inverse-transform all features
    X_train_rescaled = _inverse_transform_selected(X_train, scaler, scaled_columns)

    #fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    shap.dependence_plot(
        feature_name,
        shap_values.values,
        X_train_rescaled,
        interaction_index=most_important_feature,
        show=False  # Suppress showing the plot in interactive environments
    )

    hd.save_figure(plt.gcf(), output_path)
    hd.save_figure(plt.gcf(), output_path.replace("jpg", "pdf"))

def _plot_combined_shap_dependence(shap_pkl_path, scaler, scaled_columns):
    """
    Combine SHAP values from multiple splits and plot combined dependence.
    """
    data = hd.load_obj_from_pkl(shap_pkl_path)
    shap_values_list = data['shap_values_list']
    X_train_sampled_list = data['X_train_sampled_list']

    # define which feature should be on the x axis
    feature_name = "days_after_treatment"

    # combine all shap values and X_train samples
    combined_shap = _combine_all_shap_values(shap_values_list, class_index=1)
    combined_X = pd.concat(X_train_sampled_list, axis=0)

    # Determine most important feature
    median_abs_shap = np.median(np.abs(combined_shap), axis=0)
    most_important_feature = combined_X.columns[np.argmax(median_abs_shap)]
    print("Most important feature: " + most_important_feature)

    # Inverse scale all features using the fitted scaler
    combined_X_rescaled = _inverse_transform_selected(combined_X, scaler, scaled_columns)

    shap.dependence_plot(
        feature_name,
        combined_shap,
        combined_X_rescaled,
        interaction_index=most_important_feature,
        show=False
    )

    return plt.gcf()

def plot_all_shap_dependence(base_folder, models, scaler, scaled_columns):
    # Loop through all models
    for m in range(len(models)):
        model_name = models[m]
        print(model_name)

        # Define filename to load
        shap_pkl_path = os.path.join(base_folder, "shap_values_" + model_name + ".pkl")

        fig = _plot_combined_shap_dependence(shap_pkl_path, scaler, scaled_columns)

        full_path = os.path.join(base_folder, "SHAP_all_dependence_" + model_name + ".jpg")
        hd.save_figure(fig, full_path)

        full_path = os.path.join(base_folder, "SHAP_all_dependence_" + model_name + ".pdf")
        hd.save_figure(fig, full_path)
    
def OLDcalculate_shap_values(base_folder, model_name, X_train_list, df_train_results):
    """
    Calculate SHAP values for a single model and save them to disk.
    """
    full_path = os.path.join(base_folder, f"shap_values_{model_name}.pkl")

    if os.path.isfile(full_path):
        print("SHAP values already calculated: " + full_path)
        return True

    shap_values_list = []
    X_train_sampled_list = []

    # Filter rows for this model only
    df_model_results = df_train_results[df_train_results['Model'] == model_name].reset_index(drop=True)

    if len(df_model_results) != len(X_train_list):
        print("Mismatch between number of splits and model results.")
        return False

    for i in range(len(X_train_list)):
        split_model_row = df_model_results.iloc[i]
        best_model = split_model_row['clf_best']
        X_train = X_train_list[i]

        # Sample rows for speed if desired
        X_train_sampled = X_train  # or X_train.sample(500, random_state=42)
        X_train_sampled_list.append(X_train_sampled)

        # Create the explainer
        np.random.seed(42)
        if model_name in ["RF"]:
            background = shap.sample(X_train, 50)
            explainer = shap.TreeExplainer(best_model, data=background, feature_perturbation="interventional")
        elif model_name in ["XGboost", "LightGBM", "CatBoost"]:
            background = shap.sample(X_train, 50)
            explainer = shap.TreeExplainer(best_model, data=background, feature_perturbation="interventional", model_output="probability")
        elif model_name in ["SVM", "LR", "KNN", "MLP"]:
            background = shap.kmeans(X_train, 50)
            explainer = shap.KernelExplainer(best_model.predict_proba, background)
        elif model_name == "NB":
            background = shap.sample(X_train, 50)
            explainer = shap.Explainer(best_model.predict_proba, background)
        else:
            raise ValueError("Unknown model type: " + model_name)

        if model_name in ["RF", "XGboost", "LightGBM", "CatBoost"]:
            shap_values = explainer(X_train_sampled, check_additivity=False)
        else:
            shap_values = explainer(X_train_sampled)
        shap_values_list.append(shap_values)

        # Plot individual SHAP values
        output_path = os.path.join(base_folder, f"shap_{model_name}_split_{i}.jpg")
        _plot_shap_values(shap_values, output_path=output_path)

        output_path_dep = os.path.join(base_folder, f"shap_dependence_{model_name}_split_{i}.jpg")
        #try:
        #    _plot_shap_dependence(
        #        shap_values,
        #        X_train_sampled,
        #        feature_name="days_after_treatment",
        #        output_path=output_path_dep,
        #        interaction_index=None
        #    )
        #except Exception as e:
        #    print(e)

    # Save both SHAP values and sampled X_train
    hd.save_obj_as_pkl({
        'shap_values_list': shap_values_list,
        'X_train_sampled_list': X_train_sampled_list
    }, full_path)
    print("Saved SHAP values and sampled X_train to: " + full_path)

    return True

def calculate_shap_values(base_folder, model_name, X_train_list, chip_train_list, df_train_results):
    """
    Calculate SHAP values for a single model on outer test sets and save them to disk.
    Stores chip IDs for subject-level aggregation.
    """
    full_path = os.path.join(base_folder, f"shap_values_{model_name}.pkl")

    if os.path.isfile(full_path):
        print("SHAP values already calculated: " + full_path)
        return True

    shap_values_list = []
    X_train_sampled_list = []
    chip_id_list = []

    # Filter rows for this model only
    df_model_results = df_train_results[df_train_results['Model'] == model_name].reset_index(drop=True)

    if len(df_model_results) != len(X_train_list):
        print("Mismatch between number of splits and model results.")
        return False

    for i in range(len(X_train_list)):
        split_model_row = df_model_results.iloc[i]
        best_model = split_model_row['clf_best']
        X_train = X_train_list[i]
        chip_ids = chip_train_list[i]  # array of chip IDs (should all be same for LOGO)

        # Sample rows if desired (usually NOT for test SHAP)
        X_train_sampled = X_train
        X_train_sampled_list.append(X_train_sampled)
        chip_id_list.append(chip_ids)

        # Create the explainer
        np.random.seed(42)
        if model_name in ["RF"]:
            background = shap.sample(X_train, 50)
            explainer = shap.TreeExplainer(best_model, data=background, feature_perturbation="interventional")
        elif model_name in ["XGboost", "LightGBM", "CatBoost"]:
            background = shap.sample(X_train, 50)
            explainer = shap.TreeExplainer(
                best_model,
                data=background,
                feature_perturbation="interventional",
                model_output="probability"
            )
        elif model_name in ["SVM", "LR", "KNN", "MLP"]:
            background = shap.kmeans(X_train, 50)
            explainer = shap.KernelExplainer(best_model.predict_proba, background)
        elif model_name == "NB":
            background = shap.sample(X_train, 50)
            explainer = shap.Explainer(best_model.predict_proba, background)
        else:
            raise ValueError("Unknown model type: " + model_name)

        # Compute SHAP
        if model_name in ["RF", "XGboost", "LightGBM", "CatBoost"]:
            shap_values = explainer(X_train_sampled, check_additivity=False)
        else:
            shap_values = explainer(X_train_sampled)

        shap_values_list.append(shap_values)

        # Plot individual SHAP values
        output_path = os.path.join(base_folder, f"shap_{model_name}_split_{i}.jpg")
        _plot_shap_values(shap_values, output_path=output_path)

    # Save SHAP values, test data, and chip IDs
    hd.save_obj_as_pkl({
        'shap_values_list': shap_values_list,
        'X_train_sampled_list': X_train_sampled_list,
        'chip_id_list': chip_id_list
    }, full_path)

    print("Saved SHAP values and chip IDs to: " + full_path)
    return True

def _inverse_transform_selected(X_train, scaler, scaled_columns):
    X_rescaled = X_train.copy()

    # Apply inverse_transform only on the scaled subset
    X_rescaled[scaled_columns] = pd.DataFrame(
        scaler.inverse_transform(X_train[scaled_columns]),
        columns=scaled_columns,
        index=X_train.index
    )
    return X_rescaled

# TODO
def TODO_plot_shap_with_reference(combined_shap, combined_X, feature_name="days_after_treatment", class_index=1):
    """
    Plots a SHAP dependence plot for a feature with a horizontal reference line
    at the mean absolute SHAP value across all features, and a bar plot ranking
    features by mean absolute SHAP.
    
    Args:
        combined_shap: np.array or shap values array, concatenated across splits
        combined_X: pd.DataFrame, features corresponding to combined_shap
        feature_name: str, the feature to show on the x-axis
        class_index: int, which class for SHAP values if multiclass
    """
    # Convert to array if necessary (e.g., shap.Explanation object)
    if hasattr(combined_shap, 'values'):
        shap_values_array = combined_shap.values[..., class_index] if combined_shap.values.ndim == 3 else combined_shap.values
    else:
        shap_values_array = combined_shap

    # Compute mean absolute SHAP for each feature
    mean_abs_shap = np.mean(np.abs(shap_values_array), axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=combined_X.columns).sort_values(ascending=False)
    
    # Compute global reference line (mean absolute SHAP across all features)
    global_mean_abs = mean_abs_shap.mean()
    
    print("Feature ranking by mean absolute SHAP:")
    print(feature_importance)
    print("Global mean absolute SHAP (reference line):", global_mean_abs)
    
    # Dependence plot
    plt.figure(figsize=(10, 5))
    shap.dependence_plot(
        feature_name,
        shap_values_array,
        combined_X,
        interaction_index=None,
        show=False
    )
    
    # Add horizontal reference line
    plt.axhline(global_mean_abs, color='red', linestyle='--', alpha=0.5, label='Global mean |SHAP|')
    plt.legend()
    plt.title("SHAP Dependence with Reference Line")
    plt.tight_layout()
    plt.show()
    
    # Bar plot of feature importance ranking
    plt.figure(figsize=(12, 6))
    feature_importance.plot(kind='bar')
    plt.axhline(global_mean_abs, color='red', linestyle='--', alpha=0.5, label='Global mean |SHAP|')
    plt.ylabel("Mean Absolute SHAP Value")
    plt.title("Feature Importance Ranking")
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def _get_path_experiment_list():
    # define parameter
    bin_sizes = settings.BIN_SIZES
    window_sizes = settings.WINDOW_SIZES
    window_overlaps = settings.WINDOW_OVERLAPS

    path_experiment_list = folder_structure.generate_paths(target_data_folder=SOURCE_DATA_FOLDER,
                                                           bin_sizes=bin_sizes,
                                                           window_sizes=window_sizes,
                                                           window_overlaps=window_overlaps,
                                                           chip_names=[],
                                                           groups=[])

    return path_experiment_list


def plot_feature_groups(feature_groups,
                        n_columns=3,
                        figsize=(18, 12),
                        title="Feature Groups",
                        save_path=None):
    """
    Publication-style multi-column feature group visualization.

    Parameters
    ----------
    feature_groups : dict
        {group_name: [feature1, feature2, ...]}
    n_columns : int
        Number of columns.
    figsize : tuple
        Figure size.
    title : str
        Figure title.
    save_path : str or None
        If provided, saves figure to this path.
    """

    # Convert dict to list for column splitting
    groups = list(feature_groups.items())
    n_groups = len(groups)
    groups_per_col = math.ceil(n_groups / n_columns)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    col_width = 1 / n_columns
    y_start = 0.98

    for col in range(n_columns):
        x = col * col_width + 0.02
        y = y_start

        start_idx = col * groups_per_col
        end_idx = min((col + 1) * groups_per_col, n_groups)

        for group_name, features in groups[start_idx:end_idx]:

            # Plot group name
            ax.text(x, y, group_name,
                    fontsize=14,
                    fontweight="bold",
                    transform=ax.transAxes,
                    verticalalignment="top")
            y -= 0.035

            # Plot wrapped features
            for feature in features:
                wrapped = textwrap.fill(feature.replace("_", " "), width=30)
                ax.text(x + 0.015, y, wrapped,
                        fontsize=9,
                        transform=ax.transAxes,
                        verticalalignment="top")
                y -= 0.025

            y -= 0.02  # spacing between groups

    fig.suptitle(title, fontsize=18, y=0.995)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

if __name__ == '__main__':
    # for all machine learning folder:
    for FOLDER in settings.FEATURE_SET_LIST:

        print(FOLDER)

        SOURCE_DATA_FOLDER = FOLDER.replace("0_feature_set", "1_ML")
        TARGET_DATA_FOLDER = FOLDER.replace("0_feature_set", "2_SHAP")

        # get paths of all experiments
        path_experiment_list = _get_path_experiment_list()

        # for all experiments
        for path_experiment in path_experiment_list:

            # define path of feature-set (needed for correlation and feature grouping)
            feature_path = path_experiment.replace(SOURCE_DATA_FOLDER, FOLDER)
            # define source path
            full_path = os.path.join(feature_path, 'feature_set_preprocessed.csv')
            
            
            # load the feature dataframes
            df_features = hd.load_csv_as_df(full_path, index_col=False)

            # drop columns that are not needed for the machine learning
            df_features = df_features.drop(columns=['group', 'recording', 'file']) 


            feature_groups, corr = calculate_correlation_and_feature_groups(df_features)
            corr_path = os.path.join(path_experiment.replace(SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER), "feature_correlation_matrix.jpg")
            plot_feature_correlation(corr, save_path=corr_path)
            feature_groups_path = os.path.join(path_experiment.replace(SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER), "feature_groups.pdf")
            plot_feature_groups(feature_groups, save_path=feature_groups_path)

            # save feature groups
            df_feature_groups = pd.DataFrame([feature_groups])
            feature_groups_path = os.path.join(path_experiment.replace(SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER), "feature_groups.csv")
            hd.save_df_as_csv(df_feature_groups, feature_groups_path)

            # get all folder names in the source path
            source_path = path_experiment
            models = [f for f in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, f))]
            
            for model in models:
                
                model_path = os.path.join(source_path, model)
                days = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]
            
                for day in days:

                    day_path = os.path.join(model_path, day)

                    target_path = day_path.replace(SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER)

                    # load ml results
                    df_train_results, df_test_results, X_train_list, y_train_list, X_test_list, y_test_list, chip_train_list, y_pred_all, y_true_all = workflow_unpaired.load_df_results_from_hd(day_path)

                    # IMPORTANT: use x_train or x_test for shap calculation? 
                    # actually use x_test (=unseen data) since here we use ML as an statistical test.
                    # But in x_train are more data points (=more stable results) and it does not differ too much from x_train shap results
                    # since the cross-validation already prevented overfitting

                    # calculate shap values
                    #OLDcalculate_shap_values(target_path, model, X_train_list, df_train_results)
                    try:
                        calculate_shap_values(target_path, model, X_train_list, chip_train_list, df_train_results)
                    except Exception as e:
                        print(e)

                    # plot all shap value dependence
                    #try:
                    #    plot_all_shap_dependence(target_folder_full, models)
                    #except Exception as e:
                    #    print(e)

                    # plot shap all values
                    try:
                        plot_all_shap_values(target_path, model, feature_groups)
                    except Exception as e:
                        print(e)

                    # plot shap correlation
                    #plot_shap_correlation(target_folder, models)