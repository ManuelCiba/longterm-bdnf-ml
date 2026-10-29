from ast import If
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

if settings.WELLS_SHAM == []:
    from lib.machine_learning import workflow_paired_optatune as workflow_ml
else:
    from lib.machine_learning import workflow_unpaired_optatune as workflow_ml

TOP_N_SHAP_FEATURES = 10  # number of top features to plot 
FLAG_OVERWRITE = False  # overwrite existing SHAP values and plots


###############################
# Main shap calculation (and shap interaction) function
###############################

def calculate_shap_values(base_folder, model_name, X_train_list, chip_train_list, df_train_results, feature_groups, top_n=10):
    """
    Calculate SHAP values for a single model on outer test sets and save them to disk.
    Stores chip IDs for subject-level aggregation.
    """
    full_path = os.path.join(base_folder, f"shap_values_{model_name}.pkl")

    if os.path.isfile(full_path) and not FLAG_OVERWRITE:
        print("SHAP values already calculated: " + full_path)
        return True

    shap_values_list = []
    shap_interaction_values_list = []
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

        # Compute Shap interaction values if supported
        if model_name in ["RF", "XGboost", "LightGBM", "CatBoost"]:
            shap_interaction_values = None
            # create a new explainer for interaction values
            explainer_interaction = shap.TreeExplainer(
                            best_model,
                            data=background,
                            feature_perturbation="tree_path_dependent",
                            model_output="raw" # needed for interaction values 
                        )
            try:
                shap_interaction_values = explainer_interaction.shap_interaction_values(X_train_sampled)    
            except Exception as e:
                print(f"Error occurred while computing SHAP interaction values for {model_name}: {e}")
            shap_interaction_values_list.append(shap_interaction_values)
           

        # Plot individual SHAP values
        output_path = os.path.join(base_folder, f"shap_{model_name}_split_{i}.jpg")
        _plot_shap_values(shap_values, output_path=output_path, feature_groups=feature_groups, top_n=top_n)
        
        feature_names = list(X_train_sampled.columns)
        output_path = output_path.replace(".jpg", "_interaction.jpg")
        shap_interaction_matrix = _average_across_windows(shap_interaction_values)
        _plot_shap_interaction(shap_interaction_matrix=shap_interaction_matrix, output_path=output_path, feature_names=feature_names, feature_groups=feature_groups, top_n=top_n)

    # Save SHAP values, test data, and chip IDs
    hd.save_obj_as_pkl({
        'shap_values_list': shap_values_list,
        'X_train_sampled_list': X_train_sampled_list,
        'chip_id_list': chip_id_list,
        'shap_interaction_values_list': shap_interaction_values_list
    }, full_path)

    print("Saved SHAP values and chip IDs to: " + full_path)
    return True


###############################
# Feature Grouping
###############################

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
        hd.save_figure(fig, save_path)

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

    from scipy.spatial.distance import squareform

    # remove columns with zero variance
    df_features = df_features.loc[:, df_features.nunique() > 1]

    # drop non-feature columns and calculate correlation matrix
    corr = df_features.drop(columns=settings.NOT_NEEDED_FOR_SHAP_CORRELATION).corr(method='spearman')

    # Cluster features based on absolute Spearman correlation
    distance = 1 - np.abs(corr)

    # Ensure numerical symmetry and a zero diagonal (maybe not needed)
    #distance = (distance + distance.T) / 2
    #np.fill_diagonal(distance, 0.0)

    # Convert square distance matrix to condensed form (required by linkage function)
    condensed_distance = squareform(distance.to_numpy())

    # 'complete': only features with correlation > threshold, are added to cluster
    # 'average': features with correlation > threshold are added to cluster, but also features with lower correlation to the cluster center
    Z = linkage(condensed_distance, method='complete') 
    clusters = fcluster(Z, t=0.3, criterion='distance')

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


###############################
# Shap Values
###############################

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

def plot_all_shap_values(base_folder, model_name, feature_groups=None, max_display=10):

    # TODO Here mean over all windows? (currently each window is one data point in the shap plot)
    # AI answer: 
    # Want every observation to have equal weight? → current approach is appropriate; combine all SHAP observations.
    # Want every window/chip/subject to have equal weight? → first average within each window/chip, then average those means.
    # If your windows correspond to the same underlying chips/subjects measured repeatedly, I would generally prefer the second approach if your scientific question is "what is the typical importance across chips?" rather than "what is the importance across individual observations?"

    # Define filename to load
    full_path = os.path.join(base_folder, "shap_values_" + model_name + ".pkl")

    data = hd.load_obj_from_pkl(full_path)
    shap_values_list = data['shap_values_list']
    #X_train_sampled_list = data['X_train_sampled_list']
    combined_shap = _combine_shap_explanations(shap_values_list, class_index=1)

    # Aggregate SHAP values by feature groups if provided
    #if feature_groups is not None:
    #    grouped_shap_values = _collapse_shap_by_group(combined_shap, feature_groups)
    #    #print(grouped_shap_values)
    #    combined_shap = grouped_shap_values
    #    output_path = os.path.join(base_folder, "SHAP_all_grouped_" + model_name + ".jpg")
    #else:
    #    output_path = os.path.join(base_folder, "SHAP_all_" + model_name + ".jpg")
    
    output_path = os.path.join(base_folder, "SHAP_all_grouped_" + model_name + ".jpg")
    
    _plot_shap_values(combined_shap, output_path, feature_groups=feature_groups, top_n=max_display)
    #_plot_shap_values(combined_shap, output_path.replace("jpg", "pdf"), max_display)

def _plot_shap_values(shap_values, output_path, feature_groups, top_n=10):
    # Plot the SHAP values
    # Create subplots
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    # shap.summary_plot(shap_values, X_train, plot_type="bar", class_names=["00bic", "10bic"])

    grouped_shap_values = _collapse_shap_by_group(shap_values, feature_groups)

    shap.plots.beeswarm(grouped_shap_values, max_display=top_n)  # limit to top 20 features

    hd.save_figure(fig, output_path)
    #hd.save_figure(fig, output_path.replace("jpg", "pdf"))
    _save_grouped_shap_summary(grouped_shap_values, output_path.replace(".jpg", ".csv"), feature_groups=feature_groups)

def _save_grouped_shap_summary(
    grouped_shap_values,
    output_path,
    feature_groups=None,
):
    """
    Save all grouped features ranked by mean absolute SHAP value.

    Assumes:
      grouped_shap_values.values: (n_samples, n_groups)
      grouped_shap_values.data:   (n_samples, n_groups)

    Positive SHAP values push the explained model output higher;
    negative SHAP values push it lower.
    """

    shap_values = np.asarray(grouped_shap_values.values)
    feature_values = np.asarray(grouped_shap_values.data)
    feature_names = list(grouped_shap_values.feature_names)

    if shap_values.ndim != 2:
        raise ValueError(
            f"Expected 2D SHAP values, got {shap_values.shape}"
        )

    if feature_values.shape != shap_values.shape:
        raise ValueError(
            "Feature values and SHAP values must have the same shape. "
            f"Got {feature_values.shape} and {shap_values.shape}."
        )

    rows = []

    for j, name in enumerate(feature_names):
        sv = shap_values[:, j]
        fv = feature_values[:, j]

        valid = np.isfinite(sv) & np.isfinite(fv)
        sv = sv[valid]
        fv = fv[valid]

        # Correlation between feature value and signed SHAP value
        if len(sv) > 1 and np.std(fv) > 0 and np.std(sv) > 0:
            value_shap_corr = np.corrcoef(fv, sv)[0, 1]
        else:
            value_shap_corr = np.nan

        # Compare SHAP contributions for lower vs higher feature values
        if len(fv) > 0:
            median_value = np.median(fv)
            low = fv <= median_value
            high = fv > median_value

            mean_shap_low = (
                np.mean(sv[low]) if np.any(low) else np.nan
            )
            mean_shap_high = (
                np.mean(sv[high]) if np.any(high) else np.nan
            )

            mean_feature_value = np.mean(fv)
            median_feature_value = np.median(fv)
        else:
            mean_shap_low = np.nan
            mean_shap_high = np.nan
            mean_feature_value = np.nan
            median_feature_value = np.nan

        members = (
            feature_groups.get(name, [])
            if feature_groups is not None
            else []
        )

        rows.append({
            "feature_group": name,
            "member_features": ", ".join(map(str, members)),
            "mean_feature_value": mean_feature_value,
            #"median_feature_value": median_feature_value,
            "mean_abs_shap": np.mean(np.abs(sv)) if len(sv) else np.nan,
            #"mean_signed_shap": np.mean(sv) if len(sv) else np.nan,
            #"shap_std": np.std(sv) if len(sv) else np.nan,
            #"percent_shap_positive": (
            #    np.mean(sv > 0) * 100 if len(sv) else np.nan
            #),
            #"percent_shap_negative": (
            #    np.mean(sv < 0) * 100 if len(sv) else np.nan
            #),
            #"mean_shap_low_feature_value": mean_shap_low,
            #"mean_shap_high_feature_value": mean_shap_high,
            #"feature_value_shap_correlation": value_shap_corr,
        })

    results = pd.DataFrame(rows)

    results = results.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    results.insert(0, "rank", np.arange(1, len(results) + 1))

    hd.save_df_as_csv(results, output_path)


    return results


###############################
# Shap Interaction Values
###############################

def _plot_shap_interaction(shap_interaction_matrix, output_path, feature_names, feature_groups, top_n=10):

    grouped_matrix, group_names = _collapse_shap_interaction_by_group(
            shap_interaction_matrix,
            feature_names,
            feature_groups,
        )

    # Shape: (n_samples, n_groups, n_groups)
    # Sum signed interaction contributions to recover each group's
    # total SHAP value, including its main effect.
    importance= grouped_matrix.sum(axis=1)

    indices = np.argsort(importance)[::-1][:top_n]
    selected_names = np.array(group_names)[indices]

    # Keep the diagonal visible to compare main effects with
    # between-group interactions.
    heatmap = pd.DataFrame(
        grouped_matrix[np.ix_(indices, indices)],
        index=selected_names,
        columns=selected_names,
    )

    fig, ax = plt.subplots(
        figsize=(
            max(8, len(indices) * 0.7),
            max(6, len(indices) * 0.6),
        )
    )

    # Mask the upper triangle, but keep the diagonal visible.
    mask = np.triu(
        np.ones_like(heatmap, dtype=bool),
        k=1,
    )

    sns.heatmap(
        heatmap,
        #mask=mask,
        cmap="viridis",
        square=True,
        xticklabels=True,
        yticklabels=True,
        ax=ax,
    )

    ax.set_title(
        "SHAP Interaction Strength "
        "(Groups Ranked by Mean Absolute SHAP Value)"
    )
    ax.set_xlabel("Feature group")
    ax.set_ylabel("Feature group")

    ax.tick_params(axis="x", labelrotation=90)
    ax.tick_params(axis="y", labelrotation=0)

    fig.tight_layout()
    hd.save_figure(fig, output_path)
    hd.save_df_as_csv(heatmap, output_path.replace(".jpg", ".csv"))
    plt.close(fig)

    return heatmap, pd.Series(
        importance[indices],
        index=selected_names,
        name="mean_absolute_shap",
    )

def _average_across_windows(shap_interaction_values):
    """
    Average SHAP interaction values across windows (axis=0).
    Use absolute values since we are interested in the strength of interactions, not their direction.
    """
    return np.mean(np.abs(shap_interaction_values), axis=0)

def OLD_plot_shap_interaction(shap_interaction_values, output_path, feature_names, feature_groups, top_n=10):

    grouped_shap_values, group_names = _collapse_shap_interaction_by_group(shap_interaction_values, feature_names, feature_groups)

    # average over splits (use absolute values since we are interested in the strength, not the direction)
    mean_interactions = np.mean(np.abs(grouped_shap_values), axis=0)
    #np.fill_diagonal(mean_interactions, 0)

    # Select features with the strongest total interactions.
    importance = mean_interactions.sum(axis=0)
    indices = np.argsort(importance)[::-1][:top_n]

    heatmap = pd.DataFrame(
        mean_interactions[np.ix_(indices, indices)],
        index=np.array(group_names)[indices],
        columns=np.array(group_names)[indices],
    )

    # Keep only the lower triangle; hide the diagonal.
    #mask = np.triu(np.ones_like(heatmap, dtype=bool))

    plt.figure(figsize=(max(8, len(indices) * 0.6),
                        max(6, len(indices) * 0.5)))
    sns.heatmap(heatmap,
                #mask=mask, 
                cmap="viridis", 
                square=True,
                xticklabels=True,
                yticklabels=True,)
    plt.title("Mean Absolute SHAP Interaction Values")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    hd.save_figure(plt.gcf(), output_path)
    plt.close()

    return heatmap

def OLD_plot_shap_interaction(
    shap_interaction_values,
    output_path,
    feature_groups=None,
    top_n=10,
):
    """
    Plot a heatmap of mean absolute SHAP interaction values
    for the top-N most important features.

    Parameters
    ----------
    shap_interaction_values : np.ndarray or list
        Interaction values with shape:
        (n_samples, n_features, n_features)

        For multiclass models, may be a list of interaction
        arrays, one per class.

    output_path : str
        Base output path for the plot.

    feature_names : list, optional
        Names of the model input features.

    top_n : int
        Number of top features to display.
    """
    if shap_interaction_values is None:
        print("No SHAP interaction values to plot.")
        return

    # Handle multiclass output returned as a list.
    # For binary classification, select class 1.
    if isinstance(shap_interaction_values, list):
        shap_interaction_values = shap_interaction_values[1]

    # Handle shap.Explanation objects
    if isinstance(shap_interaction_values, shap.Explanation):
        if feature_names is None:
            feature_names = shap_interaction_values.feature_names
        shap_interaction_values = shap_interaction_values.values

    shap_interaction_values = np.asarray(shap_interaction_values)

    # Expected shape: (samples, features, features)
    if shap_interaction_values.ndim != 3:
        raise ValueError(
            "Expected SHAP interaction values with shape "
            "(n_samples, n_features, n_features), but got "
            f"{shap_interaction_values.shape}."
        )

    n_features = shap_interaction_values.shape[1]

    if shap_interaction_values.shape[2] != n_features:
        raise ValueError("Interaction matrix must be square.")

    if feature_names is None:
        feature_names = [
            f"Feature {i}" for i in range(n_features)
        ]

    if len(feature_names) != n_features:
        raise ValueError(
            "feature_names length does not match the number "
            "of features in the interaction values."
        )

    # Mean absolute interaction strength across observations
    mean_abs_interaction = np.mean(
        np.abs(shap_interaction_values),
        axis=0
    )

    # Rank features by total interaction strength
    # (includes main effects on the diagonal for ranking)
    importance = mean_abs_interaction.sum(axis=1)

    top_idx = np.argsort(importance)[::-1][:min(top_n, n_features)]

    top_names = [feature_names[i] for i in top_idx]

    # Extract top-N x top-N matrix
    interaction_matrix = mean_abs_interaction[
        np.ix_(top_idx, top_idx)
    ].copy()

    # Remove diagonal main effects to focus on pairwise interactions
    #np.fill_diagonal(interaction_matrix, 0)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        interaction_matrix,
        xticklabels=top_names,
        yticklabels=top_names,
        cmap="viridis",
        square=True,
        linewidths=0.4,
        cbar_kws={
            "label": "Mean absolute SHAP interaction value"
        },
        ax=ax,
    )

    ax.set_title(
        f"SHAP interaction heatmap — top {len(top_idx)} features"
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save JPG and PDF
    jpg_path = (
        output_path
        if output_path.lower().endswith(".jpg")
        else output_path + "_interaction.jpg"
    )
    pdf_path = jpg_path[:-4] + ".pdf"

    hd.save_figure(fig, jpg_path)
    plt.close(fig)

    return interaction_matrix

def OLD_collapse_shap_interaction_by_group(shap_interaction, feature_names, feature_groups):
    values = shap_interaction.values
    base_values = shap_interaction.base_values
    data = shap_interaction.data
    feature_names = list(shap_interaction.feature_names)

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

def _collapse_shap_interaction_by_group(shap_interaction_matrix, feature_names, feature_groups):

    feature_names = list(feature_names)


    feature_idx = {name: i for i, name in enumerate(feature_names)}
    group_names = []
    group_indices = []

    for group_name, features in feature_groups.items():
        indices = [
            feature_idx[f] for f in features if f in feature_idx
        ]
        if indices:
            group_names.append(group_name)
            group_indices.append(indices)

    n_groups = len(group_names)
    grouped = np.zeros((n_groups, n_groups))

    for i, idx_i in enumerate(group_indices):
        for j, idx_j in enumerate(group_indices):
            grouped[i, j] = (
                np.abs(shap_interaction_matrix[idx_i, :][:, idx_j]).sum(axis=(0, 1))
            )

    return grouped, group_names

def plot_all_shap_interaction(base_folder, model_name, feature_groups, max_display=10,
):
    full_path = os.path.join(
        base_folder,
        f"shap_values_{model_name}.pkl",
    )

    data = hd.load_obj_from_pkl(full_path)

    shap_interaction_values_list = data["shap_interaction_values_list"]

    # Use the actual feature names that were used by the model.
    if data.get("X_train_sampled_list"):
        feature_names = list(data["X_train_sampled_list"][0].columns)
    elif data.get("shap_values_list"):
        feature_names = list(data["shap_values_list"][0].feature_names)
    else:
        feature_names = None

    target_path = os.path.join(base_folder, "SHAP_all_interaction_" + model_name + ".jpg")

    # for each outer-crossvalidation split: combine all windows by taking the mean
    # before each list element has: n_windows*n_features*n_features
    # after : n_features*n_features
    shap_interaction_matrix_list = []
    for shap_interaction_values in shap_interaction_values_list:
        shap_interaction_matrix = _average_across_windows(shap_interaction_values)
        shap_interaction_matrix_list.append(shap_interaction_matrix)

    # Transform list into 3D array: n_splits*n_features*n_features
    shap_interaction_matrix_list = np.array(shap_interaction_matrix_list)

    # average across splits (use absolute values since we are interested in the strength, not the direction)
    # before n_splits*n_features*n_features
    # after: n_features*n_features
    shap_interaction_matrix = np.mean(shap_interaction_matrix_list, axis=0)

    return _plot_shap_interaction(
        shap_interaction_matrix=shap_interaction_matrix,
        output_path=target_path,
        feature_names=feature_names,
        feature_groups=feature_groups,
        top_n=max_display,
    )


###############################
# Currently not used
###############################

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
    feature_name = settings.X_LABEL

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
def TODO_plot_shap_with_reference(combined_shap, combined_X, feature_name="x_tick_labels", class_index=1):
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
    

###############################
# Helper
###############################

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



if __name__ == '__main__':

    top_n = TOP_N_SHAP_FEATURES  # number of top features to plot in shap plots

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



            feature_groups_path = os.path.join(path_experiment.replace(SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER), "feature_groups.csv")
            if 0: #os.path.isfile(feature_groups_path):
                print("Feature groups already calculated: " + feature_groups_path)
                df_feature_groups = hd.load_csv_as_df(feature_groups_path, index_col=False)
            else:
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
                    df_train_results, df_test_results, X_train_list, y_train_list, X_test_list, y_test_list, chip_train_list, y_pred_all, y_true_all = workflow_ml.load_df_results_from_hd(day_path)

                    # IMPORTANT: use x_train or x_test for shap calculation? 
                    # actually use x_test (=unseen data) since here we use ML as an statistical test.
                    # But in x_train are more data points (=more stable results) and it does not differ too much from x_train shap results
                    # since the cross-validation already prevented overfitting

                    # calculate shap values
                    #OLDcalculate_shap_values(target_path, model, X_train_list, df_train_results)
                    try:
                        calculate_shap_values(target_path, model, X_train_list, chip_train_list, df_train_results, feature_groups=feature_groups, top_n=top_n)
                    except Exception as e:
                        print(e)

                    # plot all shap value dependence
                    #try:
                    #    plot_all_shap_dependence(target_folder_full, models)
                    #except Exception as e:
                    #    print(e)

                    # plot shap all values
                    try:
                        plot_all_shap_values(target_path, model, feature_groups, max_display=top_n)
                        plot_all_shap_interaction(target_path, model, feature_groups, max_display=top_n)
                    except Exception as e:
                        print(e)

                    # plot shap correlation
                    #plot_shap_correlation(target_folder, models)