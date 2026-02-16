import os
import quantities as pq
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lib.machine_learning import workflow_unpaired
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
        all_values.append(values)
        all_base_values.append(expl.base_values)
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

def plot_all_shap_values(base_folder, models):
    # Loop through all models
    for m in range(len(models)):
        model_name = models[m]
        print(model_name)

        # Define filename to load
        full_path = os.path.join(base_folder, "shap_values_" + model_name + ".pkl")

        data = hd.load_obj_from_pkl(full_path)
        shap_values_list = data['shap_values_list']
        #X_train_sampled_list = data['X_train_sampled_list']
        combined_shap = _combine_shap_explanations(shap_values_list, class_index=1)

        output_path = os.path.join(base_folder, "SHAP_all_" + model_name + ".jpg")
        _plot_shap_values(combined_shap, output_path)

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
    hd.save_figure(fig, output_path.replace("jpg", "pdf"))

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
    
def calculate_shap_values(base_folder, models, X_train_list, df_train_results, scaler, scaled_columns):
    """
    Calculate SHAP values for any number of models and save them to disk.
    """
    for m in range(len(models)):
        model_name = models[m]
        full_path = os.path.join(base_folder, "shap_values_" + model_name + ".pkl")
        
        if os.path.isfile(full_path):
            print("SHAP values already calculated: " + full_path)
            continue

        shap_values_list = []
        X_train_sampled_list = []

        for i in range(len(X_train_list)):
            # Get the best model for this split
            idx_model = (i * len(models)) + m
            split_model_row = df_train_results.iloc[idx_model]
            if model_name != split_model_row['Model']:
                print("Wrong Model order -> Cancel calculation")
                return False

            best_model = split_model_row['clf_best']
            X_train = X_train_list[i]

            # Sample rows for speed (e.g. 20 or 50 ect.)
            X_train_sampled = X_train.sample(500, random_state=42)
            X_train_sampled_list.append(X_train_sampled)

            # Create the explainer
            np.random.seed(42)
            if model_name in ["RF", "XGboost"]:
                background = shap.sample(X_train, 50)
                explainer = shap.TreeExplainer(best_model, data=background, feature_perturbation="interventional")
            elif model_name in ["SVM", "LR", "KNN", "MLP"]:
                background = shap.kmeans(X_train, 50)
                explainer = shap.KernelExplainer(best_model.predict_proba, background)
            elif model_name == "NB":
                background = shap.sample(X_train, 50)
                explainer = shap.Explainer(best_model.predict_proba, background)
            else:
                raise ValueError("Unknown model type: " + model_name)

            shap_values = explainer(X_train_sampled)
            shap_values_list.append(shap_values)

            # Plot individual SHAP values
            output_path = os.path.join(base_folder, "shap_" + model_name + "_split_" + str(i) + ".jpg")
            _plot_shap_values(shap_values, output_path=output_path)

            output_path_dep = os.path.join(base_folder, "shap_dependence_" + model_name + "_split_" + str(i) + ".jpg")
            try:
                _plot_shap_dependence(
                    shap_values,
                    X_train_sampled,
                    feature_name="days_after_treatment",
                    scaler = scaler,
                    scaled_columns = scaled_columns,
                    output_path=output_path_dep,
                    interaction_index=None
                )
            except Exception as e:
                print(e)

        # Save both SHAP values and sampled X_train
        hd.save_obj_as_pkl({
            'shap_values_list': shap_values_list,
            'X_train_sampled_list': X_train_sampled_list
        }, full_path)
        print("Saved SHAP values and sampled X_train to: " + full_path)

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
    

if __name__ == '__main__':
    # for all machine learning folder:
    for FOLDER in settings.FEATURE_SET_LIST:

        print(FOLDER)

        SOURCE_DATA_FOLDER = FOLDER.replace("0_feature_set", "1_ML")
        TARGET_DATA_FOLDER = FOLDER.replace("0_feature_set", "2_SHAP")

        source_folder = os.path.join(settings.PATH_RESULTS_FOLDER, SOURCE_DATA_FOLDER)
        target_folder = os.path.join(settings.PATH_RESULTS_FOLDER, TARGET_DATA_FOLDER)

        # Define parameter set for which SHAP will be calculated
        bin_size = 1 * pq.ms
        window_size = 240 * pq.s
        window_overlap = 0.75
        models = settings.ML_MODELS

        # define path where df_results are stored
        path_experiment = os.path.join(source_folder,
                                     f"bin_{bin_size}",
                                     f"window_{window_size}",
                                     f"overlap_{window_overlap}")

        target_folder_full = path_experiment.replace(SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER)

        # load scaler to inverse transform the feature values
        scaler_path = path_experiment
        scaler_path = scaler_path.replace(SOURCE_DATA_FOLDER, FOLDER)
        scaler_path_full = os.path.join(scaler_path, "scaler.pkl")
        data = hd.load_obj_from_pkl(scaler_path_full)
        scaler = data["scaler"]
        scaled_columns = data["columns"]

        # load ml results
        df_train_results, df_test_results, X_train_list, y_train_list, X_test_list, y_test_list = workflow_unpaired.load_df_results_from_hd(path_experiment)

        # IMPORTANT: use x_train or x_test for shap calculation? 
        # actually use x_test (=unseen data) since here we use ML as an statistical test.
        # But in x_train are more data points (=more stable results) and it does not differ too much from x_train shap results
        # since the cross-validation already prevented overfitting

        # calculate shap values
        calculate_shap_values(target_folder_full, models, X_train_list, df_train_results, scaler, scaled_columns)

        # plot all shap value dependence
        try:
            plot_all_shap_dependence(target_folder_full, models, scaler, scaled_columns)
        except Exception as e:
            print(e)

        # plot shap all values
        try:
            plot_all_shap_values(target_folder_full, models)
        except Exception as e:
            print(e)

        # plot shap correlation
        #plot_shap_correlation(target_folder, models)