import os
from pathlib import Path
import shutil

def copy_plots_from_results_to_plots(plots_dir, subfolder):
    # Create corresponding folder in plots/
    output_subfolder = plots_dir / subfolder.name
    output_subfolder.mkdir(parents=True, exist_ok=True)

    # copy whole statistics folder
    source = subfolder / "6_0_statistics"
    destination = output_subfolder / "6_0_statistics"
    shutil.copytree(source, destination, dirs_exist_ok=True)
    #print("Copied statistics folder")

    for folder in subfolder.iterdir():

        if folder.name == "3_1_ML_all":
            filename = "AUC_summary_CatBoost.pdf"
            source = folder / filename
            destination = output_subfolder / filename
            shutil.copy2(source, destination)

        if folder.name == "3_2_SHAP_all":

            filename = "feature_groups.pdf"
            source = folder / "bin_1.0 ms/window_240.0 s/overlap_0.75/" /filename
            destination = output_subfolder / folder.name / filename
            shutil.copy2(source, destination)

            folder2 = folder / "bin_1.0 ms/window_240.0 s/overlap_0.75/CatBoost"
            for folder3 in folder2.iterdir():
                new_folder = output_subfolder / folder.name / folder3.name
                new_folder.mkdir(parents=True, exist_ok=True)
                filename = "SHAP_all_grouped_CatBoost.pdf"
                source = folder3 / filename
                destination = new_folder / filename
                shutil.copy2(source, destination)

            

if __name__ == '__main__':
    

    # Main folders
    results_dir = Path("results")
    plots_dir = Path("plots")

    # Create the output directory if it doesn't exist
    plots_dir.mkdir(exist_ok=True)

    # Go through all subfolders in results/
    for subfolder in results_dir.iterdir():

        if not subfolder.is_dir():
            continue

        print("Copying plots of folder: " + subfolder.name)

        try:
            copy_plots_from_results_to_plots(plots_dir, subfolder)
        except Exception as e:
            print(f"Error occurred while copying plots from {subfolder.name}: {e}")

    print("Done!")
