import os
from pathlib import Path
import shutil

def copy(source_folder_path, destination_folder_path, filename):

    source = os.path.join(source_folder_path, filename)
    destination = os.path.join(destination_folder_path, filename)

    os.makedirs(destination_folder_path, exist_ok=True)
    
    try:
        shutil.copy2(source, destination)
    except:
        print("Missing: " + destination)

def copy_plots_from_results_to_plots(plots_dir, subfolder):

    source_base = os.path.join(os.getcwd(), "results", subfolder.name)
    
    # Create corresponding folder in plots/
    destination_base = os.path.join(os.getcwd(), plots_dir.name, subfolder)
    os.makedirs(destination_base, exist_ok=True)

    # copy whole statistics folder
    source_path_full = os.path.join(source_base, "6_0_statistics")
    destination_path_full = os.path.join(destination_base, "6_0_statistics")
    shutil.copytree(source_path_full, destination_path_full, dirs_exist_ok=True)

    
    folders = [f for f in os.listdir(source_base) if os.path.isdir(os.path.join(source_base, f))]

    for folder in folders:
        
        if folder == "3_1_ML_all":
            filename = "AUC_summary_CatBoost.pdf"
            source_folder_path = os.path.join(source_base, folder)   
            destination_folder_path = os.path.join(destination_base, folder)
            copy(source_folder_path, destination_folder_path, filename)

            filename = "AUC_summary_CatBoost.csv"
            copy(source_folder_path, destination_folder_path, filename)

        if folder == "3_2_SHAP_all":

            filename = "feature_groups.pdf"
            source_folder_path = os.path.join(source_base, folder, "bin_1.0 ms/window_240.0 s/overlap_0.75/")
            destination_folder_path = os.path.join(destination_base, folder)
            copy(source_folder_path, destination_folder_path, filename)

            filename = "feature_groups.csv"
            copy(source_folder_path, destination_folder_path, filename)

            full_path_folder = os.path.join(source_base, folder, "bin_1.0 ms/window_240.0 s/overlap_0.75/CatBoost")
            folders2 = [f for f in os.listdir(full_path_folder) if os.path.isdir(os.path.join(full_path_folder, f))]

            for folder2 in folders2:
                
                destination_folder_path = os.path.join(destination_base, folder, folder2)
                os.makedirs(destination_folder_path, exist_ok=True)
            
                filename = "SHAP_all_grouped_CatBoost.jpg"
                source_folder_path = os.path.join(source_base, folder, "bin_1.0 ms/window_240.0 s/overlap_0.75/CatBoost", folder2)
                destination_folder_path = os.path.join(destination_base, folder, folder2)
                copy(source_folder_path, destination_folder_path, filename)

                filename = "SHAP_all_interaction_CatBoost.jpg"
                copy(source_folder_path, destination_folder_path, filename)

                files = [f for f in os.listdir(source_folder_path) if os.path.isfile(os.path.join(source_folder_path, f))]
                for file in files:
                    if file[-4:] != ".pkl":
                        filename = file
                        copy(source_folder_path, destination_folder_path, filename)


            

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
