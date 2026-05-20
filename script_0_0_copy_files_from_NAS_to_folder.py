import os
import shutil
from pathlib import Path

def copy_spk_files(source_root, destination_root):
    source_root = Path(source_root)
    destination_root = Path(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    copied_files = {}

    for dirpath, dirnames, filenames in os.walk(source_root):
        for filename in filenames:
            if filename.endswith(".spk"):
                src_file = Path(dirpath) / filename
                dest_file = destination_root / filename

                # Handle filename collisions
                if dest_file.exists():
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while True:
                        new_name = f"{base}_{counter}{ext}"
                        dest_file = destination_root / new_name
                        if not dest_file.exists():
                            break
                        counter += 1

                # Copy the file
                shutil.copy2(src_file, dest_file)
                print(f"Copied: {src_file} -> {dest_file}")

    print("\nDone copying .spk files.")

if __name__ == "__main__":
    # Define paths
    #source_nas_path = "/mnt/bmdata/08_Forschungsdaten/LSD_FKZ-TH1448_5-1/2025-03-15_B_Franziska-Schmitt_rat-neurons_Axion_BDNF"   # adjust to your NAS mount point
    #source_nas_path = "/mnt/bmdata/08_Forschungsdaten/ESTRANGE_FKZ-02NUK081B/2025_WM_Mayer_Slices_hOLS_Axion_X-ray_C-ions"
    #source_nas_path = "/mnt/bmdata/08_Forschungsdaten/LSD_FKZ-TH1448_5-1/2024-09-03_M_Ismael-Lozano_rat-neurons_Axion_LSD/Long_Term_Exp"
    source_nas_path =  "/mnt/bmdata/08_Forschungsdaten/LSD_FKZ-TH1448_5-1/2026-04-16_RCX_DMT/PSYNET_Cort"
    
    #destination_local_path = "/home/mciba/Documents/2025-07_BDNF/data"  # or any local folder on university PC
    #destination_local_path = "/home/mciba/Documents/2025-10-ESTRANGE/data"  # or any local folder on university PC
    #destination_local_path = "/home/mciba/Documents/2025-07_BDNF/data/Ismael"  # or any local folder on university PC
    destination_local_path = "/home/mciba/Documents/2025-07_BDNF/data/DMT/Longterm_DIV8"

    copy_spk_files(source_nas_path, destination_local_path)
