# =============================================================================
# batch_run_single_experiments.py
#
# This script automates the batch execution of a parameterized Jupyter notebook
# (analyze_single_anemotaxis.ipynb) for multiple single-path experiments with
# trx.mat files for each experiment.
# It reads a list of file paths from a text file, runs the notebook for each path
# using papermill, and exports the results to PDF files (both figures as well as
# the executed notebook in PDF format). Temporary notebooks are deleted after export.
#
# Example usage:
#   python batch_run_single_experiments.py --paths_file /path/to/single_paths.txt
# =============================================================================
import papermill as pm
from nbconvert import PDFExporter
import os
import shutil

if not shutil.which("xelatex"):
    tex_path = "/Library/TeX/texbin"
    if os.path.exists(tex_path):
        os.environ["PATH"] += os.pathsep + tex_path
        print(f"⚠️  'xelatex' not found initially. Added {tex_path} to PATH.")

def execute_and_export(input_nb, pdf_path, params=None):
    temp_nb_path = pdf_path.replace('.pdf', '.ipynb')

    print(f"📘 Executing notebook ...")
    pm.execute_notebook(
        input_path=input_nb,
        output_path=temp_nb_path,
        parameters=params or {},
        progress_bar=False,
        report_mode=False
    )
    print("✅ Notebook executed in a temporary file.")

    print(f"📝 Exporting to PDF: {pdf_path}")
    exporter = PDFExporter()
    body, resources = exporter.from_filename(temp_nb_path)

    with open(pdf_path, "wb") as f:
        f.write(body)

    print(f"🎉 PDF exported to: {pdf_path}")

    os.remove(temp_nb_path)
    print(f"🗑️ Deleted temporary notebook: {temp_nb_path}")

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Batch run and export single-path notebooks.")
    parser.add_argument("--paths_file", type=str, default="/Users/sharbat/Projects/anemotaxis/data/single_paths.txt", help="Path to the file containing single paths.")
    args = parser.parse_args()

    input_nb = "/Users/sharbat/Projects/anemotaxis/scripts/analyze_single_anemotaxis.ipynb"
    paths_file = args.paths_file

    print(f"\n📄 Reading paths from: \033[1m{paths_file}\033[0m\n")
    with open(paths_file) as f:
        paths = [line.strip() for line in f if line.strip()]

    print("🗂️  \033[1mContents of paths file:\033[0m")
    for idx, p in enumerate(paths, 1):
        print(f"  {idx:2d}. {p}")
    print()
    total = len(paths)
    print(f"🔢 \033[1m{total} files found. Starting batch processing...\033[0m\n")

    for i, single_path in enumerate(paths):
        parent_dir = os.path.dirname(single_path)
        output_dir = os.path.join(parent_dir, 'analyses')
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, f"analyze_single_anemotaxis.pdf")
        params = {"single_path": single_path}
        print(f"\n🚀 \033[1;34m[{i+1}/{total}]\033[0m Processing: \033[1m{single_path}\033[0m")
        execute_and_export(input_nb, pdf_path, params)
        print(f"✅ \033[92mFinished {i+1}/{total}\033[0m\n{'-'*60}")


# # The following code was used to generate the input text files with paths
# # for the batch processing script above. It is commented out as it is not part
# # of the main functionality but may be useful for reference.
# import os
# import re

# root_dir = "/Volumes/eq-jovanic/Personal_folders/Iara/Beh_Data/final_analysis"
# output_file_300 = "iara_trx_mat_path_t7_x300s.txt"
# output_file_420 = "iara_trx_mat_path_t7_x420s.txt"
# prefix = "/Users/sharbat/Projects/anemotaxis/data/t7"

# def is_date_folder(name):
#     return len(name) == 15 and name[:8].isdigit() and name[8] == '_' and name[9:].isdigit()

# paths_300 = []
# paths_420 = []

# for dirpath, dirnames, filenames in os.walk(root_dir):
#     for dirname in dirnames:
#         if is_date_folder(dirname):
#             date_folder = os.path.join(dirpath, dirname)
#             files = [f for f in os.listdir(date_folder) if os.path.isfile(os.path.join(date_folder, f))]
#             if files:
#                 for f in files:
#                     m = re.match(r"^\d{8}_\d{6}@(.*?)@t\d+@(.*?)@", f)
#                     if m:
#                         genotype = m.group(1)
#                         protocol = m.group(2)
#                         # Compose the new path
#                         trx_path = os.path.join(
#                             prefix,
#                             genotype,
#                             protocol,
#                             dirname,
#                             "trx.mat"
#                         )
                        
#                         if "x300s" in protocol:
#                             paths_300.append(trx_path)
#                         elif "x420s" in protocol:
#                             paths_420.append(trx_path)
                            
#                         break  # Only one file per date folder

# with open(output_file_300, "w") as f:
#     for p in paths_300:
#         f.write(p + "\n")

# with open(output_file_420, "w") as f:
#     for p in paths_420:
#         f.write(p + "\n")

# print(f"Written {len(paths_300)} paths to {output_file_300}")
# print(f"Written {len(paths_420)} paths to {output_file_420}")
# import os

# # List of root folders to search
# root_folders = [
#     "/Users/sharbat/Projects/anemotaxis/data/t7/Gr43a-GAL4@20XUAS_CsChrimsom_mVenus",
#     "/Users/sharbat/Projects/anemotaxis/data/t7/Orco-GAL4_W11_17@20XUAS_CsChrimsom_mVenus",
#     "/Users/sharbat/Projects/anemotaxis/data/t7/Or42a-GAL4_F48_1@20XUAS_CsChrimsom_mVenus",
#     "/Users/sharbat/Projects/anemotaxis/data/t7/W1118@20XUAS_CsChrimsom_mVenus"
# ]

# trx_mat_paths = []

# for root in root_folders:
#     for dirpath, dirnames, filenames in os.walk(root):
#         if "trx.mat" in filenames:
#             trx_mat_paths.append(os.path.join(dirpath, "trx.mat"))

# # Write all paths to a text file
# output_file = "all_trx_mat_path_t7.txt"
# with open(output_file, "w") as f:
#     for path in trx_mat_paths:
#         f.write(path + "\n")

# print(f"Found {len(trx_mat_paths)} trx.mat files. Paths written to {output_file}")