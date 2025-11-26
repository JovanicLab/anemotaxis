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