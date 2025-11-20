import papermill as pm
from nbconvert import PDFExporter
import tempfile
import os

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
    parser = argparse.ArgumentParser(description="Batch run and export single-path notebooks.")
    parser.add_argument("--paths_file", type=str, default="/Users/sharbat/Projects/anemotaxis/data/single_paths.txt", help="Path to the file containing single paths.")
    args = parser.parse_args()

    input_nb = "/Users/sharbat/Projects/anemotaxis/scripts/analyze_single_anemotaxis.ipynb"
    paths_file = args.paths_file

    print(f"📄 Reading paths from: {paths_file}\n")
    with open(paths_file) as f:
        paths = [line.strip() for line in f if line.strip()]
    print("Contents of paths file:")
    for p in paths:
        print(p)
    print()

    for i, single_path in enumerate(paths):
        parent_dir = os.path.dirname(single_path)
        output_dir = os.path.join(parent_dir, 'analyses')
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, f"analyze_single_anemotaxis.pdf")
        params = {"single_path": single_path}
        execute_and_export(input_nb, pdf_path, params)