# Taxis Analysis (anemotaxis)

A Python package for analyzing larval (anemo)taxis behavior from trajectory data. This package provides tools for processing, analyzing, and visualizing larval navigation behaviors in response to (wind) gradients. Written by Sharbatanu Chatterjee, PhD, as part of his postdoctoral research stint, with assistance from Github Copilot (Claude Sonent 4 and GPT-4.1).

## Features

- **Data Processing**: Load and preprocess trajectory data from `.mat` files
- **Behavioral Analysis**: Analyze run probability, turn behavior, head casting, and navigation indices
- **Visualization**: Create publication-ready plots including polar plots, time series, and behavioral matrices
- **Head Cast Detection**: Advanced algorithms for detecting and classifying head cast behaviors
- **Statistical Analysis**: Comprehensive statistical testing and bias analysis

## Project Structure

```
anemotaxis/
├── src/
│   ├── core/
│   │   ├── data_loader.py      # Data loading and saving functions
│   │   └── data_processor.py   # Analysis and processing algorithms
│   ├── viz/
│   │   └── plot_data.py        # Plotting and visualization functions
│   └── utils/
│       └── preprocessing.py    # Data filtering and preprocessing
├── scripts/
│   └── analyze_single_anemotaxis.ipynb  # Example analysis notebook
├── data/                       # Data directory (not included in repo)
├── environment.yml            # Conda environment specification
├── pyproject.toml            # Modern Python packaging configuration
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/sharbatc/anemotaxis.git
cd anemotaxis

# Create conda environment with development tools
conda env create -f environment.yml
conda activate anemotaxis

# Install additional development dependencies
pip install -e ".[dev]"
```

## Quick Start

### 1. Basic Usage

```python
import core.data_loader as data_loader
import core.data_processor as data_processor
import viz.plot_data as plot_data
import utils.preprocessing as preprocessing

# Load trajectory data
trx_data = data_loader.load_single_trx_file("path/to/your/trx.mat")

# Preprocess data
filtered_data = preprocessing.filter_larvae_by_duration(trx_data, min_total_duration=300)
filtered_data = preprocessing.filter_larvae_by_excess_stop_time(filtered_data, max_stop_percentage=0.5)
clean_data = preprocessing.merge_short_stop_sequences(filtered_data)

# Analyze behavior
run_results = data_processor.analyze_run_probability_by_orientation(clean_data)
turn_results = data_processor.analyze_turn_probability_by_orientation(clean_data)

# Create visualizations
plot_data.plot_run_probabilities(run_results)
plot_data.plot_turn_probabilities(turn_results)
```

### 2. Complete Analysis Pipeline

See the example notebook `scripts/analyze_single_anemotaxis.ipynb` for a complete analysis pipeline that includes:

- Data loading and preprocessing
- Behavioral analysis (runs, turns, head casts)
- Statistical testing
- Publication-ready figure generation
- Results saving

### 3. Head Cast Analysis

```python
# Detect head casts
cast_events = data_processor.detect_head_casts_in_casts(
    clean_data,
    peak_threshold=10.0,
    peak_prominence=10.0
)

# Analyze head cast bias
bias_results = data_processor.analyze_first_head_cast_bias_perpendicular(
    cast_events, 
    analysis_type='first'
)

# Visualize results
plot_data.plot_head_cast_bias_perpendicular(bias_results)
```

## Key Modules

### `core.data_loader`
- `load_single_trx_file()`: Load trajectory data from .mat files
- `save_analysis_results()`: Save analysis results to HDF5 format

### `core.data_processor`
- `analyze_run_probability_by_orientation()`: Analyze running behavior vs orientation
- `analyze_turn_probability_by_orientation()`: Analyze turning behavior
- `detect_head_casts_in_casts()`: Detect head casting events
- `analyze_navigational_index_over_time()`: Calculate navigation performance

### `viz.plot_data`
- `plot_run_probabilities()`: Create run probability plots
- `plot_cast_detection_results()`: Interactive head cast visualization
- `plot_global_behavior_matrix()`: Behavioral ethogram visualization

### `utils.preprocessing`
- `filter_larvae_by_duration()`: Remove short trajectories
- `filter_larvae_by_excess_stop_time()`: Remove inactive larvae
- `merge_short_stop_sequences()`: Clean up behavioral sequences

## Data Format

The package expects trajectory data in MATLAB `.mat` format with the following structure:

```
trx.mat:
├── larva_id_1/
│   ├── t                           # Time vector
│   ├── x, y                        # Position coordinates  
│   ├── angle_upper_lower_smooth_5   # Head angle
│   ├── global_state_small_large_state # Behavioral state
│   └── ...
├── larva_id_2/
│   └── ...
```

## Output

The analysis generates:
- **PDF figures**: Publication-ready plots saved automatically
- **HDF5 files**: Processed data and analysis results
- **Statistical summaries**: Printed to console and saved in results

## Batch Processing Workflow

To analyze multiple single-path experiments, use the batch script:

```bash
python scripts/batch_run_single_experiments.py --paths_file /path/to/your_paths.txt
```

This will execute the parameterized notebook (`analyze_single_anemotaxis.ipynb`) for each path in the file, saving results and PDFs in the corresponding `analyses` subfolder.

**Note:** You can generate the paths file using the provided scripts or by listing all `trx.mat` files you wish to process.

## Requirements

- Check out the `envionment.yml` and `pyproject.toml` file for the requirements.
- **xelatex** (required for PDF export)
  - On macOS, install MacTeX or BasicTeX, which provides xelatex. Make sure `/Library/TeX/texbin` is in your PATH.
  - On Linux, install with `sudo apt-get install texlive-xetex`.
  - On Windows, install MikTeX or TeX Live and ensure xelatex is on your PATH.

If you see an error like `OSError: xelatex not found on PATH`, install xelatex and restart your terminal/session.

## Citation

If you use this package in your research, please cite:

```bibtex
@misc{chatterjee2025anemotaxis,
  author = {Chatterjee, Sharbatanu},
  title = {Anemotaxis Analysis: A Python package for analyzing larval anemotaxis behavior from trajectory data},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/JovanicLab/anemotaxis}},
  note = {Version 2.1}
}
```
**For Word/Google Docs:**
Plain text citation format:
```
Chatterjee, S. (2025). Anemotaxis Analysis: A Python package for analyzing larval anemotaxis behavior from trajectory data (Version 0.1.0) [analysis software]. GitHub. https://github.com/sharbatc/anemotaxis
```

<!-- ## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request -->

## Support

For questions or issues:
- Open an issue on GitHub
- Contact: [sharbatanu.chatterjee@cnrs.fr]

## Troubleshooting

### Import Errors
If you get import errors, make sure you've installed the package properly:

```bash
# Reinstall in development mode
pip install -e .

# Or check if the package is installed
pip list | grep anemotaxis
```

### Missing Dependencies
If you encounter missing dependencies:

```bash
# Update conda environment
conda env update -f environment.yml

# Or reinstall with pip
pip install -e ".[dev]"
```

### Jupyter Notebook Issues
If imports don't work in Jupyter:

```python
# Add this to the top of your notebook
import sys
sys.path.append('.')

# Or restart the Jupyter kernel after installation

```
