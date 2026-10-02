# ARC Analysis

This repository provides a script to analyze results from different model runs on the ARC dataset. It generates a report including various computed features, correlations, and embedded plots to help understand model performance differences.

## Requirements

- The `uv` tool for isolated Python environments

## Setup

1. Install `uv` by following the instructions:
   [https://docs.astral.sh/uv/guides/install-python/](https://docs.astral.sh/uv/guides/install-python/)

## Running the Analysis

To run the analysis against **all** available model results:
```uv run analysis.py```

The script will:
- Load all model result sets found in `../results`.
- Identify common tasks that appear across all chosen result sets.
- Compute a wide range of features and correlations.
- Generate a set of plots and produce an HTML report in `plots/all/`.

If you wish to run the analysis on a **specific subset** of result sets, list their directory names as arguments:
```uv run analysis.py open_ai_o3_high_20241220 open_ai_o3_low_20241220```

In this example, only these two result sets are included. The resulting plots and report will be placed in `plots/open_ai_o3_high_20241220_open_ai_o3_low_20241220/`.

## Output

- The `plots/` directory will contain a subdirectory named based on the chosen models.
- The subdirectory will include:
  - `.png` image files for binned score plots.
  - A `report.html` file with:
    - A summary of which models were analyzed.
    - A table of top features correlated with performance.
    - Inline-embedded plots.

Open the `report.html` file in a browser to view the results.

If you are running on a remote machine, simply run and port-forward:
```uv run --python 3.12.0 -m http.server 8000``` 
and you can navigate to the `report.html` to view it in browser. 

## Notes

- The `uv run` command will automatically handle environment setup and dependency installation.
- Each run overwrites existing `.png` files in the target plot directory, ensuring a fresh report each time.

## Troubleshooting

- Ensure that `analysis.py` can find the `../data/arc-agi/data/evaluation/` directory. Adjust paths as needed (ensure you have synced the submodule)
