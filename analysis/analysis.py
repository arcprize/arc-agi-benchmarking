# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "matplotlib",
#     "numpy"
# ]
# ///

#!/usr/bin/env python3
import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log2
import sys
import shutil

########################################
# This script:
# 1. Optionally takes a subset of result sets as command-line arguments.
#    - If no arguments, use all result sets found in ../results.
#    - If arguments provided, use only those result sets.
#
# 2. Loads these results sets and ARC evaluation data.
# 3. Extracts features and creates a DataFrame with scores for each chosen result set.
# 4. Generates binned score plots for each numeric feature against all chosen score columns.
# 5. Computes correlations and prints top features.
# 6. Saves the plots and a simple HTML report in a directory under `plots/`.
#
# Run this script as:
#   python run.py
# or:
#   python run.py open_ai_o3_high_20241220 open_ai_o3_low_20241220
########################################

EVAL_DATA_PATH = '../data/arc-agi/data/evaluation'
RESULTS_DIR = '../results'
PLOTS_BASE_DIR = 'plots'

def load_all_results(results_dir: str):
    """
    Load all results.json files from the given directory.
    Returns a dict: {set_name: { 'task_results': {...} }}
    set_name is derived from the directory name.
    """
    result_sets = {}
    for subdir in os.listdir(results_dir):
        full_subdir = os.path.join(results_dir, subdir)
        if os.path.isdir(full_subdir):
            results_file = os.path.join(full_subdir, 'results.json')
            if os.path.isfile(results_file):
                with open(results_file, 'r') as fin:
                    data = json.load(fin)
                result_sets[subdir] = data
    return result_sets

def load_eval_data(eval_data_path: str):
    eval_data = {}
    for f in glob.glob(os.path.join(eval_data_path, '*.json')):
        task_name = os.path.basename(f).replace('.json', '')
        with open(f, 'r') as fin:
            eval_data[task_name] = json.load(fin)
    return eval_data

def get_test_input(eval_dict):
    return eval_dict['test'][0]['input']

def get_test_output(eval_dict):
    return eval_dict['test'][0]['output']

def get_size_from_eval(eval_dict):
    output_grid = get_test_output(eval_dict)
    if output_grid and len(output_grid) > 0 and len(output_grid[0]) > 0:
        return (len(output_grid), len(output_grid[0]))
    else:
        return (0,0)

def get_input_grid_count(eval_dict):
    return len(eval_dict['train'])

def get_average_input_grid_area(eval_dict):
    areas = []
    for sample in eval_dict['train']:
        inp = sample['input']
        if inp and len(inp[0])>0:
            areas.append(len(inp)*len(inp[0]))
    return sum(areas) / len(areas) if areas else 0

def get_unique_values(eval_dict):
    test_in = get_test_input(eval_dict)
    test_out = get_test_output(eval_dict)
    unique_in = len(set(x for row in test_in for x in row)) if test_in else 0
    unique_out = len(set(x for row in test_out for x in row)) if test_out else 0
    return unique_in, unique_out

def grid_area(grid):
    if not grid or len(grid)==0 or len(grid[0])==0:
        return 0
    return len(grid)*len(grid[0])

def unique_values_set(grid):
    vals = set()
    for row in grid:
        for v in row:
            vals.add(v)
    return vals

def unique_values_count(grid):
    return len(unique_values_set(grid))

def count_nonzero(grid):
    return sum((1 for row in grid for v in row if v!=0))

def count_connected_components_and_sizes(grid):
    if not grid or len(grid)==0 or len(grid[0])==0:
        return 0, []
    h, w = len(grid), len(grid[0])
    visited = [[False]*w for _ in range(h)]
    
    def neighbors(r, c):
        for nr,nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc
    
    def dfs(r, c, val):
        stack = [(r,c)]
        visited[r][c] = True
        size = 0
        while stack:
            rr,cc = stack.pop()
            size += 1
            for nr,nc in neighbors(rr, cc):
                if not visited[nr][nc] and grid[nr][nc] == val:
                    visited[nr][nc] = True
                    stack.append((nr,nc))
        return size

    count = 0
    sizes = []
    for rr in range(h):
        for cc in range(w):
            if not visited[rr][cc]:
                comp_size = dfs(rr, cc, grid[rr][cc])
                sizes.append(comp_size)
                count += 1
    return count, sizes

def shannon_entropy(values):
    if not values:
        return 0.0
    freq = {}
    for v in values:
        freq[v] = freq.get(v,0)+1
    total = len(values)
    entropy = 0.0
    for _,v in freq.items():
        p = v/total
        entropy -= p*log2(p)
    return entropy

def is_symmetric_horizontally(grid):
    if not grid:
        return True
    h = len(grid)
    for i in range(h//2):
        if grid[i] != grid[h-1-i]:
            return False
    return True

def is_symmetric_vertically(grid):
    if not grid or len(grid[0])==0:
        return True
    w = len(grid[0])
    for row in grid:
        for i in range(w//2):
            if row[i]!=row[w-1-i]:
                return False
    return True

def extract_basic_features(task_name, eval_data, result_sets):
    # Extract basic features common to all tasks
    e_data = eval_data[task_name]
    h, w = get_size_from_eval(e_data)
    unique_in, unique_out = get_unique_values(e_data)
    input_grid_count = get_input_grid_count(e_data)
    avg_inp_area = get_average_input_grid_area(e_data)

    # Get scores from all result sets
    row = {
        'task_name': task_name,
        'test_output_height': h,
        'test_output_width': w,
        'unique_in_values': unique_in,
        'unique_out_values': unique_out,
        'input_grid_count': input_grid_count,
        'avg_input_area': avg_inp_area
    }

    for set_name, data in result_sets.items():
        row[f"{set_name}_score"] = data['task_results'][task_name]

    return row

def extract_advanced_features(task_data):
    # TRAIN info
    train_same_size = []
    train_areas_in = []
    train_areas_out = []
    unique_in_vals_train = []
    unique_out_vals_train = []

    for ex in task_data['train']:
        inp = ex['input']
        outp = ex['output']
        same_size = (len(inp)==len(outp) and (len(inp[0])==len(outp[0]) if inp and outp else True))
        train_same_size.append(1.0 if same_size else 0.0)
        train_areas_in.append(grid_area(inp))
        train_areas_out.append(grid_area(outp))
        unique_in_vals_train.append(unique_values_count(inp))
        unique_out_vals_train.append(unique_values_count(outp))
    
    frac_train_same_size = np.mean(train_same_size) if train_same_size else 0.0
    num_train_examples = len(task_data['train'])
    avg_input_area_train = np.mean(train_areas_in) if train_areas_in else 0.0
    avg_output_area_train = np.mean(train_areas_out) if train_areas_out else 0.0
    if train_areas_in:
        valid_pairs = [(float(o)/float(i)) for i, o in zip(train_areas_in, train_areas_out) if i != 0]
        avg_in_out_area_ratio = np.mean(valid_pairs) if valid_pairs else 0.0
    else:
        avg_in_out_area_ratio = 0.0
    var_input_area_train = np.var(train_areas_in) if train_areas_in else 0.0
    avg_unique_in_train = np.mean(unique_in_vals_train) if unique_in_vals_train else 0.0
    avg_unique_out_train = np.mean(unique_out_vals_train) if unique_out_vals_train else 0.0

    # TEST info
    test_in = task_data['test'][0]['input']
    test_out = task_data['test'][0]['output']
    test_area_in = grid_area(test_in)
    test_area_out = grid_area(test_out)

    test_in_vals = unique_values_set(test_in)
    test_out_vals = unique_values_set(test_out)
    test_unique_in = len(test_in_vals)
    test_unique_out = len(test_out_vals)

    test_in_components, test_in_sizes = count_connected_components_and_sizes(test_in)
    test_out_components, test_out_sizes = count_connected_components_and_sizes(test_out)

    zero_ratio_test_in = (sum(row.count(0) for row in test_in)/test_area_in) if test_area_in>0 else 0.0
    ratio_test_in_out_area = (test_area_in / test_area_out) if (test_area_out != 0) else 0.0
    nonzero_out_test = count_nonzero(test_out)
    nonzero_in_test = count_nonzero(test_in)
    diff_elements_test = len(test_in_vals.symmetric_difference(test_out_vals))
    sum_in_area = sum(train_areas_in) if train_areas_in else 0
    sum_out_area = sum(train_areas_out) if train_areas_out else 0
    ratio_train_in_out_total = (sum_in_area / sum_out_area) if sum_out_area != 0 else 0.0

    # Entropy (test input and output)
    test_in_flat = [v for row in test_in for v in row] if test_in and len(test_in[0])>0 else []
    test_out_flat = [v for row in test_out for v in row] if test_out and len(test_out[0])>0 else []
    test_input_entropy = shannon_entropy(test_in_flat)
    test_output_entropy = shannon_entropy(test_out_flat)
    test_entropy_diff = test_output_entropy - test_input_entropy

    # Non-zero ratios
    zero_count_out = sum(row.count(0) for row in test_out) if test_area_out>0 else 0
    zero_ratio_test_out = zero_count_out/test_area_out if test_area_out>0 else 0
    nonzero_ratio_test_in = (nonzero_in_test / test_area_in) if test_area_in>0 else 0
    nonzero_ratio_test_out = (nonzero_out_test / test_area_out) if test_area_out>0 else 0
    nonzero_ratio_diff = nonzero_ratio_test_out - nonzero_ratio_test_in

    # Value intersection fraction
    intersection = len(test_in_vals & test_out_vals)
    union = len(test_in_vals | test_out_vals)
    val_intersection_fraction = intersection/union if union>0 else 0

    # Test input size differences
    test_input_height = len(test_in) if test_in else 0
    test_input_width = len(test_in[0]) if test_in and len(test_in[0])>0 else 0
    test_output_height = len(test_out) if test_out else 0
    test_output_width = len(test_out[0]) if test_out and len(test_out[0])>0 else 0
    height_diff = test_output_height - test_input_height
    width_diff = test_output_width - test_input_width

    # Symmetry checks
    test_in_horizontal_sym = int(is_symmetric_horizontally(test_in))
    test_in_vertical_sym = int(is_symmetric_vertically(test_in))
    test_out_horizontal_sym = int(is_symmetric_horizontally(test_out))
    test_out_vertical_sym = int(is_symmetric_vertically(test_out))

    # Object sizes stats
    def sizes_stats(sizes):
        if not sizes:
            return (0,0,0)
        count_distinct = len(set(sizes))
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        return (count_distinct, mean_size, std_size)

    in_obj_count_distinct, in_obj_mean_size, in_obj_std_size = sizes_stats(test_in_sizes)
    out_obj_count_distinct, out_obj_mean_size, out_obj_std_size = sizes_stats(test_out_sizes)

    return {
        'frac_train_same_size': frac_train_same_size,
        'num_train_examples': num_train_examples,
        'avg_input_area_train': avg_input_area_train,
        'avg_output_area_train': avg_output_area_train,
        'avg_in_out_area_ratio': avg_in_out_area_ratio,
        'var_input_area_train': var_input_area_train,
        'avg_unique_in_train': avg_unique_in_train,
        'avg_unique_out_train': avg_unique_out_train,
        'test_area_in': test_area_in,
        'test_area_out': test_area_out,
        'test_unique_in': test_unique_in,
        'test_unique_out': test_unique_out,
        'test_in_components': test_in_components,
        'test_out_components': test_out_components,
        'zero_ratio_test_in': zero_ratio_test_in,
        'ratio_test_in_out_area': ratio_test_in_out_area,
        'nonzero_out_test': nonzero_out_test,
        'nonzero_in_test': nonzero_in_test,
        'diff_elements_test': diff_elements_test,
        'ratio_train_in_out_total': ratio_train_in_out_total,
        'test_input_entropy': test_input_entropy,
        'test_output_entropy': test_output_entropy,
        'test_entropy_diff': test_entropy_diff,
        'zero_ratio_test_out': zero_ratio_test_out,
        'nonzero_ratio_test_in': nonzero_ratio_test_in,
        'nonzero_ratio_test_out': nonzero_ratio_test_out,
        'nonzero_ratio_diff': nonzero_ratio_diff,
        'val_intersection_fraction': val_intersection_fraction,
        'test_input_height': test_input_height,
        'test_input_width': test_input_width,
        'height_diff': height_diff,
        'width_diff': width_diff,
        'test_in_horizontal_sym': test_in_horizontal_sym,
        'test_in_vertical_sym': test_in_vertical_sym,
        'test_out_horizontal_sym': test_out_horizontal_sym,
        'test_out_vertical_sym': test_out_vertical_sym,
        'in_obj_sizes_count_distinct': in_obj_count_distinct,
        'in_obj_sizes_mean': in_obj_mean_size,
        'in_obj_sizes_std': in_obj_std_size,
        'out_obj_sizes_count_distinct': out_obj_count_distinct,
        'out_obj_sizes_mean': out_obj_mean_size,
        'out_obj_sizes_std': out_obj_std_size
    }

def plot_binned_success(feature, df, score_cols, plots_dir, n_bins=10):
    """
    Plot binned success for multiple score columns on one plot.
    """
    df_valid = df.dropna(subset=[feature] + score_cols)
    if len(df_valid) == 0:
        return
    df_valid['feature_bin'] = pd.qcut(df_valid[feature], q=n_bins, duplicates='drop')
    grouped = df_valid.groupby('feature_bin', observed=False)[score_cols].agg(['mean','std','count'])

    plt.figure(figsize=(12,6))
    x_positions = np.arange(len(grouped.index))
    x_labels = grouped.index.astype(str)

    for i, sc in enumerate(score_cols):
        mean_vals = grouped[(sc, 'mean')]
        std_vals = grouped[(sc, 'std')]
        counts = grouped[(sc, 'count')]
        plt.errorbar(x_positions, mean_vals, yerr=std_vals, fmt='o-', capsize=5, label=sc)
        # Add counts as text
        for j, cnt in enumerate(counts):
            plt.text(x_positions[j], mean_vals.iloc[j], f"n={cnt}", ha='center', va='bottom')

    plt.title(f"Scores by binned {feature} (mean ± std)")
    plt.xlabel(feature)
    plt.ylabel("Mean Score")
    plt.xticks(x_positions, x_labels, rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(plots_dir, f"{feature}_binned_scores.png"))
    plt.close()

def print_top_correlations(df, score_cols, top_n=20):
    numeric_feats = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_feats = [f for f in numeric_feats if f not in score_cols]

    all_correlations = []
    for sc in score_cols:
        corr_series = df[numeric_feats+[sc]].corr()[sc].drop(sc)
        corr_df = corr_series.to_frame().reset_index()
        corr_df.columns = ['feature', 'corr']
        corr_df['score_col'] = sc
        all_correlations.append(corr_df)
    
    all_correlations = pd.concat(all_correlations, ignore_index=True)
    # Sort by absolute correlation
    all_correlations['abs_corr'] = all_correlations['corr'].abs()
    top = all_correlations.sort_values('abs_corr', ascending=False).head(top_n)
    return top

def write_html_report(report_file, selected_sets, top_correlations, plots_dir):
    # Generate a simple HTML report.
    html = []
    html.append("<html><head><title>ARC Analysis Report</title></head><body>")
    html.append(f"<h1>ARC Analysis Report</h1>")
    html.append("<h2>Selected Result Sets</h2>")
    html.append("<ul>")
    for s in selected_sets:
        html.append(f"<li>{s}</li>")
    html.append("</ul>")

    html.append("<h2>Top Correlations</h2>")
    html.append("<table border='1' cellpadding='5' cellspacing='0'>")
    html.append("<tr><th>Feature</th><th>Score Column</th><th>Correlation</th></tr>")
    for i, row in top_correlations.iterrows():
        html.append(f"<tr><td>{row['feature']}</td><td>{row['score_col']}</td><td>{row['corr']:.3f}</td></tr>")
    html.append("</table>")

    html.append("<h2>Plots</h2>")
    html.append("<p>All generated plots:</p>")
    png_files = sorted([f for f in os.listdir(plots_dir) if f.endswith('.png')])
    for png in png_files:
        html.append(f"<h3>{png}</h3>")
        html.append(f"<img src='{png}' style='max-width:100%; height:auto; margin-bottom:20px;' />")

    html.append("</body></html>")

    with open(report_file, 'w') as f:
        f.write("\n".join(html))

def clear_directory_of_pngs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    else:
        # Remove all .png files
        for f in os.listdir(dir_path):
            if f.endswith('.png'):
                os.remove(os.path.join(dir_path, f))

def main():
    # Parse arguments: if arguments given, use them as subset; else use all
    args = sys.argv[1:]
    all_result_sets = load_all_results(RESULTS_DIR)

    if args:
        # Use subset
        selected_sets = [a for a in args if a in all_result_sets]
        if len(selected_sets) != len(args):
            missing = [a for a in args if a not in all_result_sets]
            print(f"Warning: The following sets not found: {missing}")
    else:
        # Use all
        selected_sets = list(all_result_sets.keys())

    if not selected_sets:
        print("No valid result sets selected. Exiting.")
        return

    # Create filtered result_sets dict
    result_sets = {k:v for k,v in all_result_sets.items() if k in selected_sets}

    # Determine plots_dir
    if len(selected_sets) == len(all_result_sets):
        # means all sets are chosen
        plots_dir = os.path.join(PLOTS_BASE_DIR, "all")
    else:
        # subset chosen
        subset_name = "_".join(selected_sets)
        plots_dir = os.path.join(PLOTS_BASE_DIR, subset_name)

    # Clear old plots
    clear_directory_of_pngs(plots_dir)

    # Load eval data
    eval_data = load_eval_data(EVAL_DATA_PATH)

    # Determine common tasks (appear in all chosen result sets)
    all_task_sets = [set(d['task_results'].keys()) for d in result_sets.values()]
    common_tasks = set.intersection(*all_task_sets) if all_task_sets else set()

    rows = []
    for task_name in common_tasks:
        row = extract_basic_features(task_name, eval_data, result_sets)
        rows.append(row)
    df = pd.DataFrame(rows)

    # Extract advanced features
    extra_features = []
    for i, row in df.iterrows():
        task_name = row['task_name']
        f = extract_advanced_features(eval_data[task_name])
        extra_features.append(f)
    features_df = pd.DataFrame(extra_features)
    df = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

    # Identify score columns
    score_cols = [c for c in df.columns if c.endswith('_score')]

    # Plot binned success for all numeric features
    numeric_feats = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_feats = [f for f in numeric_feats if f not in score_cols]

    for feat in numeric_feats:
        plot_binned_success(feat, df, score_cols, plots_dir, n_bins=10)

    # Compute top correlations
    top_correlations = print_top_correlations(df, score_cols, top_n=20)

    # Print top correlations to console
    print("Top 20 features by absolute correlation with scores:")
    for i, row in top_correlations.iterrows():
        print(f"{row['feature']} vs {row['score_col']}: corr={row['corr']:.3f}")

    # Write HTML report
    report_file = os.path.join(plots_dir, "report.html")
    write_html_report(report_file, selected_sets, top_correlations, plots_dir)

    print(f"\nAll plots and a report have been saved to '{plots_dir}'. Open 'report.html' to view.")


if __name__ == "__main__":
    main()
