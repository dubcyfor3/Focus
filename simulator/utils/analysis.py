#!/usr/bin/env python3

import pandas as pd
import glob
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sparse_info import SparseInfo
from models.models import ModelConfig
from arch.accelerator import Accelerator
from core.simulator import Simulator


def get_utilization(tile_size, array_width):
    return tile_size / (tile_size + array_width)



def worst_case_analysis(trace_dir, output_dir):
    num_tokens_per_tile_list = []
    models = ["llava_vid"]
    datasets = ["videomme", "mlvu", "mvbench"]
    tile_size = 1024

    for model in models:
        for dataset in datasets:
            model_config = ModelConfig(model, dataset, trace_dir)
            sparse_info = SparseInfo("focus", model, dataset, model_config, trace_dir)
            num_blocks = model_config.num_blocks
            for block in range(num_blocks):
                for layer_name in sparse_info.info_dict['mask_zero'].keys():
                    mask_zero = sparse_info.info_dict['mask_zero'][layer_name][block]
                    mask_similar = sparse_info.info_dict['mask_similar'][layer_name][block]
                    non_zero_row = torch.all(~mask_zero, dim=-1)
                    mask_similar = mask_similar[non_zero_row]

                    # pad dim 0 to multiple of tile size
                    num_tokens = mask_similar.shape[0]
                    compression_ratio = mask_similar.numel() / torch.sum(~mask_similar)
                    tokens_to_pad = tile_size - num_tokens % tile_size if num_tokens % tile_size != 0 else 0
                    if tokens_to_pad > 0:
                        mask_similar = torch.cat([mask_similar, torch.ones(tokens_to_pad, mask_similar.shape[1], dtype=mask_similar.dtype)], dim=0)
                    mask_similar = mask_similar.view(tile_size, -1, mask_similar.shape[1])
                    num_tokens_per_tile = torch.sum(~mask_similar, dim=0)
                    num_tokens_per_tile_list.append(num_tokens_per_tile.flatten())

            # compute_cycles, num_ops, dense_ops = self.sim_compute.run_attn_focus(mask_zero, mask_similar, group_idx, layer_config)

    # Create histogram of all values in num_tokens_per_tile_list
    if num_tokens_per_tile_list:
        # Concatenate all tensors into a single tensor
        
        all_values = torch.cat(num_tokens_per_tile_list, dim=0)
        # filter out values less than 10
        all_values = all_values[all_values >= 10]
        all_values_np = all_values.numpy()
        
        # Create dual-axis plot with density and utilization
        fig, ax1 = plt.subplots(figsize=(12, 4))
        
        # Set font properties
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
        plt.rcParams['font.size'] = 14
        
        # Left axis: Probability density (Histogram)
        color1 = '#67BB35'  # Darker green
        ax1.set_xlabel('Number of Vectors per Tile', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Probability Density', fontsize=16, fontweight='bold')
        ax1.hist(all_values_np, bins=50, alpha=0.7, edgecolor='black', density=True, color=color1, label='Probability Density')
        ax1.tick_params(axis='y', labelsize=14)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Right axis: Utilization
        array_width = 32
        x_range = np.linspace(all_values_np.min(), all_values_np.max(), 1000)
        utilization_values = get_utilization(x_range, array_width)
        
        # Calculate average utilization based on actual data
        actual_utilizations = get_utilization(all_values_np, array_width)
        avg_utilization = np.mean(actual_utilizations)
        
        ax2 = ax1.twinx()
        color2 = '#FF8C00'  # Orange
        ax2.set_ylabel('Utilization', fontsize=16, fontweight='bold')
        ax2.plot(x_range, utilization_values, color=color2, linewidth=2, linestyle='--', label='Utilization')
        ax2.axhline(y=avg_utilization, color=color2, linestyle='-', linewidth=2, alpha=0.7, label=f'Avg Utilization: {avg_utilization:.3f}')
        ax2.tick_params(axis='y', labelsize=14)
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=14)
        
        # Add statistics text
        mean_val = np.mean(all_values_np)
        std_val = np.std(all_values_np)
        min_val = np.min(all_values_np)
        max_val = np.max(all_values_np)
        median_val = np.median(all_values_np)
        
        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "figure_13.svg")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to: {output_path}")
        
        # Show the plot
        plt.show()
        
        # Print summary statistics
        print(f"\nSummary Statistics for Number of Tokens per Tile:")
        print(f"Total number of tiles: {len(all_values_np)}")
        print(f"Mean: {mean_val:.2f}")
        print(f"Standard deviation: {std_val:.2f}")
        print(f"Minimum: {min_val}")
        print(f"Maximum: {max_val}")
        print(f"Median: {median_val:.2f}")
        print(f"Number of blocks analyzed: {num_blocks}")
        print(f"Number of tensors in list: {len(num_tokens_per_tile_list)}")
        
        # Compute utilization for each tile size and create histogram
        array_width = 32  # Assuming array width of 32
        
        utilizations = np.mean(get_utilization(all_values_np, array_width))
        print(f"Utilization mean: {utilizations:.4f}")
        
    else:
        print("No data found in num_tokens_per_tile_list")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace_dir', type=str, default='../algorithm/output')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    worst_case_analysis(args.trace_dir, args.output_dir)
