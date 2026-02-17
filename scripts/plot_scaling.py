import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

def plot_scaling_laws(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # metrics to plot against flops_budget
    metrics = [
        'val_bpb', 
        'core_score', 
        'tokens_trained', 
        'params', 
        'train_time_sec',
        'num_iterations'
    ]
    
    # Filter out metrics that don't exist in the CSV
    metrics = [m for m in metrics if m in df.columns]

    if not metrics:
        print("No valid metrics found to plot.")
        return

    # Setup the grid
    num_plots = len(metrics)
    cols = 2
    rows = math.ceil(num_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    # Ensure axes is always iterable (even if only 1 plot)
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Get unique depths for consistent coloring
    depths = sorted(df['depth'].unique())
    # Create a custom palette if needed, or use a predefined one
    palette = sns.color_palette("viridis", len(depths))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot each depth as a separate line
        sns.lineplot(
            data=df,
            x='flops_budget',
            y=metric,
            hue='depth',
            marker='o',
            palette=palette,
            ax=ax
        )
        
        ax.set_title(f'{metric} vs FLOPs')
        ax.set_xscale('log')
        ax.set_xlabel('FLOPs Budget')
        ax.set_ylabel(metric)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        # Determine if log scale is appropriate for Y axis
        # Use log scale for these metrics generally, unless they contain <=0 values
        if metric in ['val_bpb', 'params', 'tokens_trained', 'num_iterations', 'train_time_sec']:
            if (df[metric] > 0).all():
                ax.set_yscale('log')

    # Hide empty subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.splitext(csv_path)[0] + '_scaling_plots.png'
    plt.savefig(output_path)
    print(f"Scaling plots saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot scaling laws from a results CSV file.")
    parser.add_argument("csv_path", type=str, help="Path to the results.csv file")
    args = parser.parse_args()

    plot_scaling_laws(args.csv_path)
