import matplotlib
matplotlib.use('TkAgg')  # Force GUI backend
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np
import pickle
from tms.models.autoencoder import ToyAutoencoder
from tms.plots.kgons import plot_losses_and_polygons
import warnings
from collections import defaultdict
import matplotlib.cm as cm
import pandas as pd
from scipy.spatial import ConvexHull
import torch
import torch.nn as nn
import os
from typing import List, Dict, Any, Tuple
from tms.utils.utils import load_results, get_first, iterate_container
from tms.training.experiments import run_experiments
from tms.utils.utils import generate_sparsity_values
from tms.data.dataset import SyntheticBinaryValued
from tms.models.autoencoder import ToyAutoencoder
from tms.llc import estimate_llc, get_llc_data, preaggregate_llc
from tms.plots.kgons import plot_losses_and_polygons
from tms.plots.losses import compare_dataframes_and_results

def get_or_create_preaggregated_llc_csv(results, version: str, data_dir: str) -> pd.DataFrame:
    """Load preaggregated LLC values from CSV, or generate and save them."""
    preagg_path = os.path.join(data_dir, f"llc_preagg_{version}.csv")
    
    if os.path.exists(preagg_path):
        print(f"Loading preaggregated LLC from {preagg_path}")
        return pd.read_csv(preagg_path)
    
    print("Preaggregated file not found — computing from raw results...")
    llc_estimates = get_llc_data(results, version, data_dir)
    # Preaggregate (returns dict), convert to DataFrame for CSV
    preagg = preaggregate_llc(llc_estimates)
    preagg_df = pd.DataFrame([
        {'index': k[0], 'batch_size': k[1], 'lr': k[2], 'snapshot_index': k[3], 'llc': v}
        for k, v in preagg.items()
    ])
    print(f"Saving preaggregated LLC to {preagg_path}")
    preagg_df.to_csv(preagg_path, index=False)
    return preagg_df

data_path = "../../data/"
version = "1.13.0"
results_1_13 = load_results(data_path, version)
llc_estimates_1_13 = get_or_create_preaggregated_llc_csv(results_1_13, version, data_path)

def plot_specific_index(results, index, step=-1):
    """
    Plot results for a specific index in the results list.
    
    Parameters
    ----------
    results : list
        List of experiment results
    index : int
        Specific index to plot
    step : int, optional
        Step index for loss checking. Default is -1 (last step)
    """
    if index >= len(results):
        print(f"Index {index} out of range for results of length {len(results)}")
        return
    
    result = results[index]
    
    STEPS = result['parameters']['log_ivl']
    logs = result['logs']
    losses = [logs.loc[logs['step'] == s, 'loss'].values[0] for s in STEPS]
    
    sparsity = result['parameters']['sparsity']
    
    print(f"Plotting index {index}")
    print(f"Sparsity: {sparsity}")
    print(f"Loss at step {step}: {losses[step]}")
    
    NUM_EPOCHS = 20000
    PLOT_STEPS = [min(STEPS, key=lambda s: abs(s-i)) for i in [0, 200, 2000, 10000, NUM_EPOCHS - 1]]
    PLOT_INDICES = [STEPS.index(s) for s in PLOT_STEPS]
    
    Ws = [result['weights'][i]['embedding.weight'] for i in PLOT_INDICES]
    biases = [result['weights'][i]['unembedding.bias'] for i in PLOT_INDICES]
    
    # Optional: Load model weights
    model = ToyAutoencoder(6, 2, final_bias=True)
    new_weights = {}
    for idx, ndarray in result['weights'][PLOT_INDICES[-1]].items():
        new_weights[idx] = torch.from_numpy(ndarray)
    
    plot_losses_and_polygons(STEPS, losses, PLOT_STEPS, Ws, biases)
    plt.show()

def test_click_detection():
    """Simple test to verify click detection works"""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
    ax.set_title("Click Test - Click anywhere")
    
    clicks = []
    
    def on_click(event):
        if event.inaxes == ax:
            clicks.append((event.xdata, event.ydata))
            print(f"Click detected at: ({event.xdata:.2f}, {event.ydata:.2f})")
            ax.plot(event.xdata, event.ydata, 'ro', markersize=10)
            fig.canvas.draw()
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    return clicks

class InteractiveDendrogram:
    def __init__(self, Z, loss_matrix, plot_func=None):
        print("Initializing InteractiveDendrogram...")
        self.Z = Z
        self.loss_matrix = loss_matrix
        self.plot_func = plot_func
        self.selected_items = []
        
        # Create the dendrogram
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        print("Creating dendrogram...")
        self.dend = dendrogram(Z, ax=self.ax, color_threshold=0.7*max(Z[:,2]))
        
        print(f"Dendrogram leaves: {self.dend['leaves']}")
        print(f"Number of leaves: {len(self.dend['leaves'])}")
        print(f"Loss matrix shape: {loss_matrix.shape}")
        
        self.ax.set_title("Interactive Dendrogram - Click on bottom numbers to compare")
        self.ax.set_xlabel("Model Index (click on numbers below)")
        
        # Connect click event
        print("Connecting click event...")
        self.connection = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        print(f"Event connection ID: {self.connection}")
        
        # Show selected items
        self.selection_text = self.ax.text(0.02, 0.98, "Selected: None", 
                                         transform=self.ax.transAxes, 
                                         verticalalignment='top',
                                         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Add instructions
        self.ax.text(0.02, 0.02, "Click near the leaf numbers at bottom to select models", 
                    transform=self.ax.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        print("Showing plot...")
        plt.show()
    
    def on_click(self, event):
        print(f"Click event triggered! Event: {event}")
        print(f"Event axes: {event.inaxes}")
        print(f"Event xdata: {event.xdata}, ydata: {event.ydata}")
        
        if event.inaxes != self.ax or event.xdata is None:
            print("Click outside axes or no xdata")
            return
            
        # Get the leaf order from dendrogram
        leaves = self.dend['leaves']
        print(f"Available leaves: {leaves}")
        
        # Find which leaf was clicked
        n_leaves = len(leaves)
        x_range = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
        
        # Map click position to leaf index
        click_pos = (event.xdata - self.ax.get_xlim()[0]) / x_range * n_leaves
        leaf_idx = int(round(click_pos))
        
        print(f"Click position: {click_pos}, mapped to leaf index: {leaf_idx}")
        
        if 0 <= leaf_idx < len(leaves):
            clicked_item = leaves[leaf_idx]
            print(f"Clicked item: {clicked_item}")
            
            # Add to selection
            if clicked_item not in self.selected_items:
                self.selected_items.append(clicked_item)
                print(f"Added to selection: {self.selected_items}")
                
            # Keep only last 2 selections
            if len(self.selected_items) > 2:
                self.selected_items = self.selected_items[-2:]
                print(f"Trimmed selection: {self.selected_items}")
            
            # Update display
            self.update_selection_display()
            
            # If we have 2 items, plot comparison
            if len(self.selected_items) == 2:
                print("Two items selected, comparing...")
                self.compare_items()
        else:
            print(f"Leaf index {leaf_idx} out of range")
    
    def update_selection_display(self):
        if len(self.selected_items) == 0:
            text = "Selected: None"
        elif len(self.selected_items) == 1:
            text = f"Selected: Model {self.selected_items[0]} (click another)"
        else:
            text = f"Comparing: Model {self.selected_items[0]} vs Model {self.selected_items[1]}"
        
        print(f"Updating display: {text}")
        self.selection_text.set_text(text)
        self.fig.canvas.draw()
    
    def compare_items(self):
        item1, item2 = self.selected_items
        print(f"\n🔍 Comparing Model {item1} vs Model {item2}")
        
        if self.plot_func:
            try:
                # Fix: Your plot_func expects only one argument (index)
                print(f"Calling plot_func for item1: {item1}")
                self.plot_func(item1)
                print(f"Calling plot_func for item2: {item2}")
                self.plot_func(item2)
            except Exception as e:
                print(f"Error calling plot_func: {e}")
                self.default_comparison(item1, item2)
        else:
            self.default_comparison(item1, item2)
    
    def default_comparison(self, item1, item2):
        """Default comparison plot"""
        print(f"Default comparison of {item1} vs {item2}")
        
        if item1 >= self.loss_matrix.shape[1] or item2 >= self.loss_matrix.shape[1]:
            print(f"Items out of range: {item1}, {item2} vs matrix shape {self.loss_matrix.shape}")
            return
            
        pattern1 = self.loss_matrix[:, item1]
        pattern2 = self.loss_matrix[:, item2]
        
        # Create new window for comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Model {item1} vs Model {item2}')
        
        # Plot patterns
        axes[0].plot(pattern1, 'o-', alpha=0.7, color='blue')
        axes[0].set_title(f'Model {item1} Loss Pattern')
        axes[0].set_xlabel('Input Index')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(pattern2, 's-', alpha=0.7, color='red')
        axes[1].set_title(f'Model {item2} Loss Pattern')
        axes[1].set_xlabel('Input Index')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)
        
        # Scatter comparison
        axes[2].scatter(pattern1, pattern2, alpha=0.6)
        min_val = min(pattern1.min(), pattern2.min())
        max_val = max(pattern1.max(), pattern2.max())
        axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        axes[2].set_xlabel(f'Model {item1} Loss')
        axes[2].set_ylabel(f'Model {item2} Loss')
        axes[2].set_title('Direct Comparison')
        axes[2].grid(True, alpha=0.3)
        
        correlation = np.corrcoef(pattern1, pattern2)[0, 1]
        axes[2].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=axes[2].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

def launch_interactive_dendrogram(Z, loss_matrix, plot_func=None):
    """Launch the interactive dendrogram"""
    print("Launching interactive dendrogram...")
    app = InteractiveDendrogram(Z, loss_matrix, plot_func)
    return app

def debug_dendrogram_data():
    """Debug function to check the data"""
    try:
        with open('dendrogram_data.pkl', 'rb') as f:
            data = pickle.load(f)
            print("Data keys:", data.keys())
            
            Z = data['Z_models']
            loss_matrix = data['loss_matrix']
            
            print(f"Z_models shape: {Z.shape}")
            print(f"Loss matrix shape: {loss_matrix.shape}")
            print(f"Z_models sample:\n{Z[:5]}")
            print(f"Loss matrix sample:\n{loss_matrix[:5, :5]}")
            
            return Z, loss_matrix
    except FileNotFoundError:
        print("dendrogram_data.pkl not found!")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

if __name__ == "__main__":
    print("=== DEBUGGING DENDROGRAM ===")
    
    # First test basic click detection
    print("1. Testing basic click detection...")
    # test_click_detection()  # Uncomment to test
    
    # Debug data loading
    print("2. Debugging data...")
    Z, loss_matrix = debug_dendrogram_data()
    
    if Z is not None and loss_matrix is not None:
        print("3. Launching interactive dendrogram...")
        
        # Fixed plot function - your original expects just index
        plot_func = lambda index: plot_specific_index(results_1_13, index=index)
        
        app = launch_interactive_dendrogram(Z, loss_matrix, plot_func=plot_func)
        
        # Keep the script running
        input("Press Enter to exit...")
    else:
        print("Could not load data. Please save data first from Jupyter.")