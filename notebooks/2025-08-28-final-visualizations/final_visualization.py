# Above we are adding this autoreload magic command, so we can make changes to tms and it will load them correctly
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches

import numpy as np
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

# %%
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

# %%

data_path = "../../data/"
version = "1.8.0"

results_1_8= load_results(data_path, version)
llc_estimates_1_8 = get_or_create_preaggregated_llc_csv(results_1_8, version, data_path)

version = "1.7.0"

results_1_7= load_results(data_path, version)
llc_estimates_1_7 = get_or_create_preaggregated_llc_csv(results_1_7, version, data_path)

version = "1.11.0"

results_1_11= load_results(data_path, version)
llc_estimates_1_11 = get_or_create_preaggregated_llc_csv(results_1_11, version, data_path)

version = "1.12.0"

results_1_12= load_results(data_path, version)
llc_estimates_1_12 = get_or_create_preaggregated_llc_csv(results_1_12, version, data_path)

version = "1.13.0"

results_1_13= load_results(data_path, version)
llc_estimates_1_13 = get_or_create_preaggregated_llc_csv(results_1_13, version, data_path)

version = "1.14.0"

results_1_14= load_results(data_path, version)
llc_estimates_1_14 = get_or_create_preaggregated_llc_csv(results_1_14, version, data_path)

version = "1.15.0"

results_1_15= load_results(data_path, version)
llc_estimates_1_15 = get_or_create_preaggregated_llc_csv(results_1_15, version, data_path)


# %%
results_random_init = results_1_13
llc_estimates_random_init = llc_estimates_1_13

results_optimal_init = results_1_14
llc_estimates_optimal_init = llc_estimates_1_14

result_path = '../../results/'

# %%
compare_dataframes_and_results(
    [(llc_estimates_1_13, results_1_13), (llc_estimates_1_14, results_1_14)], result_path='../../results', ymin=-0.01
)

# %%
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import torch
import torch.nn as nn

import numpy as np

from typing import Any, Dict, List, Tuple
import warnings
from collections import defaultdict

from tms.models.autoencoder import ToyAutoencoder
from tms.data.dataset import SyntheticBinaryValued
from tms.plots.kgons import plot_losses_and_polygons
from tms.utils.utils import iterate_container, get_first
import pandas as pd
Results = Dict[str, Any]
DfResultPair = Tuple[pd.DataFrame, Results]

def create_position_color_mapping(positions):
    """Create color mapping for different positions/timesteps."""
    cmap = plt.get_cmap("tab10")  
    n_colors = cmap.N 
    return {pos: cmap(i % n_colors) for i, pos in enumerate(sorted(positions))}


def plot_for_sparsity(sparsity, df_results_pairs: Tuple[DfResultPair, DfResultPair], 
                     batch_size, learning_rate, position_to_color, positions, 
                     x_scale, y_scale, sharex, sharey, ymin):
    """Plot LLC vs Loss for a specific sparsity, with different positions as colors."""
    fig, axes = plt.subplots(1, len(df_results_pairs), figsize=(15*len(df_results_pairs), 10), 
                            sharey=sharey, sharex=sharex)
    if len(df_results_pairs) == 1:
        axes = [axes]

    for pair_index, (llc_estimates, results) in enumerate(df_results_pairs):
        steps = get_first(results)['parameters']['log_ivl']
        
        llc_estimates_dict = {
            (row['index'], row['batch_size'], row['lr'], row['snapshot_index']): row['llc']
            for _, row in llc_estimates.iterrows()
        }
        
        # Group by position instead of sparsity
        llc_loss_by_position = defaultdict(list)
        
        for result in iterate_container(results):
            index = result["run_id"]
            result_sparsity = results[index]['parameters']['sparsity']
            
            # Only plot results matching the target sparsity
            if abs(result_sparsity - sparsity) > 1e-6:  # Allow for floating point precision
                continue
                
            for position in positions:
                llc = llc_estimates_dict.get((index, batch_size, learning_rate, position), np.nan)
                if position < len(results[index]['logs']):
                    loss = results[index]['logs']['loss'].values[position]
                    llc_loss_by_position[position].append((llc, loss))

        # Plot each position with different colors
        for position, llc_loss in llc_loss_by_position.items():
            if not llc_loss:  # Skip if no data
                continue
                
            arr = np.asarray(llc_loss)
            mask = ~np.isnan(arr[:, 0])
            if not mask.any():
                continue
                
            llcs, losses = arr[mask].T
            color = position_to_color.get(position, 'gray')
            epoch = steps[position] if position < len(steps) else position
            axes[pair_index].scatter(llcs, losses, label=f"Epoch {epoch}", color=color, alpha=0.7)

        if pair_index == 0:
            title = "Initialized at random 4-gon"
        elif pair_index == 1:
            title = "Initialized at optimal parameters for sparse inputs"
        else:
            title = f"Pair {pair_index}"
            
        axes[pair_index].set_title(f"{title}, Sparsity {sparsity:.3f}")
        axes[pair_index].set_xlabel("LLC")
        axes[pair_index].set_ylabel("Loss")
        axes[pair_index].legend()
        axes[pair_index].set_xscale(x_scale)
        axes[pair_index].set_yscale(y_scale)
        axes[pair_index].set_ylim(ymin=ymin)

    plt.tight_layout()
    plt.suptitle(f"Loss vs LLC for Sparsity {sparsity:.3f} (Different Colors = Different Epochs)", fontsize=16)
    plt.subplots_adjust(top=0.9)

    return fig


def compare_dataframes_by_sparsity(
    df_results_pairs: Tuple[DfResultPair, DfResultPair],
    positions=[9, 18, 27, 36, 45],
    hyperparam_combos=[(300, 0.001)],
    x_scale="linear",
    y_scale="linear",
    sharey=False,
    sharex=False,
    ymin=1e-4
):
    """
    Create separate plots for each sparsity value, with positions shown as different colors.
    This is the inverse of the original function where sparsity was colored.
    """
    warnings.simplefilter(action='ignore', category=UserWarning)

    # Collect all unique sparsities across all results
    unique_sparsities = collect_global_sparsities(df_results_pairs)
    position_to_color = create_position_color_mapping(positions)

    for batch_size, learning_rate in hyperparam_combos:
        print(f"Batch size: {batch_size}, Learning rate: {learning_rate}\n")

        for sparsity in unique_sparsities:
            fig = plot_for_sparsity(
                sparsity,
                df_results_pairs,
                batch_size,
                learning_rate,
                position_to_color,
                positions,
                x_scale,
                y_scale,
                sharex,
                sharey,
                ymin
            )

            param_string = f"bs{batch_size}_lr{learning_rate}_sparsity{sparsity:.3f}"
            if x_scale != "linear" or y_scale != "linear":
                param_string += f"_x{x_scale}_y{y_scale}"
            if ymin != 1e-4:
                param_string += f"_ymin{ymin}"

            save_path = f'{result_path}loss_vs_llc_by_sparsity_{param_string}'
            fig.savefig(f'{save_path}.svg', bbox_inches='tight', format='svg')
            fig.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', format='png')
            plt.show()


def plot_single_sparsity_position(
    sparsity: float,
    position: int,
    df_results_pairs: Tuple[DfResultPair, DfResultPair],
    batch_size: int = 300,
    learning_rate: float = 0.001,
    x_scale: str = "linear",
    y_scale: str = "linear",
    sharex: bool = False,
    sharey: bool = False,
    ymin: float = 1e-4,
    figsize: Tuple[float, float] = (15, 6)
):
    """
    Plot LLC vs Loss for a single specific sparsity and position combination.
    
    Parameters
    ----------
    sparsity : float
        The specific sparsity value to plot
    position : int
        The specific position/timestep to plot
    df_results_pairs : Tuple[DfResultPair, DfResultPair]
        The data pairs to plot
    batch_size : int, optional
        Batch size parameter, by default 300
    learning_rate : float, optional
        Learning rate parameter, by default 0.001
    x_scale : str, optional
        X-axis scale, by default "linear"
    y_scale : str, optional
        Y-axis scale, by default "linear"
    sharex : bool, optional
        Share x-axis, by default False
    sharey : bool, optional
        Share y-axis, by default False
    ymin : float, optional
        Minimum y value, by default 1e-4
    figsize : Tuple[float, float], optional
        Figure size, by default (15, 6)
    """
    warnings.simplefilter(action='ignore', category=UserWarning)
    
    fig, axes = plt.subplots(1, len(df_results_pairs), figsize=figsize, 
                            sharey=sharey, sharex=sharex)
    if len(df_results_pairs) == 1:
        axes = [axes]

    for pair_index, (llc_estimates, results) in enumerate(df_results_pairs):
        steps = get_first(results)['parameters']['log_ivl']
        
        llc_estimates_dict = {
            (row['index'], row['batch_size'], row['lr'], row['snapshot_index']): row['llc']
            for _, row in llc_estimates.iterrows()
        }
        
        llcs = []
        losses = []
        
        for result in iterate_container(results):
            index = result["run_id"]
            result_sparsity = results[index]['parameters']['sparsity']
            
            # Only include results matching the target sparsity
            if abs(result_sparsity - sparsity) > 1e-6:
                continue
                
            llc = llc_estimates_dict.get((index, batch_size, learning_rate, position), np.nan)
            if not np.isnan(llc) and position < len(results[index]['logs']):
                loss = results[index]['logs']['loss'].values[position]
                llcs.append(llc)
                losses.append(loss)

        if llcs:  # Only plot if we have data
            axes[pair_index].scatter(llcs, losses, alpha=0.7, s=50)

        if pair_index == 0:
            title = "Initialized at random 4-gon"
        elif pair_index == 1:
            title = "Initialized at optimal parameters for sparse inputs"
        else:
            title = f"Pair {pair_index}"
            
        epoch = steps[position] if position < len(steps) else position
        axes[pair_index].set_title(f"{title}")
        axes[pair_index].set_xlabel("LLC")
        axes[pair_index].set_ylabel("Loss")
        axes[pair_index].set_xscale(x_scale)
        axes[pair_index].set_yscale(y_scale)
        axes[pair_index].set_ylim(ymin=ymin)

    plt.tight_layout()
    plt.suptitle(f"Loss vs LLC - Sparsity {sparsity:.3f}, Epoch {epoch}", fontsize=16)
    plt.subplots_adjust(top=0.85)

    return fig


# Keep the existing helper functions
def collect_global_sparsities(df_results_pairs):
    sparsities = set()
    for _, results in df_results_pairs:
        for result in iterate_container(results):
            sparsity = results[result["run_id"]]['parameters']['sparsity']
            if sparsity != 0:
                sparsities.add(sparsity)
    return sorted(sparsities)




# %%
compare_dataframes_by_sparsity([(llc_estimates_1_13, results_1_13), (llc_estimates_1_14, results_1_14)], hyperparam_combos=[(300, 0.001)],ymin=1e-2,y_scale='log')

# %% [markdown]
# - lower left is not really nicely visible. 
# - Pair of plots for epoch, inside different sparsity
# - we could also do different sparsities as plots and epochs as colors
# 
# What are observations we make:
# - there seem to be clusters of loss levels, 
# 
# Observation:
# - there is a transition for sparsity 0.67. Hypothesis: Instead of directions it is encoding averages. (Note, it seems the toy models of superposition paper is saying actually something slightly differnet with their prediction?)
# - competing predictions: you can either get lower loss by only representing a subset of the features, or you can get a lower loss by predicting all of them. (How do we figure out which one makes more sense?)

# %%
def plot_results(results, plot_number =5, step =-1, loss_window = (0.14, .16), weird_indices = [], sparsities= [0.426, 0.671, 0.811, 0.892, 0.938, 0.964, 0.98 , 0.988, 0.993], epsilon=0.001):
    # loss_hist = []
    for sparse_value in sparsities:
        plotted =0
        print(f"Plot polygons for sparsity={sparse_value}")
        for index in range(len(results)):
            
            STEPS = results[index]['parameters']['log_ivl']
            logs = results[index]['logs']
            losses = [logs.loc[logs['step'] == s, 'loss'].values[0] for s in STEPS]
            
            outside_loss_window = losses[step] < loss_window[0] or losses[step] > loss_window[1]
            non_matching_sparsity = abs(results[index]['parameters']['sparsity'] - sparse_value) > epsilon
            if non_matching_sparsity or outside_loss_window:
                continue
            else:
                if plotted>=plot_number:
                    continue
                plotted+=1

            print(f"Loss: {losses[step]}")

            NUM_EPOCHS = 20000
            PLOT_STEPS = [min(STEPS, key=lambda s: abs(s-i)) for i in [0, 200, 2000, 10000, NUM_EPOCHS - 1]]
            PLOT_INDICES = [STEPS.index(s) for s in PLOT_STEPS]
            Ws = [results[index]['weights'][i]['embedding.weight'] for i in PLOT_INDICES]
            biases = [results[index]['weights'][i]['unembedding.bias'] for i in PLOT_INDICES]
            
            model = ToyAutoencoder(6, 2, final_bias=True)
            new_weights ={}
            for idx, ndarray in results[index]['weights'][PLOT_INDICES[-1]].items():
                new_weights[idx] = torch.from_numpy(ndarray)

            # criterion=nn.MSELoss()
        
            # model.load_state_dict(new_weights)

            # # print(sample)
            # # print(model(sample))
            # test_set = SyntheticBinaryValued(10000, 6, sparse_value)
            # mean_loss_test = 0
            # for sample in test_set:
            #     output = model(sample)
            #     mean_loss_test += criterion(output, sample)
            # # print("Mean loss test:")
            # loss_hist.append(mean_loss_test)
            # print(f"index: {index}")
            # print(mean_loss_test/10000)
            # # if mean_loss_test<20:
            # #     continue
            # # else:
            # #     weird_indices.append(index)
            

            print(f'index: {index}')
            #all_weights = [[results[j]['weights'][i] for i in PLOT_INDICES] for j in range(len(results))]
            plot_losses_and_polygons(STEPS, losses, PLOT_STEPS, Ws, biases)
            plt.show()

# %%
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


# %%
def get_weights(results, index, step=-1):
    """
    Get the weights of a model at a specific index and timestep.
    
    Parameters
    ----------
    results : list
        List of experiment results
    index : int
        Index of the experiment in results
    step : int, optional
        Step index to get weights from. Default is -1 (last step)
        
    Returns
    -------
    dict
        Dictionary containing the weights at the specified step
    """
    if index >= len(results):
        raise IndexError(f"Index {index} out of range for results of length {len(results)}")
    
    weights_list = results[index]['weights']
    
    if step >= len(weights_list):
        raise IndexError(f"Step {step} out of range for weights list of length {len(weights_list)}")
    
    weights = weights_list[step]
    return weights['embedding.weight'], weights['unembedding.bias']


# %%
def calculate_convex_hull_vertices(W):

    """
    Calculate the number of vertices of the convex hull of the points represented by the columns of W.
    
    Parameters:
    W (torch.Tensor): A 2xN matrix where each column represents a point in 2D space.
    
    Returns:
    int: The number of vertices of the convex hull.
    """
    if W.shape[0] != 2:
        raise ValueError("The weight matrix W must have 2 rows.")
    
    # Convert the tensor to a numpy array if it isn't already
    if isinstance(W, torch.Tensor):
        W = W.cpu().detach().numpy()
    
    hull = ConvexHull(W.T)
    return len(hull.vertices)  # The number of vertices is the same as the number of edges

def count_kgons(W):
    edge_counts = {}
    
    # Process each weight matrix
    for full_w in W:
        num_edges = classify_kgon(full_w)
        if num_edges in edge_counts:
            edge_counts[num_edges] += 1
        else:
            edge_counts[num_edges] = 1

    return edge_counts

def classify_5_gon(W, b, differentiate_5_plus=False):
    """
    Classify a 5-gon based on the weights and biases. 
    """

    # Convert tensor to numpy if it isn't already
    if isinstance(W, torch.Tensor):
        W = W.cpu().detach().numpy()
    
    if W.shape[0] == 2:
        W = W.T

    # Compute the convex hull
    hull = ConvexHull(W)
    
    # Check if the number of vertices is equal to 5
    if len(hull.vertices) != 5:
        return "not a 5-gon"
    
    # Convert biases to a numpy array if it isn't already
    if isinstance(b, torch.Tensor):
        b = b.cpu().detach().numpy()


    # Check if any of the non-vertex biases are large negative
    non_vertex_biases = np.delete(b, hull.vertices)

    # Check for any positive bias that is not part of the convex hull vertices
    non_hull_positive_bias = np.any(non_vertex_biases > 0)

    if not non_hull_positive_bias:
        return 5
    elif non_hull_positive_bias and differentiate_5_plus:
        return "5+"
    elif non_hull_positive_bias and not differentiate_5_plus:
        return 5
    else:

        return 'not a 5-gon'



def classify_kgon(W):
    embedding_w = W["embedding.weight"]
    edges = calculate_convex_hull_vertices(embedding_w)
    if edges == 5:
        return classify_5_gon(embedding_w, W["unembedding.bias"])
    return edges

# %%
def calculate_kgon_percentages(results, step =-1, sparsities= [0.426, 0.671, 0.811, 0.892, 0.938, 0.964, 0.98 , 0.988, 0.993], epsilon=0.001):
    # loss_hist = []
    for sparse_value in sparsities:
        plotted =0
        print(f"Plot polygons for sparsity={sparse_value}")
        weights = []

        STEPS = results[0]['parameters']['log_ivl']
        NUM_EPOCHS = 20000
        PLOT_STEPS = [min(STEPS, key=lambda s: abs(s-i)) for i in [0, 200, 2000, 10000, NUM_EPOCHS - 1]]
        PLOT_INDICES = [STEPS.index(s) for s in PLOT_STEPS]
        for PLOT_INDEX in range(len(STEPS)):
            weights = []
            for index in range(len(results)):
                non_matching_sparsity = abs(results[index]['parameters']['sparsity'] - sparse_value) > epsilon
                if non_matching_sparsity:
                    continue


                # Ws = [results[index]['weights'][i]['embedding.weight'] for i in PLOT_INDICES]
                # weights = [results[index]['weights'][i] for i in PLOT_INDICES]

                
                weights.append(results[index]['weights'][PLOT_INDEX])
            print(count_kgons(weights))

            


# %%
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def plot_kgon_percentages(results, sparsities=[0.426, 0.671, 0.811, 0.892, 0.938, 0.964, 0.98, 0.988, 0.993], epsilon=0.001):
    STEPS = results[0]['parameters']['log_ivl']
    NUM_EPOCHS = 20000
    PLOT_STEPS = [min(STEPS, key=lambda s: abs(s - i)) for i in [0, 200, 2000, 10000, NUM_EPOCHS - 1]]
    PLOT_INDICES = [STEPS.index(s) for s in PLOT_STEPS]

    for sparse_value in sparsities:
        print(f"Processing sparsity = {sparse_value}")
        kgon_frequencies = []

        for step_idx in PLOT_INDICES:
            weights_at_step = []
            for result in results:
                if abs(result['parameters']['sparsity'] - sparse_value) > epsilon:
                    continue
                weights_at_step.append(result['weights'][step_idx])

            # Count kgons
            counts = count_kgons(weights_at_step)
            total = sum(counts.values())
            frequencies = {k: v / total * 100 for k, v in counts.items()}
            kgon_frequencies.append(frequencies)

        # Normalize keys (ensure all k-gon types are present in each step)
        all_kgons = sorted(set(k for freqs in kgon_frequencies for k in freqs))
        for freqs in kgon_frequencies:
            for k in all_kgons:
                freqs.setdefault(k, 0)

        # Convert to plot format
        plot_data = {k: [freqs[k] for freqs in kgon_frequencies] for k in all_kgons}

        # Plot
        plt.figure()
        for k, values in plot_data.items():
            plt.plot(PLOT_STEPS, values, label=f'{k}-gon')
        plt.title(f'% Frequency of k-gons over training steps\n(sparsity={sparse_value})')
        plt.xlabel('Training Step')
        plt.ylabel('Percentage Frequency')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'kgon_frequencies_sparsity_{sparse_value}.png', dpi=300)
        plt.show()


# %%
plot_kgon_percentages(
    results_random_init
)

# %%
plot_kgon_percentages(
    results_optimal_init
)

# %%
def generate_2d_kgon_vertices(k, rot=0., pad_to=None, force_length=0.9):
    """Set the weights of a 2D k-gon to be the vertices of a regular k-gon."""
    # Angles for the vertices
    theta = np.linspace(0, 2*np.pi, k, endpoint=False) + rot

    # Generate the vertices
    x = np.cos(theta)
    y = np.sin(theta)
    result = np.vstack((x, y))

    if pad_to is not None and k < pad_to:
        num_pad = pad_to - k
        result = np.hstack([result, np.zeros((2, num_pad))])

    return (result * force_length)

def generate_init_param(m, n, init_kgon, prior_std=1., no_bias=True, init_zerobias=True, seed=0, force_negb=False, noise=0.01):
    np.random.seed(seed)

    if init_kgon is None or m != 2:
        init_W = np.random.normal(size=(m, n)) * prior_std
    else:
        assert init_kgon <= n
        rand_angle = np.random.uniform(0, 2 * np.pi, size=(1,))
        noise = np.random.normal(size=(m, n)) * noise
        init_W = generate_2d_kgon_vertices(init_kgon, rot=rand_angle, pad_to=n) + noise

    if no_bias:
        param = {"W": init_W}
    else:
        init_b = np.random.normal(size=(n, 1)) * prior_std
        if force_negb:
            init_b = -np.abs(init_b)
        if init_zerobias:
            init_b = init_b * 0
        param = {
            "W": init_W,
            "b": init_b
        }
    return param

def generate_optimal_solution(m,n,rot=0.0):
    assert m == 2
    assert n==6 # Possibly implement other values of n later. See page 46 of dynamical bayseanism paper and code that automatically finds solution(s).
    # Solutions exist for multiples of 4 and 5,6 and 7
    if n == 6:
        l =1.4142 # confusion: I get as the optimal parameter for the length: 1.4142, but the paper says 1.32053
        init_b = - np.ones((n,1)) *0.9999 # confusion: I get through training, that the optimal bias is -0.9999 instead of 0.61814
        

    init_w = generate_2d_kgon_vertices(n, rot=rot, force_length=l, pad_to=n)
    param = {
        "W": init_w,
        "b": init_b
    }
    return param

# %%

m = 6
n = 2
l = 1.4142
b = -1

w = torch.from_numpy(generate_2d_kgon_vertices(m, rot=0., force_length=l, pad_to=m)).float()
bias = torch.ones((m)) * b

sample = torch.tensor([1,0,0,0,0,0]).float()

print(sample)
output =(w.T  @ ( w @ sample)) + b
print(output)



# %%
model = lambda w, b: (lambda sample:(torch.Tensor(w).T  @ ( torch.Tensor(w) @ torch.Tensor(sample))) + torch.Tensor(b))

# %%
from tms.data.dataset import SyntheticBinarySparseValued

# %%
dataset = SyntheticBinarySparseValued(num_samples=1000, num_features=6, sparsity=0.42)

# %%
dataset[0]

# %%
w, b = get_weights(results_1_13, 0)

# %%
sample = torch.tensor([1,0,0,0,0,0]).float()

# %%
plot_specific_index(results_1_13, 0)

# %% [markdown]
# How do I qualify this solution? I guess we have the loss space:
# params -> loss (given dataset)
# We could also look at given 

# %%
model(w,b)(dataset[0])

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def get_weights(results, index, step=-1):
    """
    Get the weights of a model at a specific index and timestep.
    """
    if index >= len(results):
        raise IndexError(f"Index {index} out of range for results of length {len(results)}")
    
    weights_list = results[index]['weights']
    
    if step >= len(weights_list):
        raise IndexError(f"Step {step} out of range for weights list of length {len(weights_list)}")
    
    weights = weights_list[step]
    embedding_weight = torch.from_numpy(weights['embedding.weight']).float()
    unembedding_bias = torch.from_numpy(weights['unembedding.bias']).float()
    
    return embedding_weight, unembedding_bias

def test_get_weights(results):
    """Test the get_weights function"""
    print("=== Testing get_weights ===")
    try:
        W, b = get_weights(results, 0)
        print(f"✓ Model 0 weights shape: {W.shape}, bias shape: {b.shape}")
        
        W2, b2 = get_weights(results, 1)
        print(f"✓ Model 1 weights shape: {W2.shape}, bias shape: {b2.shape}")
        
        # Test that different models have different weights
        weights_different = not torch.allclose(W, W2)
        print(f"✓ Different models have different weights: {weights_different}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def autoencoder_forward(input_vec, W, b):
    """
    Single forward pass: output = W.T @ (W @ input_vec) + b
    
    W is shape (2, 6) - encoder weights
    For autoencoder: encode with W, decode with W.T
    """
    # W is (2, 6), input_vec is (6,)
    # Encode: W @ input_vec 
    encoded = W @ input_vec    # (2, 6) @ (6,) = (2,)
    # Decode: W.T @ encoded
    decoded = W.T @ encoded    # (6, 2) @ (2,) = (6,)
    # Add bias
    output = torch.relu(decoded + b)       # (6,) + (6,) = (6,)
    return output

def test_autoencoder_forward():
    """Test the autoencoder forward pass"""
    print("\n=== Testing autoencoder_forward ===")
    
    # Create test data with correct dimensions
    W = torch.randn(2, 6)  # Encoder weights: (latent_dim, input_dim)
    b = torch.randn(6)     # Bias for reconstruction
    input_vec = torch.tensor([1., 0., 0., 0., 0., 0.])
    
    try:
        output = autoencoder_forward(input_vec, W, b)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        # Test with different input
        input_vec2 = torch.tensor([0., 1., 0., 0., 0., 0.])
        output2 = autoencoder_forward(input_vec2, W, b)
        
        # Outputs should be different for different inputs
        outputs_different = not torch.allclose(output, output2)
        print(f"✓ Different inputs give different outputs: {outputs_different}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def generate_all_inputs():
    """Generate all 64 possible 6-bit binary inputs"""
    all_inputs = []
    for i in range(64):
        binary = format(i, '06b')
        input_vec = torch.tensor([float(int(b)) for b in binary])
        all_inputs.append(input_vec)
    return all_inputs

def test_generate_all_inputs():
    """Test input generation"""
    print("\n=== Testing generate_all_inputs ===")
    
    inputs = generate_all_inputs()
    print(f"✓ Generated {len(inputs)} inputs")
    
    # Check first few
    print(f"✓ Input 0: {inputs[0].numpy()}")  # Should be [0,0,0,0,0,0]
    print(f"✓ Input 1: {inputs[1].numpy()}")  # Should be [0,0,0,0,0,1]
    print(f"✓ Input 63: {inputs[63].numpy()}")  # Should be [1,1,1,1,1,1]
    
    # Check all are different
    unique_inputs = len(set([tuple(inp.numpy()) for inp in inputs]))
    print(f"✓ All inputs unique: {unique_inputs == 64}")
    
    return inputs

def create_loss_matrix_simple(results, max_models=200, step=-1):
    """
    Create loss matrix for first max_models models only
    """
    print(f"\n=== Creating loss matrix for first {max_models} models ===")
    
    # Generate all inputs
    all_inputs = generate_all_inputs()
    
    # Limit to first max_models
    num_models = min(max_models, len(results))
    print(f"Processing {num_models} models...")
    
    # Initialize loss matrix
    loss_matrix = torch.zeros(64, num_models)
    sparsities = []
    
    for model_idx in range(num_models):
        if model_idx % 50 == 0:
            print(f"Processing model {model_idx}...")
            
        try:
            # Get model weights
            W, b = get_weights(results, model_idx, step)
            sparsities.append(results[model_idx]['parameters']['sparsity'])
            
            # Compute loss for all 64 inputs
            for input_idx, input_vec in enumerate(all_inputs):
                output = autoencoder_forward(input_vec, W, b)
                mse_loss = torch.mean((output - input_vec) ** 2)
                loss_matrix[input_idx, model_idx] = mse_loss
                
        except Exception as e:
            print(f"Error processing model {model_idx}: {e}")
            loss_matrix[:, model_idx] = float('nan')
            sparsities.append(float('nan'))
    
    return loss_matrix.numpy(), all_inputs, np.array(sparsities)

def test_loss_matrix(results):
    """Test loss matrix creation"""
    print("\n=== Testing loss matrix creation ===")
    
    # Test with just 3 models first
    loss_matrix, inputs, sparsities = create_loss_matrix_simple(results, max_models=3)
    
    print(f"✓ Loss matrix shape: {loss_matrix.shape}")
    print(f"✓ Sparsities shape: {sparsities.shape}")
    
    # Check for variation
    loss_variance_across_models = np.var(loss_matrix, axis=1)
    loss_variance_across_inputs = np.var(loss_matrix, axis=0)
    
    print(f"✓ Loss variance across models (first 5): {loss_variance_across_models[:5]}")
    print(f"✓ Loss variance across inputs (all 3): {loss_variance_across_inputs}")
    
    # Check if we have variation (not all zeros/identical)
    has_model_variation = np.any(loss_variance_across_models > 0.001)
    has_input_variation = np.any(loss_variance_across_inputs > 0.001)
    
    print(f"✓ Has variation across models: {has_model_variation}")
    print(f"✓ Has variation across inputs: {has_input_variation}")
    
    return loss_matrix, inputs, sparsities

def visualize_first_sparsity(results, max_models=200):
    """
    Visualize the first sparsity group (first 200 models)
    """
    print(f"\n=== Visualizing first {max_models} models ===")
    
    # Create loss matrix
    loss_matrix, inputs, sparsities = create_loss_matrix_simple(results, max_models)
    
    # Remove any NaN models
    valid_models = ~np.isnan(loss_matrix).any(axis=0)
    loss_matrix = loss_matrix[:, valid_models]
    sparsities = sparsities[valid_models]
    
    print(f"Valid models: {valid_models.sum()}/{max_models}")
    
    if valid_models.sum() < 2:
        print("Not enough valid models for visualization")
        return
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Loss matrix heatmap
    ax = axes[0, 0]
    im = ax.imshow(loss_matrix, aspect='auto', cmap='viridis')
    ax.set_title(f'Loss Matrix (64 inputs × {valid_models.sum()} models)')
    ax.set_xlabel('Model Index')
    ax.set_ylabel('Input Index')
    plt.colorbar(im, ax=ax, label='Loss')
    
    # 2. PCA of models
    ax = axes[0, 1]
    if loss_matrix.shape[1] >= 2:
        pca = PCA(n_components=2)
        model_pca = pca.fit_transform(loss_matrix.T)
        
        scatter = ax.scatter(model_pca[:, 0], model_pca[:, 1], c=sparsities, cmap='plasma', alpha=0.7)
        ax.set_title(f'PCA of Models (var: {pca.explained_variance_ratio_.sum():.2f})')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
        plt.colorbar(scatter, ax=ax, label='Sparsity')
    
    # 3. Input discrimination
    ax = axes[1, 0]
    input_variance = np.var(loss_matrix, axis=1)
    bars = ax.bar(range(64), input_variance)
    
    # Highlight top 5 discriminative inputs
    top_discriminative = np.argsort(input_variance)[-5:]
    for idx in top_discriminative:
        bars[idx].set_color('red')
    
    ax.set_title('Input Discrimination Power')
    ax.set_xlabel('Input Index')
    ax.set_ylabel('Loss Variance Across Models')
    
    # 4. Sample losses across models
    ax = axes[1, 1]
    # Plot losses for a few sample inputs
    sample_inputs = [0, 1, 31, 63]  # Binary: 000000, 000001, 011111, 111111
    for inp_idx in sample_inputs:
        if inp_idx < loss_matrix.shape[0]:
            binary = format(inp_idx, '06b')
            ax.plot(loss_matrix[inp_idx, :], label=f'Input {inp_idx} ({binary})', alpha=0.7)
    
    ax.set_title('Loss Patterns for Sample Inputs')
    ax.set_xlabel('Model Index')
    ax.set_ylabel('Loss')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print analysis
    print(f"\nSparsity range: {sparsities.min():.3f} to {sparsities.max():.3f}")
    print(f"Loss range: {loss_matrix.min():.4f} to {loss_matrix.max():.4f}")
    
    print(f"\nTop 5 discriminative inputs:")
    for i, idx in enumerate(top_discriminative):
        binary = format(idx, '06b')
        num_ones = binary.count('1')
        print(f"  {i+1}. Input {idx:2d} ({binary}) - {num_ones} ones - var: {input_variance[idx]:.4f}")
    
    return loss_matrix, sparsities

def run_all_tests(results):
    """Run all test functions"""
    print("🧪 Running all tests...\n")
    
    # Test individual functions
    test1 = test_get_weights(results)
    test2 = test_autoencoder_forward()
    test3 = test_generate_all_inputs()
    test4 = test_loss_matrix(results)
    
    print(f"\n📊 Test Results:")
    print(f"✓ get_weights: {test1}")
    print(f"✓ autoencoder_forward: {test2}")
    print(f"✓ generate_all_inputs: {test3}")
    print(f"✓ loss_matrix: {test4}")
    
    if all([test1, test2, test3, test4]):
        print("\n🎉 All tests passed! Running full visualization...")
        return visualize_first_sparsity(results, max_models=2000)  # Pass max_models explicitly
    else:
        print("\n❌ Some tests failed. Fix issues before proceeding.")
        return None

# Example usage:
#loss_matrix, sparsities = run_all_tests(results)

# %%
loss_matrices=[]
sparsities=[]
for i in range(10):
    loss_matrix,_, sparsity = create_loss_matrix_simple(results_1_13[i*200:(i+1)*200], 200)
    loss_matrices.append(loss_matrix)
    sparsities.append(sparsity)

# %%
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import time

def permutation_invariant_distance_hungarian(pattern1, pattern2, metric='euclidean'):
    """
    Compute minimum distance using Hungarian algorithm (O(n³)).
    
    This is faster for larger n, but for n=6 brute force is fine.
    """
    pattern1 = np.array(pattern1)
    pattern2 = np.array(pattern2)
    n = len(pattern1)
    
    # Create cost matrix
    cost_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if metric == 'euclidean':
                cost_matrix[i, j] = (pattern1[i] - pattern2[j])**2
            elif metric == 'manhattan':
                cost_matrix[i, j] = abs(pattern1[i] - pattern2[j])
            elif metric == 'cosine':
                # For single elements, cosine doesn't make sense
                # Fall back to absolute difference
                cost_matrix[i, j] = abs(pattern1[i] - pattern2[j])
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    min_cost = cost_matrix[row_ind, col_ind].sum()
    
    if metric == 'euclidean':
        min_distance = np.sqrt(min_cost)
    else:
        min_distance = min_cost
    
    return min_distance, col_ind

def permutation_invariant_pdist(patterns, metric='euclidean'):
    """
    Compute pairwise permutation-invariant distances for clustering.
    
    Parameters:
    -----------
    patterns : array-like, shape (n_patterns, 6)
        Each row is a loss pattern for 6 states
    metric : str
        Distance metric to use
    
    Returns:
    --------
    distances : array, shape (n_patterns * (n_patterns - 1) / 2,)
        Condensed distance matrix suitable for scipy clustering
    """
    patterns = np.array(patterns)
    n_patterns = patterns.shape[0]
    
    # Compute all pairwise distances
    distances = []
    
    for i in range(n_patterns):
        for j in range(i + 1, n_patterns):
            dist, _ = permutation_invariant_distance_hungarian(
                patterns[i], patterns[j], metric
            )
            distances.append(dist)
    
    return np.array(distances)

def create_permutation_invariant_dendrogram(loss_matrix, metric='euclidean'):
    """
    Create dendrogram using permutation-invariant distances.
    
    Parameters:
    -----------
    loss_matrix : array, shape (n_inputs, n_models)
        Loss matrix where each column is a model's loss pattern
    metric : str
        Distance metric
    
    Returns:
    --------
    Z : array
        Linkage matrix for dendrogram
    distances : array
        Pairwise distances used
    """
    # Transpose so each row is a model's pattern across inputs
    patterns = loss_matrix.T
    
    print(f"Computing permutation-invariant distances for {patterns.shape[0]} models...")
    print(f"Each model has {patterns.shape[1]} loss values")
    
    # Compute permutation-invariant distances
    start_time = time.time()
    distances = permutation_invariant_pdist(patterns, metric)
    compute_time = time.time() - start_time
    
    print(f"Computed {len(distances)} pairwise distances in {compute_time:.2f}s")
    
    # Create linkage matrix
    Z = linkage(distances, method='ward')
    
    return Z, distances

# %%
large_loss_matrix, large_sparsities = visualize_first_sparsity(results_1_13, max_models=2000)

def calculate_convex_hull_vertices(W):
    """
    Calculate the number of vertices of the convex hull of the points represented by the columns of W.
    
    Parameters:
    W (torch.Tensor): A 2xN matrix where each column represents a point in 2D space.
    
    Returns:
    int: The number of vertices of the convex hull.
    """
    if W.shape[0] != 2:
        raise ValueError("The weight matrix W must have 2 rows.")
    
    # Convert the tensor to a numpy array if it isn't already
    if isinstance(W, torch.Tensor):
        W = W.cpu().detach().numpy()
    
    hull = ConvexHull(W.T)
    return len(hull.vertices)

def classify_5_gon(W, b, differentiate_5_plus=False):
    """
    Classify a 5-gon based on the weights and biases. 
    """
    # Convert tensor to numpy if it isn't already
    if isinstance(W, torch.Tensor):
        W = W.cpu().detach().numpy()
    
    if W.shape[0] == 2:
        W = W.T
    # Compute the convex hull
    hull = ConvexHull(W)
    
    # Check if the number of vertices is equal to 5
    if len(hull.vertices) != 5:
        return "not a 5-gon"
    
    # Convert biases to a numpy array if it isn't already
    if isinstance(b, torch.Tensor):
        b = b.cpu().detach().numpy()
    # Check if any of the non-vertex biases are large negative
    non_vertex_biases = np.delete(b, hull.vertices)
    # Check for any positive bias that is not part of the convex hull vertices
    non_hull_positive_bias = np.any(non_vertex_biases > 0)
    if not non_hull_positive_bias:
        return 5
    elif non_hull_positive_bias and differentiate_5_plus:
        return "5+"
    elif non_hull_positive_bias and not differentiate_5_plus:
        return 5
    else:
        return 'not a 5-gon'

def classify_kgon(W):
    embedding_w = W["embedding.weight"]
    edges = calculate_convex_hull_vertices(embedding_w)
    if edges == 5:
        return classify_5_gon(embedding_w, W["unembedding.bias"])
    return edges

def analyze_biases(bias_vector, epsilon=0.1):
    """
    Analyze bias vector and count positive, negative, and zero biases.
    
    Parameters:
    -----------
    bias_vector : array-like
        The bias vector to analyze
    epsilon : float
        Tolerance for considering a bias as zero
    
    Returns:
    --------
    dict: Counts of positive, negative, zero biases
    """
    if isinstance(bias_vector, torch.Tensor):
        bias_vector = bias_vector.cpu().detach().numpy()
    
    bias_vector = np.array(bias_vector)
    
    positive = np.sum(bias_vector > epsilon)
    negative = np.sum(bias_vector < -epsilon)
    zero = np.sum(np.abs(bias_vector) <= epsilon)
    
    return {
        'positive': int(positive),
        'negative': int(negative), 
        'zero': int(zero),
        'total': len(bias_vector)
    }

def classify_all_solutions(results, sparsities, epsilon=0.1):
    """
    Classify all solutions by k-gon type and bias patterns.
    
    Parameters:
    -----------
    results : list
        List of experimental results
    sparsities : array-like
        Array of sparsity values corresponding to results
    epsilon : float
        Tolerance for zero bias classification
    
    Returns:
    --------
    classifications : list
        List of dicts containing classification info for each model
    """
    classifications = []
    
    print(f"Classifying {len(results)} models...")
    
    for i, result in enumerate(results):
        try:
            # Get final weights (last snapshot)
            final_weights = result['weights'][-1]
            
            # Classify k-gon
            kgon_type = classify_kgon(final_weights)
            
            # Analyze biases
            bias_analysis = analyze_biases(final_weights["unembedding.bias"], epsilon)
            
            # Get sparsity
            sparsity = result['parameters']['sparsity']
            
            # Store classification
            classification = {
                'model_index': i,
                'sparsity': sparsity,
                'kgon_type': kgon_type,
                'bias_positive': bias_analysis['positive'],
                'bias_negative': bias_analysis['negative'],
                'bias_zero': bias_analysis['zero'],
                'bias_total': bias_analysis['total'],
                'bias_pattern': f"{bias_analysis['positive']}pos_{bias_analysis['negative']}neg"
            }
            
            classifications.append(classification)
            
        except Exception as e:
            print(f"Error processing model {i}: {e}")
            # Add failed classification
            classifications.append({
                'model_index': i,
                'sparsity': sparsities[i] if i < len(sparsities) else None,
                'kgon_type': 'error',
                'bias_positive': 0,
                'bias_negative': 0,
                'bias_zero': 0,
                'bias_total': 0,
                'bias_pattern': 'error'
            })
    
    return classifications

def create_classification_summary(classifications):
    """
    Create summary statistics of all classifications.
    """
    # Group by sparsity
    by_sparsity = defaultdict(list)
    for cls in classifications:
        by_sparsity[cls['sparsity']].append(cls)
    
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    
    for sparsity in sorted(by_sparsity.keys()):
        models = by_sparsity[sparsity]
        print(f"\nSparsity {sparsity} ({len(models)} models):")
        print("-" * 40)
        
        # Count k-gon types
        kgon_counts = Counter([m['kgon_type'] for m in models])
        print("K-gon types:")
        for kgon_type, count in sorted(kgon_counts.items()):
            print(f"  {kgon_type}-gon: {count}")
        
        # Count bias patterns
        bias_patterns = Counter([m['bias_pattern'] for m in models])
        print("Bias patterns:")
        for pattern, count in sorted(bias_patterns.items()):
            print(f"  {pattern}: {count}")
    
    # Overall summary
    print(f"\nOVERALL SUMMARY ({len(classifications)} total models):")
    print("-" * 40)
    all_kgons = Counter([m['kgon_type'] for m in classifications])
    for kgon_type, count in sorted(all_kgons.items()):
        print(f"{kgon_type}-gon: {count}")
    
    return by_sparsity

def create_annotated_dendrogram(results,save_path=f"{result_path}annotated_dendrogram.svg", 
                               figsize=(30, 20), dpi=300):
# def create_annotated_dendrogram(Z, classifications, ):
    """
    Create a large annotated dendrogram with k-gon and bias information.
    
    Parameters:
    -----------
    Z : array
        Linkage matrix
    classifications : list
        Classification results
    save_path : str
        Path to save SVG file
    figsize : tuple
        Figure size in inches
    dpi : int
        DPI for the figure
    """
    loss_matrix, _, sparsity = create_loss_matrix_simple(results, max_models=len(results))

    Z, _ = create_permutation_invariant_dendrogram(loss_matrix)
    classifications = classify_all_solutions(results, sparsity)
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    def ann(idx):
        cls = classifications[idx]
        return f"M{cls['model_index']}_{cls['kgon_type']}-gon_S:{cls['sparsity']:.3f}_{cls['bias_pattern']}"
    # Create dendrogram
    dend = dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=8)
    leaves = dend['leaves']
    
    # Create color map for k-gon types
    unique_kgons = list(set([c['kgon_type'] for c in classifications]))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_kgons)))
    kgon_color_map = dict(zip(unique_kgons, colors))
    
    # Annotate leaves
    leaf_positions = np.arange(len(leaves)) * 10  # Position along x-axis
    
    for i, leaf_idx in enumerate(leaves):
        if leaf_idx < len(classifications):
            cls = classifications[leaf_idx]
            
            # Position for annotation
            x_pos = i * 10
            y_pos = -0.05 * ax.get_ylim()[1]  # Below the dendrogram
            
            # K-gon type annotation
            kgon_color = kgon_color_map[cls['kgon_type']]
            
            # Create annotation text
            annotation = f"M{cls['model_index']}_{cls['kgon_type']}-gon_S:{cls['sparsity']:.3f}_{cls['bias_pattern']}"
            
            # Add colored rectangle for k-gon type
            rect = patches.Rectangle((x_pos - 4, y_pos - 0.02 * ax.get_ylim()[1]), 
                                   8, 0.01 * ax.get_ylim()[1], 
                                   facecolor=kgon_color, alpha=0.7)
            ax.add_patch(rect)
            
            # Add text annotation
            ax.text(x_pos, y_pos - 0.03 * ax.get_ylim()[1], annotation, 
                   ha='center', va='top', fontsize=6, rotation=90)
    
    # Create legend for k-gon types
    legend_elements = []
    for kgon_type, color in kgon_color_map.items():
        legend_elements.append(patches.Patch(color=color, label=f'{kgon_type}-gon'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Set title and labels
    ax.set_title('Annotated Dendrogram with K-gon Classifications and Bias Patterns', 
                fontsize=16)
    ax.set_xlabel('Model Index (with k-gon type and bias pattern)', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save as SVG
    print(f"Saving annotated dendrogram to {save_path}...")
    plt.savefig(save_path, format='svg', dpi=dpi, bbox_inches='tight')
    plt.show()
    
    return fig, ax

def quick_classification_test(results, n_models=50):
    """
    Quick test on first n_models to verify the classification works.
    """
    print(f"Testing classification on first {n_models} models...")
    
    test_results = results[:n_models]
    sparsities = [r['parameters']['sparsity'] for r in test_results]
    
    classifications = classify_all_solutions(test_results, sparsities)
    summary = create_classification_summary(classifications)
    
    return classifications, summary

# Example usage
if __name__ == "__main__":
    print("K-gon Classification and Analysis Tool")
    print("Usage:")
    print("1. classifications = classify_all_solutions(results, sparsities)")
    print("2. summary = create_classification_summary(classifications)")
    print("3. create_annotated_dendrogram(Z, classifications, 'large_dendrogram.svg')")
    print("4. Or test first: quick_classification_test(results, n_models=50)")

# %%
quick_classification_test(results_1_13, n_models=50)

# %%

# create_dendrogram_visualization_mimimum(loss_matrix, info=f'{sparsities[i][0]:.3f}')
small_results = results_random_init[:10]
create_annotated_dendrogram(small_results,save_path=f"{result_path}annotated_dendrogram_small.svg")

# %%

# create_dendrogram_visualization_mimimum(loss_matrix, info=f'{sparsities[i][0]:.3f}')
small_results = results_random_init[:10]
loss_matrix, _, sparsity = create_loss_matrix_simple(small_results, 10)
Z, distances = create_permutation_invariant_dendrogram(loss_matrix)
create_annotated_dendrogram(Z, classifications = classify_all_solutions(small_results, sparsity),save_path=f"{result_path}annotated_dendrogram_small.svg")

# %%
for i in range(10):
    create_annotated_dendrogram(results_1_13[i*200:(i+1)*200],save_path=f"{result_path}annotated_dendrogram_{i}.svg")

# %%
Z, distances = create_permutation_invariant_dendrogram(large_loss_matrix[:,:10])

# %%
create_annotated_dendrogram(Z, 
                            classifications=classify_all_solutions(results_1_13, large_sparsities),
                            save_path="{result_path}large_dendrogram.svg",
                            figsize=(30, 20), dpi=300)
