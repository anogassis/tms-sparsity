import warnings

from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull

import torch
import os

from typing import List, Dict, Any, Tuple

from tms.utils.utils import load_results, get_first, iterate_container
from tms.models.autoencoder import ToyAutoencoder
from tms.llc import get_llc_data, preaggregate_llc
from tms.plots.kgons import plot_losses_and_polygons
from tms.plots.losses import compare_dataframes_and_results, Results, DfResultPair


import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tms.models.autoencoder import ToyAutoencoder
from tms.plots.kgons import plot_losses_and_polygons
from tms.utils.utils import iterate_container, get_first


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import linkage, dendrogram
import time

result_path="../../results"

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

    def create_position_color_mapping(positions):
        """Create color mapping for different positions/timesteps."""
        cmap = plt.get_cmap("tab10")
        n_colors = cmap.N
        return {pos: cmap(i % n_colors) for i, pos in enumerate(sorted(positions))}


    def collect_global_sparsities(df_results_pairs):
        sparsities = set()
        for _, results in df_results_pairs:
            for result in iterate_container(results):
                sparsity = results[result["run_id"]]['parameters']['sparsity']
                if sparsity != 0:
                    sparsities.add(sparsity)
        return sorted(sparsities)

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

def plot_results(results, plot_number =5, step =-1, loss_window = (0.14, .16), weird_indices = [], sparsities= [0.426, 0.671, 0.811, 0.892, 0.938, 0.964, 0.98 , 0.988, 0.993], epsilon=0.001):
    """
    Variant of plot_results with loss_window, which is useful if we want to find models that one spotted through the loss-vs-llc-plot
    """
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

            print(f'index: {index}')
            plot_losses_and_polygons(STEPS, losses, PLOT_STEPS, Ws, biases)
            plt.show()

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

def get_weights(results:Results, index:int, step:int=-1)->tuple[ torch.Tensor, torch.Tensor ]:
    """
    Get the weights of a model at a specific index and timestep.
    
    """
    if index >= len(results):
        raise IndexError(f"Index {index} out of range for results of length {len(results)}")
    
    weights_list = results[index]['weights']
    
    if step >= len(weights_list):
        raise IndexError(f"Step {step} out of range for weights list of length {len(weights_list)}")
    
    weights = weights_list[step]
    return weights['embedding.weight'], weights['unembedding.bias']


def calculate_convex_hull_vertices(W:torch.Tensor)->int:

    """
    Calculate the number of vertices of the convex hull of the points represented by the columns of W.
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


def generate_2d_kgon_vertices(k, rot:float=0., pad_to=None, force_length=0.9):
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
        init_W = generate_2d_kgon_vertices(init_kgon, rot=float(rand_angle), pad_to=n) + noise

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

def generate_all_inputs():
    """Generate all 64 possible 6-bit binary inputs"""
    all_inputs = []
    for i in range(64):
        binary = format(i, '06b')
        input_vec = torch.tensor([float(int(b)) for b in binary])
        all_inputs.append(input_vec)
    return all_inputs

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

def permutation_invariant_distance_hungarian(pattern1, pattern2, metric='euclidean'):
    """
    Compute minimum distance using Hungarian algorithm (O(n³)). Faster than brute force even for n=6.
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
                'bias_pattern': f"{bias_analysis['positive']}pos_{bias_analysis['negative']}neg_{bias_analysis['zero']}_zero"
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

def create_annotated_dendrogram(results,save_path=f"{result_path}annotated_dendrogram.svg"
                               figsize=(30, 20), dpi=300):
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
    dend = dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=8, leaf_label_func=ann)

    # Create color map for k-gon types
    unique_kgons = list(set([c['kgon_type'] for c in classifications]))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_kgons)))
    kgon_color_map = dict(zip(unique_kgons, colors))
    
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
    
    print(f"Saving annotated dendrogram to {save_path}...")
    plt.savefig(save_path, format='svg', dpi=dpi, bbox_inches='tight')

    return fig, ax

def plot_everything(results_random_init: List[Any], llc_estimates_random_init:pd.DataFrame, results_optimal_init: Results, llc_estimates_optimal_init:pd.DataFrame):
    compare_dataframes_and_results(((llc_estimates_random_init, results_random_init),(llc_estimates_optimal_init, results_random_init)), ymin=0, plot=False)

    plot_kgon_percentages(
        results_random_init
    )

    plot_kgon_percentages(
        results_optimal_init
    )

    loss_matrices=[]
    sparsities=[]
    small_results = results_random_init[:10]

    create_annotated_dendrogram(small_results,save_path=f"{result_path}annotated_dendrogram_small.svg")

    for i in range(10):
        create_annotated_dendrogram(results_random_init[i*200:(i+1)*200],save_path=f"{result_path}annotated_dendrogram_{i}.svg", save=False, plot=False)

def main():
    data_path = "../../data"
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

    #TODO: check results from get_weights
    plot_everything(results_random_init=results_1_13, llc_estimates_random_init=llc_estimates_1_13, results_optimal_init=results_1_14, llc_estimates_optimal_init=llc_estimates_1_14)

main()
