import warnings

from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import operator
import functools

import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull

import torch
import torch.nn as nn
import os

from typing import List, Dict, Any, Tuple

from tms.utils.utils import load_results, get_first, iterate_container
from tms.models.autoencoder import ToyAutoencoder
from tms.llc import get_llc_data, preaggregate_llc
from tms.plots.kgons import plot_losses_and_polygons, plot_polygon
from tms.plots.losses import compare_dataframes_and_results, Results, DfResultPair


from einops import rearrange, reduce, repeat, einsum
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped #Array (We cannot used array (which by my understanding generalizes between numpy and pytorch, because that would require us to install jax))
import typeguard


import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tms.models.autoencoder import ToyAutoencoder
from tms.plots.kgons import plot_losses_and_polygons
from tms.utils.utils import iterate_container, get_first
from tms.data.dataset import SyntheticBinarySparseValued


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import linkage, dendrogram
import time

plot_path="../../results/"


sparse_value=0.426
test_set_size = 10000
test_X = torch.stack([x for x in SyntheticBinarySparseValued(test_set_size, 6, sparse_value)]).float()

def compute_loss(W,b):
    encoded = test_X @ W.T          # (N, 2)
    decoded = encoded @ W      # (N, 6)
    out = decoded + b       # (N, 6)  (bias broadcasts)
    return torch.mean((out-test_X).pow(2))   # scalar mean MSE over all samples and dims

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
    kgon = calculate_convex_hull_vertices(Ws[-1], 0.05)

    print(f"Kgon (with epsilon 0.05): {kgon}")
    # Optional: Load model weights
    model = ToyAutoencoder(6, 2, final_bias=True)
    new_weights = {}
    for idx, ndarray in result['weights'][PLOT_INDICES[-1]].items():
        new_weights[idx] = torch.from_numpy(ndarray)


    test_losses = []
    for i, s in enumerate(STEPS):
        weights = results[index]['weights'][i]
        W = weights['embedding.weight']
        b = weights['unembedding.bias']
        loss = compute_loss(W,b)
        test_losses.append((s,loss))

    loss = compute_loss(Ws[-1],biases[-1])

    plot_losses_and_polygons(STEPS, losses, PLOT_STEPS, Ws, biases, run=index, test_losses=test_losses)
    plt.show()

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


            NUM_EPOCHS = 20000
            PLOT_STEPS = [min(STEPS, key=lambda s: abs(s-i)) for i in [0, 200, 2000, 10000, NUM_EPOCHS - 1]]
            PLOT_INDICES = [STEPS.index(s) for s in PLOT_STEPS]
            Ws = [results[index]['weights'][i]['embedding.weight'] for i in PLOT_INDICES]
            biases = [results[index]['weights'][i]['unembedding.bias'] for i in PLOT_INDICES]

            test_losses = []
            for i, s in enumerate(STEPS):
                weights = results[index]['weights'][i]
                W = weights['embedding.weight']
                b = weights['unembedding.bias']
                loss = compute_loss(W,b)
                test_losses.append((s,loss))

            loss = compute_loss(Ws[-1],biases[-1])
            # print(f"Loss: {losses[step]}")
            print(f"Loss: {loss}")
            model = ToyAutoencoder(6, 2, final_bias=True)
            new_weights ={}
            for idx, ndarray in results[index]['weights'][PLOT_INDICES[-1]].items():
                new_weights[idx] = torch.from_numpy(ndarray)
            print(f"Test_losses: {test_losses}")

            # print(f'index: {index}')
            #all_weights = [[results[j]['weights'][i] for i in PLOT_INDICES] for j in range(len(results))]
            plot_losses_and_polygons(STEPS, losses, PLOT_STEPS, Ws, biases, run=index, test_losses=test_losses)
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
    return torch.Tensor(weights['embedding.weight']), torch.Tensor(weights['unembedding.bias'])

def calculate_convex_hull_vertices(W:torch.Tensor, epsilon=0.)->int:

    """
    Calculate the number of vertices of the convex hull of the points represented by the columns of W.
    """
    if W.shape[0] != 2:
        raise ValueError("The weight matrix W must have 2 rows.")
    
    # Convert the tensor to a numpy array if it isn't already
    if isinstance(W, torch.Tensor):
        W = W.cpu().detach().numpy()
    
    hull = ConvexHull(W.T)
    vertices = W[:,ConvexHull(W.T).vertices]
    l = len(vertices.T)
    vertex_count=l
    removed=[]
    for i in range(l):
        element = vertices[:,i]
        other = np.delete(vertices, removed + [i], axis=1)
        centroid = np.mean(vertices,axis=1)
        direction = centroid - element
        normalized_direction = direction /np.linalg.norm(direction)
        new_element = element+direction*epsilon

        new_full = np.hstack((other, new_element.reshape(-1,1))).T
        hull = ConvexHull(new_full)
        if len(hull.vertices) < vertex_count:
            removed.append(i)
            vertex_count-=1
    # vertices = W[:,ConvexHull(W.T).vertices]
    # l = len(vertices.T)
    # prev = vertices[:,-1]
    # vertex_count=0

    # for i in range(l):
    #     v = vertices[:,i]
    #     if np.linalg.norm(v-prev) > epsilon:
    #         vertex_count+=1
    #     prev=v
    # return vertex_count

    # return len(hull.vertices)  # The number of vertices is the same as the number of edges

    return vertex_count

def count_kgons(W, epsilon=0.):
    edge_counts = {}
    
    # Process each weight matrix
    for full_w in W:
        num_edges = classify_kgon(full_w, epsilon=epsilon)
        if num_edges in edge_counts:
            edge_counts[num_edges] += 1
        else:
            edge_counts[num_edges] = 1

    return edge_counts

def classify_kgon(W, epsilon=0.):
    embedding_w = W["embedding.weight"]
    edges = calculate_convex_hull_vertices(embedding_w, epsilon)
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

            
TEMPLATE_KGON_PERCENTAGES = "% Frequency of k-gons over training steps\n(sparsity={sparse_value:.3f})"

def plot_kgon_percentages(results, sparsities=[0.426, 0.671, 0.811, 0.892, 0.938, 0.964, 0.98, 0.988, 0.993], epsilon_sparsity=0.001, plot_path="../../results/",
    save_path_tmpl="{plot_path}kgon_frequencies_sparsity_{sparse_value:.3f}_{name}_{epsilon_kgon}.png",
    title_tmpl=TEMPLATE_KGON_PERCENTAGES,
    name="random",
    epsilon_kgon=0.,
):
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
                if abs(result['parameters']['sparsity'] - sparse_value) > epsilon_sparsity:
                    continue
                weights_at_step.append(result['weights'][step_idx])

            # Count kgons
            counts = count_kgons(weights_at_step, epsilon=epsilon_kgon)
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

        title = title_tmpl.format(sparse_value=sparse_value)
        save_path = save_path_tmpl.format(plot_path=plot_path, sparse_value=sparse_value, name=name, epsilon_kgon=epsilon_kgon)

        # Plot
        plt.figure()
        for k, values in plot_data.items():
            plt.plot(PLOT_STEPS, values, label=f'{k}-gon')
        plt.title(title)
        plt.xlabel('Training Step')
        plt.ylabel('Percentage Frequency')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)


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
    # ensure torch, correct dtypes and shapes
    if isinstance(W, np.ndarray): W = torch.from_numpy(W)
    if isinstance(b, np.ndarray): b = torch.from_numpy(b)
    if isinstance(input_vec, np.ndarray): input_vec = torch.from_numpy(input_vec)

    W = W.float()                 # (2, 6)
    b = b.float().view(-1)        # (6,)
    x = input_vec.float().view(-1)  # (6,)

    encoded = W @ x               # (2,)
    decoded = W.t() @ encoded     # (6,)
    return torch.relu(decoded + b)

def create_loss_matrix_simple(results, max_models=200, step=-1):
    """
    Create loss matrix for first max_models models only
    """
    print(f"\n=== Creating loss matrix for first {max_models} models ===")
    
    # Generate all inputs
    all_inputs = generate_all_inputs(6)
    
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
            kgon_type_precise = classify_kgon(final_weights, epsilon=0)
            kgon_type_imprecise = classify_kgon(final_weights, epsilon=.1)
            diff = kgon_type_precise - kgon_type_imprecise

            # Analyze biases
            bias_analysis = analyze_biases(final_weights["unembedding.bias"], epsilon)
            
            # Get sparsity
            sparsity = result['parameters']['sparsity']
            
            # Store classification
            classification = {
                'model_index': i,
                'sparsity': sparsity,
                'kgon_type': kgon_type_imprecise,
                'bias_positive': bias_analysis['positive'],
                'bias_negative': bias_analysis['negative'],
                'bias_zero': bias_analysis['zero'],
                'bias_total': bias_analysis['total'],
                'bias_pattern': f"{bias_analysis['positive']}pos_{bias_analysis['negative']}neg_{bias_analysis['zero']}zero",
                'diff': diff,
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
                'bias_pattern': 'error',
                'diff': 0,
            })
    
    return classifications

def create_annotated_dendrogram(results,indices=None,save_path=f"{plot_path}annotated_dendrogram.svg",
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
    if indices:
        results = [results[i] for i in indices]
    else:
        indices = range(len(results))
    loss_matrix, _, sparsity = create_loss_matrix_simple(results, max_models=len(results))

    Z, _ = create_permutation_invariant_dendrogram(loss_matrix)
    classifications = classify_all_solutions(results, sparsity)
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    def ann(idx):
        cls = classifications[idx]
        return f"M{indices[cls['model_index']]}_{cls['kgon_type']}-gon_S:{cls['sparsity']:.3f}_{cls['bias_pattern']}_diff:{cls['diff']}"
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

    compare_dataframes_and_results(((llc_estimates_random_init, results_random_init),(llc_estimates_optimal_init, results_optimal_init)), ymin=0, plot=True, result_path=plot_path,plot_test=True)

    EPSILON_KGON=0.05
    # plot_kgon_percentages(
    #     results_random_init , title_tmpl=TEMPLATE_KGON_PERCENTAGES+ "with random initialization"
    # )

    plot_kgon_percentages(
        results_random_init , title_tmpl=TEMPLATE_KGON_PERCENTAGES+ "with random initialization", epsilon_kgon=EPSILON_KGON
    )

    # plot_kgon_percentages(
    #     results_optimal_init, title_tmpl=TEMPLATE_KGON_PERCENTAGES + " with optimal initialization",name="optimal"
    # )
    plot_kgon_percentages(
        results_optimal_init, title_tmpl=TEMPLATE_KGON_PERCENTAGES + " with optimal initialization",name="optimal", epsilon_kgon=EPSILON_KGON
    )

    loss_matrices=[]
    sparsities=[]
    small_results = results_random_init[:10]

    create_annotated_dendrogram(results_random_init,indices = [0, 10, 42, 1000, 1500, 1999, -1], save_path=f"{plot_path}annotated_dendrogram_small.svg")

    for i in range(10):
        create_annotated_dendrogram(results_random_init, range(i*200, (i+1)*200),save_path=f"{plot_path}annotated_dendrogram_{i}.svg")


def autoencoder_forward_simple(
    input_vec: Float[Tensor, "... d"],      # (d,)
    W: Float[Tensor, "en d"],          # (in, d)
    b: Float[Tensor, "d"],
    ) -> Float[Tensor, "d"]:
    encoded = einsum(W,input_vec,"en d, ... d -> ... en")
    decoded = einsum(W,encoded, "en d, ... en -> ... d")
    out = decoded + b
    return torch.relu(out)

def decoder(encoded: Float[Tensor, "... en"], W: Float[Tensor, "en d"], b: Float[Tensor, "d"])-> Float[Tensor, "d"]:
    decoded = einsum(W,encoded, "en d, ... en -> ... d")
    out = decoded + b
    return torch.relu(out)

def generate_all_inputs(n:int):
    all_inputs = []

    for i in range(2**n):
        binary = format(i, f'0{n}b')
        input_vec = torch.tensor([float(int(b)) for b in binary])
        all_inputs.append(input_vec)
    print(all_inputs)
    return torch.stack(all_inputs,dim=0)

def with_interactive_plots(func):
    """
    This decorator makes sure that we do not block in case we create multiple
    visualizations inside a function and at the same time it should block at the
    end of the function in order for the plots to not vanish instantly
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # enable interactive mode
        plt.ion()
        try:
            return func(*args, **kwargs)
        finally:
            # turn interactive off again
            plt.ioff()
            plt.show()   # block at the end so windows stay open
    return wrapper

# @with_interactive_plots
def model_geometry():
    data_path = "../../data"
    model_plot_path = f"{plot_path}model-geometry/"

    version = "1.13.0"

    results_1_13= load_results(data_path, version)
    llc_estimates_1_13 = get_or_create_preaggregated_llc_csv(results_1_13, version, data_path)


    results_random_init=results_1_13
    llc_estimates_random_init=llc_estimates_1_13


    m=6
    n = 2
    index = 0

    def visualize_all(W,b):
        # v = torch.eye(n,n)
        # v = torch.zeros((1,n))
        # einsum(W,W,"en d -> en d")
        u = generate_all_inputs(2)

        # w = autoencoder_forward_simple(v,W,b)
        # print(w)
        # U, S, Vh = torch.linalg.svd(W, full_matrices=False)  # U:(en,r), S:(r,), Vh:(r,d)
        # print(f"U:\n{U}")
        def ellipse_axes(W: torch.Tensor):
            # W: (2, d)
            M = W @ W.T              # 2×2
            evals, evecs = torch.linalg.eigh(M)  # evals: λ_i = σ_i^2
            axes = torch.stack([torch.sqrt(val) * vec for val, vec in zip(evals, evecs.T)])
            return axes  # list of 2 vectors in R^2

        # v = W @ W.T
        el = ellipse_axes(W)

        print(f"v:\n{el}")
        v = einsum(el,u,"an en, ... en -> ... an")

        w = decoder(v, W, b)

        print(w.size())
        plt.imshow(w.T, aspect="auto", cmap="viridis")  # or "gray", etc.
        plt.colorbar()  # optional, adds color scale
        plt.show()
        # plt.pause(0.1)       # let the GUI event loop breathe
        print(f"v:\n{v}")

    def filter_rows_eq(v: torch.Tensor, conds: list[tuple[int, int | float]]) -> torch.Tensor:
        """
        Keep rows of v where v[:, idx] == val for every (idx, val) in conds.
        v: (N, D) tensor
        conds: [(idx, val), ...]
        """
        if not conds:
            return v  # nothing to filter

        masks = [(v[:, idx] == val) for idx, val in conds]
        combined = functools.reduce(operator.and_, masks)
        return v[combined]


    def comp(index=0, conds=[]):
        W,b = get_weights(results_random_init, index)
        v = generate_all_inputs(6)

        v = filter_rows_eq(v, conds=conds)

        w = autoencoder_forward_simple(v, W, b)

        new = rearrange(torch.stack((v,w)), "two solutions n -> solutions two n")

        print(new)

    def comp_diff(index, intervention_index, conds=[]):
        W,b = get_weights(results_random_init, index)
        v = generate_all_inputs(6)

        v = filter_rows_eq(v, conds=conds)
        mask = (v[:,intervention_index] == 1)
        mask2 = (v[:,intervention_index] == 0)
        def automap(v):
            return autoencoder_forward_simple(v, W, b)

        w = automap(v[mask]) - automap(v[mask2])

        print(reduce(w,"solutions d-> d","mean"))
        plt.imshow(w.T, aspect="auto", cmap="viridis")  # or "gray", etc.
        plt.colorbar()  # optional, adds color scale
        plt.show()

    def visualize_v(W,b):

        v = generate_all_inputs(6)
        print(f"v:\n{v}")
        # condition: keep rows where the last entry == 0

        def automap(v):
        # v = v[:10]
            return autoencoder_forward_simple(v, W, b)
        # print(f"v:\n{v}")
        # for i in range(6):
        #     print(i)
        #     mask = (v[:, i] == 0)     # tensor([False, True])
        #     mask2 = (v[:, i] == 1)     # tensor([False, True])
        #     w = automap(v[mask]) - automap(v[mask2])

        #     plt.imshow(w.T, aspect="auto", cmap="viridis")  # or "gray", etc.
        #     plt.colorbar()  # optional, adds color scale
        #     plt.show()

        mask = (v[:, 2] == 1)     # tensor([False, True])
        mask2 = (v[:, 4] == 1)     # tensor([False, True])
        mask3 = (v[:, 2] == 0)     # tensor([False, True])
        mask4 = (v[:, 4] == 0)     # tensor([False, True])

        w = automap(v[mask & mask2]) - automap(v[mask3 & mask4])

        plt.imshow(w.T, aspect="auto", cmap="viridis")  # or "gray", etc.
        plt.colorbar()  # optional, adds color scale
        plt.show()


    # visualize_v(W,b)
    # comp(W,b,0, [(1,0),(3,0),(5,0)])
    index = 1
    # comp(1, [])
    # comp_diff(1,4)





    # plt.close()
    # print(f"w:\n{w}")

    # for i in range(n):


    plot_results(results_random_init, loss_window=(0.0,0.16), sparsities=[0.426], plot_number=100)

    # plot_specific_index(results_random_init, index)




def main():
    data_path = "../../data"
    # version = "1.8.0"

    # results_1_8= load_results(data_path, version)
    # llc_estimates_1_8 = get_or_create_preaggregated_llc_csv(results_1_8, version, data_path)

    # version = "1.7.0"

    # results_1_7= load_results(data_path, version)
    # llc_estimates_1_7 = get_or_create_preaggregated_llc_csv(results_1_7, version, data_path)

    # version = "1.11.0"

    # results_1_11= load_results(data_path, version)
    # llc_estimates_1_11 = get_or_create_preaggregated_llc_csv(results_1_11, version, data_path)

    # version = "1.12.0"

    # results_1_12= load_results(data_path, version)
    # llc_estimates_1_12 = get_or_create_preaggregated_llc_csv(results_1_12, version, data_path)

    version = "1.13.0"

    results_1_13= load_results(data_path, version)
    llc_estimates_1_13 = get_or_create_preaggregated_llc_csv(results_1_13, version, data_path)

    version = "1.14.0"

    results_1_14= load_results(data_path, version)
    llc_estimates_1_14 = get_or_create_preaggregated_llc_csv(results_1_14, version, data_path)

    results_random_init=results_1_13
    llc_estimates_random_init=llc_estimates_1_13
    results_optimal_init=results_1_14
    llc_estimates_optimal_init=llc_estimates_1_14



    # version = "1.15.0"

    # results_1_15= load_results(data_path, version)
    # llc_estimates_1_15 = get_or_create_preaggregated_llc_csv(results_1_15, version, data_path)

    # for index in range(200,1000):
    # # for index in [0]:
    #     plot_specific_index(results_random_init, index)

    #TODO: check results from get_weights
    plot_everything(results_random_init=results_1_13, llc_estimates_random_init=llc_estimates_1_13, results_optimal_init=results_1_14, llc_estimates_optimal_init=llc_estimates_1_14)



def perfect_solution():
    #Notice: if we reparameterize like this, then finding the solution is probably faster?
    #Could meta-learning look like finding the right parameterization?
    m = 6
    n = 2
    l = 0.6 #
    b = .65 #

    w = torch.from_numpy(generate_2d_kgon_vertices(m, rot=0., force_length=l, pad_to=m)).float()
    bias = torch.ones((m)) * b
    sparse_value = 0.426

    # model = ToyAutoencoder(6, 2, final_bias=True)
    # test_set = SyntheticBinarySparseValued(test_set_size, 6, sparse_value)
    # test_X = test_X.to(device)   # shape (N, 6)

    mse = compute_loss(w,bias)

    print(f"mse: {mse}")

    # mean_loss_test = 0
    # for sample in test_set:
    #     output = model(sample)
    #     mean_loss_test += criterion(output, sample)
    # # print("Mean loss test:")
    # loss_hist.append(mean_loss_test)
    # print(f"index: {index}")
    # print(mean_loss_test/test_set_size)

    # print(f'index: {index}')
    # plot_polygon(w, bias)
    # plt.show()


def evaluate_mse(l, b, test_X, m=6):
    # weight matrix from polygon parameterisation
    w = torch.from_numpy(
        generate_2d_kgon_vertices(m, rot=0., force_length=l, pad_to=m)
    ).float()

    bias = torch.ones((m,)) * b

    encoded = test_X @ w.T        # (N, 2)
    decoded = encoded @ w         # (N, 6)
    out = decoded + bias          # (N, 6)
    mse = torch.mean((out - test_X).pow(2))
    return mse.item()

def grid_search(test_set_size=1000, sparse_value=0.426, m=6):
    # create one fixed test set
    test_X = torch.stack([
        x for x in SyntheticBinarySparseValued(test_set_size, m, sparse_value)
    ]).float()

    # define grid ranges
    l_values = np.linspace(0.5, 0.7, 300)     # adjust ranges and resolution
    b_values = np.linspace(0.4, 0.7, 300)

    best_mse = float("inf")
    best_params = None

    for l in l_values:
        for b in b_values:
            mse = evaluate_mse(l, b, test_X, m=m)
            if mse < best_mse:
                best_mse = mse
                best_params = (l, b)

    print(f"Best parameters: l={best_params[0]:.4f}, b={best_params[1]:.4f}, MSE={best_mse:.6f}")
    return best_params, best_mse

if __name__ == "__main__":
    best_params, best_mse = grid_search()


# perfect_solution()
# model_geometry()

main()
# calculate_convex_hull_vertices(torch.Tensor(
# [[-1.8623e+00, -1.1313e+00,  8.4201e-01,  6.8771e-03, -1.5209e-02,
#          -1.2631e+00],
#         [-1.0020e+00, -2.1917e+00, -1.5675e+00, -5.0797e-04,  9.5378e-03,
#          -6.7840e-01]]))
