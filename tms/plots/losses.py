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
from tms.data.dataset import SyntheticBinaryValued, SyntheticBinarySparseValued
from tms.plots.kgons import plot_losses_and_polygons
from tms.utils.utils import iterate_container, get_first
import pandas as pd


def compute_test_loss(W,b, sparsity, test_set_size = 10000):
    test_X = torch.stack([x for x in SyntheticBinarySparseValued(test_set_size, 6, sparsity)]).float()
    if test_X is None:
        raise ValueError("test_X must be provided")
    encoded = test_X @ W.T          # (N, 2)
    decoded = encoded @ W      # (N, 6)
    out = torch.relu(decoded + b)       # (N, 6)  (bias broadcasts)
    return torch.mean((out-test_X).pow(2))   # scalar mean MSE over all samples and dims

def plot_results_by_indices(results, indices):
    """
    Plot the results of the experiment.

    Parameters
    ----------
    results : list
        A list of dictionaries containing the experiment results.
    plot_number : list[int]
        The indices of the results to plot.
    """
    
    for index in indices:
        sparse_value = results[index]['parameters']['sparsity']
        STEPS = results[index]['parameters']['log_ivl']
        logs = results[index]['logs']

        losses = [logs.loc[logs['step'] == s, 'loss'].values[0] for s in STEPS]

        NUM_EPOCHS = results[index]['parameters']['num_epochs']
        PLOT_STEPS = [min(STEPS, key=lambda s: abs(s-i)) for i in [0, 200, 2000, 10000, NUM_EPOCHS - 1]]
        PLOT_INDICES = [STEPS.index(s) for s in PLOT_STEPS]
        Ws = [results[index]['weights'][i]['embedding.weight'] for i in PLOT_INDICES]
        biases = [results[index]['weights'][i]['unembedding.bias'] for i in PLOT_INDICES]
        model = ToyAutoencoder(6, 2, final_bias=True)
        new_weights = {}
        for idx, ndarray in results[index]['weights'][PLOT_INDICES[-1]].items():
            new_weights[idx] = torch.from_numpy(ndarray)

        criterion = nn.MSELoss()
    
        model.load_state_dict(new_weights)

        test_set = SyntheticBinaryValued(10000, 6, sparse_value)
        mean_loss_test = 0
        for sample in test_set:
            output = model(sample)
            mean_loss_test += criterion(output, sample)
        # print("Mean loss test:")
        print(f"index: {index}")
        print(mean_loss_test/10000)
    
        model.load_state_dict(new_weights)
        plot_losses_and_polygons(STEPS, losses, PLOT_STEPS, Ws, biases)
        plt.show()
            

def plot_results(results: List[Dict[str, Any]] | Dict[int,Dict[str,Any]], plot_number=5):
    """
    Plot the results of the experiment.

    Parameters
    ----------
    results : list
        A list of dictionaries containing the experiment results.
    plot_number : int, optional
        The maximum number of plots to display, by default 5.
    """

    sparsity = [result['parameters']['sparsity'] for result in iterate_container(results)]
    for sparse_value in sparsity:
        plotted = 0
        print(f"Plot polygons for sparsity={sparse_value}")
        for index in range(len(results)):
            
            STEPS = results[index]['parameters']['log_ivl']
            if results[index]['parameters']['sparsity'] != sparse_value:
                continue
            else:
                if plotted >= plot_number:
                    continue
                plotted += 1
            logs = results[index]['logs']

            losses = [logs.loc[logs['step'] == s, 'loss'].values[0] for s in STEPS]

            NUM_EPOCHS = results[index]['parameters']['num_epochs']
            PLOT_STEPS = [min(STEPS, key=lambda s: abs(s-i)) for i in [0, 200, 2000, 10000, NUM_EPOCHS - 1]]
            PLOT_INDICES = [STEPS.index(s) for s in PLOT_STEPS]
            Ws = [results[index]['weights'][i]['embedding.weight'] for i in PLOT_INDICES]
            biases = [results[index]['weights'][i]['unembedding.bias'] for i in PLOT_INDICES]
            model = ToyAutoencoder(6, 2, final_bias=True)
            new_weights = {}
            for idx, ndarray in results[index]['weights'][PLOT_INDICES[-1]].items():
                new_weights[idx] = torch.from_numpy(ndarray)

            criterion = nn.MSELoss()
        
            model.load_state_dict(new_weights)

            test_set = SyntheticBinaryValued(10000, 6, sparse_value)
            mean_loss_test = 0
            for sample in test_set:
                output = model(sample)
                mean_loss_test += criterion(output, sample)
            # print("Mean loss test:")
            print(f"index: {index}")
            print(mean_loss_test/10000)

            plot_losses_and_polygons(STEPS, losses, PLOT_STEPS, Ws, biases)
            plt.show()
            
    


def collect_global_sparsities(df_results_pairs):
    sparsities = set()
    for _, results in df_results_pairs:
        for result in iterate_container(results):
            sparsity = results[result["run_id"]]['parameters']['sparsity']
            if sparsity != 0:
                sparsities.add(sparsity)
    return sorted(sparsities)


def create_color_mapping(sparsities):
    cmap = plt.get_cmap("tab10")  
    n_colors = cmap.N 

    return {s: cmap(i % n_colors) for i, s in enumerate(sorted(sparsities))}


Results = Dict[int, Any] | List[Any]
DfResultPair =  Tuple[pd.DataFrame, Results]

def plot_for_position(position, df_results_pairs: Tuple[DfResultPair, DfResultPair], batch_size, learning_rate, sparsity_to_color, x_scale, y_scale, sharex, sharey, ymin, test_loss=True,test_set_size = 10000):
    test_X = {}

    sparsities = collect_global_sparsities(df_results_pairs)
    for sparsity in sparsities:
        test_X[sparsity] = torch.stack([x for x in SyntheticBinarySparseValued(test_set_size, 6, sparsity)]).float()

    fig, axes = plt.subplots(1, len(df_results_pairs), figsize=(15*len(df_results_pairs), 10), sharey=sharey, sharex=sharex)
    if len(df_results_pairs) == 1:
        axes = [axes]

    for pair_index, (llc_estimates, results) in enumerate(df_results_pairs):
        llc_loss_by_sparsity = defaultdict(list)
        steps = get_first(results)['parameters']['log_ivl']

        # llc_estimates_dict = llc_estimates.to_dict()

        llc_estimates_dict = {
            (row['index'], row['batch_size'], row['lr'], row['snapshot_index']): row['llc']
            for _, row in llc_estimates.iterrows()
        }
        for result in iterate_container(results):
            index = result["run_id"]
            sparsity = results[index]['parameters']['sparsity']
            if sparsity == 0:
                continue
            llc = llc_estimates_dict.get((index, batch_size, learning_rate, position), np.nan)
            # print llc indices:
            # print(index, batch_size, learning_rate, position)
            if test_loss:
                weights = results[index]['weights'][position]
                W = weights['embedding.weight']
                b = weights['unembedding.bias']
                loss = compute_test_loss(W,b, sparsity)
            else:
                loss = results[index]['logs']['loss'].values[position]
            llc_loss_by_sparsity[sparsity].append((llc, loss))
            # print("Sparsity:", sparsity)
            # print("loss:", loss)
            # print("llc:", llc)

        for sparsity, llc_loss in llc_loss_by_sparsity.items():


            arr = np.asarray(llc_loss)
            mask = ~np.isnan(arr[:, 0])
            if not mask.any():
                continue
            llcs, losses = arr[mask].T
            color = sparsity_to_color.get(sparsity, 'gray')
            axes[pair_index].scatter(llcs, losses, label=f"Sparsity: {round(sparsity, 3)}", color=color)

        if pair_index == 0:
            title = "Initialized at random 4-gon"
        if pair_index == 1:
            title = "Initialized at optimal parameters for sparse inputs"
        axes[pair_index].set_title(f"Pair {title}, Position {position}", fontsize=24)
        axes[pair_index].set_xlabel("LLC")
        axes[pair_index].set_ylabel("Loss")
        axes[pair_index].legend(fontsize=20)
        axes[pair_index].set_xscale(x_scale)
        axes[pair_index].set_yscale(y_scale)
        axes[pair_index].set_ylim(ymin=ymin)

    plt.tight_layout()
    plt.suptitle(f"Loss and LLC After Epoch {steps[position]}",#, fontsize=16
                 fontsize=30,
                 )
    plt.subplots_adjust(top=0.9)

    return fig, steps[position]


def compare_dataframes_and_results(
    df_results_pairs: Tuple[DfResultPair, DfResultPair],
    positions=[9, 18, 27, 36, 45],
    hyperparam_combos=[(300, 0.001)],
    x_scale="linear",
    y_scale="linear",
    sharey=False,
    sharex=False,
    ymin=1e-4,
    result_path='../results',
    plot:bool=True,
    plot_test:bool=False,
):
    warnings.simplefilter(action='ignore', category=UserWarning)

    # Create global sparsity-color mapping
    unique_sparsities = collect_global_sparsities(df_results_pairs)
    sparsity_to_color = create_color_mapping(unique_sparsities)

    for batch_size, learning_rate in hyperparam_combos:
        print(f"Batch size: {batch_size}, Learning rate: {learning_rate}\n")

        # Preaggregate all pairs
        # preaggs = [preaggregate_llc(est) for est, _ in df_results_pairs]
        plot_test_param = [False]
        if plot_test:
            plot_test_param = [True, False]

        positions_test = [(position, test_loss) for test_loss in plot_test_param for position in positions]
        for position,test_loss in positions_test:

            fig, step = plot_for_position(
                position,
                df_results_pairs,
                batch_size,
                learning_rate,
                sparsity_to_color,
                x_scale,
                y_scale,
                sharex,
                sharey,
                ymin,
                test_loss=test_loss,
            )

            param_string = f"bs{batch_size}_lr{learning_rate}_pos{position}_epoch{step}"
            if x_scale != "linear" or y_scale != "linear":
                param_string += f"_x{x_scale}_y{y_scale}"
            if ymin != 1e-4:
                param_string += f"_ymin{ymin}"
            if test_loss:
                param_string+= "_test"
            else:
                param_string+= "_train"

            save_path = f'{result_path}/loss_vs_llc_{param_string}'
            fig.savefig(f'{save_path}.svg', bbox_inches='tight', format='svg')
            fig.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', format='png')
            if plot:
                plt.show()
