"""Data loading and caching utilities for Streamlit app."""

import streamlit as st
import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, List, Optional
import sys
import os

# Add parent directory to path to import tms modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tms.utils.utils import load_results
from tms.plots.losses import compute_test_loss
from tms.data.dataset import SyntheticBinarySparseValued


@st.cache_resource
def load_all_results(data_path: str = "data"):
    """
    Load both random and optimal initialization results.
    
    This is cached as a resource since the results objects are large
    and should persist across reruns.
    
    Args:
        data_path: Path to data directory
        
    Returns:
        Tuple of (results_random, results_optimal)
    """
    try:
        results_random = load_results(data_path, "1.15.0")
        results_optimal = load_results(data_path, "1.14.0")
        return results_random, results_optimal
    except Exception as e:
        st.error(f"Error loading results: {e}")
        raise


@st.cache_data
def load_llc_estimates(data_path: str = "data"):
    """
    Load pre-aggregated LLC estimates from CSV files.
    
    Args:
        data_path: Path to data directory
        
    Returns:
        Tuple of (llc_random, llc_optimal) DataFrames
    """
    try:
        llc_random = pd.read_csv(f"{data_path}/llc_preagg_1.15.0.csv")
        llc_optimal = pd.read_csv(f"{data_path}/llc_preagg_1.14.0.csv")
        return llc_random, llc_optimal
    except Exception as e:
        st.error(f"Error loading LLC estimates: {e}")
        raise


@st.cache_data
def get_position_to_epoch_mapping(_results) -> Dict[int, int]:
    """
    Get mapping from snapshot position index to actual epoch number.
    
    Args:
        _results: Results list (underscore prefix prevents hashing)
        
    Returns:
        Dictionary mapping position -> epoch number
    """
    # Use first result as reference (all have same log_ivl)
    log_ivl = _results[0]['parameters']['log_ivl']
    
    # Standard positions used in LLC estimation
    # These are the indices we have LLC data for
    positions = [0, 9, 18, 27, 36, 45]
    
    return {pos: log_ivl[pos] for pos in positions if pos < len(log_ivl)}


@st.cache_data
def get_unique_sparsities(_results) -> List[float]:
    """
    Extract all unique sparsity values from results.
    
    Args:
        _results: Results list
        
    Returns:
        Sorted list of unique sparsity values
    """
    sparsities = set(r['parameters']['sparsity'] for r in _results)
    return sorted(sparsities)


@st.cache_data
def prepare_test_sets(_results, test_set_size: int = 10000) -> Dict[float, torch.Tensor]:
    """
    Pre-generate test sets for all unique sparsity values.
    
    This is cached to avoid regenerating test data on every interaction.
    
    Args:
        _results: Results list
        test_set_size: Number of test samples
        
    Returns:
        Dictionary mapping sparsity -> test tensor
    """
    unique_sparsities = get_unique_sparsities(_results)
    test_sets = {}
    
    for sparsity in unique_sparsities:
        test_sets[sparsity] = torch.stack([
            x for x in SyntheticBinarySparseValued(test_set_size, 6, sparsity)
        ]).float()
    
    return test_sets


@st.cache_data
def prepare_scatter_data(
    _results,
    _llc_df: pd.DataFrame,
    position: int,
    batch_size: int = 300,
    lr: float = 0.001,
    use_test_loss: bool = True,
    sparsity_filter: Optional[List[float]] = None,
    dataset_id: str = "default"
) -> pd.DataFrame:
    """
    Prepare scatter plot data for a specific epoch position.
    
    Args:
        _results: Results list
        _llc_df: LLC estimates DataFrame
        position: Snapshot position index
        batch_size: Batch size used in training
        lr: Learning rate used in training
        use_test_loss: If True, compute test loss; otherwise use train loss
        sparsity_filter: Optional list of sparsities to include
        dataset_id: Identifier for dataset (used for cache key differentiation)
        
    Returns:
        DataFrame with columns: model_index, llc, loss, sparsity
    """
    # Create LLC lookup dictionary for fast access
    llc_dict = {
        (row['index'], row['batch_size'], row['lr'], row['snapshot_index']): row['llc']
        for _, row in _llc_df.iterrows()
    }
    
    # Pre-generate test sets if needed
    test_sets = None
    if use_test_loss:
        test_sets = prepare_test_sets(_results)
    
    # Collect data for each model
    data = []
    for idx, result in enumerate(_results):
        sparsity = result['parameters']['sparsity']
        
        # Apply sparsity filter if provided
        if sparsity_filter is not None and sparsity not in sparsity_filter:
            continue
        
        # Get LLC value
        llc = llc_dict.get((idx, batch_size, lr, position), np.nan)
        if np.isnan(llc):
            continue
        
        # Skip if position is out of range
        if position >= len(result['weights']):
            continue
            
        # Get loss value
        try:
            if use_test_loss:
                W = result['weights'][position]['embedding.weight']
                b = result['weights'][position]['unembedding.bias']
                loss = compute_test_loss(W, b, test_sets[sparsity])
            else:
                # Get training loss at this position
                log_ivl = result['parameters']['log_ivl']
                if position < len(log_ivl):
                    step = log_ivl[position]
                    logs = result['logs']
                    loss_values = logs.loc[logs['step'] == step, 'loss'].values
                    if len(loss_values) > 0:
                        loss = loss_values[0]
                    else:
                        continue
                else:
                    continue
        except Exception as e:
            # Skip this model if there's an error computing loss
            continue
        
        data.append({
            'model_index': idx,
            'llc': llc,
            'loss': loss,
            'sparsity': sparsity
        })
    
    return pd.DataFrame(data)


@st.cache_data
def get_model_data(_results, model_index: int) -> Optional[Dict]:
    """
    Get all data needed to plot a specific model.
    
    Args:
        _results: Results list
        model_index: Index of the model to retrieve
        
    Returns:
        Dictionary with model data or None if index is invalid
    """
    if model_index < 0 or model_index >= len(_results):
        return None
    
    result = _results[model_index]
    
    return {
        'run_id': model_index,
        'sparsity': result['parameters']['sparsity'],
        'log_ivl': result['parameters']['log_ivl'],
        'logs': result['logs'],
        'weights': result['weights'],
        'parameters': result['parameters']
    }


def get_data_summary(_results, _llc_df: pd.DataFrame) -> Dict:
    """
    Get summary statistics about the loaded data.
    
    Args:
        _results: Results list
        _llc_df: LLC estimates DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    num_models = len(_results)
    unique_sparsities = get_unique_sparsities(_results)
    
    # Get position info
    position_to_epoch = get_position_to_epoch_mapping(_results)
    
    return {
        'num_models': num_models,
        'num_sparsities': len(unique_sparsities),
        'sparsity_range': (min(unique_sparsities), max(unique_sparsities)),
        'num_positions': len(position_to_epoch),
        'epoch_range': (min(position_to_epoch.values()), max(position_to_epoch.values())),
        'num_llc_estimates': len(_llc_df)
    }
