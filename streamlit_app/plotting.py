"""Plotting utilities for Streamlit interactive visualization."""

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from typing import Optional, List, Dict
import sys
import os

# Add parent directory to path to import tms modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tms.plots.kgons import plot_losses_and_polygons
from tms.plots.losses import compute_test_loss
from tms.data.dataset import SyntheticBinarySparseValued


def create_scatter_plot(
    data: pd.DataFrame,
    title: str,
    selected_index: Optional[int] = None,
    height: int = 500
) -> go.Figure:
    """
    Create interactive Plotly scatter plot for Loss vs LLC.
    
    Args:
        data: DataFrame with columns [model_index, llc, loss, sparsity]
        title: Plot title
        selected_index: Model index to highlight (optional)
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    if data.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=title,
            height=height,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        return fig
    
    # Get unique sparsities and create color mapping
    unique_sparsities = sorted(data['sparsity'].unique())
    colors = px.colors.qualitative.Plotly
    color_map = {s: colors[i % len(colors)] for i, s in enumerate(unique_sparsities)}
    
    fig = go.Figure()
    
    # Add scatter points grouped by sparsity
    for sparsity in unique_sparsities:
        mask = data['sparsity'] == sparsity
        subset = data[mask].copy()
        
        # Check if any points in this sparsity group are selected
        if selected_index is not None:
            is_selected = (subset['model_index'] == selected_index).any()
        else:
            is_selected = False
        
        # Highlight selected point
        if is_selected:
            # Non-selected points in this group
            non_selected = subset[subset['model_index'] != selected_index]
            selected = subset[subset['model_index'] == selected_index]
            
            # Add non-selected points
            if not non_selected.empty:
                fig.add_trace(go.Scatter(
                    x=non_selected['llc'],
                    y=non_selected['loss'],
                    mode='markers',
                    name=f'Sparsity {sparsity:.3f}',
                    marker=dict(
                        size=6,
                        color=color_map[sparsity],
                        opacity=0.6
                    ),
                    customdata=non_selected[['model_index', 'sparsity']],
                    hovertemplate=(
                        '<b>Model %{customdata[0]}</b><br>' +
                        'LLC: %{x:.4f}<br>' +
                        'Loss: %{y:.6f}<br>' +
                        'Sparsity: %{customdata[1]:.3f}<br>' +
                        '<extra></extra>'
                    ),
                    showlegend=True,
                    legendgroup=f'sparsity_{sparsity}'
                ))
            
            # Add selected point
            if not selected.empty:
                fig.add_trace(go.Scatter(
                    x=selected['llc'],
                    y=selected['loss'],
                    mode='markers',
                    name=f'Sparsity {sparsity:.3f} (selected)',
                    marker=dict(
                        size=12,
                        color=color_map[sparsity],
                        line=dict(width=2, color='black')
                    ),
                    customdata=selected[['model_index', 'sparsity']],
                    hovertemplate=(
                        '<b>SELECTED - Model %{customdata[0]}</b><br>' +
                        'LLC: %{x:.4f}<br>' +
                        'Loss: %{y:.6f}<br>' +
                        'Sparsity: %{customdata[1]:.3f}<br>' +
                        '<extra></extra>'
                    ),
                    showlegend=False,
                    legendgroup=f'sparsity_{sparsity}'
                ))
        else:
            # Normal display without selection
            fig.add_trace(go.Scatter(
                x=subset['llc'],
                y=subset['loss'],
                mode='markers',
                name=f'Sparsity {sparsity:.3f}',
                marker=dict(
                    size=6,
                    color=color_map[sparsity],
                    opacity=0.7
                ),
                customdata=subset[['model_index', 'sparsity']],
                hovertemplate=(
                    '<b>Model %{customdata[0]}</b><br>' +
                    'LLC: %{x:.4f}<br>' +
                    'Loss: %{y:.6f}<br>' +
                    'Sparsity: %{customdata[1]:.3f}<br>' +
                    '<extra></extra>'
                ),
                showlegend=True
            ))
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title='LLC (Learning Coefficient)',
        yaxis_title='Loss',
        hovermode='closest',
        height=height,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(l=60, r=20, t=60, b=60)
    )
    
    return fig


def plot_model_details(model_data: Dict, test_set_size: int = 10000) -> plt.Figure:
    """
    Create detailed plot for a specific model using existing TMS plotting functions.
    
    This reuses the plot_losses_and_polygons function from tms.plots.kgons
    to maintain consistency with existing visualizations.
    
    Args:
        model_data: Dictionary containing model data (from get_model_data)
        test_set_size: Number of test samples for loss computation
        
    Returns:
        Matplotlib Figure object
    """
    index = model_data['run_id']
    sparsity = model_data['sparsity']
    STEPS = model_data['log_ivl']
    logs = model_data['logs']
    
    # Extract losses from logs
    losses = [logs.loc[logs['step'] == s, 'loss'].values[0] for s in STEPS]
    
    # Select specific epochs to display polygon snapshots
    NUM_EPOCHS = model_data['parameters'].get('num_epochs', 20000)
    PLOT_STEPS = [min(STEPS, key=lambda s: abs(s-i)) 
                  for i in [0, 200, 2000, 10000, NUM_EPOCHS - 1]]
    PLOT_INDICES = [STEPS.index(s) for s in PLOT_STEPS]
    
    # Get weights and biases at those steps
    Ws = [model_data['weights'][i]['embedding.weight'] for i in PLOT_INDICES]
    biases = [model_data['weights'][i]['unembedding.bias'] for i in PLOT_INDICES]
    
    # Compute test losses for all logged steps
    test_X = torch.stack([
        x for x in SyntheticBinarySparseValued(test_set_size, 6, sparsity)
    ]).float()
    
    test_losses = []
    for i, s in enumerate(STEPS):
        W = model_data['weights'][i]['embedding.weight']
        b = model_data['weights'][i]['unembedding.bias']
        loss = compute_test_loss(W, b, test_X)
        test_losses.append((s, loss))
    
    # Create the plot using existing function
    plot_losses_and_polygons(
        STEPS, losses, PLOT_STEPS, Ws, biases,
        run=index, test_losses=test_losses, sparsity=sparsity
    )
    
    # Return the current figure
    return plt.gcf()


def create_summary_stats_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics table for the current view.
    
    Args:
        data: DataFrame with scatter plot data
        
    Returns:
        DataFrame with summary statistics
    """
    if data.empty:
        return pd.DataFrame()
    
    # Group by sparsity
    summary = data.groupby('sparsity').agg({
        'llc': ['mean', 'std', 'min', 'max'],
        'loss': ['mean', 'std', 'min', 'max'],
        'model_index': 'count'
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={'model_index_count': 'num_models'})
    
    return summary.reset_index()


def get_color_for_sparsity(sparsity: float, all_sparsities: List[float]) -> str:
    """
    Get consistent color for a sparsity value.
    
    Args:
        sparsity: Sparsity value
        all_sparsities: List of all unique sparsity values
        
    Returns:
        Color string (hex)
    """
    colors = px.colors.qualitative.Plotly
    try:
        idx = sorted(all_sparsities).index(sparsity)
        return colors[idx % len(colors)]
    except ValueError:
        return colors[0]
