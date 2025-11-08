"""
TMS Sparsity Interactive Visualization - Streamlit App

This application provides an interactive interface to explore the relationship
between Learning Coefficient (LLC) and Loss across different training epochs
for the TMS Sparsity experiments.
"""

import streamlit as st
import sys
import os

# Add parent directory to path to import tms modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from streamlit_app.data_loader import (
    load_all_results,
    load_llc_estimates,
    get_position_to_epoch_mapping,
    prepare_scatter_data,
    get_model_data,
    get_unique_sparsities,
    get_data_summary
)
from streamlit_app.plotting import (
    create_scatter_plot,
    plot_model_details,
    create_summary_stats_table
)

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="TMS Sparsity Interactive Visualization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Session State Initialization
# =============================================================================

if 'selected_model_random' not in st.session_state:
    st.session_state.selected_model_random = None
if 'selected_model_optimal' not in st.session_state:
    st.session_state.selected_model_optimal = None
if 'selected_init_type' not in st.session_state:
    st.session_state.selected_init_type = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# =============================================================================
# Header
# =============================================================================

st.title("🔬 TMS Sparsity: Interactive Loss vs LLC Visualization")
st.markdown(
    "Explore the relationship between **Learning Coefficient (LLC)** and **Loss** "
    "across different training epochs and initialization strategies."
)

# =============================================================================
# Data Loading
# =============================================================================

data_path = "data"

# Check if data directory exists
if not os.path.exists(data_path):
    st.error(
        f"❌ Data directory '{data_path}' not found. "
        "Please ensure you are running the app from the project root directory."
    )
    st.stop()

# Load data with progress indicator
if not st.session_state.data_loaded:
    with st.spinner("🔄 Loading experiment results... This may take a moment."):
        try:
            results_random, results_optimal = load_all_results(data_path)
            llc_random, llc_optimal = load_llc_estimates(data_path)
            
            # Store in session state
            st.session_state.results_random = results_random
            st.session_state.results_optimal = results_optimal
            st.session_state.llc_random = llc_random
            st.session_state.llc_optimal = llc_optimal
            
            st.session_state.data_loaded = True
            st.success("✅ Data loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            st.exception(e)
            st.stop()
else:
    # Retrieve from session state
    results_random = st.session_state.results_random
    results_optimal = st.session_state.results_optimal
    llc_random = st.session_state.llc_random
    llc_optimal = st.session_state.llc_optimal

# Get metadata
position_to_epoch = get_position_to_epoch_mapping(results_random)
all_sparsities_random = get_unique_sparsities(results_random)
all_sparsities_optimal = get_unique_sparsities(results_optimal)

# =============================================================================
# Sidebar Controls
# =============================================================================

st.sidebar.header("⚙️ Controls")

# Display data summary in expandable section
with st.sidebar.expander("📊 Data Summary", expanded=False):
    summary_random = get_data_summary(results_random, llc_random)
    summary_optimal = get_data_summary(results_optimal, llc_optimal)
    
    st.markdown("**Random Initialization (v1.15.0)**")
    st.write(f"- Models: {summary_random['num_models']}")
    st.write(f"- Sparsities: {summary_random['num_sparsities']}")
    st.write(f"- Sparsity range: {summary_random['sparsity_range'][0]:.3f} - {summary_random['sparsity_range'][1]:.3f}")
    
    st.markdown("**Optimal Initialization (v1.14.0)**")
    st.write(f"- Models: {summary_optimal['num_models']}")
    st.write(f"- Sparsities: {summary_optimal['num_sparsities']}")
    st.write(f"- Sparsity range: {summary_optimal['sparsity_range'][0]:.3f} - {summary_optimal['sparsity_range'][1]:.3f}")

# Epoch selector
st.sidebar.subheader("🕐 Select Epoch")
positions = sorted(position_to_epoch.keys())

selected_position = st.sidebar.select_slider(
    "Training Epoch",
    options=positions,
    value=positions[2] if len(positions) > 2 else positions[0],
    format_func=lambda x: f"Epoch {position_to_epoch[x]:,}",
    help="Select which training checkpoint to visualize"
)

st.sidebar.markdown(
    f"**Current Selection:**  \n"
    f"Position Index: {selected_position}  \n"
    f"Actual Epoch: {position_to_epoch[selected_position]:,}"
)

# Loss type selector
st.sidebar.subheader("📉 Loss Type")
loss_type = st.sidebar.radio(
    "Select loss calculation method",
    options=["Test Loss", "Train Loss"],
    index=0,
    help="Test loss is computed on held-out data, train loss is from training logs"
)
use_test_loss = (loss_type == "Test Loss")

# Sparsity filter
st.sidebar.subheader("🎯 Sparsity Filter")

# Combine all unique sparsities
all_sparsities = sorted(set(all_sparsities_random + all_sparsities_optimal))

# Option to select all or customize
filter_mode = st.sidebar.radio(
    "Filter mode",
    options=["Show All", "Custom Selection"],
    index=0
)

if filter_mode == "Custom Selection":
    selected_sparsities = st.sidebar.multiselect(
        "Select sparsity values to display",
        options=all_sparsities,
        default=all_sparsities,
        format_func=lambda x: f"{x:.3f}"
    )
else:
    selected_sparsities = all_sparsities

# Info box
st.sidebar.divider()
st.sidebar.info(
    "💡 **How to use:**\n\n"
    "1. Use the **epoch slider** above to select a training checkpoint\n"
    "2. Click on any point in the scatter plots to see detailed model visualization\n"
    "3. Or use the dropdown menus below each plot to manually select a model\n"
    "4. The detailed view shows loss curves and polygon evolution"
)

# =============================================================================
# Prepare Scatter Plot Data
# =============================================================================

with st.spinner("📊 Preparing visualization data..."):
    data_random = prepare_scatter_data(
        results_random, 
        llc_random, 
        selected_position,
        use_test_loss=use_test_loss,
        sparsity_filter=selected_sparsities if filter_mode == "Custom Selection" else None,
        dataset_id="random_1.15.0"
    )
    
    data_optimal = prepare_scatter_data(
        results_optimal,
        llc_optimal,
        selected_position,
        use_test_loss=use_test_loss,
        sparsity_filter=selected_sparsities if filter_mode == "Custom Selection" else None,
        dataset_id="optimal_1.14.0"
    )

# =============================================================================
# Main Content: Scatter Plots
# =============================================================================

st.header(f"📈 Loss vs LLC at Epoch {position_to_epoch[selected_position]:,}")
st.markdown(f"*Showing {loss_type}*")

# Create two columns for side-by-side plots
col1, col2 = st.columns(2)

with col1:
    st.subheader("Random 4-gon Initialization")
    
    if data_random.empty:
        st.warning("⚠️ No data available for random initialization with current filters")
    else:
        st.markdown(f"*{len(data_random)} models displayed*")
        
        fig_random = create_scatter_plot(
            data_random,
            "Random Initialization",
            st.session_state.selected_model_random
        )
        
        st.plotly_chart(fig_random, use_container_width=True, key="scatter_random")
        
        # Manual selection dropdown
        available_indices_random = sorted(data_random['model_index'].unique().tolist())
        selected_idx_random = st.selectbox(
            "Manually select model:",
            options=[None] + available_indices_random,
            format_func=lambda x: "None (click plot to select)" if x is None else f"Model {x}",
            key="manual_select_random"
        )
        
        if selected_idx_random is not None:
            st.session_state.selected_model_random = selected_idx_random
            st.session_state.selected_init_type = 'random'

with col2:
    st.subheader("Optimal Parameter Initialization")
    
    if data_optimal.empty:
        st.warning("⚠️ No data available for optimal initialization with current filters")
    else:
        st.markdown(f"*{len(data_optimal)} models displayed*")
        
        fig_optimal = create_scatter_plot(
            data_optimal,
            "Optimal Initialization",
            st.session_state.selected_model_optimal
        )
        
        st.plotly_chart(fig_optimal, use_container_width=True, key="scatter_optimal")
        
        # Manual selection dropdown
        available_indices_optimal = sorted(data_optimal['model_index'].unique().tolist())
        selected_idx_optimal = st.selectbox(
            "Manually select model:",
            options=[None] + available_indices_optimal,
            format_func=lambda x: "None (click plot to select)" if x is None else f"Model {x}",
            key="manual_select_optimal"
        )
        
        if selected_idx_optimal is not None:
            st.session_state.selected_model_optimal = selected_idx_optimal
            st.session_state.selected_init_type = 'optimal'

# =============================================================================
# Model Details Section
# =============================================================================

st.divider()

# Determine which model to show based on most recent selection
selected_model = None
results_to_use = None
init_type_label = None

if st.session_state.selected_init_type == 'random' and st.session_state.selected_model_random is not None:
    selected_model = st.session_state.selected_model_random
    results_to_use = results_random
    init_type_label = "Random 4-gon Initialization"
elif st.session_state.selected_init_type == 'optimal' and st.session_state.selected_model_optimal is not None:
    selected_model = st.session_state.selected_model_optimal
    results_to_use = results_optimal
    init_type_label = "Optimal Parameter Initialization"

# Display model details if a model is selected
if selected_model is not None and results_to_use is not None:
    model_data = get_model_data(results_to_use, selected_model)
    
    if model_data:
        st.header(f"🔍 Model Details: Index {selected_model}")
        st.markdown(f"*{init_type_label}*")
        
        # Display key metrics
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Model Index", selected_model)
        with col_b:
            st.metric("Sparsity", f"{model_data['sparsity']:.3f}")
        with col_c:
            st.metric("Total Epochs", f"{model_data['log_ivl'][-1]:,}")
        with col_d:
            # Get final loss
            final_loss = model_data['logs']['loss'].values[-1]
            st.metric("Final Loss", f"{final_loss:.6f}")
        
        # Generate and display detailed plot
        st.subheader("📊 Loss Curves and Polygon Evolution")
        st.markdown(
            "This visualization shows the training and test loss curves over time, "
            "along with snapshots of the learned weight polygons at key epochs."
        )
        
        with st.spinner("🎨 Generating detailed visualization..."):
            try:
                fig = plot_model_details(model_data)
                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error generating detailed plot: {e}")
                with st.expander("Show error details"):
                    st.exception(e)
        
        # Clear selection button
        col_clear1, col_clear2, col_clear3 = st.columns([1, 1, 2])
        with col_clear1:
            if st.button("🔄 Clear Selection", type="primary"):
                st.session_state.selected_model_random = None
                st.session_state.selected_model_optimal = None
                st.session_state.selected_init_type = None
                st.rerun()
        
        # Option to view statistics
        with st.expander("📈 View Statistics for Current View"):
            if not data_random.empty:
                st.markdown("**Random Initialization Statistics**")
                st.dataframe(create_summary_stats_table(data_random), use_container_width=True)
            
            if not data_optimal.empty:
                st.markdown("**Optimal Initialization Statistics**")
                st.dataframe(create_summary_stats_table(data_optimal), use_container_width=True)
    else:
        st.error(f"❌ Could not load data for model {selected_model}")
else:
    # No model selected - show placeholder
    st.info(
        "👆 **Select a model to see detailed visualization**\n\n"
        "Click on any point in the scatter plots above, or use the dropdown menus "
        "to manually select a model. The detailed view will show:\n"
        "- Training and test loss curves over all epochs\n"
        "- Polygon evolution snapshots at key training checkpoints\n"
        "- Bias visualizations"
    )
    
    # Show statistics even when no model is selected
    with st.expander("📈 View Statistics for Current View"):
        if not data_random.empty:
            st.markdown("**Random Initialization Statistics**")
            st.dataframe(create_summary_stats_table(data_random), use_container_width=True)
        
        if not data_optimal.empty:
            st.markdown("**Optimal Initialization Statistics**")
            st.dataframe(create_summary_stats_table(data_optimal), use_container_width=True)

# =============================================================================
# Footer
# =============================================================================

st.divider()
st.caption(
    "TMS Sparsity Interactive Visualization | "
    "Data from training runs v1.14.0 (optimal) and v1.15.0 (random) | "
    "Built with Streamlit"
)
