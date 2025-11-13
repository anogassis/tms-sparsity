# TMS Sparsity Interactive Visualization

An interactive Streamlit application for exploring the relationship between Learning Coefficient (LLC) and Loss in the TMS Sparsity experiments.

## Features

- 📊 **Interactive Scatter Plots**: Visualize Loss vs LLC at different training epochs
- 🔍 **Model Details**: Click any point to see detailed loss curves and polygon evolution
- 🎯 **Flexible Filtering**: Filter by sparsity values and switch between test/train loss
- 🔄 **Epoch Navigation**: Easily navigate through different training checkpoints
- 📈 **Comparative Analysis**: Side-by-side comparison of random vs optimal initialization

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-streamlit.txt
```

2. Ensure you have the data files in the `data/` directory:
   - `logs_loss_1.15.0_*.pkl` (random initialization)
   - `logs_loss_1.14.0_*.pkl` (optimal initialization)
   - `llc_preagg_1.15.0.csv`
   - `llc_preagg_1.14.0.csv`

## Usage

### Quick Start

Run the application using the provided launch script:

```bash
bash run_streamlit.sh
```

Or run directly with streamlit:

```bash
streamlit run streamlit_app/app.py
```

The application will open in your browser at `http://localhost:8501`

### How to Use

1. **Select Epoch**: Use the slider in the sidebar to choose which training checkpoint to visualize
2. **Choose Loss Type**: Toggle between test loss (computed on held-out data) and train loss
3. **Filter by Sparsity**: Optionally filter which sparsity values to display
4. **Explore Models**: 
   - Click on any point in the scatter plots, OR
   - Use the dropdown menus below each plot to manually select a model
5. **View Details**: Selected model's detailed visualization appears below with:
   - Loss curves (train and test)
   - Polygon evolution snapshots at key epochs
   - Bias visualizations

## Architecture

### Data Loading (`data_loader.py`)
- Efficient caching with `@st.cache_data` and `@st.cache_resource`
- Pre-computation of test sets for all sparsity values
- Fast lookup structures for LLC estimates

### Plotting (`plotting.py`)
- Interactive Plotly scatter plots with hover tooltips
- Reuses existing matplotlib-based polygon visualization functions
- Consistent color mapping across sparsity values

### Main App (`app.py`)
- Sidebar controls for epoch selection and filtering
- Side-by-side comparison of initialization strategies
- Session state management for selections
- Expandable statistics and summaries

## Performance

- **First Load**: ~10-30 seconds (loading and caching data)
- **Subsequent Interactions**: <1 second (data is cached)
- **Epoch Changes**: <2 seconds (scatter data regeneration)
- **Model Details**: ~2-3 seconds (matplotlib plot generation)

## Troubleshooting

### Data Not Found
If you see "Data directory not found", ensure you're running from the project root:
```bash
cd /path/to/tms-sparsity
bash run_streamlit.sh
```

### Memory Issues
If you encounter memory errors with large datasets:
- The app caches data efficiently, but initial load requires ~2-4 GB RAM
- Close other applications if needed
- Consider filtering to fewer sparsity values

### Plot Not Displaying
- Check browser console for errors
- Try refreshing the page
- Ensure all dependencies are installed correctly

## Development

### File Structure
```
streamlit_app/
├── __init__.py          # Package initialization
├── app.py               # Main Streamlit application
├── data_loader.py       # Data loading and caching
└── plotting.py          # Visualization functions
```

### Adding Features

To add new features:
1. Add data preparation logic to `data_loader.py`
2. Add plotting functions to `plotting.py`
3. Add UI components to `app.py`

### Caching

The app uses two types of caching:
- `@st.cache_resource`: For objects that should persist (results, models)
- `@st.cache_data`: For DataFrames and serializable data

Clear cache from the UI: Hamburger menu → "Clear cache"

## Known Limitations

1. **Click Detection**: Plotly click events in Streamlit are limited. Use dropdown menus as fallback.
2. **Large Datasets**: Very large result sets (>5000 models) may slow down scatter plots.
3. **Browser Compatibility**: Best viewed in Chrome, Firefox, or Safari.

## Credits

Built using:
- [Streamlit](https://streamlit.io/) - Interactive web app framework
- [Plotly](https://plotly.com/) - Interactive plotting library
- [Matplotlib](https://matplotlib.org/) - Detailed visualization plots

Based on the TMS Sparsity research project.
