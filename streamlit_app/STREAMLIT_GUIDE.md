# Quick Start Guide: TMS Sparsity Interactive Visualization

## Installation & Setup

### 1. Install Dependencies
```bash
# Core dependencies (if not already installed)
pip install -r requirements.txt

# Streamlit app dependencies
pip install -r requirements-streamlit.txt
```

### 2. Verify Setup
```bash
# Optional: Run the test script to verify everything is working
python streamlit_app/test_app.py
```

### 3. Launch the App
```bash
# Easy way (using the launch script)
bash run_streamlit.sh

# Or directly
streamlit run streamlit_app/app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## Basic Usage

### Navigation

1. **Sidebar (Left)**
   - 🕐 **Epoch Slider**: Select which training checkpoint to view
   - 📉 **Loss Type**: Choose between Test Loss or Train Loss
   - 🎯 **Sparsity Filter**: Show all or select specific sparsity values

2. **Main Area (Center)**
   - Two scatter plots side-by-side
   - Left: Random 4-gon initialization
   - Right: Optimal parameter initialization
   - Each point represents one trained model

3. **Model Details (Below)**
   - Appears when you select a model
   - Shows loss curves and polygon evolution
   - Includes model metadata

### Selecting a Model

**Method 1: Click on a point**
- Simply click any point in either scatter plot
- That model's details will appear below

**Method 2: Use dropdown**
- Below each scatter plot is a "Manually select model" dropdown
- Choose from the list of available model indices

### Understanding the Visualizations

**Scatter Plots (Main)**
- **X-axis**: LLC (Learning Coefficient) - measures model complexity
- **Y-axis**: Loss - how well the model performs
- **Colors**: Different sparsity values
- **Hover**: Shows model index, LLC, loss, and sparsity

**Model Details (Bottom)**
- **Top row**: Key metrics (model index, sparsity, epochs, final loss)
- **Main plot**: 
  - Bottom panel: Loss curves over training (train and test)
  - Top panels: Polygon snapshots at 5 key epochs (0, 200, 2000, 10000, 20000)
  - Shows how the learned weights evolve during training

---

## Tips & Tricks

### Performance
- **First load is slow (~10-30 seconds)**: Data is being cached
- **After that, it's fast (<1 second)**: Cached data is reused
- **Changing epochs takes a moment (~2 seconds)**: Scatter data is regenerated

### Exploration Strategies

1. **Compare Across Epochs**
   - Use the epoch slider to see how LLC-Loss relationship changes
   - Notice how models converge over training

2. **Compare Initialization Strategies**
   - Look at both plots to see differences
   - Random init often has more variance
   - Optimal init starts closer to good solutions

3. **Focus on Specific Sparsity**
   - Use the sparsity filter to isolate specific values
   - Easier to see patterns for one sparsity level

4. **Find Interesting Models**
   - Look for outliers (high LLC but low loss, or vice versa)
   - Select them to see why they're different
   - Compare polygon shapes between good and bad models

### Common Patterns to Look For

- **Low sparsity** (0.4-0.7): Higher LLC values, more complex solutions
- **High sparsity** (0.9-1.0): Lower LLC values, simpler solutions
- **Outliers**: Models that break the pattern - often interesting!
- **Polygon evolution**: Watch how polygons deform from initialization to final state

---

## Troubleshooting

### App Won't Start
```bash
# Check you're in the right directory
pwd  # Should show .../tms-sparsity

# Check Python version (needs 3.8+)
python --version

# Reinstall dependencies
pip install -r requirements-streamlit.txt --force-reinstall
```

### "Data Not Found" Error
```bash
# Verify data directory exists
ls -la data/

# Should see files like:
# - llc_preagg_1.15.0.csv
# - llc_preagg_1.14.0.csv
# - logs_loss_1.15.0_*.pkl
# - logs_loss_1.14.0_*.pkl
```

### Slow Performance
- **Clear cache**: Menu (≡) → "Clear cache" → "Clear all caches"
- **Close other tabs**: Browser memory usage
- **Reduce filters**: Show fewer sparsity values

### Plot Not Showing
- **Refresh browser**: Hard refresh (Cmd+Shift+R on Mac, Ctrl+Shift+R on Windows)
- **Check console**: Browser DevTools → Console tab for errors
- **Try different browser**: Chrome or Firefox work best

---

## Advanced Features

### Statistics View
- Click "View Statistics for Current View" expander at bottom
- See summary stats grouped by sparsity
- Mean, std, min, max for both LLC and Loss

### Multiple Models
- Select one model from random init plot
- Select another from optimal init plot
- Both remain highlighted as you explore

### Clear Selection
- Click "Clear Selection" button to reset
- Or just select a new model

---

## Keyboard Shortcuts

Streamlit has built-in shortcuts:
- `R`: Rerun the app
- `C`: Clear cache
- `/`: Focus on search

---

## Need Help?

1. Check the README: `streamlit_app/README.md`
2. Run the test script: `python streamlit_app/test_app.py`
3. Look at the sidebar info box in the app
4. Check the original plotting code in `tms/plots/`

---

## What's Next?

Some ideas for exploration:
- Compare how LLC evolves across epochs for specific sparsities
- Find the "best" models (low loss, optimal LLC)
- Study outliers and understand what makes them different
- Look at polygon shapes to understand learned representations

Happy exploring! 🚀
