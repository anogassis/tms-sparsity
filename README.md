# TMS Sparsity Experiments

## Interactive Visualization

🎉 **NEW**: Interactive Streamlit app for exploring the results!

Visualize the relationship between Learning Coefficient (LLC) and Loss across different training epochs with an interactive web interface.

**Quick Start:**
```bash
# Install dependencies
pip install -r requirements-streamlit.txt

# Launch the app
bash run_streamlit.sh
```

See [streamlit_app/README.md](streamlit_app/README.md) for more details.

---


# Results

We explored how the solutions in the problem from the toy model of superposition change in the low sparsity regime.
We first initialized the models as six-gons (the optimal solution for 6 input-parameters), which puzzelingly lead to 0 correlation between the loss and the llc within models trained on data of the same sparsity. We then ran another run where we initialized the models as 4-gons, like in *[Chen et al. Dynamical versus Bayesian Phase Transitions in a Toy Model of Superposition](https://arxiv.org/abs/2310.06301)*, which on average lead to worse solutions in the non-sparse regime, but the best solutions tended to be better.    
