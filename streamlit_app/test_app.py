"""
Test script to verify the Streamlit app can load data correctly.

Run this before launching the app to check if all dependencies and data files are in place.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        import streamlit as st
        print("✓ streamlit")
    except ImportError as e:
        print(f"✗ streamlit: {e}")
        return False
    
    try:
        import plotly
        print("✓ plotly")
    except ImportError as e:
        print(f"✗ plotly: {e}")
        return False
    
    try:
        import torch
        print("✓ torch")
    except ImportError as e:
        print(f"✗ torch: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas")
    except ImportError as e:
        print(f"✗ pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        import matplotlib
        print("✓ matplotlib")
    except ImportError as e:
        print(f"✗ matplotlib: {e}")
        return False
    
    print("\nAll imports successful!\n")
    return True


def test_data_files():
    """Test that required data files exist."""
    print("Checking data files...")
    
    data_path = "data"
    if not os.path.exists(data_path):
        print(f"✗ Data directory '{data_path}' not found")
        return False
    print(f"✓ Data directory exists")
    
    # Check for LLC CSV files
    required_files = [
        "llc_preagg_1.15.0.csv",
        "llc_preagg_1.14.0.csv"
    ]
    
    for filename in required_files:
        filepath = os.path.join(data_path, filename)
        if os.path.exists(filepath):
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} not found")
            return False
    
    # Check for at least some pickle files
    import glob
    pickle_files_random = glob.glob(os.path.join(data_path, "logs_loss_1.15.0_*.pkl"))
    pickle_files_optimal = glob.glob(os.path.join(data_path, "logs_loss_1.14.0_*.pkl"))
    
    if len(pickle_files_random) > 0:
        print(f"✓ Found {len(pickle_files_random)} random initialization result files")
    else:
        print(f"✗ No random initialization result files found")
        return False
    
    if len(pickle_files_optimal) > 0:
        print(f"✓ Found {len(pickle_files_optimal)} optimal initialization result files")
    else:
        print(f"✗ No optimal initialization result files found")
        return False
    
    print("\nAll data files found!\n")
    return True


def test_data_loading():
    """Test that data can be loaded successfully."""
    print("Testing data loading (this may take a moment)...")
    
    try:
        from streamlit_app.data_loader import load_all_results, load_llc_estimates
        
        print("Loading results...")
        results_random, results_optimal = load_all_results("data")
        print(f"✓ Loaded {len(results_random)} random initialization models")
        print(f"✓ Loaded {len(results_optimal)} optimal initialization models")
        
        print("\nLoading LLC estimates...")
        llc_random, llc_optimal = load_llc_estimates("data")
        print(f"✓ Loaded {len(llc_random)} random LLC estimates")
        print(f"✓ Loaded {len(llc_optimal)} optimal LLC estimates")
        
        print("\nData loading successful!\n")
        return True
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("TMS Sparsity Streamlit App - Pre-flight Check")
    print("=" * 60)
    print()
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        print("   pip install -r requirements-streamlit.txt")
        return False
    
    # Test data files
    if not test_data_files():
        print("\n❌ Data file check failed. Please ensure data files are in place.")
        return False
    
    # Test data loading
    if not test_data_loading():
        print("\n❌ Data loading test failed.")
        return False
    
    print("=" * 60)
    print("✅ All checks passed! You're ready to run the app.")
    print("=" * 60)
    print("\nTo start the app, run:")
    print("   bash run_streamlit.sh")
    print("\nOr directly:")
    print("   streamlit run streamlit_app/app.py")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
