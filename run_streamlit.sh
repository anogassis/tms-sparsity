#!/bin/bash
# Launch script for TMS Sparsity Interactive Visualization

cd "$(dirname "$0")"

echo "Starting TMS Sparsity Interactive Visualization..."
echo "The app will open in your browser at http://localhost:8501"
echo ""

streamlit run streamlit_app/app.py \
    --server.port 8501 \
    --server.address localhost \
    --theme.base light
