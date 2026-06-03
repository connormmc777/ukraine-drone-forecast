#!/bin/bash
# Install script for Ukraine Drone Forecast on Bazzite
# Run with: bash install.sh

set -e  # exit on error

echo "=================================================="
echo "Ukraine Drone Forecast - Bazzite Local Installation"
echo "=================================================="

# Check Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Bazzite should ship with it."
    echo "Try: rpm-ostree install python3"
    exit 1
fi

echo ""
echo "Step 1: Creating virtual environment..."
python3 -m venv venv

echo ""
echo "Step 2: Activating environment..."
source venv/bin/activate

echo ""
echo "Step 3: Upgrading pip..."
pip install --upgrade pip --quiet

echo ""
echo "Step 4: Installing required packages..."
pip install --quiet \
    streamlit \
    pandas \
    numpy \
    matplotlib \
    scipy \
    scikit-learn \
    geopandas

echo ""
echo "=================================================="
echo "Installation complete!"
echo "=================================================="
echo ""
echo "To start the app:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""
echo "The app will open in your browser at http://localhost:8501"
echo "Press Ctrl+C in the terminal to stop the app."
echo ""
echo "To save new observations, use the form at the bottom of the dashboard."
echo "All data is stored in the data/ folder."
