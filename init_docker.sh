#!/bin/bash
set -e

echo "Starting Docker Initialization..."

# Checking if requirements need to be installed or modified
# pip install -r requirements.txt is already done in base image generally,
# but can rerun to be safe if desired.

echo "Installing pypcd and other required libraries..."
if [ ! -d "pypcd" ]; then
    git clone https://github.com/klintan/pypcd.git
    cd pypcd
    python setup.py install
    cd ..
else
    echo "pypcd directory exists, assuming installed or checking."
    cd pypcd
    python setup.py install
    cd ..
fi

echo "Compiling BEVHeight CUDA operations..."
python setup.py develop

echo "Initialization complete! You can now run experiments."
