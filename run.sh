#!/bin/sh
set -e

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "Virtual environment created."
fi

. .venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt
echo "Dependencies installed."

python3 main.py

deactivate