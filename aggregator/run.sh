#!/bin/sh
set -e

if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating one..."
    uv venv -p 3.12 .venv
    echo "Virtual environment created successfully."
else
    echo "Virtual environment already exists."
fi

uv pip install -U syftbox --quiet
. .venv/bin/activate

echo "Running syftbox-netflix aggregator with $(python3 --version) at '$(which python3)'"
python3 main.py

deactivate