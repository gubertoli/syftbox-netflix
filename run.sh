#!/bin/bash
set -e

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Exiting..."
    exit 1
fi

# Load environment variables from .env
echo "Loading environment variables from .env..."
export $(grep -v '^#' .env | xargs)

# Check the operating system
OS=$(uname -s)

# Check if chromedriver is in the PATH
if ! command -v chromedriver &> /dev/null; then
    echo "chromedriver is not installed. Installing..."
    
    if [ "$OS" = "Darwin" ]; then
        # MacOS
        if ! command -v brew &> /dev/null; then
            echo "Homebrew is not installed. Please install it first."
            exit 1
        fi
        brew install chromedriver

    elif [ "$OS" = "Linux" ]; then
        # Linux
        if command -v apt &> /dev/null; then
            # For Debian/Ubuntu-based distros
            sudo apt-get update 2>/dev/null
            sudo apt-get install -y -qq chromium-driver
        elif command -v dnf &> /dev/null; then
            # For Red Hat/Fedora-based distros
            sudo dnf install -y chromium
        else
            echo "Unsupported Linux distribution. Please install chromedriver manually."
            exit 1
        fi
    else
        echo "Unsupported operating system. Please install chromedriver manually."
        exit 1
    fi
else
    echo "chromedriver is already installed."
fi

export CHROMEDRIVER_PATH=$(which chromedriver)



if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "Virtual environment created."
fi

. .venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt --quiet
echo "Dependencies installed."

python3 main.py

deactivate