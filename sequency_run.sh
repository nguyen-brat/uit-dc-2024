#!/bin/bash

# Ensure conda is initialized in the current shell session
source ~/miniconda3/etc/profile.d/conda.sh
if [ $? -ne 0 ]; then
    echo "Failed to source Conda. Please ensure Conda is installed correctly."
    exit 1
fi

# Function to check if the previous command was successful
check_error() {
    if [ $? -ne 0 ]; then
        echo "Error occurred in $1. Exiting script."
        exit 1
    fi
}

# Activate uit-ds environment and run the Python script
# echo "Activating uit-ds environment..."
# conda activate uit-ds
# check_error "conda activate uit-ds"

# echo "Running run_llm.py..."
# python run_llm.py
# check_error "python run_llm.py"

# Activate pixtral environment and run the reasoning Python script
echo "Activating pixtral environment..."
conda activate pixtral
check_error "conda activate pixtral"

echo "Running reasoning.py... batch 6"
python src/agent/reasoning.py --batch_size 6
echo "Running reasoning.py... batch 3"
python src/agent/reasoning.py --batch_size 3
check_error "python src/agent/reasoning.py"

echo "Script executed successfully."
