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
echo "Activating uit-ds environment..."
conda activate uit-ds
check_error "conda activate uit-ds"

echo "Running bash"
bash script/train.sh

PASSWORD="2907"
echo $PASSWORD | sudo -S shutdown -h now