#!/bin/bash

# PodTracker Analysis Pipeline Script
# This script runs all the analysis scripts in the correct order

# Set the base directory to the current directory
BASE_DIR=$(pwd)
CODE_DIR="$BASE_DIR/code"

# Ensure we're using the conda environment
eval "$(conda shell.bash hook)"
conda activate ml_env

# Echo function for better output formatting
echo_step() {
    echo "======================================================================"
    echo ">>> $1"
    echo "======================================================================"
}

# Function to run a Python script with error handling
run_script() {
    script_name=$1
    echo_step "Running $script_name"
    
    if [ -f "$CODE_DIR/$script_name" ]; then
        python "$CODE_DIR/$script_name"
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to execute $script_name"
            exit 1
        else
            echo "Successfully completed $script_name"
        fi
    else
        echo "ERROR: Script not found: $CODE_DIR/$script_name"
        exit 1
    fi
    
    echo ""
}

# Main execution pipeline
echo_step "Starting PodTracker Analysis Pipeline"

# Run each script in sequence
run_script "multifolder_raw_jpg_conversion.py"
run_script "peanut_maskrcnn_train_v2.py"
run_script "balance_dataset.py"
run_script "decisiontreeclassifier.py"
run_script "deepsort_analysis.py"
run_script "grade_analysis.py"
run_script "count_analysis.py"

echo_step "Analysis Pipeline Completed Successfully"
echo "Results are available in the results directory"
