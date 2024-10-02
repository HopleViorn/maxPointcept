#!/bin/bash

# Activate the conda environment
conda activate pointcept

# Loop from 0 to 6 (inclusive)
for i in {0..5}; do
    # Run the Python script with the loop variable as an argument
    python cylinders_normal.py $i &
done