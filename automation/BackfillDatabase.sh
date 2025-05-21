#!/bin/bash

# First, navigate to your project directory
cd ../data/

# Extract Python environment path from config.py
# This assumes your config.py has a variable like PYTHON_ENV_PATH="/home/tyler/anaconda3/envs/sports/bin/python"
PYTHON_PATH=$(grep -o 'PYTHON_ENV_PATH\s*=\s*"[^"]*"' ../config.py | cut -d'"' -f2)

# If config.py doesn't exist or doesn't contain the path, use default
if [ -z "$PYTHON_PATH" ]; then
    echo "Warning: Could not find PYTHON_ENV_PATH in config.py, using default"
    PYTHON_PATH="/usr/bin/python3"
fi

# Run the PBP insertion
printf 'Inserting PBP\n\n'
printf '%s\n' "$(date +'%Y-%m-%d %H:%M:%S')"
$PYTHON_PATH insert_pbp_data.py --backfill

# Run the schedule insertion
printf '\n\nInserting Schedule\n\n'
printf '%s\n' "$(date +'%Y-%m-%d %H:%M:%S')"
$PYTHON_PATH insert_schedule.py --backfill