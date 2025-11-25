#!/bin/bash

# Root directory to search for metrics.json files
SEARCH_ROOT="/mnt/shared-storage-user/zhoujiayi/boyuan/model_results/resist-collapse/1120_test_collapse_eval/safe_llama3.1-8b/paloma_collapse"

# Path to the aggregation script
AGG_SCRIPT="align_anything/evaluation/aggregate_paloma_scores.py"

# Ensure the aggregation script exists
if [ ! -f "$AGG_SCRIPT" ]; then
    echo "Error: Aggregation script not found at $AGG_SCRIPT"
    exit 1
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Searching for metrics.json files in $SEARCH_ROOT..."

# Find all metrics.json files recursively
find "$SEARCH_ROOT" -type f -name "metrics.json" | while read -r metrics_file; do
    dir_path=$(dirname "$metrics_file")
    output_file="${dir_path}/aggregate_report.txt"
    
    echo "Processing: $metrics_file"
    
    # Run the python script and redirect output to a report file in the same directory
    if python "$AGG_SCRIPT" "$metrics_file" > "$output_file"; then
        echo "  -> Report saved to: $output_file"
    else
        echo "  -> Error processing $metrics_file"
    fi
done

echo "Batch aggregation complete."

