#!/bin/bash

# Elastica Examples Runner
# Author: Seung Hyun Kim
# This script runs all example cases and report the results.
# It runs all .py files and skips any post-processing scripts.
# It will run non-run files as well. As long as they do not throw an error,
# they will be considered successful.

# Disable matplotlib to prevent GUI windows and headless issues
export MPLBACKEND=Agg
export DISPLAY=:0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Initialize counters
TOTAL_CASES=0
SUCCESSFUL_CASES=0
FAILED_CASES=0
SKIPPED_CASES=0

# Arrays to store results
SUCCESSFUL_LIST=()
FAILED_LIST=()
SKIPPED_LIST=()

# Function to print colored output
print_status() {
    local status_type=$1
    local message=$2
    case $status_type in
        "SUCCESS")
            echo -e "${GREEN}✓ SUCCESS${NC}: $message"
            ;;
        "FAILED")
            echo -e "${RED}✗ FAILED${NC}: $message"
            ;;
        "SKIPPED")
            echo -e "${YELLOW}⚠ SKIPPED${NC}: $message"
            ;;
        "INFO")
            echo -e "${BLUE}ℹ INFO${NC}: $message"
            ;;
    esac
}

# Function to run a Python script
run_script() {
    local script_path=$1
    local script_name=$(basename "$script_path")
    local dir_name=$(dirname "$script_path" | sed 's|^\./||')

    TOTAL_CASES=$((TOTAL_CASES + 1))

    # Skip postprocessing scripts
    if [[ "$script_name" == *"postprocessing"* ]] || [[ "$script_name" == *"post_processing"* ]]; then
        print_status "SKIPPED" "$dir_name/$script_name (postprocessing script)"
        SKIPPED_CASES=$((SKIPPED_CASES + 1))
        SKIPPED_LIST+=("$dir_name/$script_name")
        return 0
    fi

    print_status "INFO" "Running $dir_name/$script_name..."

    # Run the script and matplotlib disabled
    if python "$script_path" > /dev/null 2>&1; then
        print_status "SUCCESS" "$dir_name/$script_name"
        SUCCESSFUL_CASES=$((SUCCESSFUL_CASES + 1))
        SUCCESSFUL_LIST+=("$dir_name/$script_name")
    else
        print_status "FAILED" "$dir_name/$script_name"
        FAILED_CASES=$((FAILED_CASES + 1))
        FAILED_LIST+=("$dir_name/$script_name")
    fi
}

# Function to find and run all Python scripts in a directory
run_directory() {
    local dir_path=$1

    if [[ ! -d "$dir_path" ]]; then
        return
    fi

    # Find all Python files in the directory (excluding __pycache__ and hidden files)
    while IFS= read -r -d '' script; do
        run_script "$script"
    done < <(find "$dir_path" -maxdepth 1 -name "*.py" -type f -print0 2>/dev/null)
}

# Function to recursively find and run all Python scripts
run_directory_recursive() {
    local dir_path=$1

    if [[ ! -d "$dir_path" ]]; then
        return
    fi

    # Find all Python files in the directory and subdirectories (excluding __pycache__ and hidden files)
    while IFS= read -r -d '' script; do
        run_script "$script"
    done < <(find "$dir_path" -name "*.py" -type f -print0 2>/dev/null)
}

# Main execution
echo "============================="
echo "Elastica Examples Test Runner"
echo "===================================="
echo "Starting to run all example cases..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Run all directories
for dir in */; do
    if [[ -d "$dir" ]]; then
        dir_name="${dir%/}"
        echo "Processing directory: $dir_name"

        # Special handling for directories with subdirectories
        case "$dir_name" in
            "RigidBodyCases" | "RodContactCase")
                # Run recursively for these directories
                run_directory_recursive "$dir_name"
                ;;
            *)
                # Run only files in the main directory
                run_directory "$dir_name"
                ;;
        esac
        echo ""
    fi
done

# Print summary
echo "======="
echo "SUMMARY"
echo "======="
echo "Total cases found: $TOTAL_CASES"
echo "Successful: $SUCCESSFUL_CASES"
echo "Failed: $FAILED_CASES"
echo "Skipped: $SKIPPED_CASES"
echo ""

if [[ ${#SUCCESSFUL_LIST[@]} -gt 0 ]]; then
    echo "Successful cases:"
    for case in "${SUCCESSFUL_LIST[@]}"; do
        echo "  ✓ $case"
    done
    echo ""
fi

if [[ ${#FAILED_LIST[@]} -gt 0 ]]; then
    echo "Failed cases:"
    for case in "${FAILED_LIST[@]}"; do
        echo "  ✗ $case"
    done
    echo ""
fi

if [[ ${#SKIPPED_LIST[@]} -gt 0 ]]; then
    echo "Skipped cases:"
    for case in "${SKIPPED_LIST[@]}"; do
        echo "  ⚠ $case"
    done
    echo ""
fi

# Exit with appropriate code
if [[ $FAILED_CASES -eq 0 ]]; then
    echo "All runnable cases completed successfully!"
    exit 0
else
    echo "Some cases failed. Check the output above for details."
    exit 1
fi
