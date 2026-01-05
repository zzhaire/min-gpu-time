#!/bin/bash

# Utility script to run experiments and generate plots
# Usage: ./run_experiments.sh [experiment_type]

EXPERIMENT_TYPE=${1:-"all"}

echo "Running GPU Scheduling Experiments..."
echo "Experiment Type: $EXPERIMENT_TYPE"
echo "=================================="

# Create results directory if it doesn't exist
mkdir -p results

case $EXPERIMENT_TYPE in
    "sensitivity")
        echo "Running sensitivity analysis..."
        python exp/run_sensitivity.py
        python utils/plotting/plot_sensitivity.py
        ;;
    "pareto")
        echo "Running Pareto frontier analysis..."
        python exp/run_pareto.py
        python utils/plotting/plot_pareto.py
        ;;
    "scalability")
        echo "Running scalability analysis..."
        python exp/run_scalability.py
        python utils/plotting/plot_scalability.py
        ;;
    "completion")
        echo "Running run-to-completion analysis..."
        python exp/run_scalability_completion.py
        python utils/plotting/plot_scalability_completion_detailed.py
        ;;
    "all")
        echo "Running all experiments..."
        echo "1. Sensitivity Analysis..."
        python exp/run_sensitivity.py
        python utils/plotting/plot_sensitivity.py
        
        echo "2. Pareto Frontier..."
        python exp/run_pareto.py
        python utils/plotting/plot_pareto.py
        
        echo "3. Scalability Analysis..."
        python exp/run_scalability.py
        python utils/plotting/plot_scalability.py
        
        echo "4. Run-to-Completion Analysis..."
        python exp/run_scalability_completion.py
        python utils/plotting/plot_scalability_completion_detailed.py
        ;;
    "plot")
        echo "Generating plots from existing data..."
        python utils/plotting/plot_sensitivity.py
        python utils/plotting/plot_pareto.py
        python utils/plotting/plot_scalability.py
        python utils/plotting/plot_scalability_completion_detailed.py
        ;;
    *)
        echo "Usage: $0 [sensitivity|pareto|scalability|completion|all|plot]"
        echo "  sensitivity - Run interference sensitivity analysis"
        echo "  pareto      - Run Pareto frontier analysis"
        echo "  scalability - Run scalability stress tests"
        echo "  completion  - Run run-to-completion analysis"
        echo "  all         - Run all experiments (default)"
        echo "  plot        - Generate plots from existing CSV data"
        exit 1
        ;;
esac

echo "=================================="
echo "Experiments completed!"
echo "Results saved in results/ directory"
echo "Plots saved in results/ directory"
