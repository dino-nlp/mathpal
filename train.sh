#!/bin/bash
# Convenience script for training MathPal models
# Usage: ./train.sh [config_file] [additional_args...]

set -e

# Check if first argument is a config file (doesn't start with --)
if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
    CONFIG="$1"
    shift
else
    CONFIG="configs/complete_training_config.yaml"
fi

# Set Python path
export PYTHONPATH="/root/mathpal/src"

echo "🚀 Starting MathPal Training Pipeline"
echo "📋 Config: $CONFIG"
echo "🛤️  PYTHONPATH: $PYTHONPATH"
echo "==============================================="

# Run training with proper Python path
python3 -m training_pipeline.cli.train_gemma_v2 --config "$CONFIG" "$@"
