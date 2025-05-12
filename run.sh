#!/bin/bash

# Smart Meter Analytics Runner Script
# This script helps run the different components of the project

# Default values
PROCESS_DATA=false
TRAIN_MODELS=false
RUN_WEBAPP=false
SAMPLE_SIZE=""

# Display help
show_help() {
  echo "Smart Meter Analytics Runner"
  echo ""
  echo "Usage: ./run.sh [options]"
  echo ""
  echo "Options:"
  echo "  --process            Process the raw data"
  echo "  --train              Train the forecast models"
  echo "  --webapp             Run the web dashboard"
  echo "  --sample SIZE        Use a sample of specified size when processing data"
  echo "  --all                Run everything (process, train, webapp)"
  echo "  --help               Show this help message"
  echo ""
  echo "Examples:"
  echo "  ./run.sh --webapp              # Just run the web app"
  echo "  ./run.sh --process --train     # Process data and train models"
  echo "  ./run.sh --process --sample 100000 --webapp # Process a sample of data and run the web app"
  echo "  ./run.sh --all                 # Process data, train models, and run the web app"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --process)
      PROCESS_DATA=true
      shift
      ;;
    --train)
      TRAIN_MODELS=true
      shift
      ;;
    --webapp)
      RUN_WEBAPP=true
      shift
      ;;
    --sample)
      SAMPLE_SIZE="$2"
      shift 2
      ;;
    --all)
      PROCESS_DATA=true
      TRAIN_MODELS=true
      RUN_WEBAPP=true
      shift
      ;;
    --help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# If no options provided, just run the webapp
if [ "$PROCESS_DATA" = false ] && [ "$TRAIN_MODELS" = false ] && [ "$RUN_WEBAPP" = false ]; then
  RUN_WEBAPP=true
fi

# Construct the Python command
CMD="python run.py"

if [ "$PROCESS_DATA" = true ]; then
  CMD="$CMD --process"
  
  if [ -n "$SAMPLE_SIZE" ]; then
    CMD="$CMD --sample $SAMPLE_SIZE"
  fi
fi

if [ "$TRAIN_MODELS" = true ]; then
  CMD="$CMD --train"
fi

if [ "$RUN_WEBAPP" = true ]; then
  CMD="$CMD --webapp"
fi

# Run the command
echo "Running: $CMD"
eval "$CMD" 