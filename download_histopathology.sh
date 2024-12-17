#!/bin/bash

# Script to download the breast histopathology images dataset

# Function to display the help menu
function show_help {
  echo "Usage: $0 [options] <destination_directory>"
  echo ""
  echo "Options:"
  echo "  --help    Show this help message and exit."
  echo ""
  echo "Description:"
  echo "This script downloads the breast histopathology images dataset from Kaggle"
  echo "and saves it as a ZIP file in the specified destination directory."
  echo ""
  echo "Example:"
  echo "  $0 /path/to/destination"
  exit 0
}

# Check for --help option
if [ "$1" == "--help" ]; then
  show_help
fi

# Check if the destination directory is provided
if [ -z "$1" ]; then
  echo "Error: Destination directory not specified."
  echo "Use --help for usage information."
  exit 1
fi

# Set the destination directory
DEST_DIR="$1"

# Ensure the destination directory exists
mkdir -p "$DEST_DIR"

# Define the output file path
OUTPUT_FILE="$DEST_DIR/breast-histopathology-images.zip"

# Perform the download using curl
curl -L -o "$OUTPUT_FILE" \
  https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/breast-histopathology-images

# Notify the user
if [ $? -eq 0 ]; then
  echo "Download completed successfully: $OUTPUT_FILE"
else
  echo "Download failed."
  exit 1
fi
