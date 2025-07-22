#!/bin/bash -x

# Automatically export all sourced variables from .env
set -a
source "$(dirname "$0")/../.env"
set +a

# Read input arguments
STAGE=$1
LANGUAGE=$2

# Build paths based on environment variable
ZIP_PATH="$DATA_DIR/${LANGUAGE}-${STAGE}"
EXTRACT_DIR="$DATA_DIR/repositories-${LANGUAGE}-${STAGE}"

# Unzip the main archive
unzip "$ZIP_PATH" -d "$EXTRACT_DIR"

# Unzip each repo archive inside the extracted folder
for zipfile in "$EXTRACT_DIR"/*.zip; do
  unzip -o "$zipfile" -d "${zipfile%.zip}"
done
