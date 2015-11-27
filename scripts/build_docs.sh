#!/bin/bash
# Build documentation for display in web browser.

PORT=${1:-4000}

echo "usage: build_docs.sh [port]"

# Find the docs dir, no matter where the script is called
ROOT_DIR="$( cd "$(dirname "$0")"/.. ; pwd -P )"
cd $ROOT_DIR

# Gather docs.
scripts/gather_examples.sh

# Generate developer docs.
make docs

# Display docs using web server.
cd docs
jekyll serve -w -s . -d _site --port=$PORT
