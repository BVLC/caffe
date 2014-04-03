#!/bin/bash

PORT=${1:-4000}

echo "usage: build_docs.sh [port]"

# Find the docs dir, no matter where the script is called
DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR/../docs

jekyll serve -w -s . -d _site --port=$PORT
