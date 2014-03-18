#!/bin/bash

echo "usage: build_docs.sh [port]"
PORT=4000
if [ $# -gt 0 ]; then
    PORT=$1
fi
jekyll serve -w -s docs/ -d docs/_site --port $PORT
