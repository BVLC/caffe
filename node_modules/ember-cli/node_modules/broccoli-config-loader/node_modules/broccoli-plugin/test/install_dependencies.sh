#!/bin/bash

# Call from project root: test/install_dependencies.sh

set -e

cd test
rm -rf dependencies
mkdir dependencies
cd dependencies

for broccoli_version in 0.16.3; do
  (
    mkdir "broccoli-$broccoli_version"
    cd "broccoli-$broccoli_version"
    mkdir node_modules
    npm install "broccoli@$broccoli_version"
    echo "module.exports = require('broccoli')" > index.js
  )
done
