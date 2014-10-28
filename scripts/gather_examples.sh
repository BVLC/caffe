#!/bin/bash
# Assemble documentation for the project into one directory via symbolic links.

# Find the docs dir, no matter where the script is called
ROOT_DIR="$( cd "$(dirname "$0")"/.. ; pwd -P )"
cd $ROOT_DIR

# Gather docs from examples/**/readme.md
GATHERED_DIR=docs/gathered
rm -r $GATHERED_DIR
mkdir $GATHERED_DIR
for README_FILENAME in $(find examples -iname "readme.md"); do
    # Only use file if it is to be included in docs.
    if grep -Fxq "include_in_docs: true" $README_FILENAME; then
        # Make link to readme.md in docs/gathered/.
        # Since everything is called readme.md, rename it by its dirname.
        README_DIRNAME=`dirname $README_FILENAME`
        DOCS_FILENAME=$GATHERED_DIR/$README_DIRNAME.md
        mkdir -p `dirname $DOCS_FILENAME`
        ln -s $ROOT_DIR/$README_FILENAME $DOCS_FILENAME
    fi
done

# Gather docs from examples/*.ipynb and add YAML front-matter.
for NOTEBOOK_FILENAME in $(find examples -depth -iname "*.ipynb"); do
    DOCS_FILENAME=$GATHERED_DIR/$NOTEBOOK_FILENAME
    mkdir -p `dirname $DOCS_FILENAME`
    python scripts/copy_notebook.py $NOTEBOOK_FILENAME $DOCS_FILENAME
done
