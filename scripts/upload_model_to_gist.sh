#!/bin/bash

# Check for valid directory
DIRNAME=$1
if [ ! -f $DIRNAME/readme.md ]; then
    echo "usage: upload_model_to_gist.sh <dirname>"
    echo "  <dirname>/readme.md must exist"
fi
cd $DIRNAME
FILES=`find . -type f -maxdepth 1 ! -name "*.caffemodel*" | xargs echo`

# Check for gist tool.
gist -v >/dev/null 2>&1 || { echo >&2 "I require 'gist' but it's not installed. Do 'gem install gist'."; exit 1; }

NAME=`sed -n 's/^name:[[:space:]]*//p' readme.md`
if [ -z "$NAME" ]; then
    echo "  <dirname>/readme.md must contain name field in the front-matter."
fi

GIST=`sed -n 's/^gist_id:[[:space:]]*//p' readme.md`
if [ -z "$GIST" ]; then
    echo "Uploading new Gist"
    gist -p -d "$NAME" $FILES
else
    echo "Updating existing Gist, id $GIST"
    gist -u $GIST -d "$NAME" $FILES
fi

RESULT=$?
if [ $RESULT -eq 0 ]; then
    echo "You've uploaded your model!"
    echo "Don't forget to add the gist_id field to your <dirname>/readme.md now!"
    echo "Run the command again after you do that, to make sure the Gist id propagates."
    echo ""
    echo "And do share your model over at https://github.com/BVLC/caffe/wiki/Model-Zoo"
else
    echo "Something went wrong!"
fi
