#!/bin/bash
# Create databases for IQA CNN training and validation
# The first argument is the number of train-val databases to generate
# The second argument is the distortion type
ITER_START=$1
ITER_END=$2
DISTORTION_TYPE=$3
DATABASE_DIR=examples/IQA_dataset/$DISTORTION_TYPE
if [ ! -d $DATABASE_DIR ]; then
  mkdir $DATABASE_DIR
fi
for i in $(seq $ITER_START 1 $ITER_END)
do
   examples/IQA_dataset/create_db.sh "$i" "$DISTORTION_TYPE"
done
