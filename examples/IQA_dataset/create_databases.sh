#!/bin/bash
# Create databases for IQA CNN training and validation
# The first argument is the number of train-val databases to generate
for i in $(seq 1 1 $1)
do
   examples/IQA_dataset/create_db.sh "$i"
done
