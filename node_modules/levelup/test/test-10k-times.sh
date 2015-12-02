#!/bin/bash

cd ..

for (( i=1; i<=10000; i++ ))
do
    echo "Running test $i .."
    npm test
    if [ $? -ne 0 ]; then
        break
    fi
done
