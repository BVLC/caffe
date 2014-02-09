#!/bin/bash
# Usage parselog.sh caffe.log [init_iter=0] [test_interval=1000]
# It creates two files one caffe.log.test that contains the loss and test accuracy of the test and
# another one caffe.log.loss that contains the loss computed during the training
# Use init_iter when the log start from a previous snapshot
# Use test_interval when the testing interval is different than 1000
if [ "$#" -lt 1 ]
then
echo "Usage parselog.sh /path_to/caffe.log [init_iter=0] [test_interval=1000]"
fi
if [ "$#" -gt 1 ]
then
 INIT=$2
else
 INIT=0
fi
if [ "$#" -gt 2 ]
then
 INC=$3
else
 INC=1000
fi
LOG=`basename $1`
grep 'Test score #0' $1 | awk -v init=$INIT -v inc=$INC '{print (NR*inc+init), $8}' > aux0.txt
grep 'Test score #1' $1 | awk '{print $8}' > aux1.txt
grep ' loss =' $1 | awk '{print $6,$9}' | sed 's/,//g' | column -t > $LOG.loss 
paste aux0.txt aux1.txt | column -t > $LOG.test
rm aux0.txt aux1.txt