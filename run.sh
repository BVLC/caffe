#!/bin/bash

prototxt="models/intel_optimized_models/resnet50_v1/resnet50_int8_perf.prototxt"
iterations=1200
skip=200
NUM_CORE=56
s_BS=10
e_BS=10
INSTANCE=2

rm -rf logs
mkdir logs
rm -f ./temp.sh


for j in $INSTANCE 
do

  for ((bs = $s_BS; bs <= $e_BS; bs += 1))
  do

    if [ "$bs" -gt "10" ] && [ "$j" -gt "14" ];then
      continue
    fi

    sed -i "1,/dim/s/dim.*/dim:$bs/" $prototxt

    rm -rf logs/bs${bs}_inst$j
    mkdir logs/bs${bs}_inst$j

    NUM_INSTANCE=$j
    INTERVAL=$(($NUM_CORE / $NUM_INSTANCE))
    export KMP_HW_SUBSET=1t
    export KMP_AFFINITY=granularity=fine,compact,1,0
    export OMP_NUM_THREADS=$INTERVAL

    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

    for ((i = 0; i < $NUM_CORE; i += $INTERVAL))
    do
      end=$(($i + $INTERVAL - 1))
      NUMA_NUM=$(($i/28))
     echo "KMP_HW_SUBSET=1T KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=$INTERVAL numactl -C $i-$end -m $NUMA_NUM ./build/tools/caffe time --forward_only --phase TEST -model $prototxt -iterations $iterations >logs/bs${bs}_inst${j}/numa$i 2>&1 &" >>temp.sh

    done

    echo "wait" >>temp.sh
    bash ./temp.sh

    rm -f logs/bs${bs}_inst${j}/latency

    for ((i = 0; i < $NUM_CORE; i += $INTERVAL))
    do
      grep "###" logs/bs${bs}_inst${j}/numa$i |awk -F' ' '{ print $5 }'|awk -F'#' '{ print $4 }' | awk -F':' -v iter=$iterations -v s=$skip 'BEGIN {sum=0} {if($1>=s && $1<(iter-s)) sum+=$2;} END{ print (sum /(iter-s-s))'}  >> logs/bs${bs}_inst${j}/latency
    done

    latency=$(cat logs/bs${bs}_inst${j}/latency | awk 'BEGIN {sum=0} {sum+=$1} END{print (sum / NR)}')

    fps=`echo "1000 * $bs * $j / $latency" | bc `
    echo "bs: $bs instance: $j latency: $latency fps: $fps"

  done
done


