#!/bin/bash



for n in 3 5 10 25; do
  for s in 0.001 0.01 0.1 0.2; do
    python main_PIGP.py $n $s 
    python main_GT.py $n $s 
    echo doing
  done  
done
echo Bayes done
