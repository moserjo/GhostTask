#!/bin/bash



for n in 3 5 10; do
  for s in 0.01 0.1 0.2; do
    python main_Bayes.py $n $s 
    echo doing
  done  
done
echo Bayes done
