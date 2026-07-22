#!/bin/bash



for n in 3 5 7; do
  for s in 0.001 0.01 0.1 0.2; do
    for d in "standard" "extra_npoints" "extra_npoints_extra_sigma"; do
    python GT/main_inverse.py $n $s 4 $d
    python PIGP/main_inverse.py $n $s 4 $d
   
    echo doing $n $s $d
    done
  done  
done
echo Bayes done