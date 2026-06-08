#!/bin/bash




for n in 15; do
  for s in 0.2 0.1 0.01 0.001 ; do
    for r in 0; do
    python GT/main_inverse.py $n $s $r
    done
  done  
done
echo PIGP done

