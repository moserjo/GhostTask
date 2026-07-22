#!/bin/bash


echo starting
for n in 3 5 10 25; do
  for s in 0.001 0.01 0.1 0.2; do
    for r in 0 1; do
    python optimisation/GT/main.py $n $s $r "withAl"
    python optimisation/PIGP/main.py $n $s $r "withAl"
    echo doing $s $n
    done
  done  
done
echo done


