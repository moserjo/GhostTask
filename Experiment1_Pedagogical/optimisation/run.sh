#!/bin/bash




for n in 10 15; do
  for s in "yesextrapoint" "noextrapoint" ; do
    python GT/main_forward.py $n $s $r
    python PIGP/main_forward.py $n $s $r
  done  
done
echo DONE

