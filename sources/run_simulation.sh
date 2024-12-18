#!/bin/bash

workers=(2 4 8 16 32)

for w in ${workers[@]}
do
    python -m scoop -n "$w" benchmarks/deap_scoop.py $w
done

python benchmarks/deap_mp.py
