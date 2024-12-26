#!/bin/bash

source .env/bin/activate

models=('RandomForestClassifier' 'SVC' 'MLPClassifier')
workers=(1 2 4 8 16 32)
populations=(1000 2000 4000 8000 16000)

for model in ${models[@]}; do
    for worker in ${workers[@]}; do
        for population in ${populations[@]}; do
            echo "Running $model with $worker workers and $population population"
            python benchmarks/deap_mp.py $model $population $worker --log=info
        done
    done
done
