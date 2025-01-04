#!/bin/bash


python benchmarks/deap_mp.py MLPClassifier 0.0 --log=info
python benchmarks/deap_mp.py SVC 0.0 --log=info
python benchmarks/deap_mp.py RandomForestClassifier 0.0 --log=info

python benchmarks/deap_mp.py MLPClassifier 0.1 hof --log=info
python benchmarks/deap_mp.py SVC 0.1 hof --log=info
python benchmarks/deap_mp.py RandomForestClassifier 0.1 hof --log=info
