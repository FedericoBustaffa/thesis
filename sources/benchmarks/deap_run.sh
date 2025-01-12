#!/bin/bash


# python benchmarks/deap_mp.py MLPClassifier 0.0 --log=info
# python benchmarks/deap_mp.py SVC 0.0 --log=info
# python benchmarks/deap_mp.py RandomForestClassifier 0.0 --log=info

python benchmarks/deap_bm.py MLPClassifier --suffix=pop --log=info
python benchmarks/deap_bm.py SVC --suffix=pop --log=info
python benchmarks/deap_bm.py RandomForestClassifier --suffix=pop --log=info

python benchmarks/deap_bm.py MLPClassifier --suffix=feature --log=info
python benchmarks/deap_bm.py SVC --suffix=feature --log=info
python benchmarks/deap_bm.py RandomForestClassifier --suffix=feature --log=info