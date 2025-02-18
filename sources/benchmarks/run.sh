#!/bin/bash


# population
python benchmarks/deap_pop.py MLPClassifier --suffix=final5 --log=info
python benchmarks/ppga_pop.py MLPClassifier --suffix=final5 --log=info

python benchmarks/deap_pop.py SVC --suffix=final5 --log=info
python benchmarks/ppga_pop.py SVC --suffix=final5 --log=info

python benchmarks/deap_pop.py RandomForestClassifier --suffix=final5 --log=info
python benchmarks/ppga_pop.py RandomForestClassifier --suffix=final5 --log=info
