#!/bin/bash


# DEAP population
python benchmarks/deap_pop.py MLPClassifier --suffix=final2 --log=info
python benchmarks/deap_pop.py RandomForestClassifier --suffix=final2 --log=info
python benchmarks/deap_pop.py SVC --suffix=final2 --log=info

# DEAP feature
python benchmarks/deap_ft.py RandomForestClassifier --suffix=final2 --log=info
python benchmarks/deap_ft.py MLPClassifier --suffix=final2 --log=info
python benchmarks/deap_ft.py SVC --suffix=final2 --log=info

