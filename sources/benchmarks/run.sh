#!/bin/bash


# DEAP TEST
# population
python benchmarks/deap_pop.py MLPClassifier --suffix=pop --log=info
python benchmarks/deap_pop.py SVC --suffix=pop --log=info
python benchmarks/deap_pop.py RandomForestClassifier --suffix=pop --log=info

# feature
python benchmarks/deap_ft.py MLPClassifier --suffix=feature --log=info
python benchmarks/deap_ft.py SVC --suffix=feature --log=info
python benchmarks/deap_ft.py RandomForestClassifier --suffix=feature --log=info

# PPGA TEST
# population
python benchmarks/ppga_pop.py MLPClassifier --suffix=pop --log=info
python benchmarks/ppga_pop.py SVC --suffix=pop --log=info
python benchmarks/ppga_pop.py RandomForestClassifier --suffix=pop --log=info

# feature
python benchmarks/ppga_ft.py MLPClassifier --suffix=feature --log=info
python benchmarks/ppga_ft.py SVC --suffix=feature --log=info
python benchmarks/ppga_ft.py RandomForestClassifier --suffix=feature --log=info