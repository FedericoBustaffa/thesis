#!/bin/bash


# DEAP TEST
# population
python benchmarks/deap_pop.py MLPClassifier --suffix=final --log=info
python benchmarks/deap_pop.py SVC --suffix=final --log=info
python benchmarks/deap_pop.py RandomForestClassifier --suffix=final --log=info

# feature
python benchmarks/deap_ft.py MLPClassifier --suffix=final --log=info
python benchmarks/deap_ft.py SVC --suffix=final --log=info
python benchmarks/deap_ft.py RandomForestClassifier --suffix=final --log=info

# PPGA TEST
# population
python benchmarks/ppga_pop.py MLPClassifier --suffix=final --log=info
python benchmarks/ppga_pop.py SVC --suffix=final --log=info
python benchmarks/ppga_pop.py RandomForestClassifier --suffix=final --log=info

# feature
python benchmarks/ppga_ft.py MLPClassifier --suffix=final --log=info
python benchmarks/ppga_ft.py SVC --suffix=final --log=info
python benchmarks/ppga_ft.py RandomForestClassifier --suffix=final --log=info
