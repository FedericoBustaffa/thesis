#!/bin/bash


# DEAP population
# python benchmarks/deap_pop.py $1 --suffix=final2 --log=info

# DEAP feature
# python benchmarks/deap_ft.py $1 --suffix=final2 --log=info

# PPGA population
python benchmarks/ppga_pop.py $1 --suffix=final2 --log=info

# PPGA feature
python benchmarks/ppga_ft.py $1 --suffix=final2 --log=info
