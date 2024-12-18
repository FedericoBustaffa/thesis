#!/bin/bash

python -m scoop -n 2 benchmarks/deap_scoop.py 2
python -m scoop -n 4 benchmarks/deap_scoop.py 4
python -m scoop -n 8 benchmarks/deap_scoop.py 8
python -m scoop -n 16 benchmarks/deap_scoop.py 16
python -m scoop -n 32 benchmarks/deap_scoop.py 32

python benchmarks/deap_mp.py

