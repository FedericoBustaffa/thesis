#!/bin/bash


python benchmarks/parallelism.py MLPClassifier --log=info
python benchmarks/parallelism.py SVC --log=info
python benchmarks/parallelism.py RandomForestClassifier --log=info
