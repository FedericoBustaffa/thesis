#!/bin/bash


python benchmarks/ppga.py MLPClassifier --suffix=pop --log=info
python benchmarks/ppga.py SVC --suffix=pop --log=info
python benchmarks/ppga.py RandomForestClassifier --suffix=pop --log=info

python benchmarks/ppga.py MLPClassifier --suffix=feature --log=info
python benchmarks/ppga.py SVC --suffix=feature --log=info
python benchmarks/ppga.py RandomForestClassifier --suffix=feature --log=info