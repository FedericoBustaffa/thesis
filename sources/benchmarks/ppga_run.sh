#!/bin/bash


python benchmarks/ppga_bm.py MLPClassifier --suffix=pop --log=info
python benchmarks/ppga_bm.py SVC --suffix=pop --log=info
python benchmarks/ppga_bm.py RandomForestClassifier --suffix=pop --log=info

python benchmarks/ppga_bm.py MLPClassifier --suffix=feature --log=info
python benchmarks/ppga_bm.py SVC --suffix=feature --log=info
python benchmarks/ppga_bm.py RandomForestClassifier --suffix=feature --log=info