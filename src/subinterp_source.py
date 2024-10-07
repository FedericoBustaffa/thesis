import pickle

import numpy as np

# from loguru import logger
from ppga import Crossoverator, Evaluator, Mutator

crossoverator = Crossoverator(pickle.loads(r_chann.recv()))
mutator = Mutator(pickle.loads(r_chann.recv()))
evaluator = Evaluator(pickle.loads(r_chann.recv()))

while True:
    couples = pickle.loads(r_chann.recv())
    if couples is None:
        break

    offsprings = crossoverator.perform(couples)
    offsprings = mutator.perform(offsprings)
    scores = evaluator.perform(offsprings)

    s_chann.send(pickle.dumps(offsprings))
    s_chann.send(pickle.dumps(scores))

# logger.trace(f"{mp.current_process().name} terminated")
