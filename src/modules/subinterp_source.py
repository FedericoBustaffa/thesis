import pickle

# from loguru import logger

# from modules import Crossoverator, Evaluator, Mutator

crossoverator = pickle.loads(r_chann.recv())
mutator = pickle.loads(r_chann.recv())
evaluator = pickle.loads(r_chann.recv())

while True:
    couples = pickle.loads(r_chann.recv())
    if couples is None:
        break

    offsprings = crossoverator.perform(couples)
    offsprings = mutator.perform(offsprings)
    scores = evaluator.perform(offsprings)

    s_chann.send(pickle.dumps((offsprings, scores)))

# logger.trace(f"{mp.current_process().name} terminated")
