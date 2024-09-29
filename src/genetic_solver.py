import modules


class GeneticSolver:

    def __init__(
        self,
        population_size: int,
        chromosome_length: int,
        generation_func,
        fitness_func,
        selection_func,
        mating_func,
        crossover_func,
        mutation_func,
        mutation_rate,
        replace_func,
    ) -> None:
        self.generator = modules.Generator(population_size, generation_func)
        self.evaluator = modules.Evaluator(fitness_func)
        self.selector = modules.Selector(selection_func)
        self.mater = modules.Mater(mating_func)
        self.crossoverator = modules.Crossoverator(crossover_func)
        self.mutator = modules.Mutator(mutation_func, mutation_rate)
        self.replacer = modules.Replacer(replace_func)


if __name__ == "__main__":
    pass
