import numpy as np

from tea_pymoo.callbacks.data_collector import DataCollector


class Performance_Callback(DataCollector):

    def __init__(self, additional_run_info=None) -> None:
        '''
        This callback saves the fitness performance of a single-objective algorithm.

        Saves the generation, minimum fitness, mean fitness, median fitness, and maximum fitness.
        '''
        data_keys = ["generation", "min_f", "mean_f", "median_f", "max_f"]
        
        super().__init__(data_keys=data_keys, filename="performance", additional_run_info=additional_run_info)

    def notify(self, algorithm):
        super().handle_additional_run_info()

        generation = algorithm.n_gen
        population = algorithm.pop
        fitness = population.get("F")

        self.data["generation"].append(generation)
        self.data["min_f"].append(fitness.min())
        self.data["mean_f"].append(fitness.mean())
        self.data["median_f"].append(np.median(fitness))
        self.data["max_f"].append(fitness.max())

