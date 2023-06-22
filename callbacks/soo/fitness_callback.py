import numpy as np

from tea_pymoo.callbacks.data_collector import DataCollector



class Fitness_Callback(DataCollector):

    def __init__(self, additional_run_info=None)  -> None:
        data_keys = ["generation", "individual", "f"]

        super().__init__(data_keys=data_keys, filename="fitness", additional_run_info=additional_run_info)

    def notify(self, algorithm):

        generation = algorithm.n_gen
        fitness = algorithm.pop.get("F")

        for ind_id in range(len(fitness)):
            self.data["generation"].append(generation)
            self.data["f"].append(fitness[ind_id])
            self.data["individual"].append(ind_id)
            super().handle_additional_run_info()