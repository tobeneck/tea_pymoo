import numpy as np

from tea_pymoo.tracing.t_sampling import TracingTypes

from tea_pymoo.callbacks.data_collector import DataCollector


class Fitness_Impact_Pop_Callback(DataCollector):

    def __init__(self, initial_popsize, tracing_type=TracingTypes.TRACE_ID, additional_run_info=None, optimal_inds_only=True, filename="fitness_impact_pop") -> None:
        '''
        This callback saves the fitness impact of the initial population for each generation, accumulated for the whole population. It essentially wraps the Fitness_Impact_Inds_Callback.

        Parameters:
        -----------
        initial_popsize : int
            The size of the initial population (/the number of traceIDs).
        tracing_type : TracingTypes
            The type of tracing used.
        additional_run_info : dict
            An optional dictionary of additional infos for the config of this run. Usefull to save data like or the run number or other inportant configurations.
        '''

        data_keys = ["generation"]
        for i in range(initial_popsize):
            data_keys.append("traceID_"+str(i+1))
        data_keys.append("traceID_m")

        self.tracing_type = tracing_type
        self.max_traceID = initial_popsize
        self.additional_keys = additional_run_info
        self.optimal_inds_only = optimal_inds_only

        super().__init__(data_keys=data_keys, filename=filename, additional_run_info=additional_run_info)
        
    def print_traceVector_fitness_impact(self, ind_idx: int, population: np.ndarray) -> np.ndarray:
        fitness_impact = np.zeros( self.max_traceID + 1 )
        
        trace_vector = population[ind_idx].get("T") #shape: (n_var, n_traceIDs)

        worst_fitness = population.get("F").max()
        fitness_distances = np.abs( worst_fitness - population.get("F").flatten() ) + 1


        fitness_impact = trace_vector.sum(axis=0) * fitness_distances[ind_idx]
        fitness_impact = fitness_impact / fitness_impact.sum() # normalize
        return fitness_impact

    def notify(self, algorithm):
        super().handle_additional_run_info()

        if algorithm.problem.n_obj != 1:
            raise NotImplementedError("Fitness impact for each ind individually is currently only implemented for single objective problems.")

        generation = algorithm.n_gen
        population = algorithm.pop
        if self.optimal_inds_only:
            population = algorithm.opt

        fitnes_differences = np.abs( population.get("F").max() - population.get("F").flatten() ) + 1

        fitness_impact = np.zeros( self.max_traceID + 1 )
        for i in range(0, len(population)):
            if self.tracing_type == TracingTypes.NO_TRACING:
                return
            elif self.tracing_type == TracingTypes.TRACE_ID:
                raise NotImplementedError("Fitness impact for each ind individually is currently only implemented for trace vector representation.")
            elif self.tracing_type == TracingTypes.TRACE_LIST:
                raise NotImplementedError("Fitness impact for each ind individually is currently only implemented for trace vector representation.")
            elif self.tracing_type == TracingTypes.TRACE_VECTOR:
                current_ind_fitness_impact = self.print_traceVector_fitness_impact(i, population)
                current_ind_fitness_impact = current_ind_fitness_impact * fitnes_differences[i]
                fitness_impact += (current_ind_fitness_impact ) #normalization comes later!
        
        fitness_impact = fitness_impact / fitnes_differences.sum() # normalize overall

        for key in self.data.keys():
            if key == "generation":
                self.data[key].append(generation)
            elif key == "traceID_m":
                self.data[key].append(fitness_impact[-1])
            elif key[:7] == "traceID":
                trace_index = int( key.split("_")[1] ) - 1 #there is no traceID 0, we shift everything to 1
                self.data[key].append(fitness_impact[trace_index])


