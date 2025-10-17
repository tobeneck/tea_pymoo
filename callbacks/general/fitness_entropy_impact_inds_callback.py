import numpy as np

from tea_pymoo.callbacks.data_collector import DataCollector

from tea_pymoo.tracing.t_sampling import TracingTypes

from tea_pymoo.callbacks.general.entropy_impact_inds_callback import get_entropy_vector



class Fitness_Entropy_Impact_Inds_Callback(DataCollector):

    def __init__(self, initial_popsize, tracing_type=TracingTypes.TRACE_ID, additional_run_info=None, optimal_inds_only=True, filename="entropy_impact_inds", data_keys=None) -> None:
        '''
        This callback saves the entropy impact of the initial population for each generation, for each individual separately.

        Parameters:
        -----------
        initial_popsize : int
            The size of the initial population (/the number of traceIDs).
        tracing_type : TracingTypes
            The type of tracing used.
        additional_run_info : dict
            An optional dictionary of additional infos for the config of this run. Usefull to save data like or the run number or other inportant configurations.
        '''

        self.tracing_type = tracing_type
        self.max_traceID = initial_popsize
        self.additional_keys = additional_run_info
        self.optimal_inds_only = optimal_inds_only
        if data_keys is None:
            data_keys = ["generation", "individual"]
            for i in range(initial_popsize):
                data_keys.append("traceID_"+str(i+1))
            data_keys.append("traceID_m")

        super().__init__(data_keys=data_keys, filename=filename, additional_run_info=additional_run_info)

    def print_traceVector_fitness_entropy_impact(self, ind, entropy, worst_fitness):        
        fitness_entropy_impact = np.zeros( self.max_traceID + 1 )
        T = ind.get("T")
        X = ind.get("X")

        #scale T with the entropy
        T = T * entropy[:, np.newaxis]

        fd = 1 + np.abs(worst_fitness - ind.get("F")[0])

        fitness_entropy_impact = ( T.sum(axis=0) * fd) / ( entropy.sum() * (len(X) * fd) ) # essentially, we need to scale with the sum of the entropie for each row (I think)
        return fitness_entropy_impact
    
    def notify(self, algorithm):

        if algorithm.problem.n_obj != 1:
            raise NotImplementedError("Fitness_Entropy impact for each ind individually is currently only implemented for single objective problems.")

        generation = algorithm.n_gen
        population = algorithm.pop
        if self.optimal_inds_only:
            population = algorithm.opt
        
        #calc the entropy
        entr = get_entropy_vector(population)
        worst_fitness = population.get("F").max()

        for i in range(0, len(population)):
            super().handle_additional_run_info()
            fitness_entropy_impact = []
            
            if self.tracing_type == TracingTypes.NO_TRACING:
                return
            elif self.tracing_type == TracingTypes.TRACE_ID:
                raise NotImplementedError("Fitness_Entropy impact for each ind individually is currently only implemented for trace vector representation.")
            elif self.tracing_type == TracingTypes.TRACE_LIST:
                raise NotImplementedError("Fitness_Entropy impact for each ind individually is currently only implemented for trace vector representation.")
            elif self.tracing_type == TracingTypes.TRACE_VECTOR:
                fitness_entropy_impact = self.print_traceVector_fitness_entropy_impact(population[i], entr, worst_fitness)

            for key in self.data.keys():
                if key == "generation":
                    self.data[key].append(generation)
                elif key == "individual":
                    self.data[key].append(i)
                elif key == "traceID_m":
                    self.data[key].append(fitness_entropy_impact[-1])
                elif key[:7] == "traceID":
                    trace_index = int( key.split("_")[1] ) - 1 #there is no traceID 0, we shift everything to 1
                    self.data[key].append(fitness_entropy_impact[trace_index])

