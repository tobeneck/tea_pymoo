import numpy as np

from tea_pymoo.callbacks.data_collector import DataCollector

from tea_pymoo.tracing.t_sampling import TracingTypes



def get_entropy_vector(population):
    '''
    Calculates the entropy for each traceID in the population.
    '''
    T = population.get("T")  # shape (n_individuals, n_traceIDs)
    n_individuals, n_var, n_traceIDs = T.shape

    entropy_vector = np.zeros(n_var)

    for j in range(n_var):
        trace_column = T[:, j]
        unique_cols = np.unique(trace_column, axis=0)
        #print("unique cols", unique_cols)

        for col in unique_cols:
            matches = np.all(trace_column == col, axis=1) #returns how many times col appears in the trace column
            p_col = matches.sum() / n_individuals
            entropy_vector[j] += ( p_col * np.log2(p_col) )

    return -entropy_vector #don't forget to add -1



class Entropy_Impact_Inds_Callback(DataCollector):

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

    def print_traceVector_entropy_impact(self, ind, entropy):        
        entropy_impact = np.zeros( self.max_traceID + 1 )
        T = ind.get("T")

        #scale T with the entropy
        T = T * entropy[:, np.newaxis]

        entropy_impact = ( T.sum(axis=0)) / ( entropy.sum() ) # essentially, we need to scale with the sum of the entropie for each row (I think)
        return entropy_impact
    
    def notify(self, algorithm):

        if algorithm.problem.n_obj != 1:
            raise NotImplementedError("Entropy impact for each ind individually is currently only implemented for single objective problems.")

        generation = algorithm.n_gen
        population = algorithm.pop
        if self.optimal_inds_only:
            population = algorithm.opt
        
        #calc the entropy
        entr = get_entropy_vector(population)

        for i in range(0, len(population)):
            super().handle_additional_run_info()
            entropy_impact = []
            
            if self.tracing_type == TracingTypes.NO_TRACING:
                return
            elif self.tracing_type == TracingTypes.TRACE_ID:
                raise NotImplementedError("Entropy impact for each ind individually is currently only implemented for trace vector representation.")
            elif self.tracing_type == TracingTypes.TRACE_LIST:
                raise NotImplementedError("Entropy impact for each ind individually is currently only implemented for trace vector representation.")
            elif self.tracing_type == TracingTypes.TRACE_VECTOR:
                entropy_impact = self.print_traceVector_entropy_impact(population[i], entr)

            for key in self.data.keys():
                if key == "generation":
                    self.data[key].append(generation)
                elif key == "individual":
                    self.data[key].append(i)
                elif key == "traceID_m":
                    self.data[key].append(entropy_impact[-1])
                elif key[:7] == "traceID":
                    trace_index = int( key.split("_")[1] ) - 1 #there is no traceID 0, we shift everything to 1
                    self.data[key].append(entropy_impact[trace_index])

