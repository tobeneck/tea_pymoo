import numpy as np

from tea_pymoo.callbacks.data_collector import DataCollector

from tea_pymoo.tracing.t_sampling import TracingTypes


class Counting_Impact_Callback(DataCollector):

    def __init__(self, initial_popsize, tracing_type=TracingTypes.TRACE_ID, additional_run_info=None) -> None:
        '''
        This callback saves the counting impact of the initial population for each generation.

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

        data_keys = ["generation"]
        for i in range(initial_popsize):
            data_keys.append("traceID_"+str(i+1))
        data_keys.append("traceID_m")

        super().__init__(data_keys=data_keys, filename="counting_impact", additional_run_info=additional_run_info)
          
    def print_traceID_counting_impact(self, population):
        counting_impact = np.zeros(self.max_traceID+1)
        T = population.get("T")
        for i in range(self.max_traceID):
            counting_impact[i] = (T == i).sum() / self.max_traceID
        counting_impact[-1] = (T < 0).sum() / self.max_traceID

        return counting_impact

    def print_traceList_counting_impact(self, population):
        
        #calculate the impact
        counting_impact = np.zeros(shape=(self.max_traceID + 1 )) #TODO: accumulates the impact of mutation to -1
    
        for i in range(len(population)): #iterate over every individual
            for g in population[i].get("T"): #iterate over the genes of the current individual
                currentTraceList = g
                genome_length = len(population[i].X)
                for tt in currentTraceList.get_all():
                    currentTraceID = tt.traceID
                    if currentTraceID >= 0:
                        counting_impact[currentTraceID] = counting_impact[currentTraceID] + (tt.influenceFactor / genome_length)
                    if currentTraceID < 0:
                        counting_impact[self.max_traceID] = counting_impact[self.max_traceID] + (tt.influenceFactor / genome_length)
        
        return counting_impact

    def print_traceVector_counting_impact(self, population):        
        #calculate the impact
        counting_impact = np.zeros( self.max_traceID + 1 )
        T = population.get("T")
        X = population.get("X")

        counting_impact = T.sum(axis=0).sum(axis=0) / (X.shape[0] * X.shape[1])
        return counting_impact

    def notify(self, algorithm):
        super().handle_additional_run_info()

        generation = algorithm.n_gen
        population = algorithm.pop

        counting_impact = []
        if self.tracing_type == TracingTypes.NO_TRACING:
            return
        elif self.tracing_type == TracingTypes.TRACE_ID:
            counting_impact = self.print_traceID_counting_impact(population)
        elif self.tracing_type == TracingTypes.TRACE_LIST:
            counting_impact = self.print_traceList_counting_impact(population)
        elif self.tracing_type == TracingTypes.TRACE_VECTOR:
            counting_impact = self.print_traceVector_counting_impact(population)
        
        for key in self.data.keys():
            if key == "generation":
                self.data[key].append(generation)
            elif key == "traceID_m":
                #print(counting_impact[-1])
                self.data[key].append(counting_impact[-1])
            elif key[:7] == "traceID":
                trace_index = int( key.split("_")[1] ) - 1 #there is no traceID 0, we shift everything to 1
                self.data[key].append(counting_impact[trace_index])

