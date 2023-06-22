from tea_pymoo.callbacks.data_collector import DataCollector



class Genome_Callback(DataCollector):

    def __init__(self, n_var, additional_run_info=None, filename="genome") -> None:
        '''
        this class saves the gitness values, rank and genome values of the individuals
        Parameters
            n_var : int
                The genome size.
            run_number : int
                The current number of run for the data.
        ---------- 
        '''
        self.filename = "genome"

        data_keys = ["generation", "individual"]
        for i in range(0, n_var):
            data_keys.append("g_"+str(i+1))

        super().__init__(additional_run_info=additional_run_info, data_keys=data_keys, filename=filename)

    def notify(self, algorithm):

        generation = algorithm.n_gen
        population = algorithm.pop
        
        #save the population data
        for i in range(0, len(population)):
            super().handle_additional_run_info()
            for key in self.data.keys():
                if key == "generation":
                    self.data[key].append(generation)
                elif key == "individual":
                    self.data[key].append(i)
                elif key[:2] == "g_":
                    genome_index = int( key.split("_")[1] ) - 1
                    self.data[key].append(population[i].X[genome_index])