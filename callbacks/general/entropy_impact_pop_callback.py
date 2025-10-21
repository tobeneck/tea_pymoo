import numpy as np

from tea_pymoo.tracing.t_sampling import TracingTypes

from tea_pymoo.callbacks.general.entropy_impact_inds_callback import Entropy_Impact_Inds_Callback, get_entropy_vector


class Entropy_Impact_Pop_Callback(Entropy_Impact_Inds_Callback):

    def __init__(self, initial_popsize, tracing_type=TracingTypes.TRACE_ID, additional_run_info=None, optimal_inds_only=True, filename="entropy_impact_pop") -> None:
        '''
        This callback saves the entropy impact of the initial population for each generation, accumulated for the whole population. It essentially wraps the Entropy_Impact_Inds_Callback.

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
        super().__init__(
            initial_popsize=initial_popsize, 
            tracing_type=tracing_type, 
            additional_run_info=additional_run_info,
            optimal_inds_only=optimal_inds_only,
            filename=filename,
            data_keys=data_keys
            )

    def notify(self, algorithm):
        super().handle_additional_run_info()

        if algorithm.problem.n_obj != 1:
            raise NotImplementedError("Entropy impact for each ind individually is currently only implemented for single objective problems.")

        generation = algorithm.n_gen
        population = algorithm.pop
        if self.optimal_inds_only:
            population = algorithm.opt

        entropy_impact = np.zeros( self.max_traceID + 1 )

        entr = get_entropy_vector(population)

        for i in range(0, len(population)):
            if self.tracing_type == TracingTypes.NO_TRACING:
                return
            elif self.tracing_type == TracingTypes.TRACE_ID:
                raise NotImplementedError("Entropy impact for each ind individually is currently only implemented for trace vector representation.")
            elif self.tracing_type == TracingTypes.TRACE_LIST:
                raise NotImplementedError("Entropy impact for each ind individually is currently only implemented for trace vector representation.")
            elif self.tracing_type == TracingTypes.TRACE_VECTOR:
                current_ind_entropy_impact = self.print_traceVector_entropy_impact(i, population, entr)

                entropy_impact += (current_ind_entropy_impact / len(population) )

        for key in self.data.keys():
            if key == "generation":
                self.data[key].append(generation)
            elif key == "traceID_m":
                self.data[key].append(entropy_impact[-1])
            elif key[:7] == "traceID":
                trace_index = int( key.split("_")[1] ) - 1 #there is no traceID 0, we shift everything to 1
                self.data[key].append(entropy_impact[trace_index])


