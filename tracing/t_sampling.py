import numpy as np

from pymoo.core.sampling import Sampling
from pymoo.core.population import Population


from .tracing_types import TracingTypes, TraceList, TraceTuple

class T_Sampling(Sampling):

    def __init__(self, sampling, tracing_type=TracingTypes.NO_TRACING, **kwargs):
        '''
        This class acts as a wrapper for another sampling operator and just adds the tracing capabilitys on top.

        Parameters
        ----------

        sampling : pymoo.core.sampling.Sampling
            The actual sampling operator to be used
            
        tracing_type : TracingType
            Indicator for the type of representation used for tracing
        '''
        super().__init__()
        self.sampling = sampling
        self.tracing_type = tracing_type
    
    def do(self, problem, n_samples, **kwargs):
        return self.do(problem, n_samples, seeds=None, **kwargs)

    def do(self, problem, n_samples, seeds=None, **kwargs):
        '''
        This method calls the in the constructor provided sampler and adds the tracing information based on the specified trace type.

        Also, it provides the option to seed individuals in the initial population, which is usefull to avoid re-calculation of the trace information.
        The seeds will always be placed at the beginning of the population.

        Parameters:
        -----------
        problem : pymoo.core.problem
            The problem for which the samples should be created.
        n_samples : int
            The number of samples to be created.
        seeds : np.array(2D) (optional)
            Individuals to seed the returned population with.
        
        Returns:
        --------
        Population : pymoo.core.population
            The sampled population including the tracing information.
        '''

        #handle the seeding:
        if seeds is not None and len(seeds) > n_samples:
            raise Exception("You can't use more seeds than you wand to sample!")
        elif seeds is not None and len(seeds) == n_samples:
            val = seeds
        elif seeds is not None and len(seeds) < n_samples:
            val = np.concatenate((seeds, self.sampling._do(problem, n_samples-len(seeds), **kwargs)), axis=0)
        else:
            val = self.sampling._do(problem, n_samples)

        #create the offspring flags
        IsOff = np.zeros(len(val), dtype=bool) #in the initial population, all individuals are marked to be non offspring (or parents)

        ParentIDs = [] # the IDs of the parent from the previous generation #TODO: implement

        Birthday = 0 # the generation the individual was generated in #TODO: implement

        # build the tracing information based on the trace types
        if self.tracing_type == TracingTypes.NO_TRACING:
            return Population.new("X", val, "IsOff", IsOff)
        
        if self.tracing_type == TracingTypes.TRACE_ID:
            T = np.zeros( (n_samples, problem.n_var) )
            for i in range(0, n_samples):
                T[i] = np.zeros(problem.n_var) + i + 1
            return Population.new("X", val, "IsOff", IsOff, "T", T, "IsOff", IsOff)
        
        if self.tracing_type == TracingTypes.TRACE_LIST:
            T = []
            for indIndex in range(0, n_samples): #trace lists for all individuals
                curr_T = []
                for genomeIndex in range(0, problem.n_var): #trace list for each gene
                    curr_trace_list = np.array( [ TraceTuple(indIndex + 1, 1.0) ], dtype=TraceTuple )
                    curr_T.append( TraceList(curr_trace_list) )
                currT = np.array(curr_T, dtype=TraceList)
                T.append(curr_T)
            T = np.array(T, dtype=TraceTuple)
            return Population.new("X", val, "IsOff", IsOff, "T", T, "IsOff", IsOff)
        
        if self.tracing_type == TracingTypes.TRACE_VECTOR:
            T = np.zeros( (n_samples, problem.n_var, n_samples + 1) ) #TODO: this is wrong!
            for i in range(0, n_samples):
                T[i, :, i] = 1.0
            return Population.new("X", val, "IsOff", IsOff, "T", T, "IsOff", IsOff)
