import numpy as np

from .t_sampling import TracingTypes, TraceList, TraceTuple

from pymoo.core.mutation import Mutation
from pymoo.core.population import Population


class T_Mutation(Mutation):

    #calculates the new trace lists from the parents values as well as the new child  values
    def calculateOffspringTraceLists(self, parents_X, children_X, parents_T):
        """
        this method calculates the new trace lists for the offsprings of the mutation operation based on the parent and offspring gene value.
        Parameters
        ----------
        parents_X: numpy.array
            parents_X contains the genome values of the parents
        children_X: numpy.array
            children_X contains the genome values of the offspring
        parents_T: numpy.array
            parents_T contains the trace lists of the parents
            
        Returns
        -------
        offspring_TL : 
            the trace list of the child of parent 1 and parent 2, based on the distance between the parents and the child value        
        """
        def mutateTraceList(p_x, new_c_x, p_tl):
            """
            calculates the trace list of the offspring gene based on the gene values of the offspring and the parent
            Parameters
            ----------
            p_x: Gene Value
                the gene value of the parent gene
            new_c_x: Gene Value
                the gene value of the offspring
            p_tl: numpy.array
                p_tl contains the trace lists of the parents
                
            Returns
            -------
            new_traceList : 
                the trace list of the child of parent 1 and parent 2, based on the distance between the parents and the child value        
            """

            #inclrease the mutation counter if we do not accumulate the mutations
            if not self.accumulate_mutations:
                self.mutation_counter -= 1

            if p_x == new_c_x : #return the parents trace list if the value was not altered
                return p_tl

            influence_factor_mut = abs(p_x - new_c_x) / ( abs(p_x) + abs(p_x - new_c_x) )
            influence_factor_old = 1 - influence_factor_mut

            new_traceList = TraceList()

            i = 0 #index for this traceList

            #check if we accumulate and the first element is already the mutation
            if self.accumulate_mutations and p_tl.get(0).traceID == self.mutation_counter: #if we accumulate mutation and the gene is influenced by mutation, accumulate that
                oldScaledMutImpact = p_tl.get(0).influenceFactor * influence_factor_old
                new_traceList.append(TraceTuple(self.mutation_counter, influence_factor_mut + oldScaledMutImpact))
                i = i + 1 #increment i, as the first element of the traceList from p_tl is already added
            else: #add the new mutation ID if we don't accumulate or there is no mutation present bevore
                new_traceList.append(TraceTuple(self.mutation_counter, influence_factor_mut))

            while i < p_tl.len(): #this iterates over the traceList of this individual
                currentAID = p_tl.get(i).traceID
                currentAImpact = p_tl.get(i).influenceFactor

                new_traceList.append(TraceTuple(currentAID, influence_factor_old * currentAImpact))
                i = i + 1

            return new_traceList

        offspring_T = []

        for indIndex in range(0, len(parents_X)):
            parent_X = parents_X[indIndex]
            parent_T = parents_T[indIndex]
            child_X = children_X[indIndex]

            child_T = []

            for genomeIndex in range(0, len(parent_X)):
                parent_current_gene_value = parent_X[genomeIndex]
                parent_current_TL = parent_T[genomeIndex]
                child_current_gene_value = child_X[genomeIndex]

                child_T.append( mutateTraceList(parent_current_gene_value, child_current_gene_value, parent_current_TL) )
            
            offspring_T.append(child_T)

        return np.array(offspring_T, dtype=TraceList)
 
    def calculateOffspringTraceVectors(self, parents_X, children_X, parents_T):
        """
        this method calculates the new trace vector for the offsprings of the mutation operation based on the parent and offspring gene value.
        Parameters
        ----------
        parents_X: numpy.array
            parents_X contains the genome values of the parents
        children_X: numpy.array
            children_X contains the genome values of the offspring
        parents_T: numpy.array
            parents_T contains the trace lists of the parents
            
        Returns
        -------
        offspring_TL : 
            the trace vectors of the child of parent 1 and parent 2, based on the distance between the parents and the child value        
        """

        if not self.accumulate_mutations:
            raise Exception("Looks like you are trying to use the trace vector and to differntiate between mutations! That is not possible, use te trace list instead.")

        offspring_T = np.zeros(parents_T.shape)
        for indIndex in range(0, len(parents_X)):
            parent_X = parents_X[indIndex]
            child_X = children_X[indIndex]

            if np.equal(parent_X, child_X).all() : #return the parents trace list if the value was not altered. Also avoid divide by 0 error.
                offspring_T[indIndex] = parents_T[indIndex]
            else:

                for geneIndex in range(0, len(parent_X)):#build the child trace vectors based on the parents and the influence factors

                    if parent_X[geneIndex] == child_X[geneIndex]: #the not mutated case
                        offspring_T[indIndex, geneIndex, :] = parents_T[indIndex, geneIndex, :]
                    else :
                        influence_factor_mut = ( abs(parent_X[geneIndex] - child_X[geneIndex]) / ( abs(parent_X[geneIndex]) + abs(parent_X[geneIndex] - child_X[geneIndex]) ) )
                        influence_factor_old = 1.0 - influence_factor_mut
                        
                        offspring_T[indIndex, geneIndex, :-1] = parents_T[indIndex, geneIndex, :-1] * influence_factor_old
                        oldScaledMutImpact = parents_T[indIndex, geneIndex,  -1] * influence_factor_old
                        offspring_T[indIndex, geneIndex,  -1] = oldScaledMutImpact + influence_factor_mut

        return np.array(offspring_T)

    def calculateOffspringTraceIDs(self, parents_X, children_X, parents_T):
        """
        this methid assigns the mutationID to the mutated individuals
        Parameters
        ----------
        parents_X: numpy.array
            parents_X contains the genome values of the parents
        children_X: numpy.array
            children_X contains the genome values of the offspring
        parents_T: numpy.array
            parents_T contains the trace lists of the parents
            
        Returns
        -------
        offspring_TL : 
            the trace list of the child of parent 1 and parent 2, based on the distance between the parents and the child value        
        """

        def mutate_with_counter():
            if not self.accumulate_mutations:
                self.mutation_counter -= 1
            return self.mutation_counter

        if self.value_dependent_mutation:
            raise Exception("Looks like you are trying to use traceIDs with value dependent mutation! Use TraceVector or TraceList instead!")

        c_from_p = np.equal(parents_X, children_X)

        if self.accumulate_mutations: #asign -1 everywhere if accumulating mutations
            mutationIDs = np.zeros(parents_T.shape) - 1
            children_T = np.where(c_from_p, parents_T, mutate_with_counter())
        else: #distinglish between moccuring mutations otherwise
            children_T = parents_T
            for i in range(c_from_p.shape[0]):
                for j in range(c_from_p.shape[1]):
                    self.mutation_counter -= 1
                    children_T[i,j] = self.mutation_counter

        return children_T


    def __init__(self, mutation, tracing_type=TracingTypes.NO_TRACING, accumulate_mutations=True, **kwargs):
        super().__init__(prob=mutation.prob, prob_var=mutation.prob_var, **kwargs)
        self.mutation = mutation
        self.tracing_type = tracing_type
        self.accumulate_mutations = accumulate_mutations


    def _do(self, problem, X):
        return self.mutation._do(problem, X)

    def do(self, problem, pop, inplace=True, **kwargs):
        
        X_p = pop.get("X") #the parent genome values. Coppy this here to avoid inplace errors
        T_p = pop.get("T")

        offspring = self.mutation.do(problem, pop, inplace, **kwargs)


        IsOff = np.ones(len(offspring), dtype=bool) #in the operators, all individuals are marked to be offspring

        X_m = offspring.get("X") #the mutated genome values

        if self.tracing_type == TracingTypes.NO_TRACING: #return if not tracing
            return Population.new("X", X_m, "IsOff", IsOff)
        elif self.tracing_type == TracingTypes.TRACE_ID:
            T_m = self.calculateOffspringTraceIDs(X_p, X_m, T_p)
            return Population.new("X", X_m, "T", T_m), "IsOff", IsOff
        elif self.tracing_type == TracingTypes.TRACE_LIST:
            T_m = self.calculateOffspringTraceLists(X_p, X_m, T_p)
            return Population.new("X", X_m, "T", T_m, "IsOff", IsOff)
        elif self.tracing_type == TracingTypes.TRACE_VECTOR:
            T_m = self.calculateOffspringTraceVectors(X_p, X_m, T_p)
            return Population.new("X", X_m, "T", T_m, "IsOff", IsOff)

