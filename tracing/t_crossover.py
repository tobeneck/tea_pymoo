import numpy as np

from .t_sampling import TracingTypes, TraceList, TraceTuple
from pymoo.core.crossover import Crossover
from pymoo.core.population import Population



class T_Crossover(Crossover):

    def calculateOffspringTraceLists(self, parents_X, offspring_X, parents_T):
        """
        this methid calculates the new trace lists for the offsprings of the crossover operation based on the parents and offspring gene values.
        Parameters
        ----------
        parents_X: numpy.array
            parents_X[0] contains the genome values for the first half of parents and parents_X[1] for the second half 
        offspring_X: numpy.array
            offspring_X[0] contains the genome values for the first half of offspring and offspring_X[1] for the second half 
        parents_T: numpy.array
            parents_T[0] contains the trace lists for the first half of parents and parents_T[1] for the second half 
            
        Returns
        -------
        offspring_T : 
            the trace list of the child of parent 1 and parent 2, based on the distance between the parents and the child value        
        """

        def recombineTraceLists(p_x, p_tl, new_c_x):
            """
            this method calculates the new trace list for the child based on the parents values
            Parameters
            ----------
            p_x: np.array
                the genome value for all parents in an array [p1_x, p2_x,..., pn_x]
            p_tl: np.array
                the trace lists of all parents in an array [p1_tl, p2_tl,..., on_tl]
            new_c_x: genome value
                the genome value of the child
                
            Returns
            -------
            new_c_tl : TraceList
                the trace list of the child
            """
            n_parents = len(p_x)
            if n_parents != 2:
                raise Exception("Trace Lists are not jet implemented for more than two parents on the crossover operator!")

            p1_x = p_x[0]
            p2_x = p_x[1]
            p1_tl = p_tl[0]
            p2_tl = p_tl[1]


            #if aVal = bVal = newVal, both genes should have 50% impact
            influence_factor_p1 = 0.5
            influence_factor_p2 = 0.5

            if(p1_x == new_c_x and p2_x != new_c_x): #return a if it has 100% influence. Faster and avoids 0.0 impact traceTuples
                return p1_tl
            if(p2_x == new_c_x and p1_x != new_c_x): #return b if it has 100% influence. Faster and avoids 0.0 impact traceTuples
                return p2_tl
            if(p1_x != new_c_x and p2_x != new_c_x): #compute the new values if non of them are equal
                #if you don't cast here, the resulting values are integers and will be roundet!
                influence_factor_p1 = 1.0 - ( abs(p1_x - new_c_x) / ( abs(p1_x - new_c_x) + abs(p2_x - new_c_x) ) )
                influence_factor_p2 = 1.0 - ( abs(p2_x - new_c_x) / ( abs(p1_x - new_c_x) + abs(p2_x - new_c_x) ) )

            i = 0 #index for trace vector a
            j = 0 #index for trace vector b

            new_c_tl = TraceList()

            while True: #this iterates over the traceList of this individual
                if i >= p1_tl.len() and j >= p2_tl.len(): #stop if both vectors are empty
                    break
                elif i >= p1_tl.len() and not (j >= p2_tl.len()):#append if the a vector is empty and b vector is not.
                    currentP2ID = p2_tl.get(j).traceID
                    currentP2Influence = p2_tl.get(j).influenceFactor
                    new_c_tl.append(TraceTuple(currentP2ID, influence_factor_p2 * currentBInfluence))
                    j = j + 1
                elif not (i >= p1_tl.len()) and j >= p2_tl.len(): #append if the b vector is empty and a vector is not.
                    currentP1ID = p1_tl.get(i).traceID
                    currentAInfluence = p1_tl.get(i).influenceFactor
                    new_c_tl.append(TraceTuple(currentP1ID, influence_factor_p1 * currentAInfluence))
                    i = i + 1
                else: #if both arrays are not empty, append the next traceID:
                    currentP1ID = p1_tl.get(i).traceID
                    currentP2ID = p2_tl.get(j).traceID

                    currentAInfluence = p1_tl.get(i).influenceFactor
                    currentBInfluence = p2_tl.get(j).influenceFactor

                    if currentP1ID == currentP2ID: #combine the two if equal
                        new_c_tl.append(TraceTuple(currentP1ID, influence_factor_p1 * currentAInfluence + influence_factor_p2 * currentBInfluence))
                        i = i + 1
                        j = j + 1

                    if currentP1ID < currentP2ID: #add the traceID of a if its smaller than the traceID of b
                        new_c_tl.append(TraceTuple(currentP1ID, influence_factor_p1 * currentAInfluence))
                        i = i + 1

                    if currentP2ID < currentP1ID: #add the traceID of b if its smaller than the traceID of a
                        new_c_tl.append(TraceTuple(currentP2ID, influence_factor_p2 * currentBInfluence))
                        j = j + 1

            return new_c_tl


        offspring_T = np.empty(offspring_X.shape, dtype=TraceList)

        n_parents, n_matings, n_var = parents_X.shape
        n_offspring, n_parents, n_var = offspring_X.shape

        for offspring_index in range(n_offspring): #iterate over the offspring
            for parent_index in range(n_parents): #iterate over the offspring generated with the parent indices (the offspring at offspring_X[offspring_index, parent_index] has the paretns parent_X[parent_index])
                for genome_index in range(n_var): #iterate over each genome
                    curr_child_x = offspring_X[offspring_index, parent_index, genome_index]
                    curr_parents_x = parents_X[parent_index, :, genome_index]
                    curr_parents_t = parents_T[parent_index, :, genome_index]
                    offspring_T[offspring_index, parent_index, genome_index] = recombineTraceLists(curr_parents_x, curr_parents_t, curr_child_x)
        
        return offspring_T

    def calculateOffspringTraceVector(self, parents_X, offspring_X, parents_T):# future TODO: make this posible with more than 2 parents per gene!
        """
        this methid calculates the new trace vector for the offsprings of the crossover operation based on the parents and offspring gene values.
        Parameters
        ----------
        parents_X: np.array
            parents_X[0] contains the genome values for the first half of parents and parents_X[1] for the second half 
        offspring_X: np.array
            offspring_X[0] contains the genome values for the first half of offspring and offspring_X[1] for the second half 
        parents_T: np.array
            parents_T[0] contains the trace vectors for the first half of parents and parents_T[1] for the second half 
            
        Returns
        -------
        offspring_T : 
            the trace vector of the child of parent 1 and parent 2, based on the distance between the parents and the child value        
        """
        def recombineTraceVectors(parents_X, parents_T, child_X):
            """
            this method calculates the new trace vector for the childs gene values based on the parents.

            Parameters
            ----------
            parents_X: 2d np.array
                the genome values of the parents
            parents_T: 3d np.array
                the trace lists of the parents
            child_X: np.array
                the genome values of the child
                
            Returns
            -------
            child_T : 2d np.array
                the trace vector of the child
            """
            n_parents, genome_size = parents_X.shape
            if n_parents != 2:
                raise Exception("Trace Vectors are not jet implemented for more than two parents on the crossover operator!")

            parent1_X = parents_X[0]
            parent2_X = parents_X[1]
            parent1_T = parents_T[0]
            parent2_T = parents_T[1]

            child_T = np.zeros(parent1_T.shape) #TODO: reshape!
            for geneIndex in range(0, genome_size):#build the child trace vectors based on the parents and the influence factors
                influence_factor_p1 = 0.5
                influence_factor_p2 = 0.5
                if parent1_X[geneIndex] == child_X[geneIndex] and parent2_X[geneIndex] != child_X[geneIndex]:
                    child_T[geneIndex] = parent1_T[geneIndex]
                    continue
                elif parent1_X[geneIndex] != child_X[geneIndex] and parent2_X[geneIndex] == child_X[geneIndex]:
                    child_T[geneIndex] = parent2_T[geneIndex]
                    continue
                elif parent1_X[geneIndex] != child_X[geneIndex] and parent2_X[geneIndex] != child_X[geneIndex]:
                    influence_factor_p1 = 1.0 - ( abs(parent1_X[geneIndex] - child_X[geneIndex]) / ( abs(parent1_X[geneIndex] - child_X[geneIndex]) + abs(parent2_X[geneIndex] - child_X[geneIndex]) ) )
                    influence_factor_p2 = 1.0 - ( abs(parent2_X[geneIndex] - child_X[geneIndex]) / ( abs(parent1_X[geneIndex] - child_X[geneIndex]) + abs(parent2_X[geneIndex] - child_X[geneIndex]) ) )

                child_T[geneIndex] = parent1_T[geneIndex] * influence_factor_p1 + parent2_T[geneIndex] * influence_factor_p2
                
            return child_T

        offspring_T = np.zeros(offspring_X.shape + (parents_T.shape[-1],) )

        n_parents, n_matings, n_var = parents_X.shape
        n_offspring, n_parents, n_var = offspring_X.shape

        for offspring_index in range(n_offspring): #iterate over the offspring
            for parent_index in range(n_parents): #iterate over the offspring generated with the parent indices (the offspring at offspring_X[offspring_index, parent_index] has the paretns parent_X[parent_index])
                curr_child_X = offspring_X[offspring_index, parent_index]
                curr_parents_X = parents_X[parent_index]
                curr_parents_T = parents_T[parent_index]
                offspring_T[offspring_index, parent_index] = recombineTraceVectors(curr_parents_X, curr_parents_T, curr_child_X)
        
        return offspring_T

    def calculateOffspringTraceIDs(self, parents_X, offspring_X, parents_T):
        """
        this method calculates the new traceIDs for the offsprings of the crossover operation based on the parents and offspring gene values. Th
        Parameters
        ----------
        parents_X: numpy.array
            parents_X[0] contains the genome values for the first half of parents and parents_X[1] for the second half 
        offspring_X: numpy.array
            offspring_X[0] contains the genome values for the first half of offspring and offspring_X[1] for the second half 
        parents_T: numpy.array
            parents_T[0] contains the trace vectors for the first half of parents and parents_T[1] for the second half 
            
        Returns
        -------
        offspring_T : 
            the traceIDs of the child of parent 1 and parent 2      
        """

        parents1_X = parents_X[0]
        parents2_X = parents_X[1]

        parents1_T = parents_T[0]
        parents2_T = parents_T[1]

        children1_X = np.copy(offspring_X[0])
        children2_X = np.copy(offspring_X[1])

        c1_from_p1 = np.equal(parents1_X, children1_X)
        c1_from_p2 = np.equal(parents2_X, children1_X)
        c2_from_p1 = np.equal(parents1_X, children2_X)
        c2_from_p2 = np.equal(parents2_X, children2_X)
        
        if not np.all(c1_from_p1 | c1_from_p2) or not np.all(c2_from_p1 | c2_from_p2):
            raise Exception("Looks like you are trying to use a gene combining crossover operator with traceID tracking! Use TraceVector or TraceList instead!")

        children1_T = np.where(c1_from_p1, parents1_T, parents2_T)
        children2_T = np.where(c2_from_p1, parents1_T, parents2_T)

        return np.array([children1_T, children2_T])


    def __init__(self, crossover, tracing_type=TracingTypes.NO_TRACING, **kwargs):
        '''
        This class acts as a wrapper for another crossover operator and just adds the tracing capabilitys on top.

        Parameters
        ----------

        crossover : pymoo.core.crossover.Crossover
            The actual crossover operator to be used

        tracing_type : TracingType
            Indicator for the type of representation used for tracing
        '''
        super().__init__(n_parents=crossover.n_parents, n_offsprings=crossover.n_offsprings, prob=crossover.prob, **kwargs)
        self.crossover = crossover
        self.tracing_type = tracing_type
    

    def _do(self, problem, X, **kwargs):
        return self.crossover._do(problem, X, **kwargs)

    def do(self, problem, pop, parents=None, **kwargs):
        

        '''
        Note on the datastructure of the parents (pop) and the offspring:
        n_parents, n_matings, n_var = pop.get("X").shape
        n_offspring, n_parents, n_var = offspring.get("X").shape

        This is to be concidered when calculating the tracing information.
        '''
        if pop.shape[1] != 2:
            raise Exception("Tracing is not jet implemented for more than two parents on the crossover operator!")

        parents_X = pop.get("X")
        parents_T = pop.get("T")
        
        #revert the shape of the offspring back to its original form as specified in crossover.py line 36 (but withoud the genome size) to calculate the tracing information
        n_matings = len(pop)
        n_var = problem.n_var
        generated_offspring_shape=(self.crossover.n_offsprings, n_matings)
        offspring = self.crossover.do(problem, pop, parents, **kwargs).reshape(generated_offspring_shape) 
        offspring_X = offspring.get("X")

        n_offspring, n_parents = offspring.shape
        IsOff = np.ones(n_offspring*n_parents, dtype=bool) #in the operators, all individuals are marked to be offspring

        #calculate the tracing information based on the tracing type
        if self.tracing_type == TracingTypes.NO_TRACING:
            return Population.new("X", offspring_X, "IsOff", IsOff)
        
        elif self.tracing_type == TracingTypes.TRACE_ID:
            #create the trace lists of the offspring if tracing is enabled
            offspring_T = self.calculateOffspringTraceIDs(parents_X, offspring_X, parents_T)
            # flatten the genome array back to become a 2d-array
            offspring_X = offspring_X.reshape(-1, offspring_X.shape[-1])
            # also flatten the trace list array if tracing is enabled
            offspring_T = offspring_T.reshape(-1, offspring_T.shape[-1])
            
            return Population.new("X", offspring_X, "T", offspring_T, "IsOff", IsOff)
        elif self.tracing_type == TracingTypes.TRACE_LIST:
            #create the trace lists of the offspring if tracing is enabled
            offspring_T = self.calculateOffspringTraceLists(parents_X, offspring_X, parents_T)
            # flatten the genome array back to become a 2d-array
            offspring_X = offspring_X.reshape(-1, offspring_X.shape[-1])
            # also flatten the trace list array if tracing is enabled
            offspring_T = np.concatenate(offspring_T, axis=0)

            return Population.new("X", offspring_X, "T", offspring_T, "IsOff", IsOff)
        elif self.tracing_type == TracingTypes.TRACE_VECTOR:
            #create the trace lists of the offspring if tracing is enabled
            offspring_T = self.calculateOffspringTraceVector(parents_X, offspring_X, parents_T)
            # flatten the genome array back to become a 2d-array
            offspring_X = offspring_X.reshape(-1, offspring_X.shape[-1])
            # also flatten the trace list array if tracing is enabled
            offspring_T = np.concatenate(offspring_T, axis=0)

            return Population.new("X", offspring_X, "T", offspring_T, "IsOff", IsOff)

        #TODO: extract the traceing stuff here!