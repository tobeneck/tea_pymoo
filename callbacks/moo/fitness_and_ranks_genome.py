import numpy as np

from tea_pymoo.callbacks.data_collector import DataCollector

from pymoo.util.misc import vectorized_cdist #for "getDistanceToClosestPointOnPF"
from pymoo.indicators.distance_indicator import euclidean_distance #for "getDistanceToClosestPointOnPF"
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

#calculates the distance of a reference point to the closest point in pareto_front using the eucledian distance
def getDistanceToClosestPointOnPF(reference_solution, pareto_front):
    norm=1.0 #do not normalize
    D = vectorized_cdist(pareto_front, reference_solution, func_dist=euclidean_distance, norm=norm)
    return np.mean(np.min(D))


#returns the non dominated solutions of a population as an np.array of the size of the population, containing the ranks of the individuals
def rankIndividualByFonsecaAndFleming(individual, population, minimize, stop_if_dominated=False):

    rank = 0

    for indToCompareTo in population:
        dominated = True
        for fitnessIndex in range(len(individual.F)):
            if ( individual.F[fitnessIndex] <= indToCompareTo.F[fitnessIndex] and minimize ) or ( individual.F[fitnessIndex] >= indToCompareTo.F[fitnessIndex] and not minimize ):
                dominated = False
                break
        
        if dominated:
            rank = rank + 1

    return rank

#returns the ranks of the individuals of the population as an array
def getFonsecaAndFlemingRanks(population, minimize):

    if not minimize:
        raise Exception("error! Minimize on getFonsecaAndFlemingRanks not implemented jet!")

    ranks = np.zeros(len(population), dtype=int)

    for i in range(len(population)):
        ranks[i] = rankIndividualByFonsecaAndFleming(population[i], population, minimize)
    
    return ranks

#returns the ranks of the individuals of the population as an array
def getGoldbergRanks(population, minimize):

    if not minimize:
        raise Exception("error! Minimize on getGoldbergRanks not implemented jet!")

    #build the fitness only array
    fitness = []
    for individual in population:
        fitness.append(individual.F)
    fitness = np.array(fitness)
    
    #actually rank the individuals with the nds sorting method
    nds = NonDominatedSorting()
    ranks = np.array(nds.do(F=fitness, return_rank=True)[1])

    return ranks

#returns the ranks of the individuals of the population as an array
def getBelegunduRanks(population, minimize):

    if not minimize:
        raise Exception("error! Minimize on getBelegunduRanks (getGoldbergRanks...) not implemented jet!")

    #just rank by another ranking method like goldbergs
    ranks = getGoldbergRanks(population, minimize)

    #there are only non-dominated solutions and dominated solutions, so re-rank
    for rank_index in range(len(ranks)):
        if ranks[rank_index] > 1: 
            ranks[rank_index] = 1

    return ranks



class Fitness_and_Ranks_Callback(DataCollector):

    def __init__(self, n_obj, additional_run_info=None, minimize=True, fonsecaAndFlemingRank=False, goldbergRank=True, belegunduRank=False, dist_to_pf=False, fitness=True, filename="fitness_and_ranks") -> None:
        '''
        this class saves the gitness values, rank and genome values of the individuals
        Parameters
            n_obj : int
                the number of objective functions
            minimize : boolean
                if the problem is a minimization problem
            fonsecaAndFlemingRank : boolean
                if the rank according to finseca and flemming should be calculated
            goldbergRank : boolean
                if the rank according to goldberg should be calculated
            belegunduRank : boolean
                if the rank according to belegundu should be calculated
            fitness : boolean
                if the fitness values of the individuals should be saved
            genome : boolean
                if the genome values of the individuals should be saved
    
        ---------- 
        '''
        self.filename = "ranks_and_fitness"
        self.minimize = minimize

        self.fonsecaAndFlemingRank = fonsecaAndFlemingRank
        self.goldbergRank = goldbergRank
        self.belegunduRank = belegunduRank
        self.dist_to_pf = dist_to_pf

        data_keys = ["generation", "individual"]
        if fitness:
            for i in range(0, n_obj):
                data_keys.append("f_"+str(i+1))
        if self.fonsecaAndFlemingRank:
            data_keys.append("fonseca_fleming_rank")
        if self.goldbergRank:
            data_keys.append("goldberg_rank")
        if self.belegunduRank:
            data_keys.append("belegundu_rank")
        if self.dist_to_pf:
            data_keys.append("dist_to_pf")

        super().__init__(data_keys=data_keys, filename=filename, additional_run_info=additional_run_info)

    def notify(self, algorithm):

        generation = algorithm.n_gen
        population = algorithm.pop
        pareto_front = algorithm.problem.pareto_front()


        #calculate all ranks
        if self.fonsecaAndFlemingRank:
            fonseca_fleming_ranks = getFonsecaAndFlemingRanks(population, self.minimize)
        if self.goldbergRank:
            goldberg_ranks = getGoldbergRanks(population, self.minimize)
        if self.belegunduRank:
            belegundu_ranks = getBelegunduRanks(population, self.minimize)
        
        #save the population data
        for i in range(0, len(population)):
            super().handle_additional_run_info()
            for key in self.data.keys():
                if key == "generation":
                    self.data[key].append(generation)
                elif key == "individual":
                    self.data[key].append(i)
                elif key == "fonseca_fleming_rank":
                    self.data[key].append(fonseca_fleming_ranks[i])
                elif key == "goldberg_rank":
                    self.data[key].append(goldberg_ranks[i])
                elif key == "belegundu_rank":
                    self.data[key].append(belegundu_ranks[i])
                elif key == "dist_to_pf":
                    self.data[key].append(getDistanceToClosestPointOnPF(population[i].F, pareto_front))
                elif key[:2] == "f_":
                    fitness_index = int( key.split("_")[1] ) - 1
                    self.data[key].append(population[i].F[fitness_index])
                elif key[:2] == "g_":
                    genome_index = int( key.split("_")[1] ) - 1
                    self.data[key].append(population[i].X[genome_index])