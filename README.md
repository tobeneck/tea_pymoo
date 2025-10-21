# The traceable evolutionary algorithm in pymoo

TODO: short introduction





# Measuring the Performance

## Callbacks
There are three kinds of callbacks. Logging performance is split into **MOO** and **SOO** categories. The tracing callbacks are under the **General** category.

### General

**genome_callback.py**
Loggs the genome values of the individuals.

#### Impact Metrics

##### Population Whide

**counting_impact_pop_callback.py**
Loggs the counting impact of the whole population, or just the non-dominated individuals.

**fitness_impact_pop_callback.py**
Loggs the fitness impact of the whole population, or just the non-dominated individuals.

**entropy_impact_pop_callback.py**
Loggs the entropy impact of the whole population, or just the non-dominated individuals.

**fitness_entropy_impact_pop_callback.py**
Loggs the fitness+entropy impact of the whole population, or just the non-dominated individuals.

##### For each Individual

**counting_impact_inds_callback.py**
Loggs the counting impact, for each individual separately.

(Fitness impact for each ind does not exist, as scaling the one individual by its own fitness is just returning the counting impact)

**entropy_impact_inds_callback.py**
Loggs the entropy impact, for each individual separately.

(Fitness+entropy impact for each ind does not exist, as scaling the one individual by its own fitness is just returning the entropy impact)



### MOO

**fitness_and_ranks_callback.py**
Saves the fitness of each objective and has several implementations for logging the ranks of individuals. <> Specify which exactly! provide references.

**performance_indicators_callback.py**
Saves the performance indicators for MOEAs.

### SOO

**fitness_callback.py**
Saves the fitness of each individual, similar to fitness_and_ranks for MOEAs.

**performance_callback**
Saves the lowest, mean, median and highest fitness of each generation.