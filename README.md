# The traceable evolutionary algorithm in pymoo

TODO: short introduction





# Measuring the Performance

## Callbacks
There are three kinds of callbacks. Logging performance is split into **MOO** and **SOO** categories. The tracing callbacks are under the **General** category.

### General

**counting_impact_inds_callback.py**
Loggs the counting impact, for each individual separately.

**counting_impact_pop_callback.py**
Loggs the counting impact of the whole population, or just the non-dominated individuals.

**genome_callback.py**
Loggs the genome values of the individuals.

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