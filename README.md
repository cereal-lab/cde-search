# Experimentation framework for objective search 

The goal of the project is to provide base algorithms and benchmarks for objective search in environments where contrained co-evolution happens. 
Here we look onto case when one population do not cooperate with another. It is basically a sequence of individuals appearing in the process asynchronously.
The question is would be the second population to "adapt" to general knowledge pertained to first population.
We assume that the first population even though uncontrolable, posses low-dimensional interaction behavior. In other words, 
individuals share some knowledge of how to interact with the second population. Adaptation of the second population should discover the knowledge structure therefore. 


# Code structure 

1. Module cli.py provide entrypoints for knowledge space generation and simulation running. The simulations are defines in SIM config.py. 
   Games could be number games or space games!

2. 