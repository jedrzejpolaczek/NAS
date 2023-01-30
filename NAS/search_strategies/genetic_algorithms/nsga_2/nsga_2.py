#!/usr/bin/env python
"""
NSGA2 class. NSGA2 stands for A fast elitist non-dominated sorting genetic algorithm for multi-objective optimization: NSGAII.

"""

import random
import functools
import numpy as np
from loguru import logger

try:
    from ..genetic_algorithm import GeneticAlgorithm
except ImportError:
    pass


class NSGA2(GeneticAlgorithm):  # TODO: add typing and docstring
    """
    Class contain implementation of:
    A fast elitist non-dominated sorting genetic algorithm for multi-objective optimization: NSGAII 
    (Non-dominated Sorting Genetic Algorithm) by Kalyanmoy Deb, Samir Agrawal, Amrit Pratap, and T Meyarivan from
    Kanpur Genetic Algorithms Laboratory (KanGAL) Indian Institute of Technology Kanpur.

    Link to the papers: http://repository.ias.ac.in/83498/1/2-a.pdf
    """

    def __init__(self, population, crossover_probability: float=0.2, mutation_probability: float=0.8):
        super(NSGA2, self).__init__(population)
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.fitnesses: np.array = None
        self.fronts: list = None
        self.ranks: dict = None
        self.crowding_distance_metrics = None
        self.non_domiated_sorted_indicies = None
    
    def run(self, max_generations: int) -> None:
        """
        Execute the main genetic algorithm loop established a number of times.
        The main genetic algorithm loop contains evolving population method.

        :param max_generations (int): value to set how many times the main genetic algorithm loop need to be done.
        """
        logger.info("Starting genetic algorithm.")
        while self.generation <= max_generations:
            self.evolve_population()
            self.generation += 1

    def evolve_population(self):  # TODO: add typing and docstring
        if self.population.get_size() < self.tournament_size:
            msg = "Population size is smaller than tournament size."
            logger.error(msg)
            raise ValueError(msg)
        
        logger.info("Evaluating generation #{}".format(self.generation))
        logger.info("Population size: {}".format(self.population.get_size()))
        logger.info("Fittest individual is: {}".format(self.population.get_fittest()))
        logger.info("Fitness value is: {}".format((self.population.get_fittest().get_fitness())))

        # Calculate fitnesses
        self.fitnesses = self.calculate_fitnesses()

        # Calculate the pareto fronts
        self.fronts = self.fast_nondominated_sort()

        # Calculate ranks
        self.ranks = self.fronts_to_nondomination_ranks()
        
        # Calculate crowding distance
        self.crowding_distance_metrics = self.calculate_crowding_distance_metrics()

        # Sorting population indexes using nondominated sort
        self.non_domiated_sorted_indicies = self.nondominated_sort()
        
        # Evolving population
        new_population = self.create_new_population()
        
        # Saving new population in place of current population
        self.population.individuals = new_population

        self.guard("evolve_population", "self.population.individuals", self.population.individuals)
    
    def calculate_fitnesses(self) -> np.array:  # TODO: add typing and docstring
        fitnesses = []
        for individual in self.population.individuals:
            fitnesses.append(individual.get_fitness())

        self.guard("normalize_fitnesses", "fitnesses", len(fitnesses), self.population.get_size())
        return np.array(fitnesses)
    
    def normalize_fitnesses(self) -> np.array:  # TODO: add typing and docstring
        self.fitnesses = (self.fitnesses - np.min(self.fitnesses)) / (np.max(self.fitnesses) - np.min(self.fitnesses))
        self.guard("normalize_fitnesses", "fitnesses", self.fitnesses)
    
    def fast_nondominated_sort(self) :  # TODO: add typing and docstring
        """
        Calculate Pareto fronts
        The population is sorted and partitioned into fronts (F1, F2, etc.), 
        where F1 (first front) indicates the approximated Pareto front.

        :return list: fronts is a list of fronts, each front contain index of each individual for self.population.individuals
        """
        # Calculate dominated set for each individual
        domination_sets = []
        domination_counts = []
        for individual in self.population.individuals:
            current_domination_set = set()
            domination_counts.append(0)
            for i, other_individual in enumerate(self.population.individuals):
                if self.dominates(individual, other_individual):
                    current_domination_set.add(i)
                elif self.dominates(other_individual, individual):
                    domination_counts[-1] += 1

            domination_sets.append(current_domination_set)
        domination_counts = np.array(domination_counts)

        fronts = []
        while True:
            current_front = np.where(domination_counts==0)[0]
            if len(current_front) == 0:
                logger.debug("Current front have no individuals. Therefore we stop looking for new fronts.")
                break
            self.guard("fast_nondominated_sort", "front", current_front)
            fronts.append(current_front)

            for individual in current_front:
                # this individual is already accounted for, make it -1 so  ==0 will not find it anymore
                domination_counts[individual] = -1 
                dominated_by_current_set = domination_sets[individual]
                for dominated_by_current in dominated_by_current_set:
                    domination_counts[dominated_by_current] -= 1
        
        self.guard("fast_nondominated_sort", "fronts", fronts)
        return fronts
    
    def dominates(self, individual, other_individual):  # TODO: add typing and docstring
        larger_or_equal = individual.get_fitness() >= other_individual.get_fitness()
        larger = individual.get_fitness() > other_individual.get_fitness()

        return np.all(larger_or_equal) and np.any(larger)  # We are using np.all in case fitness would be array
    
    def fronts_to_nondomination_ranks(self) -> dict:  # TODO: add typing and docstring
        """
        
        :return dict: dictinary of indexes of each individual for self.population.individuals
        """
        nondomination_rank_dict = {}
        for i, front in enumerate(self.fronts):
            for x in front:   
                nondomination_rank_dict[x] = i

        self.guard("fronts_to_nondomination_ranks", "nondomination_rank_dict", nondomination_rank_dict)
        return nondomination_rank_dict

    def calculate_crowding_distance_metrics(self):  # TODO: add typing and docstring
        """
        Crowding Distance is a mechanism of ranking among members of a front, 
        which are dominating or dominated by each other.
        """
        # Prepare necessary data
        number_of_individuals = self.population.get_size()
        crowding_distance_metrics = np.zeros(number_of_individuals)
        number_of_objectives = len(self.fitnesses[0])
        
        # Normalise each objectives, so they are in the range [0,1]
        # This is necessary, so each objective's contribution have the same magnitude to the crowding metric.
        self.normalize_fitnesses()

        # Calculate crowding distance metrics
        for front in self.fronts:
            for objective_i in range(number_of_objectives):
                sorted_front = sorted(front, key = lambda x : self.fitnesses[x, objective_i])

                crowding_distance_metrics[sorted_front[0]] = np.inf
                crowding_distance_metrics[sorted_front[-1]] = np.inf
                if len(sorted_front) > 2:
                    for i in range(1,len(sorted_front)-1):
                        crowding_distance_metrics[sorted_front[i]] += self.fitnesses[sorted_front[i+1], objective_i] - self.fitnesses[sorted_front[i-1], objective_i]

        self.guard("calculate_crowding_distance_metrics", "crowding_distance_metrics", len(crowding_distance_metrics), number_of_individuals)
        return crowding_distance_metrics
    
    def nondominated_sort(self):  # TODO: add typing and docstring
        
        number_of_individuals = len(self.crowding_distance_metrics)
        indicies = list(range(number_of_individuals))

        def nondominated_compare(a, b):
            # returns 1 if a dominates b, or if they equal, but a is less crowded
            # return -1 if b dominates a, or if they equal, but b is less crowded
            # returns 0 if they are equal in every sense
            
            if self.ranks[a] > self.ranks[b]:  # domination rank, smaller better
                return -1
            elif self.ranks[a] < self.ranks[b]:
                return 1
            else:
                # crowding metrics, larger better
                if self.crowding_distance_metrics[a] < self.crowding_distance_metrics[b]:   
                    return -1
                elif self.crowding_distance_metrics[a] > self.crowding_distance_metrics[b]:
                    return 1
                else:
                    return 0

        # decreasing order, the best is the first
        non_domiated_sorted_indicies = sorted(indicies, key = functools.cmp_to_key(nondominated_compare), reverse=True)

        self.guard("nondominated_sort", "non_domiated_sorted_indicies", non_domiated_sorted_indicies)
        return non_domiated_sorted_indicies

    def create_new_population(self):  # TODO: add typing and docstring
        """Survive current population, tournament, crossover them and mutate their children."""

        # New population first half is from survivals from previous generation
        surviving_individuals = []
        for i in range(int(self.population.get_size()/2)):
            surviving_individuals.append(self.population.individuals[self.non_domiated_sorted_indicies[i]])

        # Other half of new population is offspring of winners of tournament
        offsprings = []
        while len(offsprings) != (self.population.get_size()/2):
            parent_first = self.tournament_select()
            parent_second = self.tournament_select()
            offspring = parent_first.reproduce(parent_second)
            offspring.mutate()
            offsprings.append(offspring)

        new_population = surviving_individuals + offsprings
        
        self.guard("create_new_population", "new_population", len(new_population), self.population.get_size())
        return new_population
    
    def tournament_select(self):  # TODO: add typing and docstring
        tournament = self.get_population_type()(
            self.population.get_species(), 
            self.x_train, 
            self.y_train, 
            individual_list=[
                self.population[i] for i in random.sample(range(self.population.get_size()), self.tournament_size)
            ], 
            maximize=self.population.get_fitness_criteria()
        )

        fittest_individual = tournament.get_fittest()

        self.guard("tournament_select", "fittest_individual", fittest_individual)
        return tournament.get_fittest()

    @staticmethod
    def guard(fun_name, name, main_object, object_to_compare=None):  # TODO: add typing and docstring
        if object_to_compare is not None:
            assert main_object == object_to_compare
        logger.debug("{}:{} (type: {}): {}".format(fun_name, name, type(main_object), main_object))
