import logging
import sys
from datetime import datetime
import numpy as np
import random
import time
from tqdm import tqdm

# Create a unique log file name with the current date and time

current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_name = f"logs/output_{current_datetime}.log"

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a file handler for output.log
file_handler = logging.FileHandler(log_file_name)
file_handler.setLevel(logging.INFO)

# Create a console handler for printing messages
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

np.random.seed(42)
random.seed(42)


class GeneticAlgorithm:
    def __init__(
        self,
        entity_factory,
        population_size=1000,
        iterations=10000,
        mutation_chance=0.1,
        fixed_length_genome=False,
        fitness_exponent=1.0,
        mutation_rate=0.1,
        log_level=logging.INFO,
        show_progress_bars=True,
    ):
        self.entity_factory = entity_factory
        self.population_size = population_size
        self.iterations = iterations
        self.mutation_chance = mutation_chance
        self.fixed_length_genome = fixed_length_genome
        self.fitness_exponent = fitness_exponent
        self.mutation_rate = mutation_rate
        logger.setLevel(log_level)
        self.show_progress_bars = show_progress_bars
        self.population = []
        self.fittest = None
        self.fittest_ever = None

    def log_generation(self, total_cycles=0, show_fittest_ever=False):
        log = ""
        fitnesses = np.array([item.fitness for item in self.population])
        avg = np.average(fitnesses)
        median = np.median(fitnesses)
        std = np.std(fitnesses)
        max_val = max(fitnesses)
        min_val = min(fitnesses)
        log += f"Iteration {total_cycles}: max: {max_val}, min: {min_val}, avg: {avg}, median: {median}, std: {std}, var: {std ** 2.0}\n"
        if not show_fittest_ever:
            log += f"Top Performer This Generation:\n{self.fittest.final_stats()}\n"
        else:
            log += f"Top Performer All Time:\n{self.fittest_ever.final_stats()}\n"
        return log

    def run(self):
        start = time.process_time()
        total_cycles = 0

        try:  # Wrap the main loop in a try-except block to handle crashes
            self.generate_initial_population()
            self.fittest = self.select_fittest()
            self.fittest_ever = self.fittest

            for _ in range(self.iterations):
                log = self.log_generation(total_cycles)
                logger.info(log)
                total_cycles += 1
                self.evolve_population()
                for member in tqdm(self.population, desc=f"Evaluating gen {total_cycles}", ncols=100, disable=not self.show_progress_bars):
                    member.evaluate()
                time.sleep(0.5)
                self.fittest = self.select_fittest()

                # Update the fittest member ever, if necessary
                if self.fittest.fitness > self.fittest_ever.fitness:
                    self.fittest_ever = self.fittest
                    logger.info(f"New fittest ever!")

            end = time.process_time()
            total_time = end - start

            logger.info("\nWinner:\n")
            if self.fittest_ever:
                logger.info(self.fittest_ever.value)
            logger.info(f"{total_cycles} iterations")
            logger.info(f"{total_time:.2f} seconds")
            logger.info(f"Total iterations: {total_cycles}")
            logger.info(f"Total time: {total_time:.2f} seconds")
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.info("Interrupted:", e)
            logger.error("Interrupted by exception", e)

        if self.fittest_ever:
            # Print and log the final information for the fittest member ever encountered
            logger.info(self.fittest_ever.final_stats())
            logger.info("Terminating with fittest ever:")
            logger.info(f"Fittest ever: {self.fittest_ever.value}\n{self.fittest_ever.final_stats()}")
            for trade_history_item in self.fittest_ever.trade_history:
                logger.info(trade_history_item)

    def generate_initial_population(self):
        for _ in tqdm(range(self.population_size), desc="Generating", ncols=100, disable=not self.show_progress_bars):
            new_entity = self.entity_factory(self)
            new_entity.evaluate()
            self.population.append(new_entity)

    def roulette_wheel_selection(self):
        probability_distribution = np.array([ind.fitness ** self.fitness_exponent for ind in self.population],
                                            dtype='float64')
        probs_sum = probability_distribution.sum()
        if probs_sum != 0:
            probability_distribution /= probability_distribution.sum()
            weighted_random_choice = np.random.choice(self.population, 1, p=probability_distribution)[0]
            return weighted_random_choice
        else:
            raise RuntimeError("Probability sum cannot equal zero.")

    def evolve_population(self, keep_fittest=False):
        num_new_children = self.population_size
        if keep_fittest and self.fittest is not None:
            new_population = [self.fittest]
            num_new_children -= 1
        else:
            new_population = []
        while len(new_population) < self.population_size:
            parent1 = self.roulette_wheel_selection()
            parent2 = self.roulette_wheel_selection()
            children = parent1.breed(parent2, mutation_chance=self.mutation_chance,
                                     mutation_rate=self.mutation_rate,
                                     fixed_length_genome=self.fixed_length_genome)
            new_population += children
        self.population = new_population

    def select_fittest(self):
        self.population.sort(reverse=True)
        return self.population[0]


class Entity:
    def __init__(self):
        self.values = []
        self.fitness = 0
        self.value = None

    def mutate(self, value, mutation_amount):
        raise NotImplementedError()

    def copy(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def split(self, point):
        max_index = (len(self.values) - 1)
        split_point = int(round(max_index * point))
        return [self.values[:split_point], self.values[split_point:]]

    def breed(self, other, mutation_chance=0.001, mutation_rate=1.0, fixed_length_genome=False):
        split_point = random.random()
        self_values_split = self.split(split_point)
        other_values_split = other.split(split_point)

        child1_values = self_values_split[0] + other_values_split[1]
        child2_values = other_values_split[0] + self_values_split[1]

        children_values = [child1_values, child2_values]
        children = []

        for new_values in children_values:
            child = self.copy()
            # Iterate through each value in new_values.
            for i in range(len(new_values)):
                # Check mutation probability for each value.
                if random.random() < mutation_chance:
                    if fixed_length_genome:
                        # Apply mutation to the fixed-length genome
                        new_values[i] = self.mutate(new_values[i], mutation_rate)
                    else:
                        # Apply mutation based on the options (insertion, deletion)
                        options = random.choice([1, 2, 3])
                        if options == 1:
                            new_values[i] = self.mutate(new_values[i], mutation_rate)
                        elif options == 2:
                            del new_values[i]
                            break  # Breaking here as the list length has changed, need to restart iteration. Mutating multiple values is still possible.
                        else:
                            # Insert one or more new mutated values after the current value
                            for c in range(1, random.randint(1, 30)):
                                mutated_value = self.mutate(new_values[i], mutation_rate)
                                new_values.insert(i + c, mutated_value)

            child.values = new_values
            children.append(child)
        return children

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness
