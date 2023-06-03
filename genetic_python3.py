import random
import time


def string_diff(string1, string2):
    string1_length = len(string1)
    string2_length = len(string2)
    total_matches = 0.0

    if string1_length >= string2_length:
        total_length = string1_length
        for i in range(string2_length):
            if string1[i] == string2[i]:
                total_matches += 1
        match_ratio = total_matches / total_length
    else:
        total_length = string2_length
        for i in range(string1_length):
            if string1[i] == string2[i]:
                total_matches += 1
        match_ratio = total_matches / total_length

    return match_ratio


class Entity:
    def __init__(self):
        self.values = []
        self.fitness = 0
        self.value = None

    def mutate(self):
        raise NotImplementedError()

    def split(self, point):
        max_index = (len(self.values) - 1)
        split_point = int(round(max_index * point))
        return [self.values[:split_point], self.values[split_point:]]

    def breed(self, other, mutation_chance=0.001):
        child = self.__class__()
        split_point = random.random()
        self_values_split = self.split(split_point)
        other_values_split = other.split(split_point)

        if random.random() < 0.5:
            new_values = self_values_split[0] + other_values_split[1]
        else:
            new_values = other_values_split[0] + self_values_split[1]

        if random.random() < mutation_chance:
            new_values_length = len(new_values)
            options = random.choice([1, 2, 3])
            possible_indexes = range(0, new_values_length)
            random_index = random.choice(possible_indexes)
            if options == 1:
                new_values[random_index] = self.mutate()
            elif options == 2:
                del new_values[random_index]
            else:
                for c in range(1, random.randint(1, 30)):
                    new_values.append(self.mutate())
        child.values = new_values
        return child

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def evaluate(self):
        raise NotImplementedError("Subclasses should implement the evaluate() function")


class StringEntity(Entity):
    target = None
    alphabet = None

    def __init__(self,
                 length=100):
        Entity.__init__(self)
        self.length = length
        self.value = ""
        assert self.target is not None
        if self.alphabet is None:
            self.alphabet = ''.join(sorted(list(set(self.target))))

    def __str__(self):
        return self.value + ' ' + str(self.fitness)

    def __repr__(self):
        return self.__str__()

    def mutate(self):
        return random.choice(self.alphabet)
        
    def generate_values(self):
        for _ in range(self.length):
            self.values.append(self.mutate())

    def evaluate(self):
        self.value = "".join(self.values)
        self.fitness = string_diff(self.value, self.target)


class GeneticAlgorithm:
    def __init__(
        self,
        entity_factory,
        population_size=1000,
        iterations=10000,
        mutation_chance=0.1,
    ):
        self.entity_factory = entity_factory
        self.population_size = population_size
        self.iterations = iterations
        self.mutation_chance = mutation_chance
        self.population = []

    def run(self):
        start = time.process_time()
        total_cycles = 0
        fittest = None

        self.generate_initial_population()

        for _ in range(self.iterations):
            total_cycles += 1

            fittest = self.select_fittest()
            print(fittest.value)
            print(repr(fittest.fitness))

            if fittest.fitness == 1.0:
                break

            self.evolve_population(fittest)

        end = time.process_time()
        total_time = end - start

        print("\nWinner:\n")
        if fittest:
            print(fittest.value)
        print(f"{total_cycles} iterations")
        print(f"{total_time:.2f} seconds")
    
    def generate_initial_population(self):
        for _ in range(self.population_size):
            new_entity = self.entity_factory()
            new_entity.evaluate()
            self.population.append(new_entity)

    def evolve_population(self, fittest):
        new_population = []
        for entity in self.population:
            new_ent = fittest.breed(entity, self.mutation_chance)
            new_ent.evaluate()
            new_population.append(new_ent)
        self.population = new_population

    def select_fittest(self):
        self.population.sort(reverse=True)
        return self.population[0]


def string_entity_factory():
    new_entity = StringEntity()
    new_entity.generate_values()
    return new_entity


def main():
    fit_string = "a quick brown fox jumps over the lazy dog"
    population_size = 1000
    iterations = 10000

    StringEntity.target = fit_string

    ga = GeneticAlgorithm(
        entity_factory=string_entity_factory,
        population_size=population_size,
        iterations=iterations,
    )
    ga.run()


if __name__ == "__main__":
    main()
