import random
import genetic_algorithm as ga


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


class StringEntity(ga.Entity):
    target = None
    alphabet = None

    def __init__(self,
                 length=100):
        ga.Entity.__init__(self)
        self.length = length
        self.value = ""
        assert self.target is not None
        if self.alphabet is None:
            self.alphabet = ''.join(sorted(list(set(self.target))))

    def __str__(self):
        return self.value + ' ' + str(self.fitness)

    def __repr__(self):
        return self.__str__()

    def mutate(self, **kwargs):
        return random.choice(self.alphabet)
        
    def generate_values(self):
        for _ in range(self.length):
            self.values.append(self.mutate())

    def evaluate(self):
        self.value = "".join(self.values)
        self.fitness = string_diff(self.value, self.target)


def string_entity_factory():
    new_entity = StringEntity()
    new_entity.generate_values()
    return new_entity


def main():
    fit_string = "a quick brown fox jumps over the lazy dog"
    population_size = 1000
    iterations = 10000
    mutation_chance = 0.01
    fitness_exponent = 1.0
    mutation_rate = 1.0

    StringEntity.target = fit_string

    g_a = ga.GeneticAlgorithm(
        entity_factory=string_entity_factory,
        population_size=population_size,
        iterations=iterations,
        mutation_chance=mutation_chance,
        fixed_length_genome=False,
        fitness_exponent=fitness_exponent,
        mutation_rate=mutation_rate,
        show_progress_bars=True
    )
    g_a.run()


if __name__ == "__main__":
    main()
