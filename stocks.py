import numpy as np
import random
import genetic_algorithm as ga
import investor_portfolio as ip

np.random.seed(42)
random.seed(42)


iterations = 50
population_size = 5000
fitness_exponent = 1.0
mutation_chance = 0.01
mutation_rate = 1.0

starting_cash = 100000
stock_points = 300


def factory():
    return lambda: ip.factory(starting_cash)


def main():
    stock_list = [
        ip.Stock("AAPL", 150, 0.1, 0.05, 1, 1 / stock_points, random_seed=42),
        ip.Stock("GOOGL", 200, 0.08, 0.06, 1, 1 / stock_points, random_seed=42),
        ip.Stock("AMZN", 100, 0.12, 0.07, 1, 1 / stock_points, random_seed=42),
        ip.Stock("TSLA", 600, 0.15, 0.08, 1, 1 / stock_points, random_seed=42),
        ip.Stock("MSFT", 250, 0.07, 0.04, 1, 1 / stock_points, random_seed=42),
        ip.Stock("NFLX", 180, 0.13, 0.09, 1, 1 / stock_points, random_seed=42),
        ip.Stock("FB", 300, 0.06, 0.03, 1, 1 / stock_points, random_seed=42),
        ip.Stock("NVDA", 800, 0.11, 0.06, 1, 1 / stock_points, random_seed=42),
        ip.Stock("AMD", 100, 0.14, 0.07, 1, 1 / stock_points, random_seed=42),
        ip.Stock("INTL", 50, 0.05, 0.02, 1, 1 / stock_points, random_seed=42),
        ip.Stock("JPM", 130, 0.05, 0.04, 1, 1 / stock_points, random_seed=42),
        ip.Stock("V", 210, 0.09, 0.05, 1, 1 / stock_points, random_seed=42),
        ip.Stock("MA", 350, 0.1, 0.05, 1, 1 / stock_points, random_seed=42)
    ]
    ip.InvestorPortfolio.set_stocks(stock_list)  # Make sure the stocks are set before creating an instance
    g_a = ga.GeneticAlgorithm(
        entity_factory=factory,
        population_size=population_size,
        iterations=iterations,
        mutation_chance=mutation_chance,
        fixed_length_genome=True,
        fitness_exponent=fitness_exponent,
        mutation_rate=mutation_rate,
        show_progress_bars=True
    )
    g_a.run()


if __name__ == "__main__":
    main()
