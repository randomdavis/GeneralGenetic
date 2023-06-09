from genetic_algorithm import Entity
from typing import List
import numpy as np
import random
from math import floor


class Stock:
    def __init__(self, name: str, initial_stock_price: float, expected_return: float, volatility: float,
                 time_period: float, time_step: float, random_seed: int = 42):
        self.name = name
        self.prices = np.array([], dtype="float32")
        self.parameters = (initial_stock_price, expected_return, volatility, time_period, time_step, random_seed)
        self.prices = self.calculate_geometric_brownian_motion(*self.parameters)

    @staticmethod
    def calculate_geometric_brownian_motion(initial_stock_price: float, expected_return: float, volatility: float,
                                            time_period: float, time_step: float, random_seed: int) -> np.ndarray:
        np.random.seed(random_seed)
        time_array = np.linspace(0, time_period, int(time_period / time_step), dtype="float32")
        num_steps = len(time_array)
        random_walk = np.random.standard_normal(size=num_steps)
        random_walk = np.cumsum(random_walk, dtype="float32") * np.sqrt(time_step)
        return np.array(initial_stock_price * np.exp(
            (expected_return - 0.5 * volatility ** 2) * time_array + volatility * random_walk), dtype="float32")


class InvestorPortfolio(Entity):
    stocks = []
    TRADE_HISTORY = True

    @classmethod
    def set_stocks(cls, stocks: List[Stock]):
        cls.stocks = stocks

    def __init__(self):
        Entity.__init__(self)
        self.reset()

    def __eq__(self, other):
        same = self.sell_threshold == other.sell_threshold and \
               self.buy_threshold == other.buy_threshold and \
               self.stop_loss_ratio == other.stop_loss_ratio and \
               self.buy_ratio == other.buy_ratio and \
               self.sell_ratio == other.sell_ratio

        return same

    def __str__(self):
        return self.string_representation(-1)

    def copy(self):
        copied_self = InvestorPortfolio()
        copied_self.initial_cash = self.initial_cash
        copied_self.initialize()
        return copied_self

    def setup(self, initial_cash: float, sell_threshold: float, buy_threshold: float, stop_loss_ratio: float,
                 buy_ratio: float, sell_ratio: float):
        self.initial_cash = initial_cash
        self.target_cash = self.initial_cash
        self.sell_threshold = sell_threshold
        self.buy_threshold = buy_threshold
        self.stop_loss_ratio = stop_loss_ratio
        self.buy_ratio = buy_ratio
        self.sell_ratio = sell_ratio
        self.init_values_from_parameters()

        self.initialize()

    def initialize(self):
        self.cash = self.initial_cash
        self.target_cash = self.initial_cash
        self.owned_stocks = {stock.name: 0 for stock in self.stocks}

    def reset(self):
        self.num_buys = 0
        self.num_sells = 0
        self.trade_history = []
        self.previous_buy_or_sell_prices = {}
        self.trade_history = []
        self.num_sells = 0
        self.num_buys = 0

    def set_value_from_parameters(self):
        self.value = f"sell_threshold {repr(self.sell_threshold)} " + \
                     f"buy_threshold {repr(self.buy_threshold)} " + \
                     f"stop_loss_ratio {repr(self.stop_loss_ratio)} " + \
                     f"buy_ratio {repr(self.buy_ratio)} " + \
                     f"sell_ratio {repr(self.sell_ratio)}"

    def init_values_from_parameters(self):
        self.values = [
            self.sell_threshold,
            self.buy_threshold,
            self.stop_loss_ratio,
            self.buy_ratio,
            self.sell_ratio,
        ]
        self.set_value_from_parameters()

    def set_parameters_from_values(self):
        self.sell_threshold = self.values[0]
        self.buy_threshold = self.values[1]
        self.stop_loss_ratio = self.values[2]
        self.buy_ratio = self.values[3]
        self.sell_ratio = self.values[4]
        self.set_value_from_parameters()

    def mutate(self, original_value, mutation_rate):
        mutation_value = random.uniform(-mutation_rate, mutation_rate)
        mutation_value += original_value
        return max(0.0, min(1.0, mutation_value))

    def stocks_value(self, price_point):
        total_stock_value = 0.0
        for stock in self.stocks:
            stock_price = stock.prices[price_point]
            stocks_owned = self.owned_stocks[stock.name]
            total_stock_value += stock_price * stocks_owned
        return total_stock_value

    def portfolio_value(self, price_point):
        total_portfolio_value = self.cash + self.stocks_value(price_point)
        return total_portfolio_value

    def final_value(self):
        return self.portfolio_value(-1)

    def string_representation(self, price_point):
        string_representation = f"Cash: {self.cash}\n"
        string_representation += f"Stocks Value: {self.stocks_value(-1)}\n"
        for stock_name in self.owned_stocks:
            shares = self.owned_stocks[stock_name]
            price = None
            for stock in self.stocks:
                if stock.name == stock_name:
                    price = stock.prices[price_point]
                    break
            string_representation += f"{stock_name}: {shares} shares at ${price}/share, total ${shares * price}\n"
        return string_representation

    def final_stats(self):
        return f"\tInitial Cash ${repr(self.initial_cash)}\n" + \
              f"\tSell Threshold {repr(self.sell_threshold * 100)}%\n" + \
              f"\tBuy Threshold {repr(self.buy_threshold * 100)}%\n" + \
              f"\tStop Loss Ratio {repr(self.stop_loss_ratio * 100)}%\n" + \
              f"\tBuy Ratio {repr(self.buy_ratio * 100)}%\n" + \
              f"\tSell Ratio {repr(self.sell_ratio * 100)}%\n" + \
              f"\tFinal Cash: ${repr(self.final_value())}\n" + \
              f"\tFitness: {repr(self.fitness)}\n" + \
              f"\ttotal buys: {self.num_buys}\n" + \
              f"\ttotal sells: {self.num_sells}"

    def update_cash_stocks_owned(self, operation, stock_name, current_price, n_stocks, price_point_num):
        if n_stocks > 0:
            total_stock_value = current_price * n_stocks
            if operation == "buy":
                self.cash -= total_stock_value
                self.owned_stocks[stock_name] += n_stocks
                self.num_buys += 1
            elif operation == "sell":
                self.cash += total_stock_value
                self.owned_stocks[stock_name] -= n_stocks
                self.num_sells += 1
            assert self.cash >= 0.0
            if self.TRADE_HISTORY:
                self.trade_history.append(f'{price_point_num}: {operation.title()} {n_stocks} share{"s" if n_stocks != 1 else ""} of {stock_name} at ${current_price}/share for ${round(current_price * n_stocks, 2)}, cash ${round(self.cash, 2)}')
            self.previous_buy_or_sell_prices[stock_name] = current_price

    def sell_condition_met(self, stock_name, current_price, portfolio_val):
        change_from_previous_point = (current_price - self.previous_buy_or_sell_prices[stock_name]) / self.previous_buy_or_sell_prices[stock_name]
        return change_from_previous_point >= self.sell_threshold or portfolio_val <= (1 - self.stop_loss_ratio) * self.target_cash

    def buy_condition_met(self, stock_name, current_price):
        change_from_previous_point = (current_price - self.previous_buy_or_sell_prices[stock_name]) / self.previous_buy_or_sell_prices[stock_name]
        return change_from_previous_point <= -self.buy_threshold

    def execute_decision(self, stock_name, current_price, price_point_num):
        n_stocks_original = self.owned_stocks[stock_name]
        if n_stocks_original > 0 and self.sell_condition_met(stock_name, current_price, price_point_num):
            n_stocks = floor(self.sell_ratio * n_stocks_original)
            if n_stocks > 0:
                self.update_cash_stocks_owned("sell", stock_name, current_price, n_stocks, price_point_num)
        elif self.buy_condition_met(stock_name, current_price):
            n_stocks = floor(self.cash * self.buy_ratio / current_price)
            if n_stocks > 0:
                self.update_cash_stocks_owned("buy", stock_name, current_price, n_stocks, price_point_num)

    def backtest_strategy(self):
        self.previous_buy_or_sell_prices = {stock.name: stock.prices[0] for stock in self.stocks}
        stock_prices_dict = {stock.name: stock.prices for stock in self.stocks}
        num_price_points = len(self.stocks[0].prices)
        range_price_points = range(num_price_points)
        for price_point_num in range_price_points:
            portfolio_val = self.portfolio_value(price_point_num)
            for stock_name in self.owned_stocks:
                current_price = stock_prices_dict[stock_name][price_point_num]
                self.execute_decision(stock_name, current_price, price_point_num)
            if portfolio_val > self.target_cash:
                self.target_cash = portfolio_val

    def evaluate(self):
        self.set_parameters_from_values()
        self.reset()
        self.initialize()
        self.backtest_strategy()
        self.fitness = self.final_value()


def factory(initial_cash):
    initial_cash = initial_cash
    sell_threshold = random.random() * 0.1
    buy_threshold = random.random() * 0.1
    stop_loss_ratio = (random.random() * 0.1) + 0.9
    buy_ratio = (random.random() * 0.1) + 0.9
    sell_ratio = (random.random() * 0.1) + 0.9

    new_entity = InvestorPortfolio()
    new_entity.setup(initial_cash, sell_threshold, buy_threshold, stop_loss_ratio, buy_ratio, sell_ratio)
    return new_entity
