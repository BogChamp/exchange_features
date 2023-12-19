import numpy as np
from numba import njit

@njit
def compute_imbalances(ask_prices, ask_amounts, bid_prices, bid_amounts):
    ask_eligible = [ask_amounts[i] for i, v in enumerate(ask_prices) \
                    if v < (ask_prices[0] * 1.05)]
    median_index = len(ask_eligible) // 2
    if len(ask_eligible) % 2:
        ask_median = ask_eligible[median_index]
    else:
        ask_median = (ask_eligible[median_index] + ask_eligible[median_index - 1]) / 2
    
    bid_eligible = [bid_amounts[i] for i, v in enumerate(bid_prices) \
                    if v > (bid_prices[0] * 0.95)]
    median_index = len(bid_eligible) // 2
    if len(bid_eligible) % 2:
        bid_median = bid_eligible[median_index]
    else:
        bid_median = (bid_eligible[median_index] + bid_eligible[median_index - 1]) / 2

    median = (ask_median + bid_median) / 2

    size = median
    money = 0
    for i, amount in enumerate(ask_amounts):
        if np.isclose(size, 0):
            break
        else:
            if amount < size:
                size -= amount
                money += ask_prices[i] * amount
            else:
                money += ask_prices[i] * size
                size = 0
    ask_imbalance = ((money / median) / ask_prices[0] - 1) * 10**5

    size = median
    money = 0
    for i, amount in enumerate(bid_amounts):
        if np.isclose(size, 0):
            break
        else:
            if amount < size:
                size -= amount
                money += bid_prices[i] * amount
            else:
                money += bid_prices[i] * size
                size = 0
                
    bid_imbalance = (bid_prices[0] / (money / median) - 1) * 10**5

    return (ask_imbalance, bid_imbalance)

@njit
def compute_improved_imbalance(ob_snapshot):
    ts, data = ob_snapshot[0], ob_snapshot[1:]
    ask_prices = data[::4]
    ask_amounts = data[1::4]
    bid_prices = data[2::4]
    bid_amounts = data[3::4]
    ask_imbalance, bid_imbalance = compute_imbalances(ask_prices, ask_amounts, 
                                                        bid_prices, bid_amounts)
    
    return (ts, ask_imbalance, bid_imbalance)

@njit
def numba_imb(dataset):
    tuples = [(0, 0.0, 0.0)] * dataset.shape[0]
    
    for i, row in enumerate(dataset):
        tuples[i] = compute_improved_imbalance(row)
    
    return tuples

@njit
def numba_calculate_past_returns(trades_avg, delta):
    past_returns = [0.0 for _ in range(trades_avg.shape[0])]
    
    start_index = 0
    delta_ms = delta * 10**6
    
    for i, v in enumerate(trades_avg):
        while (v[0] - trades_avg[start_index][0]) > delta_ms:
            start_index += 1

        past_returns[i] = (v[1] / trades_avg[start_index][1] - 1) * 10**5
    
    return past_returns