import numpy as np
from numba import njit

@njit
def numba_imb_simple(ticker):
    return (ticker[:, 1] - ticker[:, 3]) / (ticker[:, 1] + ticker[:, 3]) 

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
def get_averaged_trades(trades, delta):
    res = [(0, 0) for _ in range(trades.shape[0])]

    start_index = 0
    delta_ms = delta * 10**6

    for i, v in enumerate(trades):
        while (v[0] - trades[start_index][0]) > delta_ms:
            start_index += 1

        if i > start_index:
            res[i] = (v[0], np.sum(trades[start_index:i][1] * trades[start_index:i][2]) / np.sum(trades[start_index:i][2]))
        else:
            res[i] = (v[0], 0)
    
    return res

@njit
def numba_calculate_past_returns(trades, delta):
    trades_avg = np.array(get_averaged_trades(trades, delta))
    past_returns = [0.0 for _ in range(trades_avg.shape[0])]
    
    start_index = 0
    delta_ms = delta * 10**6
    
    for i, v in enumerate(trades_avg):
        while (v[0] - trades_avg[start_index][0]) > delta_ms:
            start_index += 1

        
        if np.isclose(trades_avg[start_index][1], 0):
            past_returns[i] = 0
        else:
            past_returns[i] = (v[1] / trades_avg[start_index][1] - 1) * 10**5
            
    
    return past_returns

@njit
def numba_log_returns(prices):
    log_prices = np.log(prices)
    return log_prices[1:] - log_prices[:-1]

@njit
def shift(xs, n):
    if n == 0:
        return xs.copy()
    e = np.empty_like(xs, np.float64)
    e[:n] = 0.0
    e[n:] = xs[:-n]
    return e

@njit
def numba_data_autocorrelation(time_series, 
                         lags, 
                         time_window):
    autocorrelations = [[0.0 for i in range(time_series.shape[0])] for j in lags]
    ts = time_series[:, 0]
    prices = time_series[:, 1]
    lag_prices_prod = [np.cumsum(prices * shift(prices, lag)) for lag in lags]

    cum_prices = np.cumsum(prices)
    cum_prices_2 = np.cumsum(prices**2)
    
    start_index = 0
    delta_ms = time_window * 10**6
    
    for i, v in enumerate(ts):
        while (v - ts[start_index]) > delta_ms:
            start_index += 1
        
        for j, lag in enumerate(lags):
            n = i - start_index + 1 - lag
            if n <= 1 or start_index == 0:
                autocorrelations[j][i] = 0
            else:
                sum_x_2 = cum_prices_2[i] - cum_prices_2[start_index + lag - 1]
                sum_x = cum_prices[i] - cum_prices[start_index + lag - 1]
                sum_y_2 = cum_prices_2[i - lag] - cum_prices_2[start_index - 1]
                sum_y = cum_prices[i - lag] - cum_prices[start_index - 1]
                denominator = (n * sum_x_2 - sum_x**2) * (n * sum_y_2 - sum_y**2)
                
                sum_xy = lag_prices_prod[j][i] - lag_prices_prod[j][start_index + lag - 1]
                numerator = n * sum_xy - sum_x * sum_y
                
                if np.isclose(numerator, 0):
                    autocorrelations[j][i] = 0
                elif denominator > 0:
                    autocorrelations[j][i] = np.divide(numerator, np.sqrt(denominator))
                else:
                    autocorrelations[j][i] = 0
             
    
    return autocorrelations

@njit
def parzen_kernel(x):
    x = abs(x)
    if x >= 1:
        return 0
    elif x >= 0.5:
        return 2 * (1 - x)**3
    else:
        return 1 - 6 * x**2 * (1 - x)

@njit
def numba_data_realized_kernel(time_series, 
                         H, 
                         time_window):

    autocorrelations = [0.0 for l in range(time_series.shape[0])]
    
    ts = time_series[:, 0]
    prices = time_series[:, 1]
    
    lag_prices_prod = [np.cumsum(prices * shift(prices, lag)) for lag in range(H+1)]
    kernel_values = [parzen_kernel(k / H) for k in range(1, H + 1)]

    start_index = 0
    delta_ms = time_window * 10**6

    for i, v in enumerate(ts):
        while (v - ts[start_index]) > delta_ms:
            start_index += 1
        
        if start_index == 0:
            autocorrelations[i] = 0
        else:
            kernel_range = min(i + 1 - start_index, H)
            res = lag_prices_prod[0][i] - lag_prices_prod[0][start_index - 1]
            for j in range(1, kernel_range+1):
                res += 2 * kernel_values[j - 1] * (lag_prices_prod[j][i] - \
                                            lag_prices_prod[j][start_index + j - 1])
            autocorrelations[i] = res
    
    return autocorrelations

