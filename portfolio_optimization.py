import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Select stocks
stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Download historical data
data = yf.download(stocks, start="2020-01-01")["Close"]

# Calculate daily returns
returns = data.pct_change().dropna()

# Mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

num_portfolios = 5000

results = np.zeros((3, num_portfolios))
weights_record = []

for i in range(num_portfolios):

    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)

    portfolio_return = np.sum(mean_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    sharpe_ratio = portfolio_return / portfolio_volatility

    results[0, i] = portfolio_volatility
    results[1, i] = portfolio_return
    results[2, i] = sharpe_ratio

    weights_record.append(weights)

# Identify best Sharpe ratio
max_sharpe_idx = np.argmax(results[2])
best_return = results[1, max_sharpe_idx]
best_volatility = results[0, max_sharpe_idx]

print("Best Portfolio Return:", best_return)
print("Best Portfolio Risk:", best_volatility)

# Plot Efficient Frontier
plt.figure(figsize=(10,6))
plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap="viridis")
plt.xlabel("Risk (Volatility)")
plt.ylabel("Expected Return")

best_weights = weights_record[max_sharpe_idx]

print("\nOptimal Portfolio Allocation:")
for stock, weight in zip(stocks, best_weights):
    print(stock, ":", round(weight*100,2), "%")
plt.title("Efficient Frontier - Portfolio Optimization")
plt.colorbar(label="Sharpe Ratio")
plt.scatter(best_volatility, best_return, color="red", s=100)

plt.show()
