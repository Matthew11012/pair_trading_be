import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class PairTradingModel:
    """
    A class that implements a statistical arbitrage pair trading strategy.
    The model calculates the spread between two correlated securities and
    generates trading signals based on mean reversion principles.
    """
    
    def __init__(self, ticker1, ticker2, lookback_period=30, z_threshold=2.0):
        """
        Initialize the pair trading model.
        
        Parameters:
        ticker1 (str): Symbol for the first stock
        ticker2 (str): Symbol for the second stock
        lookback_period (int): Number of days to look back for calculating metrics
        z_threshold (float): Z-score threshold for trading signals
        """
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.lookback_period = lookback_period
        self.z_threshold = z_threshold
        self.data = None
        self.hedge_ratio = None
        self.spread = None
        self.z_scores = None
        self.signals = None
        self.performance = None
        
    def fetch_data(self, start_date=None, end_date=None):
        """
        Fetch historical price data for the pair of stocks.
        
        Parameters:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
        Returns:
        pd.DataFrame: DataFrame with price data for both stocks
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            # Calculate start date based on lookback period plus additional buffer
            start = datetime.now() - timedelta(days=self.lookback_period * 24)
            start_date = start.strftime('%Y-%m-%d')
            
        # Fetch data using yfinance
        stock1 = yf.download(self.ticker1, start=start_date, end=end_date)
        stock2 = yf.download(self.ticker2, start=start_date, end=end_date)

        print(f"{self.ticker1} - stock1.head():\n", stock1.head())
        # print(f"{self.ticker2} - stock2.head():\n", stock2.head())

        print("Checking for empty stock data...")
        print("stock1.empty:", stock1.empty)
        print("stock2.empty:", stock2.empty)

        if stock1.empty or stock2.empty:
            raise ValueError("One or both tickers returned no data")
        
        stock1_close = stock1['Close'].squeeze()
        stock2_close = stock2['Close'].squeeze()
        # print(type(stock1_close))
        # print(stock1_close.head())


        # Create a dataframe with adjusted close prices
        df = pd.DataFrame({
            self.ticker1: stock1_close,
            self.ticker2: stock2_close
        })
        
        # print("Final dataframe head:")
        # print(df)
        
        # Remove any rows with NaN values
        self.data = df.dropna()
        return self.data
    
    def calculate_hedge_ratio(self, method='rolling'):
        """
        Calculate the hedge ratio between the two stocks.
        
        Parameters:
        method (str): Method to calculate hedge ratio ('rolling' or 'static')
        
        Returns:
        pd.Series: Hedge ratios over time
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please call fetch_data() first.")
        
        if method == 'static':
            # Calculate a single hedge ratio using OLS regression
            X = sm.add_constant(self.data[self.ticker1])
            model = sm.OLS(self.data[self.ticker2], X).fit()
            self.hedge_ratio = pd.Series(model.params[1], index=self.data.index)
            
        elif method == 'rolling':
            # Calculate rolling hedge ratio
            endog = self.data[self.ticker2]
            exog = sm.add_constant(self.data[self.ticker1])
            
            rolling_ols = RollingOLS(endog, exog, window=self.lookback_period)
            rolling_res = rolling_ols.fit()
            
            # Extract the coefficient (hedge ratio)
            self.hedge_ratio = rolling_res.params[self.ticker1]
            
        return self.hedge_ratio
    
    def calculate_spread(self):
        """
        Calculate the spread between the two stocks using the hedge ratio.
        
        Returns:
        pd.Series: Spread values over time
        """
        if self.hedge_ratio is None:
            self.calculate_hedge_ratio()
            
        # Calculate spread as the difference between actual price and hedge ratio * price
        self.spread = self.data[self.ticker2] - self.hedge_ratio * self.data[self.ticker1]
        return self.spread
    
    def calculate_z_scores(self):
        """
        Calculate the z-scores of the spread using the lookback period.
        
        Returns:
        pd.Series: Z-scores of the spread
        """
        if self.spread is None:
            self.calculate_spread()
            
        # Calculate rolling mean and standard deviation
        rolling_mean = self.spread.rolling(window=self.lookback_period).mean()
        rolling_std = self.spread.rolling(window=self.lookback_period).std()
        
        # Calculate z-scores
        self.z_scores = (self.spread - rolling_mean) / rolling_std
        return self.z_scores
    
    def generate_signals(self):
        """
        Generate trading signals based on z-scores and threshold.
        
        Returns:
        pd.DataFrame: DataFrame with trading signals
        """
        if self.z_scores is None:
            self.calculate_z_scores()
            
        # Initialize signals DataFrame
        self.signals = pd.DataFrame(index=self.data.index)
        self.signals['z_score'] = self.z_scores
        
        # Generate signals
        # 1: Long ticker2, Short ticker1
        # -1: Short ticker2, Long ticker1
        # 0: No position
        self.signals['signal'] = 0
        
        # When z-score is above threshold, go short the spread (Short ticker2, Long ticker1)
        self.signals.loc[self.z_scores > self.z_threshold, 'signal'] = -1
        
        # When z-score is below negative threshold, go long the spread (Long ticker2, Short ticker1)
        self.signals.loc[self.z_scores < -self.z_threshold, 'signal'] = 1
        
        # Add the prices for reference
        self.signals[self.ticker1] = self.data[self.ticker1]
        self.signals[self.ticker2] = self.data[self.ticker2]
        self.signals['hedge_ratio'] = self.hedge_ratio
        self.signals['spread'] = self.spread
        
        return self.signals
    
    def backtest(self, initial_capital=100000):
        """
        Backtest the pair trading strategy.
        
        Parameters:
        initial_capital (float): Initial capital for the backtest
        
        Returns:
        pd.DataFrame: DataFrame with performance metrics
        """
        if self.signals is None:
            self.generate_signals()
            
        # Create performance DataFrame
        self.performance = pd.DataFrame(index=self.signals.index)
        self.performance['signal'] = self.signals['signal']
        
        # Calculate daily returns for both stocks
        self.performance[f'{self.ticker1}_returns'] = self.data[self.ticker1].pct_change()
        self.performance[f'{self.ticker2}_returns'] = self.data[self.ticker2].pct_change()
        
        # Calculate strategy returns
        # When signal is 1: Long ticker2, Short ticker1
        # When signal is -1: Short ticker2, Long ticker1
        self.performance['strategy_returns'] = (
            self.performance['signal'].shift(1) * (
                self.performance[f'{self.ticker2}_returns'] -
                self.hedge_ratio * self.performance[f'{self.ticker1}_returns']
            )
        )
        
        # Calculate cumulative returns
        self.performance['cumulative_returns'] = (1 + self.performance['strategy_returns']).cumprod()
        
        # Calculate portfolio value
        self.performance['portfolio_value'] = initial_capital * self.performance['cumulative_returns']
        
        # Calculate drawdown
        self.performance['peak'] = self.performance['portfolio_value'].cummax()
        self.performance['drawdown'] = (self.performance['portfolio_value'] - self.performance['peak']) / self.performance['peak']
        
        # Calculate additional performance metrics
        total_return = self.performance['cumulative_returns'].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(self.performance)) - 1
        sharpe_ratio = np.sqrt(252) * (
            self.performance['strategy_returns'].mean() / 
            self.performance['strategy_returns'].std()
        )
        max_drawdown = self.performance['drawdown'].min()
        
        
        self.metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        return self.performance, self.metrics
    
    def plot_results(self):
        """
        Plot the results of the pair trading strategy.
        
        Returns:
        matplotlib.figure.Figure: Figure with the plotted results
        """
        if self.performance is None:
            self.backtest()
            
        # Create figure with 4 subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 16))
        
        # Plot 1: Stock prices
        axes[0].plot(self.data.index, self.data[self.ticker1], label=self.ticker1)
        axes[0].plot(self.data.index, self.data[self.ticker2], label=self.ticker2)
        axes[0].set_title('Stock Prices')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot 2: Spread and z-scores
        ax2 = axes[1]
        ax2.plot(self.spread.index, self.spread, label='Spread')
        ax2.set_title('Spread between Stocks')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(self.z_scores.index, self.z_scores, 'r-', label='Z-Score')
        ax2_twin.axhline(y=self.z_threshold, color='g', linestyle='--', alpha=0.5)
        ax2_twin.axhline(y=-self.z_threshold, color='g', linestyle='--', alpha=0.5)
        ax2_twin.set_ylabel('Z-Score')
        ax2_twin.legend(loc='upper right')
        
        # Plot 3: Trading signals
        axes[2].plot(self.signals.index, self.signals['signal'])
        axes[2].set_title('Trading Signals')
        axes[2].set_ylabel('Position (-1, 0, 1)')
        axes[2].grid(True)
        
        # Plot 4: Portfolio performance
        axes[3].plot(self.performance.index, self.performance['portfolio_value'])
        axes[3].set_title('Portfolio Value')
        axes[3].grid(True)
        
        # Add metrics as text
        metrics_text = (
            f"Total Return: {self.metrics['total_return']:.2%}\n"
            f"Annual Return: {self.metrics['annual_return']:.2%}\n"
            f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {self.metrics['max_drawdown']:.2%}"
        )
        axes[3].annotate(metrics_text, xy=(0.05, 0.05), xycoords='axes fraction', 
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        plt.tight_layout()
        return fig

    def export_results(self, filename='pair_trading_results.csv'):
        """
        Export the trading signals and performance to a CSV file.
        
        Parameters:
        filename (str): Name of the CSV file
        
        Returns:
        str: Path to the saved file
        """
        if self.performance is None:
            self.backtest()
            
        # Combine signals and performance into one DataFrame
        export_df = pd.concat([
            self.signals[['signal', self.ticker1, self.ticker2, 'spread', 'z_score']],
            self.performance[['strategy_returns', 'cumulative_returns', 'portfolio_value']]
        ], axis=1)
        
        # Save to CSV
        export_df.to_csv(filename)
        return filename