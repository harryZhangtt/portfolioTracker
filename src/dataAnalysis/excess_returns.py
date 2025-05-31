import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
import time
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor

class ExcessReturn:
    """
    A modular class for analyzing and visualizing excess returns of assets compared to a benchmark.
    """

    def __init__(self, tickers, start_date=None, end_date=None, benchmark='^IXIC', ticker_names=None, 
                 db_file=None, plot_immediately=True, plot_dir=None, benchmark_map=None):
        """
        Initialize the ExcessReturn analyzer with tickers and date range.
        
        Args:
            tickers (list): List of ticker symbols to analyze
            start_date (str or datetime): Start date for analysis, defaults to 1 year ago
            end_date (str or datetime): End date for analysis, defaults to today
            benchmark (str): Default ticker symbol for the benchmark, defaults to NASDAQ (^IXIC)
            ticker_names (dict): Dictionary mapping ticker symbols to display names
            db_file (str): Path to SQLite database with price data
            plot_immediately (bool): Whether to plot charts immediately after processing data
            plot_dir (str): Directory to save plots to, defaults to PROJECT_ROOT/plot/
            benchmark_map (dict): Dictionary mapping ticker symbols to their benchmarks
        """
        # Set up date range
        if end_date is None:
            self.end_date = datetime.now()
        elif isinstance(end_date, str):
            self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            self.end_date = end_date
            
        if start_date is None:
            self.start_date = self.end_date - timedelta(days=365)
        elif isinstance(start_date, str):
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            self.start_date = start_date
        
        # Set up plot directory
        if plot_dir is None:
            # Default to PROJECT_ROOT/plot/
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.plot_dir = os.path.join(current_dir, "plot")
        else:
            self.plot_dir = plot_dir
            
        # Ensure plot directory exists
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Store benchmark mapping
        self.benchmark_map = benchmark_map or {}
        self.default_benchmark = benchmark
        
        # Ensure all benchmarks are included in tickers list
        self.tickers = list(tickers)
        if benchmark not in self.tickers:
            self.tickers.append(benchmark)
            
        # Add all benchmarks from the mapping to the tickers list
        all_benchmarks = set(self.benchmark_map.values())
        for bench in all_benchmarks:
            if bench not in self.tickers:
                self.tickers.append(bench)
            
        # Set up ticker names for display
        self.ticker_names = ticker_names or {}
        self._setup_default_ticker_names()
        
        # Set database file path
        self.db_file = db_file
        
        # Initialize data placeholders
        self.data = None
        self.aligned_data = None
        self.normalized_data = None
        self.excess_returns = {}
        
        # Set plot style
        plt.style.use('fivethirtyeight')
        sns.set_palette("viridis")
        
        # Generate timestamp for unique filenames
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Immediately fetch data when initialized
        self.fetch_data()
        if self.process_data() and plot_immediately:
            try:
                self.plot_excess_returns(save=True)
                self.plot_cumulative_excess(save=True)
            except Exception as e:
                print(f"Error while plotting: {e}")
                print("Analysis completed but could not save plots.")

    def _setup_default_ticker_names(self):
        """Add default names for common tickers if not provided"""
        default_names = {
            'ETH-USD': 'Ethereum', 'BTC-USD': 'Bitcoin', 'DOGE-USD': 'Dogecoin',
            'TSLA': 'Tesla', 'AAPL': 'Apple', 'MSFT': 'Microsoft',
            'AMZN': 'Amazon', 'GOOGL': 'Google', 'META': 'Meta',
            'NVDA': 'NVIDIA', 'AMD': 'AMD', 'INTC': 'Intel',
            'TSM': 'TSMC', 'ASML': 'ASML', 'AMAT': 'Applied Materials',
            '^IXIC': 'NASDAQ', '^GSPC': 'S&P 500', '^DJI': 'Dow Jones'
        }
        
        for ticker, name in default_names.items():
            if ticker not in self.ticker_names:
                self.ticker_names[ticker] = name

    def _fetch_from_db(self, ticker):
        """Attempt to fetch pricing data for a ticker from the database"""
        if not self.db_file or not os.path.exists(self.db_file):
            return None
            
        try:
            conn = sqlite3.connect(self.db_file)
            
            # Try to get data from the expanded_df table - use date format consistently
            query = """
                SELECT date, price 
                FROM expanded_df 
                WHERE ticker = ? 
                AND date BETWEEN ? AND ?
                ORDER BY date
            """
            
            start_date_str = self.start_date.strftime('%Y-%m-%d')
            end_date_str = self.end_date.strftime('%Y-%m-%d')
            
            params = (ticker, start_date_str, end_date_str)
            
            # Debug query
            print(f"DB Query for {ticker}: date between {start_date_str} and {end_date_str}")
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if not df.empty:
                print(f"Found {len(df)} price records in DB for {ticker}")
                # Convert to datetime with consistent timezone handling
                df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
                df = df.set_index('date')
                # Ensure price is numeric
                df['price'] = pd.to_numeric(df['price'], errors='coerce')
                series = df['price']
                series.name = ticker
                return series
            else:
                print(f"No price data in DB for {ticker} in date range")
                
        except Exception as e:
            print(f"Error fetching {ticker} from database: {e}")
            import traceback
            traceback.print_exc()
            
        return None

    def fetch_data(self, threads=1):
        """
        Fetch data for all tickers, first trying from database, then from Yahoo Finance.
        
        Args:
            threads (int): Number of threads to use for downloading data
            
        Returns:
            bool: True if data was successfully fetched
        """
        print(f"Fetching data from {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}...")
        
        all_data = {}  # Use a dictionary to store ticker:series pairs
        self.available_tickers = []
        self.missing_tickers = []
        
        def download_ticker(ticker):
            # First try to get data from database
            db_series = self._fetch_from_db(ticker)
            if db_series is not None and not db_series.empty:
                print(f"Using database data for {ticker} ({len(db_series)} data points)")
                return db_series, ticker
                
            # If not in database, download from Yahoo Finance
            try:
                print(f"Downloading {ticker} from Yahoo Finance...")
                # Set auto_adjust=False to match previous behavior
                ticker_data = yf.download(
                    ticker, 
                    start=self.start_date, 
                    end=self.end_date,
                    auto_adjust=False,  # Explicitly set to False
                    progress=False
                )
                    
                # Add a delay to avoid rate limiting
                time.sleep(1)
            
                # Check if we got data
                if ticker_data is None or ticker_data.empty:
                    return None, ticker
                
                # Make sure we have the Close column
                if 'Close' in ticker_data.columns:
                    # Get just the Close prices as a Series with the ticker name
                    close_series = ticker_data['Close']
                    # Ensure it's a proper 1D Series
                    if not isinstance(close_series, pd.Series):
                        close_series = pd.Series(close_series.values.flatten(), index=ticker_data.index)
                    close_series.name = ticker
                    return close_series, ticker
                
                return None, ticker
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
                import traceback
                traceback.print_exc()
                return None, ticker
        
        # Process each ticker individually to better handle failures
        for ticker in self.tickers:
            result, ticker_name = download_ticker(ticker)
            if result is not None:
                # Make sure result is a proper 1D Series before adding it
                if isinstance(result, pd.Series) and result.ndim == 1:
                    all_data[ticker_name] = result  # Store in dictionary with ticker as key
                    self.available_tickers.append(ticker_name)
                else:
                    print(f"Warning: Data for {ticker_name} is not a 1D Series. Skipping.")
            else:
                self.missing_tickers.append(ticker_name)
        
        # Check if we have any data
        if not all_data:
            print("Failed to fetch any data.")
            return False
        
        try:
            # Store the raw data as a DataFrame - with better error handling
            self.data = pd.DataFrame(all_data)
            
            # Print data info for debugging
            print(f"Data shape: {self.data.shape}")
            print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
            
            # Check if benchmark is available
            if self.default_benchmark not in self.data.columns:
                print(f"Warning: Benchmark {self.default_benchmark} not available in dataset.")
                if len(self.available_tickers) > 0:
                    print(f"Using {self.available_tickers[0]} as benchmark instead.")
                    self.default_benchmark = self.available_tickers[0]
                else:
                    return False
                    
            return True
        except ValueError as e:
            print(f"Error creating DataFrame: {e}")
            # Debug info to help identify problematic series
            for ticker, series in all_data.items():
                print(f"Ticker: {ticker}, Type: {type(series)}, Shape: {getattr(series, 'shape', 'N/A')}, ndim: {getattr(series, 'ndim', 'N/A')}")
            return False

    def process_data(self):
        """
        Process the fetched data - align dates and normalize values.
        
        Returns:
            bool: True if processing was successful
        """
        if self.data is None or self.data.empty:
            print("No data to process. Run fetch_data() first.")
            return False
            
        print("\nAligning time series data...")
        
        # For stocks vs crypto alignment, we need consistent handling of weekends/holidays
        # First, ensure data is sorted by date
        self.data = self.data.sort_index()
        
        # Filter to date range while preserving all tickers
        self.aligned_data = self.data[
            (self.data.index >= self.start_date) & 
            (self.data.index <= self.end_date)
        ]
        
        # For any missing dates by ticker, forward-fill values (handles weekend gaps)
        self.aligned_data = self.aligned_data.fillna(method='ffill')
        
        print(f"Original data shape: {self.data.shape}")
        print(f"Aligned data shape: {self.aligned_data.shape}")
        print(f"Aligned date range: {self.aligned_data.index.min()} to {self.aligned_data.index.max()}")
        
        # Check if we lost too many data points
        if len(self.aligned_data) < 0.5 * len(self.data):
            print("Warning: More than 50% of data points lost during alignment.")
            
        # Re-normalize the aligned data
        self.normalized_data = self.aligned_data.copy()
        
        for ticker in self.normalized_data.columns:
            # Skip columns with all NaN values
            if self.normalized_data[ticker].notna().any():
                # Find the first non-NaN value to normalize
                first_valid = self.normalized_data[ticker].first_valid_index()
                if first_valid is not None:
                    base_value = self.normalized_data.loc[first_valid, ticker]
                    self.normalized_data[ticker] = self.normalized_data[ticker] / base_value * 100
        
        # Calculate excess returns compared to appropriate benchmark for each ticker
        self.excess_returns = {}
        for ticker in self.aligned_data.columns:
            # Skip processing benchmarks as assets
            if ticker in set(self.benchmark_map.values()) or ticker == self.default_benchmark:
                continue
                
            # Determine which benchmark to use for this ticker
            ticker_benchmark = self.benchmark_map.get(ticker, self.default_benchmark)
            
            if ticker_benchmark in self.aligned_data.columns:
                # Fix for FutureWarning in pct_change
                benchmark_returns = self.aligned_data[ticker_benchmark].pct_change(fill_method=None).dropna()
                
                # Calculate excess returns for this asset
                asset_returns = self.aligned_data[ticker].pct_change(fill_method=None).dropna()
                
                # Align the index for proper calculation
                common_idx = asset_returns.index.intersection(benchmark_returns.index)
                if len(common_idx) > 0:
                    excess = asset_returns[common_idx] - benchmark_returns[common_idx]
                    self.excess_returns[ticker] = {
                        'mean': excess.mean(),
                        'cumulative': (1 + excess).cumprod() - 1,
                        'values': excess,
                        'benchmark': ticker_benchmark  # Store which benchmark was used
                    }
        
        return True

    def plot_excess_returns(self, figsize=(18, 12), max_cols=3, save=True, show=False):
        """
        Create plots showing asset performance vs benchmark and excess returns.
        
        Args:
            figsize (tuple): Size of the figure (width, height)
            max_cols (int): Maximum number of columns in the plot grid
            save (bool): Whether to save the plot to disk
            show (bool): Whether to display the plot
            
        Returns:
            str: Path to saved plot file if save=True, otherwise None
        """
        if self.normalized_data is None:
            print("No normalized data available.")
            return None
            
        try:
            # Filter out benchmarks from being plotted as assets
            benchmarks_set = set(self.benchmark_map.values())
            if self.default_benchmark not in benchmarks_set:
                benchmarks_set.add(self.default_benchmark)
                
            plot_tickers = [t for t in self.available_tickers if t not in benchmarks_set]
            
            # Calculate rows and cols for the plot
            num_tickers = len(plot_tickers)
            rows = (num_tickers + max_cols - 1) // max_cols
            
            # Create figure with calculated dimensions; adjust height based on row count
            fig, axes = plt.subplots(rows, max_cols, figsize=(figsize[0], 4*rows))
            axes = axes.flatten() if hasattr(axes, 'flatten') else np.array([axes])
            
            plot_index = 0
            
            # Plot each ticker with its appropriate benchmark
            for ticker in plot_tickers:
                if plot_index < len(axes):
                    # Get the data for this asset
                    asset_data = self.normalized_data[ticker]
                    
                    # Determine which benchmark to use
                    ticker_benchmark = self.benchmark_map.get(ticker, self.default_benchmark)
                    benchmark_data = self.normalized_data[ticker_benchmark]
                    
                    # Plot both lines
                    axes[plot_index].plot(asset_data, linewidth=2, color='black', 
                                        label=self.ticker_names.get(ticker, ticker))
                    axes[plot_index].plot(benchmark_data, linewidth=1.5, color='blue', 
                                        linestyle='--', alpha=0.7, 
                                        label=f"{self.ticker_names.get(ticker_benchmark, ticker_benchmark)} (Benchmark)")
                    
                    # Fill areas between curves segment by segment
                    for j in range(len(asset_data)-1):
                        if j+1 < len(asset_data.index) and j < len(benchmark_data.index):
                            if asset_data.iloc[j] >= benchmark_data.iloc[j]:
                                # Asset above benchmark - fill green
                                axes[plot_index].fill_between(
                                    [asset_data.index[j], asset_data.index[j+1]],
                                    [asset_data.iloc[j], asset_data.iloc[j+1]],
                                    [benchmark_data.iloc[j], benchmark_data.iloc[j+1]],
                                    color='green', alpha=0.3
                                )
                            else:
                                # Asset below benchmark - fill red
                                axes[plot_index].fill_between(
                                    [asset_data.index[j], asset_data.index[j+1]],
                                    [asset_data.iloc[j], asset_data.iloc[j+1]],
                                    [benchmark_data.iloc[j], benchmark_data.iloc[j+1]],
                                    color='red', alpha=0.3
                                )
                    
                    # Add annotation about excess return and which benchmark was used
                    if ticker in self.excess_returns:
                        excess_mean = self.excess_returns[ticker]['mean']
                        used_benchmark = self.excess_returns[ticker]['benchmark']
                        benchmark_name = self.ticker_names.get(used_benchmark, used_benchmark)
                        
                        is_outperforming = excess_mean > 0
                        performance = "Outperforming" if is_outperforming else "Underperforming"
                        excessive_text = f"{performance} {benchmark_name}\nExcess Return: {excess_mean:.2%}"
                        
                        # Calculate font size based on figure size
                        font_size = max(8, min(11, figsize[0] / 2))
                        
                        # Position the annotation in top right
                        axes[plot_index].annotate(
                            excessive_text, 
                            xy=(0.95, 0.95), xycoords='axes fraction', 
                            ha='right', va='top',
                            bbox=dict(boxstyle="round,pad=0.3", 
                                    fc='green' if is_outperforming else 'red', 
                                    alpha=0.3),
                            fontsize=font_size, color='black'
                        )
                    
                    axes[plot_index].set_title(f"{self.ticker_names.get(ticker, ticker)} ({ticker})")
                    axes[plot_index].set_ylabel('Normalized Price (100 = Start)')
                    axes[plot_index].grid(True, alpha=0.3)
                    axes[plot_index].legend(loc='upper left')
                    
                    plot_index += 1
            
            # Remove any empty subplots
            for i in range(plot_index, len(axes)):
                fig.delaxes(axes[i])
            
            # Using tight_layout with safer padding to avoid errors
            try:
                plt.tight_layout(pad=1.5)
            except Exception:
                print("Warning: tight_layout failed, using default layout")
                
            plt.suptitle(
                'Asset Performance vs. Benchmark (Green = Outperforming, Red = Underperforming)', 
                fontsize=16
            )
            plt.subplots_adjust(top=0.92)
            
            # Save the figure to the plots directory
            plot_path = None
            if save:
                period = f"{self.start_date.strftime('%Y%m%d')}_to_{self.end_date.strftime('%Y%m%d')}"
                plot_filename = f"excess_returns_{period}_{self.timestamp}.png"
                plot_path = os.path.join(self.plot_dir, plot_filename)
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"Plot saved to: {plot_path}")
            
            # Display the plot if requested - use non-blocking mode to avoid hanging
            if show:
                plt.show(block=False)
                plt.pause(0.1)  # Give a small pause to render but not block
            else:
                plt.close(fig)
            
            return plot_path
            
        except Exception as e:
            print(f"Error plotting excess returns: {e}")
            import traceback
            traceback.print_exc()
            return None

    def plot_cumulative_excess(self, figsize=(12, 8), save=True, show=False):
        """
        Plot the cumulative excess return for each ticker compared to the benchmark.
        
        Args:
            figsize (tuple): Figure size
            save (bool): Whether to save the plot to disk
            show (bool): Whether to display the plot
            
        Returns:
            str: Path to saved plot file if save=True, otherwise None
        """
        if not self.excess_returns:
            print("No excess returns calculated yet.")
            return None
        
        try:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Group tickers by their benchmark for better visualization
            benchmark_groups = {}
            for ticker, data in self.excess_returns.items():
                benchmark = data['benchmark']
                if benchmark not in benchmark_groups:
                    benchmark_groups[benchmark] = []
                benchmark_groups[benchmark].append(ticker)
            
            # Use different line styles for each benchmark group
            line_styles = ['-', '--', '-.', ':']
            colors = plt.cm.tab10.colors
            
            # Plot each benchmark group with distinct styles
            for i, (benchmark, ticker_group) in enumerate(benchmark_groups.items()):
                benchmark_name = self.ticker_names.get(benchmark, benchmark)
                line_style = line_styles[i % len(line_styles)]
                
                # Plot each ticker in this benchmark group
                for j, ticker in enumerate(ticker_group):
                    data = self.excess_returns[ticker]
                    color = colors[(i*len(ticker_group) + j) % len(colors)]
                    if 'cumulative' in data:
                        cumulative = data['cumulative']
                        label = f"{self.ticker_names.get(ticker, ticker)} (vs. {benchmark_name})"
                        ax.plot(cumulative.index, cumulative.values * 100, 
                              label=label, linestyle=line_style, color=color)
            
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.set_title('Cumulative Excess Return vs. Respective Benchmarks')
            ax.set_ylabel('Excess Return (%)')
            
            # Add legend with smaller font size and better placement
            if len(self.excess_returns) > 10:
                ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))
            else:
                ax.legend(loc='best')
            
            plt.tight_layout()
            
            # Save the figure to the plots directory
            plot_path = None
            if save:
                period = f"{self.start_date.strftime('%Y%m%d')}_to_{self.end_date.strftime('%Y%m%d')}"
                plot_filename = f"cumulative_excess_{period}_{self.timestamp}.png"
                plot_path = os.path.join(self.plot_dir, plot_filename)
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"Plot saved to: {plot_path}")
            
            # Display the plot if requested - use non-blocking mode to avoid hanging
            if show:
                plt.show(block=False)
                plt.pause(0.1)  # Give a small pause to render but not block
            else:
                plt.close(fig)
            
            return plot_path
            
        except Exception as e:
            print(f"Error plotting cumulative excess: {e}")
            return None

    def get_summary_stats(self):
        """
        Return summary statistics for all tickers.
        
        Returns:
            DataFrame: Summary statistics
        """
        if self.aligned_data is None:
            print("No data available. Run fetch_data() and process_data() first.")
            return None
        
        stats = []
        
        for ticker in self.available_tickers:
            # Skip benchmarks
            if ticker in set(self.benchmark_map.values()) or ticker == self.default_benchmark:
                continue
                
            if ticker in self.aligned_data.columns:
                # Fix for FutureWarning in pct_change
                returns = self.aligned_data[ticker].pct_change(fill_method=None).dropna()
                
                ticker_stats = {
                    'ticker': ticker,
                    'name': self.ticker_names.get(ticker, ticker),
                    'total_return': (self.aligned_data[ticker].iloc[-1] / self.aligned_data[ticker].iloc[0] - 1) * 100,
                    'annualized_return': (((self.aligned_data[ticker].iloc[-1] / self.aligned_data[ticker].iloc[0]) ** 
                                        (252 / len(self.aligned_data))) - 1) * 100,
                    'volatility': returns.std() * np.sqrt(252) * 100,  # Annualized volatility
                    'sharpe': ((returns.mean() * 252) / (returns.std() * np.sqrt(252))) if returns.std() > 0 else 0
                }
                
                # Add excess returns if available
                if ticker in self.excess_returns:
                    ticker_stats['excess_return'] = self.excess_returns[ticker]['mean'] * 100
                    ticker_stats['benchmark'] = self.excess_returns[ticker]['benchmark']
                else:
                    ticker_stats['excess_return'] = None
                    ticker_stats['benchmark'] = self.benchmark_map.get(ticker, self.default_benchmark)
                    
                stats.append(ticker_stats)
        
        if stats:
            df_stats = pd.DataFrame(stats)
            # Reorder columns for better display
            cols = ['ticker', 'name', 'total_return', 'annualized_return', 
                   'volatility', 'sharpe', 'excess_return', 'benchmark']
            df_stats = df_stats[[col for col in cols if col in df_stats.columns]]
            return df_stats
        
        return None

