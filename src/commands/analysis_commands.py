import os
import datetime
import sqlite3
import pandas as pd
import signal
import matplotlib.pyplot as plt
# Use absolute imports
from src.commands.command_base import Command
# Import ExcessReturn from the dataAnalysis module
from src.dataAnalysis.excess_returns import ExcessReturn

class ExcessReturnCommand(Command):
    def __init__(self, tracker):
        self.tracker = tracker

    def execute(self):
        print("Executing Excess Return Analysis...")
        
        # Get tickers from the portfolio
        try:
            with sqlite3.connect(self.tracker.PORTFOLIO_FILE) as conn:
                df_portfolio = pd.read_sql("SELECT ticker FROM portfolio", conn)
                tickers = df_portfolio['ticker'].tolist()
        except Exception as e:
            print(f"Error fetching portfolio tickers: {e}")
            tickers = []
            
        if not tickers:
            print("No tickers found in portfolio. Please add some stocks/assets first.")
            return
        
        # Identify crypto vs stock tickers
        crypto_tickers = [t for t in tickers if t.endswith('-USD')]
        stock_tickers = [t for t in tickers if not t.endswith('-USD')]
        
        # Display information about portfolio composition
        if crypto_tickers:
            print(f"\nCrypto assets detected: {len(crypto_tickers)} tickers")
            print(f"  {', '.join(crypto_tickers)}")
        if stock_tickers:
            print(f"\nStock assets detected: {len(stock_tickers)} tickers")
            print(f"  {', '.join(stock_tickers)}")
        
        # Ask user if they want to override default benchmarks
        print("\nDefault benchmarks:")
        print("- BTC-USD for cryptocurrency assets")
        print("- ^IXIC (NASDAQ) for stock assets")
        
        override = input("\nDo you want to override these defaults? (y/n, default: n): ").strip().lower() == 'y'
        
        if override:
            crypto_benchmark = input("Enter benchmark for crypto assets (default: BTC-USD): ").strip() or "BTC-USD"
            stock_benchmark = input("Enter benchmark for stock assets (default: ^IXIC): ").strip() or "^IXIC"
        else:
            crypto_benchmark = "BTC-USD"
            stock_benchmark = "^IXIC"
        
        print(f"\nUsing {crypto_benchmark} as benchmark for crypto assets")
        print(f"Using {stock_benchmark} as benchmark for stock assets\n")
        
        # Get user's preference for date range
        days_str = input("Enter number of days to analyze (default: 100): ").strip() or "100"
        try:
            days = int(days_str)
        except ValueError:
            print("Invalid days value. Using default 100 days.")
            days = 100
        
        # Default date range based on user preference
        end_date = datetime.datetime.now()
        default_start_date = end_date - datetime.timedelta(days=days)
        
        # Determine actual date range based on aggregate portfolio data
        start_date = default_start_date
        
        # Check aggregate portfolio for earliest dates of these tickers
        try:
            with sqlite3.connect(self.tracker.AGGREGATE_PORTFOLIO_FILE) as conn:
                # Get the earliest date any of these tickers appears in the aggregate portfolio
                placeholders = ','.join(['?'] * len(tickers))
                query = f"""
                    SELECT MIN(aggregated_date) as earliest_date
                    FROM aggregate_portfolio
                    WHERE ticker IN ({placeholders})
                """
                df_dates = pd.read_sql_query(query, conn, params=tickers)
                
                if not df_dates.empty and df_dates['earliest_date'].iloc[0] is not None:
                    earliest_date = datetime.datetime.fromisoformat(df_dates['earliest_date'].iloc[0])
                    # Use the later of the default start date or earliest portfolio date
                    # This ensures we don't try to fetch data from before the ticker was in portfolio
                    if earliest_date > default_start_date:
                        start_date = earliest_date
                    print(f"Using start date: {start_date.date()} based on portfolio history")
        except Exception as e:
            print(f"Error checking aggregate portfolio for dates: {e}")
            print(f"Falling back to default date range of {days} days")
        
        # Create a plot directory in the root of RetailPortfolioTracking
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        plot_dir = os.path.join(root_dir, "plot")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Create a specific folder for this analysis with date in name
        current_date = datetime.datetime.now().strftime('%Y%m%d')
        analysis_dir = os.path.join(plot_dir, f"{current_date}_excessive")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Prepare all tickers and their appropriate benchmarks for combined analysis
        analysis_tickers = tickers.copy()
        
        # Make sure both benchmarks are included in the analysis list
        benchmarks = []
        if crypto_tickers and crypto_benchmark not in analysis_tickers:
            analysis_tickers.append(crypto_benchmark)
            benchmarks.append(crypto_benchmark)
        
        if stock_tickers and stock_benchmark not in analysis_tickers:
            analysis_tickers.append(stock_benchmark)
            benchmarks.append(stock_benchmark)
            
        # Create a mapping of tickers to their appropriate benchmarks
        ticker_benchmark_map = {}
        for ticker in tickers:
            if ticker.endswith('-USD'):
                ticker_benchmark_map[ticker] = crypto_benchmark
            else:
                ticker_benchmark_map[ticker] = stock_benchmark
                
        print(f"\nRunning combined analysis for {len(tickers)} tickers from {start_date.date()} to {end_date.date()}...")
        
        try:
            # Properly save and disable the alarm during analysis
            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, signal.SIG_IGN)
            signal.alarm(0)  # Cancel any existing alarm
            
            print(f"Running analysis for {len(tickers)} tickers with asset-specific benchmarks...")
            
            # Create analyzer with plot_immediately=False and pass the benchmark_map
            analyzer = ExcessReturn(
                tickers, 
                start_date=start_date, 
                end_date=end_date, 
                benchmark=stock_benchmark,  # Default benchmark (for any unmatched tickers)
                db_file=self.tracker.DB_FILE,
                plot_dir=analysis_dir,
                plot_immediately=False,
                benchmark_map=ticker_benchmark_map  # Pass the mapping of tickers to benchmarks
            )
            
            # Explicitly call plotting methods with show=False
            print(f"Generating plots in {analysis_dir}...")
            analyzer.plot_excess_returns(save=True, show=False)
            analyzer.plot_cumulative_excess(save=True, show=False)
            
            # Get and display summary statistics
            all_stats = analyzer.get_summary_stats()
            if all_stats is not None:
                # Add a column showing which benchmark was used for each ticker
                all_stats['benchmark_used'] = all_stats['ticker'].map(
                    lambda t: ticker_benchmark_map.get(t, stock_benchmark) if t in tickers else 'N/A'
                )
                # Ask if user wants to export stats to CSV
                export = input("\nExport summary statistics to CSV? (y/n): ").strip().lower()
                if export == 'y':
                    try:
                        csv_path = os.path.join(analysis_dir, f"excess_returns_{current_date}.csv")
                        all_stats.to_csv(csv_path, index=False)
                        print(f"Statistics exported to {csv_path}")
                    except Exception as e:
                        print(f"Error exporting statistics: {e}")
                
                # Display notice about benchmark limitations
                print("\nNOTE: Due to technical limitations, the excess return plots use a single benchmark.")
                print(f"The plots use {stock_benchmark} as the common benchmark.")
                print("The correct benchmark-specific excess returns are shown in the summary statistics.")
            
            print("\nAnalysis complete. Plots saved in the 'plot' directory.")
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Close all matplotlib figures to free resources
            plt.close('all')
            # Restore the original signal handler
            signal.signal(signal.SIGALRM, old_handler)
            # We don't set an alarm here - the CommandLine class will handle that
