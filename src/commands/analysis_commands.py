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
            
        # Allow user to specify a different benchmark
        benchmark = input(f"Enter benchmark ticker (default: ^IXIC for NASDAQ): ").strip() or "^IXIC"
        
        # Allow user to specify a custom date range
        days_str = input("Enter number of days to analyze (default: 100): ").strip() or "100"
        try:
            days = int(days_str)
        except ValueError:
            print("Invalid days value. Using default 100 days.")
            days = 100
            
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
        print(f"Running analysis for {len(tickers)} tickers from {start_date.date()} to {end_date.date()}...")
        
        # Create a plot directory in the root of RetailPortfolioTracking
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        plot_dir = os.path.join(root_dir, "plot")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Create a specific folder for this analysis with date in name
        current_date = datetime.datetime.now().strftime('%Y%m%d')
        analysis_dir = os.path.join(plot_dir, f"{current_date}_excessive")
        os.makedirs(analysis_dir, exist_ok=True)
        
        try:
            # Set timeout to None to prevent SIGALRM from interfering with analysis
            old_alarm = signal.signal(signal.SIGALRM, signal.SIG_IGN)
            signal.alarm(0)  # Cancel any existing alarm
            
            # Create analyzer with plot_immediately=False to avoid blocking
            analyzer = ExcessReturn(
                tickers, 
                start_date=start_date, 
                end_date=end_date, 
                benchmark=benchmark,
                db_file=self.tracker.DB_FILE,
                plot_dir=analysis_dir,
                plot_immediately=False  # Turn off automatic plotting
            )
            
            # Explicitly call plotting methods with show=False
            print(f"Generating plots in {analysis_dir}...")
            analyzer.plot_excess_returns(save=True, show=False)
            analyzer.plot_cumulative_excess(save=True, show=False)
            
            # Get and display summary statistics
            stats = analyzer.get_summary_stats()
            if stats is not None:
                # Ask if user wants to export stats to CSV
                export = input("Export summary statistics to CSV? (y/n): ").strip().lower()
                if export == 'y':
                    try:
                        csv_path = os.path.join(analysis_dir, f"excess_returns_{current_date}.csv")
                        stats.to_csv(csv_path, index=False)
                        print(f"Statistics exported to {csv_path}")
                    except Exception as e:
                        print(f"Error exporting statistics: {e}")
                        
            print("\nAnalysis complete. Plots saved in the 'plot' directory.")
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Close all matplotlib figures to free resources
            plt.close('all')
            # Restore the original signal handler and reset the alarm
            signal.signal(signal.SIGALRM, old_alarm)
