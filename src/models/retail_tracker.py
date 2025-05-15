import os
import datetime
import sqlite3
import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
import time
from .state_manager import StateManager

class RetailTracker:
    # Update file paths to use database directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/
    DATABASE_DIR = os.path.join(os.path.dirname(BASE_DIR), "database")
    
    # Make sure the database directory exists
    os.makedirs(DATABASE_DIR, exist_ok=True)
    
    STATE_FILE = os.path.join(DATABASE_DIR, "tracker_state.json")
    PORTFOLIO_FILE = os.path.join(DATABASE_DIR, "portfolio.db")
    DB_FILE = os.path.join(DATABASE_DIR, "price_data.db")
    AGGREGATE_PORTFOLIO_FILE = os.path.join(DATABASE_DIR, "aggregate_portfolio.db")

    def __init__(self):
        self.state_manager = StateManager(self.STATE_FILE)
        self.state_manager.tracker = self  # Set tracker reference in StateManager
        self._available_fund = self.state_manager.available_balance
        self._asset_amount = 0
        self.df = None              # Portfolio DataFrame
        self.expanded_df = None     # Expanded DataFrame with historical prices

        # Mapping for crypto tickers to Yahoo Finance format
        self.crypto_map = {
            "ETH": "ETH-USD",
            "BTC": "BTC-USD",
            "DOGE": "DOGE-USD",
            "Sui": "SUI20947-USD",
            "Trump": "TRUMP-OFFICIAL-USD",
            "ADA": "ADA-USD",
            'XRP': 'XRP-USD',
        }

    # ----- Property Methods -----
    @property
    def asset_amount(self):
        return self._asset_amount

    @asset_amount.setter
    def asset_amount(self, value):
        self._asset_amount = value

    @property
    def available_fund(self):
        return self._available_fund

    @available_fund.setter
    def available_fund(self, value):
        self._available_fund = value

    @property
    def total_amount(self):
        if isinstance(self._asset_amount, pd.Series):
            return self._asset_amount + self._available_fund
        return self._asset_amount + self._available_fund

    def save_to_db(self):
        if self.expanded_df is None or self.expanded_df.empty:
            print("[INFO] No data available in expanded_df to save.")
            return

        # Optionally drop the aggregated_date column if it's not needed.
        if "aggregated_date" in self.expanded_df.columns:
            self.expanded_df = self.expanded_df.drop("aggregated_date", axis=1)

        # Ensure the 'date' column is of type date.
        if "date" in self.expanded_df.columns:
            # Convert to datetime then to date (in case it isn't already)
            self.expanded_df["date"] = pd.to_datetime(self.expanded_df["date"]).dt.date
        else:
            print("[ERROR] 'date' column not found in expanded_df.")
            return

        # Determine start and end date from the DataFrame.
        start_date = self.expanded_df["date"].min()
        end_date = self.expanded_df["date"].max()

        # Pre-condition: Create the DB file and table if they do not exist.
        if not os.path.exists(self.DB_FILE):
            print(f"[INFO] Database file {self.DB_FILE} does not exist. Creating a new DB and table.")
            with sqlite3.connect(self.DB_FILE) as conn:
                try:
                    # Create an empty table based on the DataFrame's structure.
                    self.expanded_df.head(0).to_sql("expanded_df", conn, if_exists="replace", index=False)
                    conn.commit()
                except Exception as e:
                    print(f"[ERROR] Error creating table in the new database: {e}")

        # Iterate over each day in the date range.
        current_date = start_date
        while current_date <= end_date:
            # Filter rows for the current_date.
            day_df = self.expanded_df[self.expanded_df["date"] == current_date]
            # Convert current_date to ISO string for SQLite binding.
            date_str = current_date.isoformat()

            try:
                with sqlite3.connect(self.DB_FILE) as conn:
                    cur = conn.cursor()
                    # Check if data for this date already exists.
                    try:
                        cur.execute("SELECT COUNT(*) FROM expanded_df WHERE date = ?", (date_str,))
                        count = cur.fetchone()[0]
                    except sqlite3.OperationalError:
                        count = 0

                    if count > 0:
                        cur.execute("DELETE FROM expanded_df WHERE date = ?", (date_str,))
                        conn.commit()
                        print(f"[INFO] Existing data for date {date_str} found and overwritten.")
                    else:
                        print(f"[INFO] Date {date_str} not found in DB. Inserting new data in proper order.")

                    # Insert the data for the current date.
                    day_df.to_sql("expanded_df", conn, if_exists="append", index=False)
                    print(f"[SUCCESS] Data for date {date_str} inserted/updated.")
            except Exception as e:
                print(f"[ERROR] Error saving data for date {date_str}: {e}")

            # Move to the next day.
            current_date += datetime.timedelta(days=1)

    def load_from_db(self):
        if not os.path.exists(self.DB_FILE):
            print(f"[INFO] No database file found at {self.DB_FILE}. Running tracker.")
            return None
        conn = None
        try:
            conn = sqlite3.connect(self.DB_FILE)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='expanded_df'")
            if not cursor.fetchone():
                print("[INFO] Database exists but 'expanded_df' table not found. Running tracker.")
                return None
            self.expanded_df = pd.read_sql("SELECT * FROM expanded_df", conn)
            if self.expanded_df.empty:
                print("[INFO] Database table exists but is empty. Running tracker.")
                return None
            if 'date' in self.expanded_df.columns:
                self.expanded_df['date'] = pd.to_datetime(self.expanded_df['date'])
            print(f"[SUCCESS] Loaded {len(self.expanded_df)} rows from {self.DB_FILE}")
            return self.expanded_df
        except Exception as e:
            print(f"[ERROR] Error loading from database: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def load_portfolio_data(self):
        if not os.path.exists(self.PORTFOLIO_FILE):
            print(f"[ERROR] Database file not found: {self.PORTFOLIO_FILE}")
            self.df = None
            return
        conn = sqlite3.connect(self.PORTFOLIO_FILE)
        try:
            self.df = pd.read_sql("SELECT * FROM portfolio", conn)
            required_cols = {'ticker', 'share', 'avg_price'}
            if not required_cols.issubset(self.df.columns):
                raise ValueError(f"[ERROR] Database is missing required columns: {required_cols}")
            self.df['ticker'] = self.df['ticker'].astype(str).str.strip()
            self.df['ticker'] = self.df['ticker'].map(self.crypto_map).fillna(self.df['ticker'])
        except Exception as e:
            print(f"[ERROR] Failed to load data from database: {e}")
            self.df = None
        finally:
            conn.close()
        print(f"✅ Data successfully loaded from database. {len(self.df)} records found.")

    def update_aggregate_portfolio(self):
        # Determine starting date for aggregation from state_manager.last_render_time
        if self.state_manager.last_render_time is None:
            start_date = datetime.datetime.now().date()
        else:
            # Start from the day after the last render date
            start_date = self.state_manager.last_render_time.date() + datetime.timedelta(days=1)
        current_date = datetime.datetime.now().date()

        while start_date <= current_date:
            if not os.path.exists(self.PORTFOLIO_FILE):
                print(f"[ERROR] Portfolio file not found: {self.PORTFOLIO_FILE}")
                return

            try:
                # Read portfolio data from the portfolio database
                with sqlite3.connect(self.PORTFOLIO_FILE) as conn:
                    df_portfolio = pd.read_sql("SELECT * FROM portfolio", conn)

                if df_portfolio.empty:
                    print(f"[INFO] No portfolio data to aggregate for {start_date}")
                else:
                    # Add aggregated_date column to the DataFrame
                    df_portfolio['aggregated_date'] = start_date.isoformat()

                    with sqlite3.connect(self.AGGREGATE_PORTFOLIO_FILE) as agg_conn:
                        cur = agg_conn.cursor()
                        # Check if there's already data for this aggregated_date.
                        try:
                            cur.execute("SELECT COUNT(*) FROM aggregate_portfolio WHERE aggregated_date = ?",
                                        (start_date.isoformat(),))
                            count = cur.fetchone()[0]
                        except sqlite3.OperationalError:
                            # Table doesn't exist yet.
                            count = 0

                        if count > 0:
                            # If data exists for this date, remove it (overwrite mechanism)
                            cur.execute("DELETE FROM aggregate_portfolio WHERE aggregated_date = ?",
                                        (start_date.isoformat(),))
                            agg_conn.commit()
                            print(f"[INFO] Existing aggregated data for {start_date} found and overwritten.")

                        # Insert the new aggregated data for this day
                        df_portfolio.to_sql("aggregate_portfolio", agg_conn, if_exists="append", index=False)
                        print(
                            f"[SUCCESS] Aggregated portfolio updated for {start_date} in {self.AGGREGATE_PORTFOLIO_FILE}.")
            except Exception as e:
                print(f"[ERROR] Failed to update aggregate portfolio for {start_date}: {e}")

            start_date += datetime.timedelta(days=1)

    # ----- Price Data Methods -----
    def get_historical_prices(self, ticker):
        try:
            ticker = self.crypto_map.get(ticker, ticker)
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")
            if hist.empty:
                raise ValueError("No historical data available")
            hist = hist[['Close']].reset_index()
            hist.columns = ['date', 'price']
            hist['date'] = pd.to_datetime(hist['date'])
            hist['ticker'] = ticker
            return hist
        except Exception as e:
            print(f"Error fetching historical prices for {ticker}: {e}")
            return None

    def expand_prices(self):
        price_dfs = []
        for ticker in self.df['ticker'].unique():
            price_data = self.get_historical_prices(ticker)
            if price_data is not None:
                price_dfs.append(price_data)
        if not price_dfs:
            raise ValueError("No price data available for any tickers")
        price_df = pd.concat(price_dfs, ignore_index=True)
        price_df['date'] = pd.to_datetime(price_df['date'], utc=True).dt.tz_convert(None)
        unique_dates = price_df['date'].unique()
        expanded_df = self.df.assign(key=1).merge(pd.DataFrame({'date': unique_dates, 'key': 1}), on='key').drop(columns=['key'])
        expanded_df = expanded_df.merge(price_df, on=['ticker', 'date'], how='left')
        expanded_df.sort_values(by=['ticker', 'date'], inplace=True)
        expanded_df['price'] = expanded_df.groupby('ticker')['price'].ffill()
        expanded_df.dropna(subset=['price'], inplace=True)
        self.expanded_df = expanded_df.sort_values(by=['date', 'ticker'])

    # ----- Portfolio Calculation Methods -----
    def calculate_portfolio_distribution(self):
        self.expanded_df['amount'] = self.expanded_df['price'] * self.expanded_df['share']
        self.expanded_df.dropna(subset=['date'], inplace=True)
        self.expanded_df['date'] = pd.to_datetime(self.expanded_df['date'])
        self._asset_amount = self.expanded_df.groupby('date')['amount'].sum()
        self.expanded_df['percentage'] = self.expanded_df.apply(
            lambda row: (row['amount'] / self.total_amount.get(row['date'], 1)) * 100
            if self.total_amount.get(row['date'], 0) > 0 else 0,
            axis=1
        )
        self.expanded_df['percentage'] = self.expanded_df['percentage'].apply(lambda x: f"{x:.2f} %")

    def calculate_pnl(self):
        self.expanded_df['pnl_amount'] = (self.expanded_df['price'] - self.expanded_df['avg_price']) * self.expanded_df['share']
        self.expanded_df['pnl_percentage'] = ((self.expanded_df['price'] - self.expanded_df['avg_price']) / self.expanded_df['avg_price']) * 100
        self.expanded_df['pnl_percentage'] = self.expanded_df['pnl_percentage'].apply(lambda x: f"{x:.2f} %")

    # ----- Tracker Execution Method -----
    def run_tracker(self):
        if not self.state_manager.can_run_tracker():
            if self.load_from_db() is not None:
                print(f"Tracker was already run today ({self.state_manager.last_render_time.date()}). Using saved data.")
                self.asset_amount = self.expanded_df.groupby('date')['amount'].sum()
                print("Total amount per day:", self.total_amount)
                return self.expanded_df
            else:
                print("No data found, please rerun the program.")

        self.run_tracker_without_statecheck()

    def update_price_db(self):
        # Determine the date range: from last render time to current date.
        if self.state_manager.last_render_time is None:
            last_time = datetime.datetime.now().date()
        else:
            last_time = self.state_manager.last_render_time.date()
        current_date = datetime.datetime.now().date()

        # Fetch the aggregated portfolio rows for the date range.
        try:
            with sqlite3.connect(self.AGGREGATE_PORTFOLIO_FILE) as conn:
                query = f"""
                    SELECT * FROM aggregate_portfolio 
                    WHERE aggregated_date BETWEEN '{last_time.isoformat()}' AND '{current_date.isoformat()}'
                """
                agg_df = pd.read_sql(query, conn)
        except Exception as e:
            print(f"[ERROR] Unable to fetch aggregated portfolio: {e}")
            return None

        if agg_df.empty:
            print(f"[INFO] No aggregated portfolio data for date range {last_time} to {current_date}.")
            return self.expanded_df if hasattr(self, 'expanded_df') else None

        # Work on a copy of the aggregated data.
        expanded_df = agg_df.copy()

        # Convert 'aggregated_date' to a proper datetime column for merging.
        expanded_df['date'] = pd.to_datetime(expanded_df['aggregated_date'])

        def fetch_for_ticker(group, start, end):
            ticker_str = group['ticker'].iloc[0]
            ticker = yf.Ticker(ticker_str)
            print(f"Fetching prices for ticker: {ticker_str}")
            start_str = start.strftime("%Y-%m-%d")
            # Extend end date by one day to include the intended date.
            end_str = (end).strftime("%Y-%m-%d")
            
            # Add retry mechanism with exponential backoff
            max_retries = 5
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    hist = ticker.history(start=start_str, end=end_str, interval='1d')
                    break  # Exit the retry loop if successful
                except YFRateLimitError:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"Failed to fetch data for {ticker_str} after {max_retries} retries.")
                        group['price'] = None
                        return group
                    
                    wait_time = 2 ** retry_count  # Exponential backoff
                    print(f"Rate limit hit for {ticker_str}. Retrying in {wait_time} seconds... ({retry_count}/{max_retries})")
                    time.sleep(wait_time)
                except Exception as e:
                    print(f"Error fetching data for {ticker_str}: {e}")
                    group['price'] = None
                    return group

            if hist.empty:
                print(f"[WARNING] No price data for {ticker_str} between {start_str} and {end_str}")
                group['price'] = None
            else:
                # Reset index to get the date as a column.
                hist = hist.reset_index()
                # Map dates (date only) to the closing price.
                hist['date_only'] = pd.to_datetime(hist['Date']).dt.date
                price_map = hist.set_index('date_only')['Close'].to_dict()
                group['date_only'] = group['date'].dt.date
                group['price'] = group['date_only'].map(price_map)
                group.loc[:, 'price'] = group['price'].ffill()
                mode_series = group['price'].dropna().mode()
                if not mode_series.empty:
                    mode_value = mode_series.iloc[0]
                    group.loc[:, 'price'] = group['price'].fillna(mode_value)

                group.drop(columns=['date_only'], inplace=True)
            return group

        # Group by ticker and apply the fetch_for_ticker function with the start and end dates.
        expanded_df = expanded_df.groupby('ticker', group_keys=False).apply(
            lambda g: fetch_for_ticker(g, last_time, current_date))

        # Update the in-memory expanded_df.
        self.expanded_df = expanded_df.copy()
        # Recalculate portfolio distribution and PnL with the new price data.
        self.calculate_portfolio_distribution()
        self.calculate_pnl()

        # Save the updated expanded DataFrame to the database.
        self.save_to_db()

        return self.expanded_df

    def print_top_gainers_and_losers(self):
        current_date_str = datetime.datetime.now().date().isoformat()
        query_gainers = f"""
            SELECT ticker, ((price - avg_price) * share) as pnl
            FROM expanded_df
            WHERE date = '{current_date_str}'
            ORDER BY pnl DESC
            LIMIT 3;
        """
        with sqlite3.connect(self.DB_FILE) as conn:
            top_gainers = pd.read_sql(query_gainers, conn)
        query_losers = f"""
            SELECT ticker, ((price - avg_price) * share) as pnl
            FROM expanded_df
            WHERE date = '{current_date_str}'
            ORDER BY pnl ASC
            LIMIT 3;
        """
        with sqlite3.connect(self.DB_FILE) as conn:
            top_losers = pd.read_sql(query_losers, conn)
        print(f"Top 3 Gainers for {current_date_str}:")
        print(top_gainers)
        print(f"Top 3 Losers for {current_date_str}:")
        print(top_losers)

    def run_tracker_without_statecheck(self):
        # Load current portfolio data.
        self.load_portfolio_data()
        # Update the aggregated portfolio.
        self.update_aggregate_portfolio()
        # Update the prices for the new date range.
        self.update_price_db()

        # Update state after processing.
        self.state_manager.last_render_time = datetime.datetime.now()
        self.state_manager.saved_data_path = self.DB_FILE
        self.state_manager.save_state()

        print(f"Tracker successfully updated on {self.state_manager.last_render_time.date()}")
        print("Total amount per day:", self.total_amount)
        self.print_top_gainers_and_losers()
        return self.expanded_df
