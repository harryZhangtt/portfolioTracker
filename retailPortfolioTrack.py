import datetime
import os
import json
from symbol import continue_stmt

import pandas as pd
import yfinance as yf
import sqlite3


class RetailTracker:
    STATE_FILE = "tracker_state.json"
    PORTFOLIO_FILE="portfolio.db"
    DB_FILE = "price_data.db"
    def __init__(self):
        """Initialize RetailTracker with the CSV file path."""
        self._available_fund = 2053
        self._asset_amount = 0
        self.df = None
        self.expanded_df = None

        self.last_render_time = None
        self.saved_data_path = None


        # Load saved state
        self.load_state()

        # Mapping for crypto tickers to Yahoo Finance format
        self.crypto_map = {
            "ETH": "ETH-USD",
            "BTC": "BTC-USD",
            "DOGE": "DOGE-USD",
            "Sui": "SUI20947-USD",
            "Trump": "TRUMP-OFFICIAL-USD",
            "SOl": "SOL-USD",
            "ADA": "ADA-USD"
        }

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
        return self._asset_amount + self._available_fund

    def load_state(self):
        """Load last render time and saved data path from a persistent file."""
        if os.path.exists(self.STATE_FILE):
            with open(self.STATE_FILE, "r") as file:
                try:
                    state = json.load(file)
                    self.last_render_time = datetime.datetime.fromisoformat(state.get("last_render_time"))
                    self.saved_data_path = state.get("saved_data_path")
                except Exception as e:
                    print(f"Error loading state: {e}")

    def save_state(self):
        """Save last render time and saved data path to a persistent file."""
        state = {
            "last_render_time": self.last_render_time.isoformat() if self.last_render_time else None,
            "saved_data_path": self.saved_data_path
        }
        with open(self.STATE_FILE, "w") as file:
            json.dump(state, file)

    def can_run_tracker(self):
        """Check if the tracker can be run based on the last execution date."""
        if self.last_render_time is None:
            return True  # First-time run

        return self.last_render_time.date() < datetime.datetime.now().date()  # Ensure it's a new day

    def save_to_db(self):
        """Save expanded_df to an SQLite database for persistence."""
        if self.expanded_df is None or self.expanded_df.empty:
            print("[ERROR] No data to save to database.")
            return False

        conn = None
        try:
            conn = sqlite3.connect(self.DB_FILE)

            # Explicitly create table if not exists
            self.expanded_df.to_sql("expanded_df", conn, if_exists="replace", index=False)

            # Verify data was written correctly
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM expanded_df")
            row_count = cursor.fetchone()[0]

            if row_count != len(self.expanded_df):
                raise Exception(
                    f"Data integrity issue: {row_count} rows in DB vs {len(self.expanded_df)} rows in DataFrame")

            conn.commit()
            print(f"[SUCCESS] Expanded DataFrame successfully saved to {self.DB_FILE} ({row_count} rows)")
            return True

        except Exception as e:
            print(f"[ERROR] Error saving to database: {e}")
            if conn:
                conn.rollback()  # Rollback any changes if there was an error
            return False
        finally:
            if conn:
                conn.close()

    def load_from_db(self):
        """Load expanded_df from SQLite if available."""
        if not os.path.exists(self.DB_FILE):
            print(f"[INFO] No database file found at {self.DB_FILE}. Running tracker.")
            return None

        conn = None
        try:
            conn = sqlite3.connect(self.DB_FILE)

            # Check if the table exists
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='expanded_df'")
            if not cursor.fetchone():
                print("[INFO] Database exists but 'expanded_df' table not found. Running tracker.")
                return None

            # Load the data
            self.expanded_df = pd.read_sql("SELECT * FROM expanded_df", conn)

            if self.expanded_df.empty:
                print("[INFO] Database table exists but is empty. Running tracker.")
                return None

            # Convert date column to datetime if it exists
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
        """Load the portfolio data from the SQLite database into a DataFrame."""
        db_path = self.PORTFOLIO_FILE

        # Check if the database file exists
        import os
        if not os.path.exists(db_path):
            print(f"[ERROR] Database file not found: {db_path}")
            self.df = None
            return

        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)

        try:
            # Read the portfolio data from the "portfolio" table
            self.df = pd.read_sql("SELECT * FROM portfolio", conn)

            # Ensure required columns exist
            required_cols = {'ticker', 'share', 'avg_price'}
            if not required_cols.issubset(self.df.columns):
                raise ValueError(f"[ERROR] Database is missing required columns: {required_cols}")

            # Clean ticker symbols
            self.df['ticker'] = self.df['ticker'].astype(str).str.strip()
            self.df['ticker'] = self.df['ticker'].map(self.crypto_map).fillna(self.df['ticker'])

        except Exception as e:
            print(f"[ERROR] Failed to load data from database: {e}")
            self.df = None  # Set to None if there's an issue

        finally:
            conn.close()  # Always close the connection

        print(f"✅ Data successfully loaded from database. {len(self.df)} records found.")

    def get_historical_prices(self, ticker):
        """Fetch historical prices for the last month for a given ticker."""
        try:
            # Convert crypto symbols if needed
            ticker = self.crypto_map.get(ticker, ticker)

            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")  # Get last month's data

            if hist.empty:
                raise ValueError("No historical data available")

            hist = hist[['Close']].reset_index()  # Keep only Date and Close price
            hist.columns = ['date', 'price']
            hist['date'] = pd.to_datetime(hist['date'])  # Ensure datetime format
            hist['ticker'] = ticker  # Add ticker for merging
            return hist
        except Exception as e:
            print(f"Error fetching historical prices for {ticker}: {e}")
            return None

    def expand_prices(self):
        """Expand DataFrame with historical prices, ensuring alignment on date and ticker."""
        price_dfs = []
        for ticker in self.df['ticker'].unique():
            price_data = self.get_historical_prices(ticker)
            if price_data is not None:
                price_dfs.append(price_data)

        if not price_dfs:
            raise ValueError("No price data available for any tickers")

        # Merge all price data
        price_df = pd.concat(price_dfs, ignore_index=True)

        # Convert to datetime and force UTC if timezone exists, then remove timezone info
        price_df['date'] = pd.to_datetime(price_df['date'], utc=True).dt.tz_convert(None)

        # Step 1: Expand self.df with all unique dates from price_df
        unique_dates = price_df['date'].unique()
        expanded_df = self.df.assign(key=1).merge(pd.DataFrame({'date': unique_dates, 'key': 1}), on='key').drop(
            columns=['key'])

        # Step 2: Merge expanded_df with price_df to get historical prices
        expanded_df = expanded_df.merge(price_df, on=['ticker', 'date'], how='left')

        # Step 3: Forward-fill missing prices for each ticker
        expanded_df.sort_values(by=['ticker', 'date'], inplace=True)
        expanded_df['price'] = expanded_df.groupby('ticker')['price'].ffill()

        # Step 4: Drop any rows where price is still NaN (if some tickers had no data at all)
        expanded_df.dropna(subset=['price'], inplace=True)

        # Store the final expanded dataframe
        self.expanded_df = expanded_df
        self.expanded_df.sort_values(by=['date','ticker'], inplace=True)

    def calculate_portfolio_distribution(self):
        """Calculate amount, PnL, and percentage for each holding."""
        self.expanded_df['amount'] = self.expanded_df['price'] * self.expanded_df['share']

        # Drop rows with NaN in date before calculating total amount
        self.expanded_df.dropna(subset=['date'], inplace=True)

        # Ensure date is in datetime format
        self.expanded_df['date'] = pd.to_datetime(self.expanded_df['date'])

        # Group by date to calculate total amount per day
        self.asset_amount = self.expanded_df.groupby('date')['amount'].sum()

        # Calculate percentage allocation per day safely
        self.expanded_df['percentage'] = self.expanded_df.apply(
            lambda row: (row['amount'] / self.total_amount.get(row['date'], 1)) * 100 if self.total_amount.get(
                row['date'], 0) > 0 else 0,
            axis=1
        )

        # Format percentage
        self.expanded_df['percentage'] = self.expanded_df['percentage'].apply(lambda x: f"{x:.2f} %")

    def calculate_pnl(self):
        """Calculate profit and loss (PnL) in both amount and percentage."""
        self.expanded_df['pnl_amount'] = (self.expanded_df['price'] - self.expanded_df['avg_price']) * self.expanded_df['share']
        self.expanded_df['pnl_percentage'] = ((self.expanded_df['price'] - self.expanded_df['avg_price']) / self.expanded_df['avg_price']) * 100

        # Format PnL percentage
        self.expanded_df['pnl_percentage'] = self.expanded_df['pnl_percentage'].apply(lambda x: f"{x:.2f} %")


    def run_tracker(self):
        """Execute the full pipeline: load, expand prices, calculate, and save."""
        if not self.can_run_tracker():
            if self.load_from_db() is not None:
                print(f"Tracker was already run today ({self.last_render_time.date()}). Using saved data.")
                print(self.expanded_df)
                self.asset_amount = self.expanded_df.groupby('date')['amount'].sum()
                print(self.total_amount)
                return self.expanded_df
            else:
                print("no data found, rerun the program")


        self.load_portfolio_data()
        self.expand_prices()
        self.calculate_portfolio_distribution()
        self.calculate_pnl()
        self.save_to_db()  # Save data to SQLite

        self.last_render_time = datetime.datetime.now()  # Update last render time
        self.saved_data_path = self.DB_FILE  # Store the saved data path

        self.save_state()  # Persist changes

        print(f"Tracker successfully updated on {self.last_render_time.date()}")
        print('Total amount per day:', self.total_amount)
        return self.expanded_df


# Usage Example:
if __name__ == "__main__":
    tracker = RetailTracker()
    tracker.run_tracker()
