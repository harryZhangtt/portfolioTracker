import datetime
import os
import json
import sqlite3
import pandas as pd
import yfinance as yf
from abc import ABC, abstractmethod

# ===== State Management Component =====
class StateManager:
    def __init__(self, state_file="tracker_state.json"):
        self.state_file = state_file
        self.available_balance=None
        self.last_render_time = None
        self.saved_data_path = None
        self.load_state()

    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as file:
                try:
                    state = json.load(file)
                    self.last_render_time = datetime.datetime.fromisoformat(state.get("last_render_time"))
                    self.saved_data_path = state.get("saved_data_path")
                    self.available_balance= state.get("available_balance")
                    print(self.available_balance)

                except Exception as e:
                    print(f"Error loading state: {e}")

    def save_state(self):
        state = {
            "last_render_time": self.last_render_time.isoformat() if self.last_render_time else None,
            "saved_data_path": self.saved_data_path,
            "available_balance": tracker.available_fund
        }
        with open(self.state_file, "w") as file:
            json.dump(state, file)

    def can_run_tracker(self):
        if self.last_render_time is None:
            return True  # First-time run
        return self.last_render_time.date() < datetime.datetime.now().date()

# ===== Main RetailTracker Class =====
class RetailTracker:
    STATE_FILE = "tracker_state.json"
    PORTFOLIO_FILE = "portfolio.db"
    DB_FILE = "price_data.db"
    AGGREGATE_PORTFOLIO_FILE = "aggregate_portfolio.db"


    def __init__(self):
        self.state_manager = StateManager(self.STATE_FILE)
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

    # ----- Database Persistence Methods -----
    def save_to_db(self):
        print(self.expanded_df[self.expanded_df['ticker'=='SQQQ']])
        if self.expanded_df is None or self.expanded_df.empty:
            print("[ERROR] No data to save to database.")
            return False
        conn = None
        try:
            conn = sqlite3.connect(self.DB_FILE)
            self.expanded_df.to_sql("expanded_df", conn, if_exists="replace", index=False)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM expanded_df")
            row_count = cursor.fetchone()[0]
            if row_count != len(self.expanded_df):
                raise Exception(f"Data integrity issue: {row_count} rows in DB vs {len(self.expanded_df)} rows in DataFrame")
            conn.commit()
            print(f"[SUCCESS] Expanded DataFrame successfully saved to {self.DB_FILE} ({row_count} rows)")
            return True
        except Exception as e:
            print(f"[ERROR] Error saving to database: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()

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


    def run_tracker_without_statecheck(self):
        self.load_portfolio_data()
        self.expand_prices()
        self.calculate_portfolio_distribution()
        self.calculate_pnl()
        self.save_to_db()
        self.update_aggregate_portfolio()
        self.state_manager.last_render_time = datetime.datetime.now()
        self.state_manager.saved_data_path = self.DB_FILE
        self.state_manager.save_state()
        print(f"Tracker successfully updated on {self.state_manager.last_render_time.date()}")
        print("Total amount per day:", self.total_amount)
        return self.expanded_df


# ===== Command Pattern Components =====
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass


class TransactionCommand(Command):
    def __init__(self, tracker):
        self.tracker = tracker

    def execute(self):
        print("Executing Transaction Command...")
        # Expected format: BUY/SELL ticker price share [YYYY/MM/DD]
        cmd_str = input("Enter transaction (format: BUY/SELL ticker price share [YYYY/MM/DD]): ").strip()
        parts = cmd_str.split()
        if len(parts) not in [4, 5]:
            print("Invalid command format. Expected: BUY/SELL ticker price share [YYYY/MM/DD]")
            return

        action, ticker_input, price_str, share_str = parts[:4]
        action = action.upper()
        if action not in ["BUY", "SELL"]:
            print("Invalid action. Please enter BUY or SELL.")
            return

        try:
            price = float(price_str)
            trans_share = float(share_str)
        except ValueError:
            print("Price and share must be numbers.")
            return

        # Parse transaction date if provided; otherwise, default to today.
        if len(parts) == 5:
            try:
                transaction_date = datetime.datetime.strptime(parts[4], "%Y/%m/%d").date()
            except ValueError:
                print("Invalid date format. Please use YYYY/MM/DD.")
                return
        else:
            transaction_date = datetime.datetime.now().date()

        # Map ticker if available in crypto_map.
        ticker = self.tracker.crypto_map.get(ticker_input, ticker_input)

        # ----- Update portfolio.db -----
        port_conn = sqlite3.connect(self.tracker.PORTFOLIO_FILE)
        try:
            cursor = port_conn.cursor()
            df_portfolio = pd.read_sql("SELECT * FROM portfolio WHERE ticker = ?", port_conn, params=(ticker,))
            if action == "BUY":
                cost = trans_share * price
                if self.tracker.available_fund < cost:
                    print("Insufficient funds for this purchase.")
                    return
                self.tracker.available_fund -= cost
                self.tracker.state_manager.save_state()
                if not df_portfolio.empty:
                    current_shares = df_portfolio.iloc[0]['share']
                    current_avg = df_portfolio.iloc[0]['avg_price']
                    new_total_shares = current_shares + trans_share
                    new_avg = ((current_shares * current_avg) + (trans_share * price)) / new_total_shares
                    cursor.execute(
                        "UPDATE portfolio SET share = ?, avg_price = ? WHERE ticker = ?",
                        (new_total_shares, new_avg, ticker)
                    )
                    print(f"Updated portfolio for {ticker}: New shares = {new_total_shares}, New avg = {new_avg:.2f}")
                else:
                    cursor.execute(
                        "INSERT INTO portfolio (ticker, share, avg_price) VALUES (?, ?, ?)",
                        (ticker, trans_share, price)
                    )
                    print(f"Inserted new ticker {ticker} with shares = {trans_share} and avg price = {price:.2f}")
            elif action == "SELL":
                if df_portfolio.empty:
                    print(f"No holdings found for ticker {ticker} to sell.")
                    return
                current_shares = df_portfolio.iloc[0]['share']
                current_avg = df_portfolio.iloc[0]['avg_price']
                if trans_share > current_shares:
                    print(f"Cannot sell {trans_share} shares; only {current_shares} available.")
                    return
                revenue = trans_share * price
                self.tracker.available_fund += revenue
                self.tracker.state_manager.save_state()
                new_total_shares = current_shares - trans_share
                if new_total_shares == 0:
                    cursor.execute("DELETE FROM portfolio WHERE ticker = ?", (ticker,))
                    print(f"Sold all shares of {ticker}. Removed from portfolio.")
                else:
                    new_avg = ((current_shares * current_avg) - (trans_share * price)) / new_total_shares
                    cursor.execute(
                        "UPDATE portfolio SET share = ?, avg_price = ? WHERE ticker = ?",
                        (new_total_shares, new_avg, ticker)
                    )
                    print(f"Updated portfolio for {ticker}: New shares = {new_total_shares}, New avg = {new_avg:.2f}")
            port_conn.commit()
            # ----- Overwrite aggregate_portfolio from trade day to current day -----
            self.update_aggregate_from_trade_day(transaction_date)
            print("Portfolio update successful.")
        except Exception as e:
            print(f"Error updating portfolio: {e}")
            port_conn.rollback()
            return
        finally:
            port_conn.close()


    def update_aggregate_from_trade_day(self, trade_day):
        """
        Overwrite the aggregate_portfolio table for every day from trade_day to today with the current
        portfolio snapshot.
        """
        current_date = datetime.datetime.now().date()
        # Build a list of all dates from trade_day to current_date.
        day_list = list(pd.date_range(start=trade_day, end=current_date).date)

        # Read the current portfolio snapshot.
        with sqlite3.connect(self.tracker.PORTFOLIO_FILE) as port_conn:
            df_portfolio = pd.read_sql("SELECT * FROM portfolio", port_conn)
        if df_portfolio.empty:
            print("[INFO] No portfolio data available to aggregate.")
            return

        # Connect to the aggregate database.
        with sqlite3.connect(self.tracker.AGGREGATE_PORTFOLIO_FILE) as agg_conn:
            # Delete existing aggregate records in the date range.
            agg_conn.execute(
                "DELETE FROM aggregate_portfolio WHERE aggregated_date BETWEEN ? AND ?",
                (trade_day.isoformat(), current_date.isoformat())
            )
            agg_conn.commit()

            # Prepare a list of rows to insert.
            snapshots = []
            for day in day_list:
                day_str = day.isoformat()
                for _, row in df_portfolio.iterrows():
                    snapshots.append((
                        row['ticker'],
                        row['share'],
                        row['avg_price'],
                        day_str
                    ))
            if snapshots:
                agg_conn.executemany(
                    "INSERT INTO aggregate_portfolio (ticker, share, avg_price, aggregated_date) VALUES (?, ?, ?, ?)",
                    snapshots
                )
                agg_conn.commit()
                print(f"[SUCCESS] Overwrote aggregate portfolio for dates {trade_day.isoformat()} to {current_date.isoformat()}.")


class RerenderCommand(Command):
    def __init__(self, tracker):
        self.tracker = tracker

    def execute(self):
        print("Re-rendering tracker...")
        self.tracker.run_tracker_without_statecheck()

class CommandLine:
    def __init__(self, tracker):
        self.tracker = tracker
        self.commands = {
            "transaction": TransactionCommand(tracker),
            "rerender": RerenderCommand(tracker)
        }

    def run(self):
        print("Command Line Interface. Available commands: transaction, rerender, exit")
        while True:
            cmd = input("Command> ").strip()

            if cmd == "exit":
                print("Exiting command line.")
                break
            elif cmd in self.commands:
                self.commands[cmd].execute()
            else:
                print("Unknown command. Please try again.")


if __name__ == "__main__":
    tracker = RetailTracker()
    tracker.run_tracker()
    cli = CommandLine(tracker)
    cli.run()

