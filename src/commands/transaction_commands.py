import datetime
import sqlite3
import pandas as pd
from .command_base import Command

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
