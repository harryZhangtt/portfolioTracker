import sqlite3
import pandas as pd
from .command_base import Command

class RerenderCommand(Command):
    def __init__(self, tracker):
        self.tracker = tracker

    def execute(self):
        print("Re-rendering tracker...")
        self.tracker.run_tracker_without_statecheck()

class ImportPortfolioCommand(Command):
    def __init__(self, tracker):
        self.tracker = tracker

    def execute(self):
        filename = input("Please provide the filename to import portfolio: ").strip()
        try:
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filename)
            elif filename.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(filename)
            else:
                print("Unsupported file format. Please provide a CSV or Excel file.")
                return
        except Exception as e:
            print(f"Error reading file: {e}")
            return

        try:
            conn = sqlite3.connect(self.tracker.PORTFOLIO_FILE)
            df.to_sql("portfolio", conn, if_exists='replace', index=False)
            conn.close()
            print(f"Portfolio imported successfully from {filename}.")
        except Exception as e:
            print(f"Error importing portfolio into database: {e}")

class ExportFileCommand(Command):
    def __init__(self, tracker):
        self.tracker = tracker

    def execute(self):
        print("Export options: 1) Portfolio, 2) Aggregate Portfolio, 3) DB File")
        choice = input("Select which file to export (enter 1, 2, or 3): ").strip()
        if choice == "1":
            db_file = self.tracker.PORTFOLIO_FILE
            table_name = "portfolio"
            output_filename = "portfolio.csv"
        elif choice == "2":
            db_file = self.tracker.AGGREGATE_PORTFOLIO_FILE
            table_name = "aggregate_portfolio"
            output_filename = "aggregate_portfolio.csv"
        elif choice == "3":
            db_file = self.tracker.DB_FILE
            table_name = input("Enter the table name to export from DB_FILE: ").strip()
            output_filename = f"{table_name}.csv"
        else:
            print("Invalid choice.")
            return

        try:
            conn = sqlite3.connect(db_file)
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            conn.close()
            df.to_csv(output_filename, index=False)
            print(f"Exported {table_name} to {output_filename}.")
        except Exception as e:
            print(f"Error exporting file: {e}")
