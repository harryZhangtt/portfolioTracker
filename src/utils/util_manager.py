import sqlite3
import pandas as pd

class UtilManager:
    def __init__(self, tracker, admin_password="admin123"):
        self.tracker = tracker
        self.admin_password = admin_password

    def execute_util_command(self):
        password = input("Enter admin password: ").strip()
        if password != self.admin_password:
            print("Incorrect password.")
            return
        table = input("Enter table name (portfolio, aggregate_portfolio, expanded_df): ").strip()
        table_to_db = {
            "portfolio": self.tracker.PORTFOLIO_FILE,
            "aggregate_portfolio": self.tracker.AGGREGATE_PORTFOLIO_FILE,
            "expanded_df": self.tracker.DB_FILE
        }
        if table not in table_to_db:
            print("Invalid table name.")
            return
        db_file = table_to_db[table]
        sql_command = input("Enter SQL command: ").strip()
        # Remove trailing semicolon if present to prevent incomplete input errors.
        if sql_command.endswith(";"):
            sql_command = sql_command[:-1]
        with sqlite3.connect(db_file) as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql_command)
                if sql_command.lstrip().upper().startswith("SELECT"):
                    df = pd.read_sql_query(sql_command, conn)
                    print(df)
                else:
                    conn.commit()
                    print("SQL command executed successfully.")
            except Exception as e:
                print("Error executing SQL command:", e)
