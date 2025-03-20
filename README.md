Retail Portfolio Tracker
This repository provides a Python-based command-line application to manage and track a personal investment portfolio. The system automatically updates the portfolio, fetches historical price data, calculates gains/losses, and aggregates daily snapshots of holdings in a SQLite database. The project is designed to be easy to use and extend, while keeping your portfolio data local.




Features
Portfolio Management

Store ticker symbols, number of shares, and average cost in a local SQLite database (portfolio.db).
Supports both traditional stocks (e.g., AAPL, NVDA) and cryptocurrencies (via a custom mapping to Yahoo Finance tickers, e.g., ETH -> ETH-USD).
Automated Price Updates

Fetches historical or latest prices from Yahoo Finance.
Merges fetched prices with your portfolio data to calculate daily valuations.
Gain/Loss and Percentage Allocation

Calculates each position’s share of the total portfolio.
Computes PnL (profit and loss) amounts and percentages for each holding.
State Management

Uses tracker_state.json to store the last run date, so it won’t repeatedly fetch the same data.
Keeps track of available funds (cash balance) between sessions.
Command-Line Interface

Provides commands for transactions (buy/sell), re-running the tracker, importing new portfolios, and exporting data.
A simple menu-driven system that sets a 30-second timeout to avoid hanging.
Aggregate and Historical Data

Daily snapshots of your portfolio are stored in aggregate_portfolio.db.
Historical price data is saved in price_data.db, avoiding repeated downloads from Yahoo Finance.
Directory and File Structure

your-project/
│
├── retailPortfolioTracker3.0.py    # Main script containing all classes & CLI
├── tracker_state.json              # JSON file to store the last run date, available balance, etc.
├── portfolio.db                    # SQLite database for the portfolio (created on first run if not existing)
├── aggregate_portfolio.db          # SQLite database for daily snapshots of the portfolio
├── price_data.db                   # SQLite database for historical price data
├── requirements.txt                # (Optional) Python dependencies
└── README.md                       # This file (the project documentation)
Note: Some of these files (like .db files) will be created automatically on first run if not present.

Dependencies
Python 3.7+
pandas
yfinance
sqlite3 (bundled with Python)
signal (bundled with Python)
Install them via pip if needed:
pip install pandas yfinance
(Or install all from requirements.txt if you maintain one.)

How It Works
Initialization

RetailTracker loads the current state from tracker_state.json, including your last render time and available cash balance.
Loading Portfolio

On startup, the tracker reads portfolio.db to load your portfolio of tickers, shares, and average prices.
If you have a new CSV or Excel file with updated holdings, you can import it (see the import command).
Aggregating Portfolio Data

A snapshot of your holdings is stored daily in aggregate_portfolio.db. This allows you to track changes day by day.
Fetching Prices

The tracker queries Yahoo Finance (via yfinance) to get recent or historical price data for each ticker in your portfolio.
The fetched data is merged into a single DataFrame (expanded_df), which is saved to price_data.db.
Calculations

For each day, the tracker calculates total portfolio value, each position’s percentage weight, and profit/loss relative to average cost.
State Saving

Once the update completes, the code writes the current date/time and your updated available cash balance back to tracker_state.json.
This prevents unnecessary re-fetches if you run the script multiple times per day.



Usage
Initial Setup
Clone the repository:

git clone https://github.com/yourusername/RetailPortfolioTracker.git
cd RetailPortfolioTracker
Install dependencies:


pip install -r requirements.txt
(Or just install pandas and yfinance manually.)

(Optional) Create initial portfolio.db

If you don’t already have a portfolio.db, you can create one via the import command or simply let the script create an empty database and insert data via transactions.
Running the Tracker
Execute:


python retailPortfolioTracker3.0.py
On first run, it will:
Check if portfolio.db exists; if not, it will complain or create an empty DB.
Attempt to fetch prices for your tickers (if any).
Save results in price_data.db and aggregate_portfolio.db.
Print out total portfolio value by day, plus any relevant messages.
Command-Line Interface (CLI)
After the script initializes, it will load the command-line interface. The prompt says:


Command Line Interface. Available commands: transaction, rerender, exit
Command>
We have extended it to also include import and export.


transaction
Prompts: Enter transaction (format: BUY/SELL ticker price share [YYYY/MM/DD])

Example:


BUY AAPL 150 10
This buys 10 shares of AAPL at $150 per share using your available funds.
If you omit the date, it defaults to today’s date.
You can also do:


SELL NVDA 250 5 2025/03/19
to sell 5 shares of NVDA at $250 on March 19, 2025.

Transactions adjust your available_fund in tracker_state.json and update portfolio.db accordingly.


rerender
Forces the tracker to re-run run_tracker_without_statecheck(), which:
Reloads the portfolio.
Updates aggregated portfolio data.
Fetches any missing price data from Yahoo Finance.
Saves updated info to the local databases.


import
Prompts for a filename. You can provide a CSV or Excel file containing your portfolio data.
Expected columns: at minimum ticker, share, avg_price.
The script will overwrite the portfolio table in portfolio.db with the file’s contents.

export
Prompts:
Export options: 1) Portfolio, 2) Aggregate Portfolio, 3) DB File
You can choose which database/table to export.
1 exports portfolio.db -> portfolio.csv
2 exports aggregate_portfolio.db -> aggregate_portfolio.csv
3 exports a table from price_data.db (e.g., the expanded_df table) into CSV.


exit
Closes the command-line interface and ends the program.
Portfolio Format
The portfolio is stored in the portfolio.db database with a table named portfolio. Required columns:

ticker (string)
share (float)
avg_price (float)
If you import from CSV/Excel, ensure you have these columns.
Additional columns like percentage might appear in the UI, but they are optional for the DB table.

An example CSV might look like:
ticker,share,avg_price
AAPL,10,150
NVDA,5,230
ETH,2,1800
(The script automatically maps known crypto tickers to Yahoo Finance format, e.g., ETH -> ETH-USD.)

Database Schema
portfolio.db

Table: portfolio
ticker (TEXT)
share (REAL)
avg_price (REAL)
(Potentially more columns if you manually add them)
aggregate_portfolio.db

Table: aggregate_portfolio
ticker (TEXT)
share (REAL)
avg_price (REAL)
aggregated_date (TEXT / ISO date string)
price_data.db

Table: expanded_df
ticker (TEXT)
date (DATE)
price (REAL)
share (REAL)
avg_price (REAL)
(and other columns like percentage, pnl_amount, pnl_percentage if the code has processed them)
Troubleshooting
No Data in Portfolio

If you haven’t added any positions, or your portfolio.db is empty, the script may complain about missing data. Use the transaction or import command to populate it.
Insufficient Funds

If you get an error during a BUY transaction, you may not have enough available_fund. Either SELL something first or manually edit the available_balance in tracker_state.json if you want to simulate adding more cash.
Yahoo Finance Ticker Not Found

If a ticker is invalid or not supported by Yahoo Finance, you may see a warning about no price data. Double-check the spelling, or add a mapping to self.crypto_map if it’s a cryptocurrency.
Date Issues

If you see errors about date formatting, ensure you use YYYY/MM/DD. For example, 2025/03/19.
Timeout

The CLI has a 30-second input timeout. If you don’t type a command within 30 seconds, it will exit automatically.
License
This project is licensed under the MIT License. You are free to use, modify, and distribute this code in personal or commercial projects, provided you include the original license.

Thank you for using the Retail Portfolio Tracker!

If you have any questions or suggestions, feel free to open an issue or submit a pull request. Happy tracking!
