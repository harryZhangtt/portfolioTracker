import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Define a dictionary mapping tech sub-sectors to representative ETF tickers.
tech_subsectors = {
    "Artificial Intelligence": "BOTZ",
    "Aerospace & Defense": "ITA",
    "Robotics & Automation": "ROBO",
    "Quantum Computing": "QTUM",  # Assumed ticker for quantum computing ETF
    "Cybersecurity": "CIBR",
    "Cloud Computing": "SKYY",
     # Assumed ticker for IoT ETF
    "5G & Communications": "FIVG",  # Assumed ticker for 5G ETF
    "E-commerce": "IBUY",
    "Fintech": "ARKF"
}

# Define the time window: last 3 years.
end_date = datetime.today()
start_date = end_date - timedelta(days=3 * 365)

# Dictionary to store monthly aggregated trading volume for each sub-sector.
subsector_monthly_volume = {}

# Loop through each ETF, download data, and resample the 'Volume' column to monthly sums.
for subsector, ticker in tech_subsectors.items():
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        print(f"No data for {ticker} in the period.")
        continue

    # Resample using 'ME' (month end) frequency and sum the volume.
    monthly_volume = data['Volume'].resample('ME').sum()

    # If the resampling returns a DataFrame, squeeze it to a Series.
    if isinstance(monthly_volume, pd.DataFrame):
        monthly_volume = monthly_volume.squeeze(axis=1)

    # Ensure the monthly volume is a float Series.
    monthly_volume = monthly_volume.astype(float)

    subsector_monthly_volume[subsector] = monthly_volume

# Create a DataFrame from the dictionary. The DataFrame will have a datetime index (all months)
# and one column per sub-sector.
volume_df = pd.DataFrame(subsector_monthly_volume)

# Plot the evolution of monthly trading volume for each tech sub-sector.
plt.figure(figsize=(12, 6))
for subsector in volume_df.columns:
    plt.plot(volume_df.index, volume_df[subsector], label=subsector)

plt.xlabel("Date")
plt.ylabel("Total Monthly Trading Volume")
plt.title("Evolution of Monthly Trading Volume for Tech Sub-Sectors Over the Last 3 Years")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
