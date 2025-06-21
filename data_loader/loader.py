import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import requests
import io
import re

CSV_URL = ("https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/"
           "1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund")

def get_sp500_tickers(style="wikipedia"):
    """
    Return a list of S&P‑500 symbols.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, flavor="bs4", header=0)[0]
    tickers = table["Symbol"].tolist()
    tickers = [t.replace(".", "-") for t in tickers]
    return sorted(set(tickers))

def get_russell3000_tickers() -> list[str]:
    """
    Return today's Russell 3000 constituents as Yahoo‑compatible tickers.
    """
    raw = requests.get(CSV_URL, timeout=30).content

    # The first real header line starts with "Ticker,Name,..."
    start = raw.find(b"Ticker")
    df = pd.read_csv(io.BytesIO(raw[start:]))
    tickers = (
        df["Ticker"]
        .dropna()
        .astype(str)
        .apply(lambda t: re.sub(r"\.", "-", t))  # Yahoo uses '-' instead of '.'
        .unique()
        .tolist()
    )
    return sorted(tickers)

def download_returns(years_back=10, batch_size=50):
    """
    Download S&P‑500 daily returns.
    """
    # get latest tickers - not accurate historically
    #tickers = get_sp500_tickers()
    tickers = get_russell3000_tickers()

    end = date.today()
    start = end - timedelta(days=365 * years_back)

    # batch download data
    # a lot of R3000 tickers are not covered in yfinance
    frames = []
    batches = (tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size))
    print(f"Downloading data from ({start} → {end})")
    for batch in batches:
        prices = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            group_by="column",
            progress=False,
            threads=True,
            rounding=True,
        )

        # transform
        prices  = prices.stack(level=1).reset_index().sort_values(['Date', 'Ticker'])
        prices['Date'] = prices['Date'].dt.date
        prices['Return'] = prices.groupby('Ticker')['Close'].pct_change()
    
        frames.append(prices)
    
    prices_all = pd.concat(frames, ignore_index=True)

    return prices_all

if __name__ == "__main__":
    returns = download_returns(years_back=10)
    returns.to_parquet("stock_returns_us.parquet")
