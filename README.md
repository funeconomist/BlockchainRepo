# 💱 Institutional Stablecoin Flow Analysis Dashboard

An interactive Streamlit dashboard tracking USDC/USDT flows between major centralized exchanges (CEXs) and DeFi protocols to identify institutional money movement patterns.

## Features

### 📊 Flow Volume Chart
- Time series visualization of daily net flows (CEX → DeFi vs DeFi → CEX)
- 30-day lookback period
- Stacked area chart showing USDC vs USDT separately

### 🔝 Top Movers Table
- Displays largest single transfers (>$10M) from the last 7 days
- Shows: Date, Amount, From (CEX/Protocol), To (CEX/Protocol), Token type

### 📈 Correlation Analysis
- Scatter plot showing relationship between net stablecoin inflow and ETH price changes
- 48-hour lag analysis
- Pearson correlation coefficient displayed

### 🏦 Protocol Breakdown
- Horizontal bar chart of total inflows by destination protocol
- Covers Aave, Compound, Curve, Uniswap, and other major DeFi protocols
- Last 30 days data

## Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly
- **APIs**:
  - Dune Analytics (stablecoin transfer data)
  - CoinGecko (ETH price data)

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)

For production use with real data:

**Dune Analytics API:**
1. Create a free account at [dune.com](https://dune.com)
2. Get your API key from account settings
3. Set environment variable:
   ```bash
   export DUNE_API_KEY="your_api_key_here"
   ```

**Note:** The dashboard works without API keys using mock data for testing and demonstration purposes.

### 3. Create Dune Query (For Real Data)

To use actual blockchain data, create a Dune query that returns:
- `date`: Transaction date
- `token`: Token symbol (USDC/USDT)
- `from_type`: Source type (CEX/DeFi)
- `to_type`: Destination type (CEX/DeFi)
- `from_name`: Specific CEX or protocol name
- `to_name`: Specific CEX or protocol name
- `amount`: Transfer amount in USD

Example Dune SQL query structure:
```sql
SELECT
    DATE_TRUNC('day', block_time) as date,
    CASE
        WHEN contract_address = '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48' THEN 'USDC'
        WHEN contract_address = '0xdac17f958d2ee523a2206206994597c13d831ec7' THEN 'USDT'
    END as token,
    -- Add logic to identify CEX wallets and DeFi protocols
    -- ...
FROM erc20_ethereum.evt_Transfer
WHERE block_time >= NOW() - INTERVAL '30' DAY
```

Update the `query_id` in `stablecoin_dashboard.py` (line 90) with your query ID.

### 4. Run the Dashboard

```bash
streamlit run stablecoin_dashboard.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## Usage

The dashboard automatically:
- Fetches the latest stablecoin transfer data (or uses mock data)
- Retrieves ETH price history from CoinGecko
- Calculates net flows and correlations
- Generates insights based on recent trends

### Understanding the Insights

The top section provides a 2-3 sentence summary including:
- Net directional flow over 30 days (into or out of DeFi)
- Recent trend (accelerating or moderating)
- Correlation strength between stablecoin flows and ETH price movements

## Data Sources

- **Stablecoin Transfers**: Dune Analytics (blockchain data)
  - Tracks transfers from known CEX wallets: Binance, Coinbase, Kraken
  - Monitors flows to DeFi protocols: Aave, Compound, Uniswap, Curve
- **ETH Prices**: CoinGecko API (free tier, no key required)

## Project Structure

```
.
├── stablecoin_dashboard.py   # Main dashboard application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── logolove.png              # Repository logo
```

## Notes

- The dashboard uses a 1-hour cache for API data to respect rate limits
- Mock data is automatically generated when API keys are not provided
- All monetary values are displayed in USD
- CEX wallets and DeFi protocol addresses need to be identified in your Dune query

## Future Enhancements

Potential improvements:
- Add more CEXs and DeFi protocols
- Include stablecoin supply changes
- Add alerts for unusual flow patterns
- Historical comparison (month-over-month)
- Export data to CSV
- Additional tokens (DAI, FRAX, etc.)

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
