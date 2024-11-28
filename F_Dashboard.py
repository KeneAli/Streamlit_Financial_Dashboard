import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


st.set_page_config(layout="wide", page_title="Stock Analysis Dashboard", page_icon="📈")

st.title("📈 Stock Analysis Dashboard")
# Initialize session state variables
if 'menu' not in st.session_state:
    st.session_state.menu = "Menu"
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = 'AAPL'
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = yf.Ticker(st.session_state.selected_ticker)
if 'financial_type' not in st.session_state:
    st.session_state.financial_type = 'Income Statement'
if 'period_type' not in st.session_state:
    st.session_state.period_type = 'Annual'

# S&P 500 initialisation
@st.cache_data
def list_wikipedia_sp500() -> pd.DataFrame:
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(url)[0]
    sp500_table.set_index('Symbol', inplace=True)  # Utiliser les tickers comme index
    return sp500_table

# Load tickers
if 'ticker_list' not in st.session_state:
    df_ticker = list_wikipedia_sp500()
    st.session_state.ticker_list = sorted(df_ticker.index.to_list())  # Liste triée

# Subsettinh the page in 3 columns
col_header1, col_header2, col_header3 = st.columns([2, 1, 1])

with col_header3:
    selected_ticker_temp = st.selectbox(
        "Select a Stock",
        st.session_state.ticker_list,
        index=st.session_state.ticker_list.index(st.session_state.selected_ticker)
        if 'selected_ticker' in st.session_state else 0
    )
    # Removed the automatic update of session state here

with col_header2:
    st.markdown(
        "<style>div.stButton > button {width: 100%; height: 50px; font-size: 18px;}</style>",
        unsafe_allow_html=True
    )
    if st.button("Update"):
        st.session_state.selected_ticker = selected_ticker_temp
        st.session_state.stock_data = yf.Ticker(st.session_state.selected_ticker)

# Display the stock name in col_header1
with col_header1:
    stock_data = st.session_state.get('stock_data', None)
    if stock_data:
        st.markdown(
            f"<h2 style='text-align: left; color: black;'>{stock_data.info.get('longName', 'N/A')}</h2>",
            unsafe_allow_html=True
        )

# Function to update date range based on selected period 
def update_dates(period):
    end_date = datetime.now()
    if period == '1M':
        start_date = end_date - timedelta(days=30)
    elif period == '3M':
        start_date = end_date - timedelta(days=90)
    elif period == '6M':
        start_date = end_date - timedelta(days=180)
    elif period == 'YTD':
        start_date = datetime(end_date.year, 1, 1)
    elif period == '1Y':
        start_date = end_date - timedelta(days=365)
    elif period == '3Y':
        start_date = end_date - timedelta(days=3*365)
    elif period == '5Y':
        start_date = end_date - timedelta(days=5*365)
    elif period == 'MAX':
        hist_data = stock_data.history(period="max")
        start_date = hist_data.index.min().to_pydatetime() if not hist_data.empty else datetime(2022, 1, 1)
    return start_date, end_date

# Main Content based on selected menu from the 5 buttons in the header
if st.session_state.stock_data is not None:
    stock_data = st.session_state.stock_data

    # Create tabs
    tabs = st.tabs(["Summary", "Chart", "Financial", "Monte Carlo", "Analysis"])

    # Menu Page
    with tabs[0]:
        col_info, col_chart = st.columns([1, 2])        #Divide space in 2
        with col_info:
            col1, col2 = st.columns([1, 1])     #Divide the information part in two as well
            with col1:
                st.markdown("<hr>", unsafe_allow_html=True)         #Adding Bar and space to get a good layout
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.write(f"**Previous Close** : {stock_data.info.get('previousClose', 'N/A')}")                                                         #Adding Each information from yahoo finance in the left column
                st.write(f"**Open** : {stock_data.info.get('open', 'N/A')}")
                st.write(f"**Bid** : {stock_data.info.get('bid', 'N/A')}")
                st.write(f"**Ask** : {stock_data.info.get('ask', 'N/A')}")
                st.write(f"**Day's Range** : {stock_data.info.get('dayLow', 'N/A')} - {stock_data.info.get('dayHigh', 'N/A')}")
                st.write(f"**52 Week Range** : {stock_data.info.get('fiftyTwoWeekLow', 'N/A')} - {stock_data.info.get('fiftyTwoWeekHigh', 'N/A')}")
                st.write(f"**Volume** : {stock_data.info.get('volume', 'N/A'):,}")
                st.write(f"**Avg. Volume** : {stock_data.info.get('averageVolume', 'N/A'):,}")
                st.write(f"**Major Shareholders** : {stock_data.info.get('majorHoldersBreakdown', 'N/A')}")

            with col2:
                st.markdown("<hr>", unsafe_allow_html=True)                     #Adding Bar and space to get a good layout
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.write(f"**Market Cap** : {stock_data.info.get('marketCap', 'N/A'):,} USD")                                                           #Adding Each information from yahoo finance in the right column
                st.write(f"**PE Ratio (TTM)** : {stock_data.info.get('trailingPE', 'N/A')}")
                st.write(f"**EPS (TTM)** : {stock_data.info.get('trailingEps', 'N/A')}")
                st.write(f"**Dividend** : {stock_data.info.get('dividendYield', 'N/A')}")
                st.write(f"**Beta** : {stock_data.info.get('beta', 'N/A')}")
                st.write(f"**Earnings Date** : {stock_data.info.get('earningsDate', 'N/A')}")
                st.write(f"**Forward Dividend & Yield** : {stock_data.info.get('dividendYield', 'N/A')}")
                st.write(f"**Company Profile** : {stock_data.info.get('industry', 'N/A')} - {stock_data.info.get('sector', 'N/A')}")

        with col_chart:                 #2nd column of the menu page : Selectbox and chart 
                    period = st.selectbox("Select the period :", ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'MAX'], key = 'key_one')        #Differents period availables
                    period_dict = {             #Dictionnary to match with '.history' function of yahoo finance afterwards
                        '1M': '1mo',
                        '3M': '3mo',
                        '6M': '6mo',
                        'YTD': 'ytd',
                        '1Y': '1y',
                        '3Y': '3y',
                        '5Y': '5y',
                        'MAX': 'max'
                    }
                    yf_period = period_dict.get(period, '1d')
                    hist = stock_data.history(period=yf_period)         #downloding the necessary data
                    fig = go.Figure(data=[go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close')])       #Displaying the chart       
                    fig.update_layout(title=f"Price of {st.session_state.selected_ticker} on {period}", xaxis_title="Date", yaxis_title="Prix (USD)")
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)             #Outside any column of the menu page : the description information from the selected ticker.
        st.subheader("Company Description")
        st.write(f"{stock_data.info.get('longBusinessSummary', 'N/A')}")

    # Tab 2: Chart
    with tabs[1]:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Chart of Stock Price")

        # Divide the page into three columns for period, start date, and end date
        col_period, col_start_date, col_end_date = st.columns([1, 1, 1])

        # Period selection (predefined ranges like Yahoo Finance)
        with col_period:
            period = st.selectbox(
                "Select the period:", 
                ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'MAX']
            )
        # Update start and end dates based on the selected period
        start_date, end_date = update_dates(period)

        # Allow manual adjustment of start and end dates
        with col_start_date:
            start_date = st.date_input("Starting Date", value=start_date)
        with col_end_date:
            end_date = st.date_input("Ending Date", value=end_date)

        # Interval selection (day, week, month)
        interval = st.selectbox(
            "Select an Interval:", 
            ['1d', '1wk', '1mo']
        )

        # Chart type selection (line plot or candlestick plot)
        chart_type = st.radio(
            "Chart type:", 
            ['Line plot', 'Candle plot']
        )

        # Download the stock data based on user inputs
        data = stock_data.history(start=start_date, end=end_date, interval=interval)

        # Calculate the 50-day moving average
        data['MA50'] = data['Close'].rolling(window=50).mean()

        # Initialize Plotly figure
        fig = go.Figure()

        # Add the selected chart type
        if chart_type == 'Line plot':
            fig.add_trace(go.Scatter(
                x=data.index, 
                y=data['Close'], 
                mode='lines', 
                name='Close'
            ))
        else:  # Candlestick plot
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Candlestick'
            ))

        # Add the 50-day moving average line
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['MA50'], 
            mode='lines', 
            name='MA50', 
            line=dict(color='purple')
        ))

        # Add trading volume as a bar chart with conditional coloring
        colors = ['green' if row.Close > row.Open else 'red' for _, row in data.iterrows()]
        fig.add_trace(go.Bar(
            x=data.index, 
            y=data['Volume'], 
            name="Volume", 
            marker=dict(color=colors, opacity=0.3), 
            yaxis='y2'
        ))

        # Update layout for dual y-axis and other settings
        fig.update_layout(
            title=f"{st.session_state.selected_ticker} Stock Price",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            yaxis2=dict(
                title="Volume",
                overlaying='y',
                side='right',
                showgrid=False,
                range=[0, max(data['Volume']) * 6]  # Adjust scale for volume
            ),
            height=700,
            xaxis_rangeslider_visible=False,  # Remove the default range slider
        )

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)


    # Tab 3: Financials
    with tabs[2]:
        st.header("Financials")
        st.markdown("<hr>", unsafe_allow_html=True)

        # Dropdowns for selecting financial type and period
        financial_type = st.selectbox("Select Financial Type", ["Income Statement", "Balance Sheet", "Cash Flow"])
        period_type = st.selectbox("Select Period", ["Annual", "Quarterly"])

        # Mapping financial type and period to the appropriate data
        if financial_type == "Income Statement":
            financials = stock_data.financials if period_type == "Annual" else stock_data.quarterly_financials
        elif financial_type == "Balance Sheet":
            financials = stock_data.balance_sheet if period_type == "Annual" else stock_data.quarterly_balance_sheet
        elif financial_type == "Cash Flow":
            financials = stock_data.cashflow if period_type == "Annual" else stock_data.quarterly_cashflow
        else:
            financials = None

        # Displaying the selected data
        st.subheader(f"{financial_type} - {period_type}")
        st.dataframe(financials if financials is not None else "No data available")

 # Page Monte Carlo Simulation
    with tabs[3]:
        st.markdown("<hr>", unsafe_allow_html=True)

        # Options selection
        col_simulations, col_days, col_startprice, col_var = st.columns([1, 1, 1, 1])  # Dividing into 4 columns
        with col_simulations:  # Number of simulations
            num_simulations = st.selectbox("Select number of simulations (n):", [200, 500, 1000], index=1)
        with col_days:  # Time horizon
            time_horizon = st.selectbox("Select time horizon (days):", [30, 60, 90], index=0)

        # Downloading the required data
        stock_data = st.session_state.stock_data.history(period="1y")
        closing_prices = stock_data['Close']

        # Compute daily returns and statistics
        daily_returns = closing_prices.pct_change().dropna()
        mean_return = daily_returns.mean()
        std_dev = daily_returns.std()

        # Initialize the last price
        last_price = closing_prices[-1]

        # Simulate price trajectories
        simulation_results = []
        for _ in range(num_simulations):
            prices = [last_price]
            for _ in range(time_horizon):
                price_change = prices[-1] * (1 + np.random.normal(mean_return, std_dev))
                prices.append(price_change)
            simulation_results.append(prices)

        # Calculate Value at Risk (95%)
        final_prices = [simulation[-1] for simulation in simulation_results]
        VaR_95 = np.percentile(final_prices, 5)

        # Display starting price and VaR
        with col_startprice:
            st.markdown(f"<h4 style='font-size:15px; font-weight:bold;'>Starting price: ${last_price:.2f}</h4>", unsafe_allow_html=True)
        with col_var:
            st.markdown(f"<h4 style='font-size:15px; font-weight:bold;'>Value at Risk (VaR) at 95% confidence: ${VaR_95:.2f}</h4>", unsafe_allow_html=True)

        # Monte Carlo simulation chart
        fig_mc = go.Figure()
        for simulation in simulation_results:
            fig_mc.add_trace(go.Scatter(
                x=list(range(time_horizon + 1)),
                y=simulation,
                mode='lines',
                line=dict(width=1),
                opacity=0.5
            ))

        # Add starting price as a line
        fig_mc.add_trace(go.Scatter(
            x=[0],
            y=[last_price],
            mode='markers',
            marker=dict(color='red', size=8),
            name='Current Price'
        ))
        fig_mc.add_shape(
            type="line",
            x0=0,
            x1=time_horizon,
            y0=last_price,
            y1=last_price,
            line=dict(color="red", width=2, dash="dash"),
        )

        # Update chart layout
        fig_mc.update_layout(
            title=f"Monte Carlo Simulation for {st.session_state.selected_ticker} Stock Price in Next {time_horizon} Days",
            xaxis_title="Day",
            yaxis_title="Price (USD)",
            height=570,
            showlegend=False
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        # Histogram of final prices
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Distribution of Ending Prices")
        fig_hist = go.Figure()

        # Add histogram of ending prices
        fig_hist.add_trace(go.Histogram(
            x=final_prices,
            nbinsx=50,
            marker=dict(color='blue', opacity=0.7),
            name='Ending Prices'
        ))

        # Add vertical line for VaR (5th percentile)
        fig_hist.add_shape(
            type="line",
            x0=VaR_95,
            x1=VaR_95,
            y0=0,
            y1=max(np.histogram(final_prices, bins=50)[0]),
            line=dict(color="red", width=2, dash="dash"),
        )

        # Update layout for histogram
        fig_hist.update_layout(
            title="Histogram of Ending Prices with VaR (95%)",
            xaxis_title="Ending Price (USD)",
            yaxis_title="Frequency",
            height=570,
            showlegend=False
        )

        # Display histogram as a separate figure
        st.plotly_chart(fig_hist, use_container_width=True)


    # Comparison Page
    with tabs[4]:    
        col_title, col_selectbox = st.columns([1, 1])               # Dividing the page in two columns

        with col_selectbox:                                         # 2nd SelectBox for comparison
            selected_ticker_2 = st.selectbox(
                "Select a stock to compare",
                st.session_state.ticker_list,
                index=0  
            )

        # Initialisation of the stock_data_2 and the 2nd company's longName 
        stock_data_2 = yf.Ticker(selected_ticker_2)
        company_name_2 = stock_data_2.info.get('longName', 'N/A')

        # Period Initialisation
        if 'comparison_period' not in st.session_state:
            st.session_state.comparison_period = '1y'               # Défault Value
        st.markdown("<hr>", unsafe_allow_html=True)


        # Buttons for selecting the period for the comparison chart
        col_button1, col_button2, col_button3, col_button4, col_button5, col_button6, col_button7, col_button8= st.columns(8)
        with col_button1:
            if st.button("1M"):
                st.session_state.comparison_period = '1mo'
        with col_button2:
            if st.button("3M"):
                st.session_state.comparison_period = '3mo'
        with col_button3:
            if st.button("6M"):
                st.session_state.comparison_period = '6mo'
        with col_button4:
            if st.button("1YTD"):
                st.session_state.comparison_period = 'ytd'
        with col_button5:
            if st.button("1Y"):
                st.session_state.comparison_period = '1y'
        with col_button6:
            if st.button("3Y"):
                st.session_state.comparison_period = '3y'
        with col_button7:
            if st.button("5Y"):
                st.session_state.comparison_period = '5y'
        with col_button8:
            if st.button("MAX"):
                st.session_state.comparison_period = 'max'

        # Getting the data thorugh 'history' function with yahoo finance from stock_data variables
        stock_data = yf.Ticker(st.session_state.selected_ticker)
        data_1 = stock_data.history(period=st.session_state.comparison_period)
        data_2 = stock_data_2.history(period=st.session_state.comparison_period)


        # Daily performance computation
        returns_1 = data_1['Close'].pct_change().dropna()
        returns_2 = data_2['Close'].pct_change().dropna()

        # Computing volatility
        volatility_1 = returns_1.std() * (252 ** 0.5) * 100  # Annualised on 252 days of stock exchange
        volatility_2 = returns_2.std() * (252 ** 0.5) * 100  # Annualised on 252 days of stock exchange
       
        # Calculate average annual yield
        mean_return_1 = returns_1.mean() * 252 * 100  # Annualised on 252 days of stock exchange
        mean_return_2 = returns_2.mean() * 252 * 100  # Annualised on 252 days of stock exchange

        # bêta computing 
        beta_1 = stock_data.info.get('beta', 'Unavailable')
        beta_2 = stock_data_2.info.get('beta', 'Unavailable')

        # Sharpe Ratio 
        sharpe_ratio_1 = mean_return_1 / volatility_1
        sharpe_ratio_2 = mean_return_2 / volatility_2

        # Market Cap 
        market_cap_1 = stock_data.info.get('marketCap', 'N/A')
        market_cap_2 = stock_data_2.info.get('marketCap', 'N/A')

        # PE Ratio (TTM)
        pe_ratio_1 = stock_data.info.get('trailingPE', 'N/A')
        pe_ratio_2 = stock_data_2.info.get('trailingPE', 'N/A')

        # EPS (TTM)
        eps_1 = stock_data.info.get('trailingEps', 'N/A')
        eps_2 = stock_data_2.info.get('trailingEps', 'N/A')

        # Dividend Yield
        divident_yield_1 = stock_data.info.get('dividendYield', 'N/A')
        divident_yield_2 = stock_data_2.info.get('dividendYield', 'N/A')

        # Comparison Line Chart
        fig = go.Figure()

        # First trace in black
        fig.add_trace(go.Scatter(
            x=data_1.index, 
            y=data_1['Close'], 
            mode='lines', 
            name=f"{st.session_state.selected_ticker} Close",
            line=dict(color='blue')  # Line color set to black
        ))

        # Second trace in grey
        fig.add_trace(go.Scatter(
            x=data_2.index, 
            y=data_2['Close'], 
            mode='lines', 
            name=f"{selected_ticker_2} Close",
            line=dict(color='grey')  # Line color set to grey
        ))

        # Graph layout
        fig.update_layout(
            title="Comparison of Stock Prices",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=700,
            template="plotly_white"  # Optional: Use a white background template for contrast
        )

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)


        st.markdown("<hr>", unsafe_allow_html=True)
        # Displaying Key statistics below the comparison chart
        st.markdown("### Key Statistics")
        comparison_data = {
        "Metric": [
            "📊 Average Return (%)",
            "📉 Volatility (%)",
            "💸 Market Cap",
            "📈 Beta",
            "📏 Sharpe Ratio",
            "🧮 PE Ratio",
            "💵 EPS (TTM)",
            "💰 Dividend Yield"
        ],
        
        f"{st.session_state.selected_ticker}": [
        f"{mean_return_1:.2f}%",  # Average Return for stock 1
        f"{volatility_1:.2f}%",  # Volatility for stock 1
        f"{market_cap_1:.2f}%",  # market cap for stock 1
        f"{beta_1:.2f}",          # Beta for stock 1
        f"{sharpe_ratio_1:.2f}",   # Sharpe Ratio for stock 1
        f"{pe_ratio_1:.2f}",   # Sharpe Ratio for stock 2
        f"{eps_1:.2f}",   # eps for stock 1
        f"{divident_yield_1:.2f}"   # divident for stock 2
        
        ],
        f"{selected_ticker_2}": [
        f"{mean_return_2:.2f}%",  # Average Return for stock 2
        f"{volatility_2:.2f}%",  # Volatility for stock 2
        f"{market_cap_2:.2f}%",  # market cap for stock 1
        f"{beta_2:.2f}",          # Beta for stock 2
        f"{sharpe_ratio_2:.2f}",   # Sharpe Ratio for stock 2
        f"{pe_ratio_2:.2f}",   # Sharpe Ratio for stock 2
        f"{eps_2:.2f}",   # eps for stock 1
        f"{divident_yield_2:.2f}"   # divident for stock 2
         ]
        }
        # Create a DataFrame
        comparison_df = pd.DataFrame(comparison_data)

        # Display the table
        st.table(comparison_df)