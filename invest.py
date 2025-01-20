import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go

# Add a title at the beginning of the UI
st.title('Real-Time Stock Data & Charting Tool')

# Function to retrieve S&P 500 stock tickers
def get_sp500_stocks():
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB',
        'TSLA', 'BRK.B', 'JNJ', 'V', 'JPM',
        'NVDA', 'NFLX', 'DIS', 'PYPL', 'PG', 'KO',
        # Add other tickers as necessary
    ]

def calculate_risk_score(answers):
    score = sum(answers)
    return score

def risk_tolerance_quiz():
    st.title("Risk Tolerance Quiz")
    
    questions = [
        "How would you describe your current investment knowledge?",
        "What is your investment time horizon?",
        "How do you react to market fluctuations?",
        "What is your primary investment goal?",
        "How comfortable are you with the idea of losing money?"
    ]

    options = {
        "How would you describe your current investment knowledge?": ["Novice", "Intermediate", "Expert"],
        "What is your investment time horizon?": ["Less than 1 year", "1-5 years", "5+ years"],
        "How do you react to market fluctuations?": ["Panic", "Stay calm", "View it as an opportunity"],
        "What is your primary investment goal?": ["Preserve capital", "Generate income", "Grow wealth"],
        "How comfortable are you with the idea of losing money?": ["Very uncomfortable", "Neutral", "Comfortable"],
    }

    answers = []
    for question in questions:
        answer = st.radio(question, options[question], key=question)
        answer_values = {
            "Novice": 1, "Intermediate": 2, "Expert": 3,
            "Less than 1 year": 1, "1-5 years": 2, "5+ years": 3,
            "Panic": 1, "Stay calm": 2, "View it as an opportunity": 3,
            "Preserve capital": 1, "Generate income": 2, "Grow wealth": 3,
            "Very uncomfortable": 1, "Neutral": 2, "Comfortable": 3
        }
        answers.append(answer_values[answer])

    if st.button("Submit"):
        risk_score = calculate_risk_score(answers)
        st.write("Your Risk Tolerance Score is:", risk_score)
        
        if risk_score <= 8:
            risk_tolerance = "Low"
        elif risk_score <= 12:
            risk_tolerance = "Medium"
        else:
            risk_tolerance = "High"
        
        st.write(f"Risk level: {risk_tolerance}")
        return risk_tolerance

# Get the user's risk level and start the quiz if necessary
risk_level = st.selectbox("What is your risk level?", ["High", "Medium", "Low", "No idea, Quiz me"])

if risk_level == "No idea, Quiz me":
    chosen_risk_level = risk_tolerance_quiz()
    risk_level = chosen_risk_level

# Investment Amount input
investment_amount = st.number_input('Enter amount for investment', min_value=1, value=1000, step=100)

# Time Frame Selection
selected_timeframe = st.selectbox("Select your investment time frame:", [
    "Day trading (Inactive)", 
    "Short-term trading (2-30 days)",  
    "Medium-term investing (Inactive)", 
    "Long-term investing (Inactive)" 
])

# Initialize days variable
days = 0
if selected_timeframe == "Short-term trading (2-30 days)":
    if investment_amount < 1:
        st.error("Please enter a valid investment amount.")
    else:
        days = st.number_input("Enter number of days (2-30)", min_value=2, max_value=30, value=20)
else:
    st.warning("Please select 'Short-term trading (2-30 days)' as the active strategy for now.")

# Initialize risk-based stock lists with more options
high_risk_stocks = ['AAPL', 'TSLA', 'GOOGL', 'NVDA', 'NFLX']
medium_risk_stocks = ['AMZN', 'FB', 'MSFT', 'DIS', 'PYPL']
low_risk_stocks = ['JNJ', 'JPM', 'BRK.B', 'V', 'PG', 'KO']

if days > 0 and investment_amount > 0:
    # Determine the list of stocks to analyze based on the user's risk level
    stocks_to_analyze = []
    if risk_level == "High":
        stocks_to_analyze = high_risk_stocks
    elif risk_level == "Medium":
        stocks_to_analyze = medium_risk_stocks
    else:
        stocks_to_analyze = low_risk_stocks

    stock_forecasts = []
    
    for ticker in stocks_to_analyze:
        data = yf.Ticker(ticker)
        hist = data.history(period='max')

        if not hist.empty:
            # Filter to get only the last 60 days of data
            now = pd.Timestamp.now(tz='America/New_York')
            last_60_days = now - pd.Timedelta(days=60)
            hist = hist[hist.index > last_60_days]

            # Prepare data for Prophet
            prophet_data = hist.reset_index()[['Date', 'Close']]
            prophet_data.columns = ['ds', 'y']  
            prophet_data['ds'] = prophet_data['ds'].dt.tz_localize(None)

            # Create and fit the model
            model = Prophet(daily_seasonality=True)
            model.fit(prophet_data)

            # Create a DataFrame for future predictions
            future = model.make_future_dataframe(periods=days)  # Forecasting for specified days
            forecast = model.predict(future)

            # Calculate buy and sell prices and expected returns
            buy_price = forecast['yhat'].iloc[0]  
            sell_price = forecast['yhat'].iloc[-1]  
            buy_date = forecast['ds'].iloc[0]  
            sell_date = forecast['ds'].iloc[-1]  

            # Ensure that the sell date comes after the buy date is always true here
            if buy_price < sell_price:
                expected_return = (sell_price - buy_price) / buy_price
                num_shares = investment_amount // buy_price

                # Store forecast results
                stock_forecasts.append((
                    ticker,
                    buy_price,
                    sell_price,
                    expected_return,
                    num_shares,
                    buy_date,
                    sell_date,
                    forecast
                ))

    # Create a DataFrame for the analysis
    forecast_df = pd.DataFrame(stock_forecasts, columns=['Ticker', 'Buy Price', 'Sell Price', 'Expected Return', 'Shares', 'Buy Date', 'Sell Date', 'Forecast'])
    forecast_df = forecast_df.sort_values(by='Expected Return', ascending=False).head(3)

    # Display results
    st.subheader('Top 3 Stock Recommendations')
    st.write(forecast_df[['Ticker', 'Buy Price', 'Sell Price', 'Expected Return', 'Shares', 'Buy Date', 'Sell Date']])

    # Plotting the predictions for each stock
    for index, row in forecast_df.iterrows():
        forecast = row['Forecast']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecasted Price'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dash')))

        # Customizing the layout
        fig.update_layout(
            title=f'{row["Ticker"]} Closing Prices with {days} Day Forecast',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            hovermode='closest'
        )
        
        # Show the plot for each stock
        st.plotly_chart(fig)

else:
    st.warning("Please ensure you have selected a timeframe and entered a valid investment amount.")

# New Question at the end
st.subheader("How do you want to proceed?")
option = st.selectbox("Choose an option:", ["Take me to a trading platform to invest.", "Take me to an expert to consult."])
