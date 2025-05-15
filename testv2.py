import streamlit as st
import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import json
import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from yahooquery import Ticker
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simplified path configuration - use current directory for files
DATA_DIR = "."
PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio.csv")
DEFAULT_PORTFOLIO_COLUMNS = [
    'Ticker', 'Name', 'Current Price', 'Purchase Price', 'Shares', 'Total Value',
    'Profit/Loss', 'Profit/Loss (%)', 'YoY Growth (%)', 'P/E Ratio', 'Dividend Yield (%)', 
    'Beta', 'Sector', 'Added Date'
]
CRYPTO_TICKERS = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'DOGE-USD', 'ADA-USD', 'DOT-USD', 'SOL-USD', 'AVAX-USD', 'MATIC-USD', 'LINK-USD', 'UNI-USD']

# Create empty portfolio file if it doesn't exist
if not os.path.exists(PORTFOLIO_FILE):
    try:
        # Create empty DataFrame
        empty_portfolio = pd.DataFrame(columns=DEFAULT_PORTFOLIO_COLUMNS)
        # Save empty DataFrame
        empty_portfolio.to_csv(PORTFOLIO_FILE, index=False)
        logger.info(f"Created empty portfolio file at {PORTFOLIO_FILE}")
    except Exception as e:
        logger.error(f"Failed to create empty portfolio file: {e}")

# Load environment variables
load_dotenv()

# Custom JSON encoder for handling non-serializable objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'to_json'):
            return json.loads(obj.to_json())
        return super().default(obj)

# Initialize LLM
def init_llm():
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            return None
        
        return ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.2,
            api_key=api_key
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

# Function to get stock data
def get_stock_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Get historical data for YoY growth calculation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        hist_data = ticker.history(start=start_date, end=end_date)
        
        if hist_data.empty:
            return None
        
        # Calculate YoY growth
        first_price = hist_data['Close'].iloc[0] if not hist_data.empty else 0
        last_price = hist_data['Close'].iloc[-1] if not hist_data.empty else 0
        yoy_growth = ((last_price - first_price) / first_price * 100) if first_price > 0 else 0
        
        # Extract relevant information
        return {
            'Name': info.get('shortName', ticker_symbol),
            'Current Price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Market Cap': info.get('marketCap', 0),
            'P/E Ratio': info.get('trailingPE', 0),
            'EPS': info.get('trailingEps', 0),
            'Dividend Yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            '52 Week High': info.get('fiftyTwoWeekHigh', 0),
            '52 Week Low': info.get('fiftyTwoWeekLow', 0),
            'YoY Growth': round(yoy_growth, 2),
            'Beta': info.get('beta', 0)
        }
    except Exception as e:
        logger.error(f"Error fetching data for {ticker_symbol}: {e}")
        return None

# Function to get crypto data
def get_crypto_data(crypto_symbol):
    if not crypto_symbol.endswith('-USD'):
        crypto_symbol = f"{crypto_symbol}-USD"
    
    try:
        ticker = yf.Ticker(crypto_symbol)
        info = ticker.info
        
        # Get historical data for YoY growth calculation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        hist_data = ticker.history(start=start_date, end=end_date)
        
        if hist_data.empty:
            return None
        
        # Calculate YoY growth
        first_price = hist_data['Close'].iloc[0] if not hist_data.empty else 0
        last_price = hist_data['Close'].iloc[-1] if not hist_data.empty else 0
        yoy_growth = ((last_price - first_price) / first_price * 100) if first_price > 0 else 0
        
        return {
            'Name': info.get('name', crypto_symbol.replace('-USD', '')),
            'Current Price': info.get('regularMarketPrice', 0),
            'Sector': 'Cryptocurrency',
            'Industry': 'Cryptocurrency',
            'Market Cap': info.get('marketCap', 0),
            'P/E Ratio': 'N/A',
            'EPS': 'N/A',
            'Dividend Yield': 0,
            '52 Week High': info.get('fiftyTwoWeekHigh', 0),
            '52 Week Low': info.get('fiftyTwoWeekLow', 0),
            'YoY Growth': round(yoy_growth, 2),
            'Beta': 'N/A'
        }
    except Exception as e:
        logger.error(f"Error fetching data for crypto {crypto_symbol}: {e}")
        return None

# Function to load portfolio
def load_portfolio():
    try:
        if os.path.exists(PORTFOLIO_FILE):
            return pd.read_csv(PORTFOLIO_FILE)
        return pd.DataFrame(columns=DEFAULT_PORTFOLIO_COLUMNS)
    except Exception as e:
        logger.error(f"Error loading portfolio: {e}")
        return pd.DataFrame(columns=DEFAULT_PORTFOLIO_COLUMNS)

# Function to save portfolio
def save_portfolio(portfolio_df):
    try:
        portfolio_df.to_csv(PORTFOLIO_FILE, index=False)
        return True
    except Exception as e:
        logger.error(f"Error saving portfolio: {e}")
        return False

# Function to update portfolio with current prices
def update_portfolio_prices(portfolio_df):
    updated_df = portfolio_df.copy()
    
    for index, row in updated_df.iterrows():
        ticker = row['Ticker']
        try:
            if ticker in CRYPTO_TICKERS or ticker.endswith('-USD'):
                data = get_crypto_data(ticker)
            else:
                data = get_stock_data(ticker)
                
            if data:
                updated_df.at[index, 'Current Price'] = data['Current Price']
                updated_df.at[index, 'Total Value'] = data['Current Price'] * row['Shares']
                updated_df.at[index, 'Profit/Loss'] = (data['Current Price'] - row['Purchase Price']) * row['Shares']
                updated_df.at[index, 'Profit/Loss (%)'] = ((data['Current Price'] - row['Purchase Price']) / row['Purchase Price'] * 100) if row['Purchase Price'] > 0 else 0
                updated_df.at[index, 'YoY Growth (%)'] = data['YoY Growth']
                updated_df.at[index, 'P/E Ratio'] = data['P/E Ratio']
                updated_df.at[index, 'Dividend Yield (%)'] = data['Dividend Yield']
                updated_df.at[index, 'Beta'] = data['Beta']
        except Exception as e:
            logger.error(f"Error updating {ticker}: {e}")
    
    return updated_df

# Function to generate investment analysis
def generate_investment_analysis(ticker, query=""):
    llm = init_llm()
    if not llm:
        return "Unable to initialize AI model. Please check your API key."
    
    try:
        # Get financial data
        if ticker in CRYPTO_TICKERS or ticker.endswith('-USD'):
            financial_data = get_crypto_data(ticker)
        else:
            financial_data = get_stock_data(ticker)
        
        if not financial_data:
            return f"Could not retrieve data for {ticker}"
        
        # Get news and sentiment
        ticker_obj = Ticker(ticker)
        news = ticker_obj.news(5)
        
        # Prepare data for analysis
        summarized_data = {"tickers": [ticker], ticker: {}}
        summarized_data[ticker]["financials"] = financial_data
        
        # Add news data
        news_items = []
        for item in news:
            news_items.append({
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "link": item.get("link", ""),
                "published": item.get("published", "")
            })
        
        summarized_data[ticker]["news"] = news_items
        
        # Create analysis prompt
        analysis_prompt = ChatPromptTemplate.from_template(
            """You are a professional Wall Street investment analyst. Generate a comprehensive investment analysis based on the following financial data.
            
            Ticker: {ticker}
            
            Financial Data: {financial_data}
            
            Your analysis should include:
            1. Executive Summary
            2. Company/Asset Overview
            3. Industry Analysis
            4. Financial Analysis
            5. Valuation
            6. Risk Factors
            7. Investment Recommendation (Buy/Hold/Sell) with clear reasoning
            
            Format your response in Markdown with appropriate headings, bullet points, and emphasis.
            """
        )
        
        analysis_chain = analysis_prompt | llm
        
        # Use the custom encoder with summarized data
        financial_data_json = json.dumps(summarized_data, cls=CustomJSONEncoder, indent=2)
        
        response = analysis_chain.invoke({
            "ticker": ticker,
            "financial_data": financial_data_json
        })
        
        return response.content
    except Exception as e:
        logger.error(f"Error generating analysis: {str(e)}")
        return f"Error generating analysis: {str(e)}"

# Function to grade portfolio
def grade_portfolio(portfolio_df):
    if portfolio_df.empty:
        return "No assets in portfolio to grade."
    
    llm = init_llm()
    if not llm:
        return "Unable to initialize AI model. Please check your API key."
    
    try:
        # Prepare portfolio data
        portfolio_summary = {
            "total_value": portfolio_df["Total Value"].sum(),
            "total_profit_loss": portfolio_df["Profit/Loss"].sum(),
            "total_profit_loss_percent": (portfolio_df["Profit/Loss"].sum() / portfolio_df["Purchase Price"].mul(portfolio_df["Shares"]).sum()) * 100 if portfolio_df["Purchase Price"].mul(portfolio_df["Shares"]).sum() > 0 else 0,
            "assets": []
        }
        
        # Add each asset
        for _, row in portfolio_df.iterrows():
            portfolio_summary["assets"].append({
                "ticker": row["Ticker"],
                "name": row["Name"],
                "sector": row["Sector"],
                "shares": row["Shares"],
                "purchase_price": row["Purchase Price"],
                "current_price": row["Current Price"],
                "total_value": row["Total Value"],
                "profit_loss": row["Profit/Loss"],
                "profit_loss_percent": row["Profit/Loss (%)"],
                "yoy_growth": row["YoY Growth (%)"],
                "pe_ratio": row["P/E Ratio"],
                "dividend_yield": row["Dividend Yield (%)"],
                "beta": row["Beta"]
            })
        
        # Calculate sector diversification
        sector_allocation = portfolio_df.groupby("Sector")["Total Value"].sum() / portfolio_df["Total Value"].sum() * 100
        portfolio_summary["sector_allocation"] = sector_allocation.to_dict()
        
        # Create grading prompt
        grading_prompt = ChatPromptTemplate.from_template(
            """You are a professional portfolio manager. Grade the following investment portfolio and provide detailed feedback.
            
            Portfolio Data: {portfolio_data}
            
            Your assessment should include:
            1. Overall Portfolio Grade (A, B, C, D, or F) with clear reasoning
            2. Diversification Analysis
            3. Risk Assessment
            4. Performance Evaluation
            5. Specific Recommendations for Improvement
            
            Format your response in Markdown with appropriate headings, bullet points, and emphasis.
            """
        )
        
        grading_chain = grading_prompt | llm
        
        # Use the custom encoder with portfolio data
        portfolio_data_json = json.dumps(portfolio_summary, cls=CustomJSONEncoder, indent=2)
        
        response = grading_chain.invoke({
            "portfolio_data": portfolio_data_json
        })
        
        return response.content
    except Exception as e:
        logger.error(f"Error grading portfolio: {str(e)}")
        return f"Error grading portfolio: {str(e)}"

# Main app
def main():
    st.set_page_config(page_title="InvestSageAI", page_icon="📈", layout="wide")
    
    st.title("📈 InvestSageAI - Smart Investment Portfolio")
    st.markdown("Manage your investment portfolio with AI-powered analysis")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Portfolio Dashboard", "Add Asset", "Asset Analysis", "Portfolio Grading"])
    
    # Load portfolio
    portfolio_df = load_portfolio()
    
    if page == "Portfolio Dashboard":
        st.header("Portfolio Dashboard")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh Prices"):
                with st.spinner("Updating prices..."):
                    portfolio_df = update_portfolio_prices(portfolio_df)
                    save_portfolio(portfolio_df)
                st.success("Prices updated!")
        
        with col2:
            if not portfolio_df.empty:
                total_value = portfolio_df["Total Value"].sum()
                total_profit_loss = portfolio_df["Profit/Loss"].sum()
                total_investment = portfolio_df["Purchase Price"].mul(portfolio_df["Shares"]).sum()
                total_profit_loss_percent = (total_profit_loss / total_investment * 100) if total_investment > 0 else 0
                
                st.metric("Total Portfolio Value", f"${total_value:,.2f}", 
                          f"{total_profit_loss_percent:.2f}%" if total_profit_loss_percent != 0 else "0.00%")
        
        if portfolio_df.empty:
            st.info("Your portfolio is empty. Go to 'Add Asset' to add stocks or cryptocurrencies.")
        else:
            # Display portfolio table
            st.subheader("Your Assets")
            st.dataframe(portfolio_df, use_container_width=True)
            
            # Portfolio visualizations
            st.subheader("Portfolio Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Asset Allocation Pie Chart
                fig_allocation = px.pie(
                    portfolio_df, 
                    values="Total Value", 
                    names="Ticker",
                    title="Asset Allocation",
                    hover_data=["Name", "Total Value", "Profit/Loss (%)"]
                )
                st.plotly_chart(fig_allocation, use_container_width=True)
            
            with col2:
                # Sector Allocation Pie Chart
                sector_data = portfolio_df.groupby("Sector")["Total Value"].sum().reset_index()
                fig_sector = px.pie(
                    sector_data,
                    values="Total Value",
                    names="Sector",
                    title="Sector Allocation"
                )
                st.plotly_chart(fig_sector, use_container_width=True)
            
            # Performance Bar Chart
            fig_performance = px.bar(
                portfolio_df,
                x="Ticker",
                y="Profit/Loss (%)",
                color="Profit/Loss (%)",
                color_continuous_scale=["red", "green"],
                title="Asset Performance (%)",
                hover_data=["Name", "Total Value", "Profit/Loss"]
            )
            st.plotly_chart(fig_performance, use_container_width=True)
    
    elif page == "Add Asset":
        st.header("Add Asset to Portfolio")
        
        asset_type = st.radio("Asset Type", ["Stock", "Cryptocurrency"])
        
        if asset_type == "Stock":
            ticker = st.text_input("Stock Ticker Symbol (e.g., AAPL, MSFT)")
        else:
            ticker = st.selectbox("Select Cryptocurrency", CRYPTO_TICKERS)
        
        shares = st.number_input("Number of Shares/Units", min_value=0.0001, step=0.0001)
        purchase_price = st.number_input("Purchase Price per Share/Unit ($)", min_value=0.01, step=0.01)
        
        if st.button("Add to Portfolio"):
            if not ticker or shares <= 0 or purchase_price <= 0:
                st.error("Please fill in all fields correctly.")
            else:
                with st.spinner(f"Fetching data for {ticker}..."):
                    if asset_type == "Cryptocurrency":
                        data = get_crypto_data(ticker)
                    else:
                        data = get_stock_data(ticker)
                    
                    if data:
                        # Check if ticker already exists in portfolio
                        if not portfolio_df.empty and ticker in portfolio_df["Ticker"].values:
                            st.error(f"{ticker} already exists in your portfolio. Please use update functionality instead.")
                        else:
                            # Calculate values
                            current_price = data["Current Price"]
                            total_value = current_price * shares
                            profit_loss = (current_price - purchase_price) * shares
                            profit_loss_percent = ((current_price - purchase_price) / purchase_price * 100) if purchase_price > 0 else 0
                            
                            # Create new row
                            new_row = {
                                'Ticker': ticker,
                                'Name': data["Name"],
                                'Current Price': current_price,
                                'Purchase Price': purchase_price,
                                'Shares': shares,
                                'Total Value': total_value,
                                'Profit/Loss': profit_loss,
                                'Profit/Loss (%)': profit_loss_percent,
                                'YoY Growth (%)': data["YoY Growth"],
                                'P/E Ratio': data["P/E Ratio"],
                                'Dividend Yield (%)': data["Dividend Yield"],
                                'Beta': data["Beta"],
                                'Sector': data["Sector"],
                                'Added Date': datetime.now().strftime("%Y-%m-%d")
                            }
                            
                            # Add to portfolio
                            portfolio_df = pd.concat([portfolio_df, pd.DataFrame([new_row])], ignore_index=True)
                            
                            # Save updated portfolio
                            if save_portfolio(portfolio_df):
                                st.success(f"{ticker} added to your portfolio!")
                            else:
                                st.error("Failed to save portfolio. Please try again.")
                    else:
                        st.error(f"Could not retrieve data for {ticker}. Please check the ticker symbol.")
    
    elif page == "Asset Analysis":
        st.header("AI-Powered Asset Analysis")
        
        if portfolio_df.empty:
            st.info("Your portfolio is empty. Add assets to analyze them.")
        else:
            ticker = st.selectbox("Select Asset to Analyze", portfolio_df["Ticker"].tolist())
            
            if st.button("Generate Analysis"):
                with st.spinner("Generating professional investment analysis..."):
                    analysis = generate_investment_analysis(ticker)
                    st.markdown(analysis)
    
    elif page == "Portfolio Grading":
        st.header("AI Portfolio Grading")
        
        if portfolio_df.empty:
            st.info("Your portfolio is empty. Add assets to receive a portfolio grade.")
        else:
            if st.button("Grade My Portfolio"):
                with st.spinner("Analyzing your portfolio..."):
                    # Update prices first
                    portfolio_df = update_portfolio_prices(portfolio_df)
                    save_portfolio(portfolio_df)
                    
                    # Generate grade
                    grade = grade_portfolio(portfolio_df)
                    st.markdown(grade)

if __name__ == "__main__":
    main()
