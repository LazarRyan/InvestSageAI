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
import traceback
import plotly.express as px
import plotly.graph_objects as go
import csv

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
def get_stock_data(ticker):
    # Check if ticker might be a cryptocurrency
    is_crypto = ticker.upper() in [t.split('-')[0] for t in CRYPTO_TICKERS]
    
    if is_crypto:
        return get_crypto_data(ticker)
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Basic validation - if we can't get basic info, try as crypto
        if not info or len(info) < 5:
            crypto_data = get_crypto_data(ticker)
            if crypto_data:
                return crypto_data
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        historical_data = stock.history(start=start_date, end=end_date)
        
        # Calculate YoY growth
        if not historical_data.empty:
            current_price = historical_data['Close'].iloc[-1]
            year_ago_price = historical_data['Close'].iloc[0] if len(historical_data) > 20 else current_price
            yoy_growth = ((current_price - year_ago_price) / year_ago_price * 100) if year_ago_price > 0 else 0
        else:
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            yoy_growth = 0
        
        # Extract key financials
        financials = {
            "Name": info.get('shortName', ticker),
            "Sector": info.get('sector', 'Unknown'),
            "Industry": info.get('industry', 'Unknown'),
            "Current Price": current_price,
            "Market Cap": info.get('marketCap', 0),
            "P/E Ratio": info.get('trailingPE', info.get('forwardPE', 0)),
            "EPS": info.get('trailingEps', 0),
            "Dividend Yield": info.get('dividendYield', 0),
            "52 Week High": info.get('fiftyTwoWeekHigh', 0),
            "52 Week Low": info.get('fiftyTwoWeekLow', 0),
            "YoY Growth": round(yoy_growth, 2),
            "Beta": info.get('beta', 1)
        }
        
        return {
            "financials": financials,
            "historical_data": historical_data
        }
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        # Try as crypto as fallback
        try:
            return get_crypto_data(ticker)
        except:
            return None

# Function to get crypto data
def get_crypto_data(ticker):
    try:
        # Add -USD suffix if not already present for crypto tickers
        if not ticker.endswith('-USD') and not ticker.endswith('-USDT'):
            crypto_ticker = f"{ticker}-USD"
        else:
            crypto_ticker = ticker
        
        # Use yahooquery to get crypto data
        crypto = Ticker(crypto_ticker)
        
        # Get summary data
        summary = crypto.summary_detail
        
        if crypto_ticker not in summary:
            # Try alternative suffix
            crypto_ticker = f"{ticker}-USDT"
            crypto = Ticker(crypto_ticker)
            summary = crypto.summary_detail
            
            if crypto_ticker not in summary:
                logger.warning(f"Could not find data for crypto {ticker}")
                return None
        
        info = summary[crypto_ticker]
        
        # Get price history
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Format dates for yahooquery
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        historical_data = crypto.history(start=start_str, end=end_str)
        
        # Calculate YoY growth
        if not historical_data.empty:
            current_price = historical_data['close'].iloc[-1]
            year_ago_price = historical_data['close'].iloc[0] if len(historical_data) > 20 else current_price
            yoy_growth = ((current_price - year_ago_price) / year_ago_price * 100) if year_ago_price > 0 else 0
        else:
            current_price = info.get('regularMarketPrice', 0)
            yoy_growth = 0
        
        # Get additional data
        quote = crypto.price
        
        # Extract key financials
        financials = {
            "Name": info.get('shortName', crypto_ticker),
            "Sector": "Cryptocurrency",
            "Industry": "Digital Assets",
            "Current Price": current_price,
            "Market Cap": info.get('marketCap', 0),
            "P/E Ratio": None,  # Not applicable for crypto
            "EPS": None,  # Not applicable for crypto
            "Dividend Yield": 0,  # Not applicable for crypto
            "52 Week High": info.get('fiftyTwoWeekHigh', 0),
            "52 Week Low": info.get('fiftyTwoWeekLow', 0),
            "YoY Growth": round(yoy_growth, 2),
            "Beta": info.get('beta', 1.5) if info.get('beta') else 1.5  # Default higher beta for crypto
        }
        
        return {
            "financials": financials,
            "historical_data": historical_data
        }
    except Exception as e:
        logger.error(f"Error fetching crypto data for {ticker}: {str(e)}")
        return None

# Function to load portfolio
def load_portfolio():
    try:
        if os.path.exists(PORTFOLIO_FILE):
            # Check file size to ensure it's not empty
            if os.path.getsize(PORTFOLIO_FILE) > 0:
                # Try reading with pandas
                try:
                    portfolio_df = pd.read_csv(PORTFOLIO_FILE)
                    if len(portfolio_df) > 0:
                        print(f"Successfully loaded portfolio with {len(portfolio_df)} rows")
                        return portfolio_df
                except Exception as e:
                    print(f"Error reading with pandas: {e}")
                    
                # Fallback: try reading with csv module
                try:
                    rows = []
                    with open(PORTFOLIO_FILE, 'r', newline='') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            rows.append(row)
                    
                    if rows:
                        portfolio_df = pd.DataFrame(rows)
                        print(f"Loaded portfolio using CSV module: {len(portfolio_df)} rows")
                        return portfolio_df
                except Exception as csv_error:
                    print(f"Error reading with CSV module: {csv_error}")
        
        # If we get here, either file doesn't exist or couldn't be read
        print("Creating new empty portfolio")
        return pd.DataFrame(columns=DEFAULT_PORTFOLIO_COLUMNS)
    except Exception as e:
        print(f"Error in load_portfolio: {e}")
        return pd.DataFrame(columns=DEFAULT_PORTFOLIO_COLUMNS)

# Function to save portfolio
def save_portfolio(portfolio_df):
    try:
        # Print the absolute path for debugging
        abs_path = os.path.abspath(PORTFOLIO_FILE)
        print(f"Saving portfolio to: {abs_path}")
        
        # Use a direct filepath in the current directory as a fallback
        backup_path = "portfolio_backup.csv"
        
        # Try both paths
        try:
            # Primary save location
            portfolio_df.to_csv(PORTFOLIO_FILE, index=False)
            print(f"Portfolio saved to: {os.path.abspath(PORTFOLIO_FILE)}")
            file_size = os.path.getsize(PORTFOLIO_FILE)
            print(f"File size: {file_size} bytes")
            return True
        except Exception as primary_error:
            print(f"Error saving to primary location: {primary_error}")
            # Backup save location
            portfolio_df.to_csv(backup_path, index=False)
            print(f"Portfolio saved to backup location: {os.path.abspath(backup_path)}")
            return True
    except Exception as e:
        print(f"Critical error saving portfolio: {str(e)}")
        logger.error(f"Error saving portfolio: {str(e)}")
        st.error(f"Could not save portfolio: {str(e)}")
        return False

# Function to update portfolio with current prices
def update_portfolio_prices(portfolio_df):
    updated_df = portfolio_df.copy()
    
    for index, row in updated_df.iterrows():
        ticker = row['Ticker']
        try:
            if ticker in [t.split('-')[0] for t in CRYPTO_TICKERS] or ticker.endswith('-USD'):
                data = get_crypto_data(ticker)
            else:
                data = get_stock_data(ticker)
                
            if data:
                financials = data.get("financials", {})
                updated_df.at[index, 'Current Price'] = financials.get('Current Price', row['Current Price'])
                updated_df.at[index, 'Total Value'] = financials.get('Current Price', row['Current Price']) * row['Shares']
                updated_df.at[index, 'Profit/Loss'] = (financials.get('Current Price', row['Current Price']) - row['Purchase Price']) * row['Shares']
                updated_df.at[index, 'Profit/Loss (%)'] = ((financials.get('Current Price', row['Current Price']) - row['Purchase Price']) / row['Purchase Price'] * 100) if row['Purchase Price'] > 0 else 0
                updated_df.at[index, 'YoY Growth (%)'] = financials.get('YoY Growth', row['YoY Growth (%)'])
                updated_df.at[index, 'P/E Ratio'] = financials.get('P/E Ratio', row['P/E Ratio'])
                updated_df.at[index, 'Dividend Yield (%)'] = financials.get('Dividend Yield', row['Dividend Yield (%)'])
                updated_df.at[index, 'Beta'] = financials.get('Beta', row['Beta'])
        except Exception as e:
            logger.error(f"Error updating {ticker}: {e}")
    
    return updated_df

# Function to add to portfolio
def add_to_portfolio(ticker_data, ticker, shares, purchase_price):
    try:
        # Get portfolio file path
        file_path = PORTFOLIO_FILE
        print(f"Attempting to add {ticker} to portfolio at: {os.path.abspath(file_path)}")
        
        # Get data from ticker_data 
        financials = ticker_data.get("financials", {})
        if not financials:
            st.error(f"No financial data available for {ticker}")
            return False
        
        current_price = float(financials.get("Current Price", 0))
        
        # Create row data
        total_value = shares * current_price
        profit_loss = shares * (current_price - purchase_price)
        profit_loss_pct = ((current_price - purchase_price) / purchase_price * 100) if purchase_price > 0 else 0
        
        row_data = {
            'Ticker': ticker,
            'Name': financials.get("Name", ticker),
            'Current Price': current_price,
            'Purchase Price': purchase_price,
            'Shares': shares,
            'Total Value': total_value,
            'Profit/Loss': profit_loss,
            'Profit/Loss (%)': profit_loss_pct,
            'YoY Growth (%)': float(financials.get("YoY Growth", 0)),
            'P/E Ratio': float(financials.get("P/E Ratio", 0)) if financials.get("P/E Ratio") else 0,
            'Dividend Yield (%)': float(financials.get("Dividend Yield", 0) * 100) if financials.get("Dividend Yield") else 0,
            'Beta': float(financials.get("Beta", 1)) if financials.get("Beta") else 1,
            'Sector': str(financials.get("Sector", "Unknown")),
            'Added Date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Try with the main approach
        try:
            # Check if file exists
            file_exists = os.path.exists(file_path)
            print(f"File exists: {file_exists}, Path: {file_path}")
            
            # Open the file in append mode if it exists, or write mode if it doesn't
            mode = 'a' if file_exists else 'w'
            with open(file_path, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=DEFAULT_PORTFOLIO_COLUMNS)
                
                # Write headers if this is a new file
                if not file_exists:
                    writer.writeheader()
                
                # Write the data row
                writer.writerow(row_data)
            
            print(f"Successfully wrote to file: {file_path}")
            
            # Now update session state by reading the file we just wrote to
            try:
                updated_portfolio = pd.read_csv(file_path)
                st.session_state.portfolio = updated_portfolio
                print(f"Updated session state portfolio: {len(updated_portfolio)} rows")
                return True
            except Exception as e:
                print(f"Error updating session state, but file was written: {e}")
                # Try to continue since we know the file was written
                return True
                
        except Exception as e:
            print(f"Error in standard CSV approach: {e}")
            traceback.print_exc()
            
            # Try the fallback with pandas approach
            try:
                # If file exists, load it first
                if file_exists:
                    existing_df = pd.read_csv(file_path)
                    # Add new row
                    new_df = pd.concat([existing_df, pd.DataFrame([row_data])], ignore_index=True)
                else:
                    # Create new DataFrame with just this row
                    new_df = pd.DataFrame([row_data])
                
                # Save back to file
                new_df.to_csv(file_path, index=False)
                
                # Update session state
                st.session_state.portfolio = new_df
                print(f"Used pandas fallback approach, portfolio now has {len(new_df)} rows")
                return True
                
            except Exception as pandas_error:
                print(f"Both CSV and pandas approaches failed: {pandas_error}")
                # Last resort - try the direct method that works
                success, direct_path = direct_add_to_portfolio(ticker, purchase_price, shares)
                if success:
                    try:
                        # Copy the direct file to the main file
                        direct_df = pd.read_csv(direct_path)
                        direct_df.to_csv(file_path, index=False)
                        st.session_state.portfolio = direct_df
                        print(f"Used direct fallback approach, portfolio now has {len(direct_df)} rows")
                        return True
                    except:
                        print("Failed to copy direct file to main file")
                return False
            
    except Exception as e:
        print(f"Error in add_to_portfolio: {e}")
        traceback.print_exc()
        return False

# Function to directly add to portfolio (the working method)
def direct_add_to_portfolio(ticker, price, shares):
    """Directly add a row to the portfolio CSV file without using session state"""
    try:
        # Define column headers
        headers = DEFAULT_PORTFOLIO_COLUMNS
        
        # Data for the new row
        row_data = {
            'Ticker': ticker,
            'Name': ticker,  # Simplified for direct add
            'Current Price': price,
            'Purchase Price': price,
            'Shares': shares,
            'Total Value': price * shares,
            'Profit/Loss': 0,
            'Profit/Loss (%)': 0,
            'YoY Growth (%)': 0,
            'P/E Ratio': 0,
            'Dividend Yield (%)': 0,
            'Beta': 1,
            'Sector': 'Unknown',
            'Added Date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Use direct file path
        file_path = "portfolio_direct.csv"
        
        # Check if file exists
        file_exists = os.path.exists(file_path)
        
        # Open the file in append mode if it exists, or write mode if it doesn't
        mode = 'a' if file_exists else 'w'
        with open(file_path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            
            # Write headers if this is a new file
            if not file_exists:
                writer.writeheader()
            
            # Write the data row
            writer.writerow(row_data)
        
        # Read the file to confirm it has data
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            print(f"Portfolio now has {len(rows)-1} rows of data")
            
        return True, file_path
    except Exception as e:
        print(f"Error in direct_add_to_portfolio: {e}")
        traceback.print_exc()
        return False, str(e)

# Function to remove asset from portfolio
def remove_from_portfolio(ticker):
    try:
        portfolio_df = st.session_state.portfolio
        if ticker not in portfolio_df['Ticker'].values:
            st.session_state.remove_success = {"status": "warning", "message": f"{ticker} not found in portfolio"}
            return False
            
        st.session_state.portfolio = portfolio_df[portfolio_df['Ticker'] != ticker]
        print(f"Removed {ticker} from portfolio")
        
        # Save updated portfolio to CSV
        if save_portfolio(st.session_state.portfolio):
            st.session_state.remove_success = {"status": "success", "message": f"Removed {ticker} from your portfolio!"}
            return True
        else:
            st.session_state.remove_success = {"status": "error", "message": f"Failed to save portfolio after removing {ticker}"}
            return False
    except Exception as e:
        logger.error(f"Error removing {ticker} from portfolio: {str(e)}")
        st.session_state.remove_success = {"status": "error", "message": f"Error removing {ticker}: {str(e)}"}
        return False

# Function to generate investment analysis with visualizations
def generate_investment_analysis(ticker, query=""):
    llm = init_llm()
    if not llm:
        return "Unable to initialize AI model. Please check your API key."
    
    try:
        # Get financial data
        if ticker in [t.split('-')[0] for t in CRYPTO_TICKERS] or ticker.endswith('-USD'):
            financial_data = get_crypto_data(ticker)
        else:
            financial_data = get_stock_data(ticker)
        
        if not financial_data:
            return f"Could not retrieve data for {ticker}", None
        
        # Get news and sentiment
        ticker_obj = Ticker(ticker)
        news = ticker_obj.news(5)
        
        # Prepare data for analysis
        summarized_data = {"tickers": [ticker], ticker: {}}
        summarized_data[ticker]["financials"] = financial_data.get("financials", {})
        
        # Add news data
        news_items = []
        for item in news:
            if isinstance(item, dict):  # Make sure item is a dictionary before accessing with .get()
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
        
        return response.content, financial_data
    except Exception as e:
        logger.error(f"Error generating analysis: {str(e)}")
        traceback.print_exc()  # Add this to get more detailed error information
        return f"Error generating analysis: {str(e)}", None

# Function to grade portfolio
def grade_portfolio(portfolio_df):
    if portfolio_df.empty:
        return {
            "grade": "N/A",
            "rationale": "Portfolio is empty",
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
    
    try:
        # Calculate portfolio metrics considering position sizes
        total_value = portfolio_df["Total Value"].sum()
        
        # Calculate weighted metrics
        weighted_pe = ((portfolio_df['P/E Ratio'] * portfolio_df['Total Value']).sum() / 
                      total_value) if total_value > 0 else 0
        weighted_growth = ((portfolio_df['YoY Growth (%)'] * portfolio_df['Total Value']).sum() / 
                          total_value) if total_value > 0 else 0
        weighted_dividend = ((portfolio_df['Dividend Yield (%)'] * portfolio_df['Total Value']).sum() / 
                            total_value) if total_value > 0 else 0
        weighted_beta = ((portfolio_df['Beta'] * portfolio_df['Total Value']).sum() / 
                        total_value) if total_value > 0 else 0
        
        sector_diversity = len(portfolio_df['Sector'].unique())
        asset_count = len(portfolio_df)
        
        # Calculate sector concentration
        sector_weights = portfolio_df.groupby('Sector')['Total Value'].sum() / total_value * 100
        max_sector_weight = sector_weights.max() if not sector_weights.empty else 0
        
        # Calculate position concentration - top 3 positions
        position_weights = portfolio_df['Total Value'] / total_value * 100
        top_positions_weight = position_weights.nlargest(3).sum() if not position_weights.empty else 0
        
        # Determine grade based on metrics
        grade = "C"  # Default grade
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Asset count analysis
        if asset_count < 5:
            weaknesses.append("Portfolio has few positions, increasing specific risk")
            recommendations.append("Add more assets to diversify company-specific risk")
        elif asset_count >= 15:
            strengths.append(f"Well-diversified across {asset_count} positions")
        
        # Sector diversity analysis
        if sector_diversity <= 2:
            weaknesses.append(f"Limited sector diversification across only {sector_diversity} sectors")
            recommendations.append("Add assets from different sectors to reduce sector risk")
        elif sector_diversity >= 5:
            strengths.append(f"Good sector diversification across {sector_diversity} sectors")
        
        # Growth analysis
        if weighted_growth > 15:
            strengths.append(f"Strong overall growth potential ({weighted_growth:.1f}%)")
            grade = "B" if grade == "C" else grade
        elif weighted_growth < 0:
            weaknesses.append(f"Negative average growth rate ({weighted_growth:.1f}%)")
            recommendations.append("Consider adding growth-oriented positions")
            grade = "D" if grade != "F" else grade
        
        # P/E analysis
        if 10 <= weighted_pe <= 20:
            strengths.append(f"Balanced P/E ratio indicates reasonable valuation ({weighted_pe:.1f})")
        elif weighted_pe > 30:
            weaknesses.append(f"High average P/E ratio may indicate overvaluation ({weighted_pe:.1f})")
            recommendations.append("Consider adding some value stocks with lower P/E ratios")
        
        # Risk analysis
        if weighted_beta > 1.5:
            weaknesses.append(f"High portfolio beta indicates above-average risk (β={weighted_beta:.2f})")
            recommendations.append("Add lower-beta stocks to improve stability")
        elif weighted_beta < 0.8:
            strengths.append(f"Low portfolio beta provides stability (β={weighted_beta:.2f})")
        
        # Dividend analysis
        if weighted_dividend > 3:
            strengths.append(f"Good dividend yield for income ({weighted_dividend:.1f}%)")
        
        # Concentration analysis
        if max_sector_weight > 40:
            weaknesses.append(f"Over-concentration in {sector_weights.idxmax()} sector ({max_sector_weight:.1f}%)")
            recommendations.append(f"Reduce exposure to {sector_weights.idxmax()} sector")
            
        if top_positions_weight > 60:
            weaknesses.append(f"Top 3 positions account for {top_positions_weight:.1f}% of portfolio")
            recommendations.append("Consider reducing largest positions to limit concentration risk")
        
        # Final grade determination
        if len(strengths) >= 4 and len(weaknesses) <= 1:
            grade = "A"
        elif len(strengths) >= 3 and len(weaknesses) <= 2:
            grade = "B"
        elif len(weaknesses) >= 4:
            grade = "D"
        elif len(weaknesses) >= 5:
            grade = "F"
        
        return {
            "grade": grade,
            "rationale": f"Portfolio of {asset_count} assets across {sector_diversity} sectors with {weighted_growth:.1f}% avg growth and {weighted_beta:.2f} beta",
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"Error grading portfolio: {str(e)}")
        return {
            "grade": "C",
            "rationale": "Unable to fully analyze portfolio",
            "strengths": ["Portfolio exists"],
            "weaknesses": ["Analysis error occurred"],
            "recommendations": ["Review portfolio manually"]
        }

# Main app
def main():
    st.set_page_config(page_title="InvestSageAI", page_icon="📈", layout="wide")
    
    st.title("📈 InvestSageAI - Smart Investment Portfolio")
    st.markdown("Manage your investment portfolio with AI-powered analysis")
    
    # Initialize session state for portfolio if not already done
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = load_portfolio()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Portfolio Dashboard", "Asset Explorer & Analysis"])
    
    # Load portfolio from session state
    portfolio_df = st.session_state.portfolio
    
    if page == "Portfolio Dashboard":
        st.header("Portfolio Dashboard")
        
        # Create tabs for different portfolio views
        overview_tab, analysis_tab = st.tabs(["Portfolio Overview", "Portfolio Analysis & Grading"])
        
        with overview_tab:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Refresh Prices"):
                    with st.spinner("Updating prices..."):
                        portfolio_df = update_portfolio_prices(portfolio_df)
                        save_portfolio(portfolio_df)
                        st.session_state.portfolio = portfolio_df
                        st.success("Prices updated successfully!")
            
            with col2:
                if not portfolio_df.empty:
                    total_value = portfolio_df["Total Value"].sum()
                    total_profit_loss = portfolio_df["Profit/Loss"].sum()
                    total_investment = portfolio_df["Purchase Price"].mul(portfolio_df["Shares"]).sum()
                    total_profit_loss_percent = (total_profit_loss / total_investment * 100) if total_investment > 0 else 0
                    
                    st.metric("Total Portfolio Value", f"${total_value:,.2f}", 
                              f"{total_profit_loss_percent:.2f}%" if total_profit_loss_percent != 0 else "0.00%")
            
            if portfolio_df.empty:
                st.info("Your portfolio is empty. Go to 'Asset Explorer & Analysis' to add stocks or cryptocurrencies.")
            else:
                # Add remove asset functionality
                with st.expander("Remove Asset"):
                    asset_to_remove = st.selectbox("Select asset to remove", portfolio_df["Ticker"].tolist())
                    if st.button("Remove Asset"):
                        if remove_from_portfolio(asset_to_remove):
                            st.success(f"Removed {asset_to_remove} from your portfolio!")
                            # Refresh the page to show updated portfolio
                            st.rerun()
                
                # Display portfolio table
                st.subheader("Your Assets")
                # Convert numeric columns to appropriate types
                numeric_cols = ['Current Price', 'Purchase Price', 'Shares', 'Total Value', 
                               'Profit/Loss', 'Profit/Loss (%)', 'YoY Growth (%)', 
                               'P/E Ratio', 'Dividend Yield (%)', 'Beta']
                
                for col in numeric_cols:
                    if col in portfolio_df.columns:
                        portfolio_df[col] = pd.to_numeric(portfolio_df[col], errors='coerce')
                
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
                    fig_allocation.update_traces(textposition='inside', textinfo='percent+label')
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
                    fig_sector.update_traces(textposition='inside', textinfo='percent+label')
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
                
                # Add a historical performance line chart
                st.subheader("Portfolio Composition")
                fig_composition = px.bar(
                    portfolio_df,
                    x="Ticker",
                    y="Total Value",
                    color="Sector",
                    title="Portfolio Composition by Value",
                    hover_data=["Name", "Shares", "Current Price"]
                )
                st.plotly_chart(fig_composition, use_container_width=True)
        
        with analysis_tab:
            st.subheader("Portfolio Analysis & Grading")
            
            if portfolio_df.empty:
                st.info("Your portfolio is empty. Add assets to receive a portfolio grade.")
            else:
                if st.button("Analyze My Portfolio"):
                    with st.spinner("Analyzing your portfolio..."):
                        # Update prices first
                        portfolio_df = update_portfolio_prices(portfolio_df)
                        save_portfolio(portfolio_df)
                        st.session_state.portfolio = portfolio_df
                        
                        # Generate grade
                        grade_result = grade_portfolio(portfolio_df)
                        
                        # Display grade with visual elements
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            # Display grade in a large circle
                            grade_color = {
                                "A": "#28a745",  # Green
                                "B": "#5cb85c",  # Light green
                                "C": "#ffc107",  # Yellow
                                "D": "#fd7e14",  # Orange
                                "F": "#dc3545"   # Red
                            }.get(grade_result["grade"], "#6c757d")
                            
                            st.markdown(f"""
                            <div style="width:120px;height:120px;border-radius:50%;background-color:{grade_color};
                            display:flex;align-items:center;justify-content:center;margin:auto;">
                                <span style="font-size:60px;font-weight:bold;color:white;">{grade_result["grade"]}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.subheader("Portfolio Grade Assessment")
                            st.write(grade_result["rationale"])
                            
                            # Strengths
                            if grade_result["strengths"]:
                                st.markdown("#### Strengths")
                                for strength in grade_result["strengths"]:
                                    st.markdown(f"✅ {strength}")
                            
                            # Weaknesses
                            if grade_result["weaknesses"]:
                                st.markdown("#### Areas for Improvement")
                                for weakness in grade_result["weaknesses"]:
                                    st.markdown(f"⚠️ {weakness}")
                        
                        # Recommendations
                        if grade_result["recommendations"]:
                            st.markdown("#### Recommendations")
                            for recommendation in grade_result["recommendations"]:
                                st.markdown(f"💡 {recommendation}")
                        
                        # Add portfolio metrics visualizations
                        st.subheader("Portfolio Metrics Analysis")
                        
                        # Create metrics for visualization
                        metrics_data = {
                            "Metric": ["Sector Diversity", "Asset Count", "Avg P/E Ratio", "Avg Dividend Yield", "Avg Beta", "Avg YoY Growth"],
                            "Value": [
                                len(portfolio_df['Sector'].unique()),
                                len(portfolio_df),
                                portfolio_df['P/E Ratio'].mean(),
                                portfolio_df['Dividend Yield (%)'].mean(),
                                portfolio_df['Beta'].mean(),
                                portfolio_df['YoY Growth (%)'].mean()
                            ]
                        }
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        
                        # Create a radar chart for portfolio metrics
                        fig = px.line_polar(
                            metrics_df, 
                            r="Value", 
                            theta="Metric", 
                            line_close=True,
                            title="Portfolio Metrics Overview"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add a sector diversification chart
                        sector_allocation = portfolio_df.groupby('Sector')['Total Value'].sum().reset_index()
                        sector_allocation['Percentage'] = sector_allocation['Total Value'] / sector_allocation['Total Value'].sum() * 100
                        
                        fig = px.bar(
                            sector_allocation, 
                            x='Sector', 
                            y='Percentage',
                            title='Sector Allocation Analysis',
                            labels={'Percentage': 'Allocation (%)'},
                            color='Percentage',
                            color_continuous_scale=px.colors.sequential.Viridis
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk vs Return scatter plot
                        st.subheader("Risk vs Return Analysis")
                        fig = px.scatter(
                            portfolio_df,
                            x="Beta",
                            y="YoY Growth (%)",
                            size="Total Value",
                            color="Sector",
                            hover_name="Ticker",
                            hover_data=["Name", "Current Price", "Profit/Loss (%)"],
                            title="Risk (Beta) vs Return (Growth) by Asset",
                            labels={"Beta": "Risk (Beta)", "YoY Growth (%)": "Return (YoY Growth %)"}
                        )
                        fig.add_shape(
                            type="line",
                            x0=1, y0=0,
                            x1=1, y1=portfolio_df["YoY Growth (%)"].max() * 1.1,
                            line=dict(color="red", width=1, dash="dash")
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    
    elif page == "Asset Explorer & Analysis":
        st.header("Asset Explorer & Analysis")
        
        # Create tabs for different functionalities
        explore_tab, analyze_tab = st.tabs(["Add to Portfolio", "Analyze Asset"])
        
        with explore_tab:
            st.subheader("Find and Add Assets")
            
            asset_type = st.radio("Asset Type", ["Stock", "Cryptocurrency"])
            
            if asset_type == "Stock":
                ticker = st.text_input("Stock Ticker Symbol (e.g., AAPL, MSFT)")
            else:
                ticker = st.selectbox("Select Cryptocurrency", CRYPTO_TICKERS)
            
            if ticker:
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Fetch Data"):
                        with st.spinner(f"Fetching data for {ticker}..."):
                            if asset_type == "Cryptocurrency":
                                data = get_crypto_data(ticker)
                            else:
                                data = get_stock_data(ticker)
                            
                            if data and data.get("financials"):
                                financials = data.get("financials")
                                st.success(f"Data retrieved for {financials.get('Name', ticker)}")
                                
                                # Store data in session state for later use
                                st.session_state.ticker_data = data
                                
                                # Display asset information in a nice format
                                st.markdown("### Asset Information")
                                
                                info_col1, info_col2 = st.columns(2)
                                with info_col1:
                                    st.metric("Current Price", f"${financials.get('Current Price', 0):.2f}")
                                    st.metric("Market Cap", f"${financials.get('Market Cap', 0):,.0f}" if financials.get('Market Cap', 0) > 0 else "N/A")
                                    st.metric("52-Week Range", f"${financials.get('52 Week Low', 0):.2f} - ${financials.get('52 Week High', 0):.2f}")
                                
                                with info_col2:
                                    st.metric("YoY Growth", f"{financials.get('YoY Growth', 0):.2f}%")
                                    st.metric("P/E Ratio", f"{financials.get('P/E Ratio', 0):.2f}" if financials.get('P/E Ratio') else "N/A")
                                    st.metric("Dividend Yield", f"{financials.get('Dividend Yield', 0) * 100:.2f}%" if financials.get('Dividend Yield') else "N/A")
                                
                                # Display historical chart if available
                                if "historical_data" in data and not data["historical_data"].empty:
                                    st.markdown("### Price History")
                                    hist_data = data["historical_data"]
                                    
                                    if "Close" in hist_data.columns:
                                        price_col = "Close"
                                    elif "close" in hist_data.columns:
                                        price_col = "close"
                                    else:
                                        price_col = hist_data.columns[0]
                                    
                                    fig = px.line(
                                        hist_data, 
                                        y=price_col, 
                                        title=f"{ticker} Price History"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error(f"Could not retrieve data for {ticker}. Please check the ticker symbol.")
                
                # Add to portfolio section
                with col2:
                    st.markdown("### Add to Portfolio")
                    shares = st.number_input("Number of Shares/Units", min_value=0.0001, step=0.0001, format="%.4f")
                    purchase_price = st.number_input("Purchase Price per Share/Unit ($)", min_value=0.01, step=0.01, format="%.2f")
                    
                    if st.button("Add to Portfolio"):
                        if not ticker or shares <= 0 or purchase_price <= 0:
                            st.error("Please fill in all fields correctly.")
                        else:
                            # Check if we have data in session state, otherwise fetch it
                            if hasattr(st.session_state, 'ticker_data') and st.session_state.ticker_data:
                                ticker_data = st.session_state.ticker_data
                            else:
                                with st.spinner(f"Fetching data for {ticker}..."):
                                    if asset_type == "Cryptocurrency":
                                        ticker_data = get_crypto_data(ticker)
                                    else:
                                        ticker_data = get_stock_data(ticker)
                            
                            if ticker_data:
                                # Check if ticker already exists in portfolio
                                if not portfolio_df.empty and ticker in portfolio_df["Ticker"].values:
                                    st.error(f"{ticker} already exists in your portfolio. Please use update functionality instead.")
                                else:
                                    # Add to portfolio
                                    if add_to_portfolio(ticker_data, ticker, shares, purchase_price):
                                        st.success(f"{ticker} added to your portfolio!")
                                        # Refresh the page to show updated portfolio
                                        st.rerun()
                                    else:
                                        st.error("Failed to add to portfolio. Please try again.")
                            else:
                                st.error(f"Could not retrieve data for {ticker}. Please check the ticker symbol.")
        
        with analyze_tab:
            st.subheader("AI-Powered Asset Analysis")
            
            # Allow analysis of any ticker, not just portfolio assets
            analysis_type = st.radio("Analysis Type", ["Portfolio Asset", "Any Asset"])
            
            if analysis_type == "Portfolio Asset":
                if portfolio_df.empty:
                    st.info("Your portfolio is empty. Add assets to analyze them.")
                else:
                    ticker_to_analyze = st.selectbox("Select Asset to Analyze", portfolio_df["Ticker"].tolist())
            else:
                ticker_to_analyze = st.text_input("Enter Ticker Symbol to Analyze")
            
            if ticker_to_analyze:
                if st.button("Generate Analysis"):
                    with st.spinner("Generating professional investment analysis..."):
                        analysis_text, financial_data = generate_investment_analysis(ticker_to_analyze)
                        
                        # Create tabs for text analysis and visualizations
                        analysis_tab, visuals_tab = st.tabs(["Analysis Report", "Data Visualizations"])
                        
                        with analysis_tab:
                            st.markdown(analysis_text)
                        
                        with visuals_tab:
                            if financial_data:
                                st.subheader(f"Visual Analysis for {ticker_to_analyze}")
                                
                                # Extract data for visualizations
                                financials = financial_data.get("financials", {})
                                historical_data = financial_data.get("historical_data")
                                
                                # Create columns for metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Current Price", f"${financials.get('Current Price', 0):.2f}")
                                    st.metric("Market Cap", f"${financials.get('Market Cap', 0):,.0f}" if financials.get('Market Cap', 0) > 0 else "N/A")
                                with col2:
                                    st.metric("P/E Ratio", f"{financials.get('P/E Ratio', 0):.2f}" if financials.get('P/E Ratio') else "N/A")
                                    st.metric("EPS", f"${financials.get('EPS', 0):.2f}" if financials.get('EPS') else "N/A")
                                with col3:
                                    st.metric("Dividend Yield", f"{financials.get('Dividend Yield', 0) * 100:.2f}%" if financials.get('Dividend Yield') else "N/A")
                                    st.metric("Beta", f"{financials.get('Beta', 0):.2f}" if financials.get('Beta') else "N/A")
                                
                                # Historical price chart
                                if historical_data is not None and not historical_data.empty:
                                    st.subheader("Price History")
                                    
                                    # Determine price column name
                                    if "Close" in historical_data.columns:
                                        price_col = "Close"
                                    elif "close" in historical_data.columns:
                                        price_col = "close"
                                    else:
                                        price_col = historical_data.columns[0]
                                    
                                    # Create price chart
                                    fig = px.line(
                                        historical_data,
                                        y=price_col,
                                        title=f"{ticker_to_analyze} Price History (Last 12 Months)"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Add volume chart if available
                                    if "Volume" in historical_data.columns or "volume" in historical_data.columns:
                                        vol_col = "Volume" if "Volume" in historical_data.columns else "volume"
                                        fig_vol = px.bar(
                                            historical_data,
                                            y=vol_col,
                                            title=f"{ticker_to_analyze} Trading Volume"
                                        )
                                        st.plotly_chart(fig_vol, use_container_width=True)
                                
                                # 52-week range visualization
                                if all(k in financials for k in ['Current Price', '52 Week Low', '52 Week High']):
                                    st.subheader("52-Week Price Range")
                                    
                                    current = financials.get('Current Price', 0)
                                    low = financials.get('52 Week Low', 0)
                                    high = financials.get('52 Week High', 0)
                                    
                                    # Create a gauge chart
                                    fig = go.Figure(go.Indicator(
                                        mode = "gauge+number",
                                        value = current,
                                        domain = {'x': [0, 1], 'y': [0, 1]},
                                        title = {'text': "Current Price in 52-Week Range"},
                                        gauge = {
                                            'axis': {'range': [low, high]},
                                            'bar': {'color': "darkblue"},
                                            'steps': [
                                                {'range': [low, (high-low)/3 + low], 'color': "red"},
                                                {'range': [(high-low)/3 + low, 2*(high-low)/3 + low], 'color': "yellow"},
                                                {'range': [2*(high-low)/3 + low, high], 'color': "green"}
                                            ],
                                            'threshold': {
                                                'line': {'color': "black", 'width': 4},
                                                'thickness': 0.75,
                                                'value': current
                                            }
                                        }
                                    ))
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Valuation metrics comparison (if available)
                                st.subheader("Key Metrics Visualization")
                                
                                # Create data for radar chart
                                metrics = {
                                    'YoY Growth (%)': financials.get('YoY Growth', 0),
                                    'P/E Ratio': min(financials.get('P/E Ratio', 0), 50) if financials.get('P/E Ratio') else 0,  # Cap at 50 for visualization
                                    'Dividend Yield (%)': financials.get('Dividend Yield', 0) * 100 if financials.get('Dividend Yield') else 0,
                                    'Beta': min(financials.get('Beta', 1), 3) if financials.get('Beta') else 1,  # Cap at 3 for visualization
                                    'Market Position': 0.7,  # Placeholder value
                                    'Growth Potential': 0.8   # Placeholder value
                                }
                                
                                # Create radar chart
                                categories = list(metrics.keys())
                                values = list(metrics.values())
                                
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatterpolar(
                                    r=values,
                                    theta=categories,
                                    fill='toself',
                                    name=ticker_to_analyze
                                ))
                                
                                fig.update_layout(
                                    polar=dict(
                                        radialaxis=dict(
                                            visible=True,
                                        )
                                    ),
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("No financial data available for visualization")
    
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
                    st.session_state.portfolio = portfolio_df
                    
                    # Generate grade
                    grade_result = grade_portfolio(portfolio_df)
                    
                    # Display grade with visual elements
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        # Display grade in a large circle
                        grade_color = {
                            "A": "#28a745",  # Green
                            "B": "#5cb85c",  # Light green
                            "C": "#ffc107",  # Yellow
                            "D": "#fd7e14",  # Orange
                            "F": "#dc3545"   # Red
                        }.get(grade_result["grade"], "#6c757d")
                        
                        st.markdown(f"""
                        <div style="width:120px;height:120px;border-radius:50%;background-color:{grade_color};
                        display:flex;align-items:center;justify-content:center;margin:auto;">
                            <span style="font-size:60px;font-weight:bold;color:white;">{grade_result["grade"]}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.subheader("Portfolio Grade Assessment")
                        st.write(grade_result["rationale"])
                        
                        # Strengths
                        if grade_result["strengths"]:
                            st.markdown("#### Strengths")
                            for strength in grade_result["strengths"]:
                                st.markdown(f"✅ {strength}")
                        
                        # Weaknesses
                        if grade_result["weaknesses"]:
                            st.markdown("#### Areas for Improvement")
                            for weakness in grade_result["weaknesses"]:
                                st.markdown(f"⚠️ {weakness}")
                    
                    # Recommendations
                    if grade_result["recommendations"]:
                        st.markdown("#### Recommendations")
                        for recommendation in grade_result["recommendations"]:
                            st.markdown(f"💡 {recommendation}")
                    
                    # Add a sector diversification chart
                    if not portfolio_df.empty:
                        sector_allocation = portfolio_df.groupby('Sector')['Total Value'].sum().reset_index()
                        sector_allocation['Percentage'] = sector_allocation['Total Value'] / sector_allocation['Total Value'].sum() * 100
                        
                        fig = px.bar(
                            sector_allocation, 
                            x='Sector', 
                            y='Percentage',
                            title='Sector Allocation Analysis',
                            labels={'Percentage': 'Allocation (%)'},
                            color='Percentage',
                            color_continuous_scale=px.colors.sequential.Viridis
                        )
                        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()