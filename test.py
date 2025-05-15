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
CRYPTO_TICKERS = ['BTC', 'ETH', 'XRP', 'LTC', 'DOGE', 'ADA', 'DOT', 'SOL', 'AVAX', 'MATIC', 'LINK', 'UNI']

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
            return obj.to_json()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

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

# Updated load_portfolio function
def load_portfolio():
    try:
        file_path = PORTFOLIO_FILE
        if os.path.exists(file_path):
            # Check file size to ensure it's not empty
            if os.path.getsize(file_path) > 0:
                # Try reading with pandas
                try:
                    portfolio_df = pd.read_csv(file_path)
                    if len(portfolio_df) > 0:
                        print(f"Successfully loaded portfolio with {len(portfolio_df)} rows")
                        return portfolio_df
                except Exception as e:
                    print(f"Error reading with pandas: {e}")
                    
                # Fallback: try reading with csv module
                try:
                    rows = []
                    with open(file_path, 'r', newline='') as f:
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

# Function to safely save portfolio to CSV
def save_portfolio(portfolio_df):
    try:
        # Print the absolute path for debugging
        abs_path = os.path.abspath(PORTFOLIO_FILE)
        print(f"Saving portfolio to: {abs_path}")
        
        # Try direct CSV saving with headers
        try:
            # Open the file in write mode
            with open(PORTFOLIO_FILE, 'w', newline='') as f:
                # Create a CSV writer
                writer = csv.DictWriter(f, fieldnames=DEFAULT_PORTFOLIO_COLUMNS)
                # Write the header
                writer.writeheader()
                # Write each row
                for _, row in portfolio_df.iterrows():
                    writer.writerow(row.to_dict())
            
            print(f"Portfolio saved to: {abs_path}")
            file_size = os.path.getsize(PORTFOLIO_FILE)
            print(f"File size: {file_size} bytes")
            return True
        except Exception as e:
            print(f"Error saving with CSV writer: {e}")
            
            # Fallback to pandas
            try:
                portfolio_df.to_csv(PORTFOLIO_FILE, index=False)
                print(f"Portfolio saved with pandas to: {abs_path}")
                return True
            except Exception as pandas_error:
                print(f"Error saving with pandas: {pandas_error}")
                return False
    except Exception as e:
        print(f"Critical error saving portfolio: {str(e)}")
        logger.error(f"Error saving portfolio: {str(e)}")
        st.error(f"Could not save portfolio: {str(e)}")
        return False

# Updated add_to_portfolio function with more robust error handling
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
        total_value = portfolio_df['Total Value'].sum()
        
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
# Function to get cryptocurrency data
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
            yoy_growth = ((current_price - year_ago_price) / year_ago_price) * 100
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
            "YoY Growth": yoy_growth,
            "Beta": info.get('beta', 1.5) if info.get('beta') else 1.5  # Default higher beta for crypto
        }
        
        return {
            "financials": financials,
            "historical_data": historical_data
        }
    except Exception as e:
        logger.error(f"Error fetching crypto data for {ticker}: {str(e)}\n{traceback.format_exc()}")
        return None
        
# Function to get stock data
def get_stock_data(ticker):
    # Check if ticker might be a cryptocurrency
    is_crypto = ticker.upper() in CRYPTO_TICKERS
    
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
            yoy_growth = ((current_price - year_ago_price) / year_ago_price) * 100
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
            "YoY Growth": yoy_growth,
            "Beta": info.get('beta', 1)
        }
        
        return {
            "financials": financials,
            "historical_data": historical_data
        }
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}\n{traceback.format_exc()}")
        # Try as crypto as fallback
        try:
            return get_crypto_data(ticker)
        except:
            return None

# Function to get news sentiment
def get_news_sentiment(ticker):
    try:
        # Simulated news sentiment analysis (fallback if API fails)
        sentiment_scores = np.random.normal(0, 0.3, 1)[0]  # Random score between -1 and 1
        sentiment_score = min(max(sentiment_scores, -1), 1)  # Clamp between -1 and 1
        
        # Generate sentiment summary based on score
        if sentiment_score > 0.3:
            sentiment_summary = "Positive news sentiment with strong market confidence"
            sentiment_label = "Positive"
        elif sentiment_score > 0:
            sentiment_summary = "Slightly positive news sentiment with cautious optimism"
            sentiment_label = "Slightly Positive"
        elif sentiment_score > -0.3:
            sentiment_summary = "Neutral to slightly negative news sentiment"
            sentiment_label = "Neutral"
        else:
            sentiment_summary = "Negative news sentiment with market concerns"
            sentiment_label = "Negative"
        
        # Generate random topics based on ticker
        topics = np.random.choice(
            ["Earnings", "Growth", "Innovation", "Market Share", "Regulation", 
             "Competition", "Leadership", "Products", "Services", "Strategy"],
            size=3, replace=False
        )
        
        # Try to get real sentiment if API key is available
        if os.getenv('ALPHA_VANTAGE_API_KEY'):
            try:
                news_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={os.getenv('ALPHA_VANTAGE_API_KEY')}"
                response = requests.get(news_url, timeout=5)  # Add timeout
                news_data = response.json()
                
                if "feed" in news_data and len(news_data["feed"]) > 0:
                    # Process news items
                    news_items = news_data["feed"][:10]  # Limit to 10 most recent
                    
                    # Calculate average sentiment
                    real_sentiment_scores = []
                    for item in news_items:
                        if "ticker_sentiment" in item:
                            for ticker_sentiment in item["ticker_sentiment"]:
                                if ticker_sentiment["ticker"] == ticker:
                                    real_sentiment_scores.append(float(ticker_sentiment["ticker_sentiment_score"]))
                    
                    if real_sentiment_scores:
                        # Calculate average sentiment
                        avg_sentiment = sum(real_sentiment_scores) / len(real_sentiment_scores)
                        
                        # Determine sentiment label
                        if avg_sentiment > 0.25:
                            sentiment_label = "Very Positive"
                        elif avg_sentiment > 0.1:
                            sentiment_label = "Positive"
                        elif avg_sentiment > -0.1:
                            sentiment_label = "Neutral"
                        elif avg_sentiment > -0.25:
                            sentiment_label = "Negative"
                        else:
                            sentiment_label = "Very Negative"
                        
                        sentiment_score = avg_sentiment
            except Exception as e:
                logger.warning(f"Error getting news sentiment from API: {str(e)}")
                # If API call fails, use the simulated data
                pass
        
        return {
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "sentiment_summary": sentiment_summary,
            "key_topics": topics.tolist()
        }
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        # Fallback sentiment
        return {
            "sentiment_score": 0,
            "sentiment_label": "Neutral",
            "sentiment_summary": "Unable to analyze sentiment due to an error.",
            "key_topics": ["Market", "Trading", "Stocks"]
        }

# Function to extract tickers from query
def extract_tickers(query):
    if 'llm' not in st.session_state or not st.session_state.llm:
        # Fallback if LLM is not available
        # Simple regex to find potential ticker symbols
        import re
        potential_tickers = re.findall(r'\b[A-Z]{1,5}\b', query.upper())
        return [t for t in potential_tickers if t not in ['A', 'I', 'AM', 'BE', 'DO', 'FOR', 'IN', 'IS', 'IT', 'OR', 'THE', 'TO']]
    
    try:
        # Use LLM to extract potential tickers
        extract_prompt = ChatPromptTemplate.from_template(
            """Extract all stock and cryptocurrency ticker symbols from the following query:
            
            {query}
            
            Return only the ticker symbols as a comma-separated list. If no specific tickers are mentioned but companies or cryptocurrencies are referenced, provide their ticker symbols.
            
            For cryptocurrencies, use the standard ticker symbol (like BTC for Bitcoin, ETH for Ethereum).
            """
        )
        
        extract_chain = extract_prompt | st.session_state.llm
        response = extract_chain.invoke({"query": query})
        
        # Parse the response
        tickers = [ticker.strip().upper() for ticker in response.content.split(',')]
        
        # Filter out non-tickers (simple validation)
        valid_tickers = []
        for ticker in tickers:
            # Remove any non-alphanumeric characters
            ticker = ''.join(c for c in ticker if c.isalnum())
            
            # Basic validation (tickers are typically 1-5 characters)
            if 1 <= len(ticker) <= 5:
                valid_tickers.append(ticker)
        
        return valid_tickers
    except Exception as e:
        logger.error(f"Error extracting tickers: {str(e)}")
        # Fallback to simple extraction
        import re
        potential_tickers = re.findall(r'\b[A-Z]{1,5}\b', query.upper())
        return [t for t in potential_tickers if t not in ['A', 'I', 'AM', 'BE', 'DO', 'FOR', 'IN', 'IS', 'IT', 'OR', 'THE', 'TO']]

# Function to generate investment thesis
def generate_thesis(query, financial_data):
    if 'llm' not in st.session_state or not st.session_state.llm:
        return "Unable to generate thesis: LLM not available. Please check your OpenAI API key."
    
    try:
        # Create a summarized version of the financial data
        summarized_data = {}
        
        # Process tickers data
        if "tickers" in financial_data:
            summarized_data["tickers"] = financial_data["tickers"]
        
        # For each ticker, include only the most important financial metrics
        for ticker in financial_data.get("data", {}):
            if ticker not in summarized_data:
                summarized_data[ticker] = {}
            
            # Include basic financials
            if "financials" in financial_data["data"][ticker]:
                financials = financial_data["data"][ticker]["financials"]
                summarized_data[ticker]["financials"] = {
                    "Name": financials.get("Name", ""),
                    "Sector": financials.get("Sector", ""),
                    "Industry": financials.get("Industry", ""),
                    "Current Price": financials.get("Current Price", 0),
                    "Market Cap": financials.get("Market Cap", 0),
                    "P/E Ratio": financials.get("P/E Ratio", 0),
                    "EPS": financials.get("EPS", 0),
                    "Dividend Yield": financials.get("Dividend Yield", 0),
                    "52 Week High": financials.get("52 Week High", 0),
                    "52 Week Low": financials.get("52 Week Low", 0),
                    "YoY Growth": financials.get("YoY Growth", 0),
                    "Beta": financials.get("Beta", 0)
                }
            
            # Include sentiment summary if available
            if "sentiment" in financial_data["data"][ticker]:
                sentiment = financial_data["data"][ticker]["sentiment"]
                summarized_data[ticker]["sentiment"] = {
                    "sentiment_score": sentiment.get("sentiment_score", 0),
                    "sentiment_summary": sentiment.get("sentiment_summary", "")
                }
        
        thesis_prompt = ChatPromptTemplate.from_template(
            """You are a professional investment analyst. Generate a comprehensive investment thesis based on the following query and financial data.
            
            Query: {query}
            
            Financial Data: {financial_data}
            
            Your thesis should include:
            1. Executive Summary
            2. Company Overview
            3. Industry Analysis
            4. Financial Analysis
            5. Valuation
            6. Risk Factors
            7. Investment Recommendation (Buy/Hold/Sell)
            
            Format your response in Markdown with appropriate headings, bullet points, and emphasis.
            """
        )
        
        thesis_chain = thesis_prompt | st.session_state.llm
        
        # Use the custom encoder with summarized data
        try:
            financial_data_json = json.dumps(summarized_data, cls=CustomJSONEncoder, indent=2)
        except Exception as e:
            logger.error(f"Error serializing financial data: {str(e)}")
            # Provide a simplified version as fallback
            financial_data_json = json.dumps({"error": "Data too complex to serialize", "tickers": financial_data.get("tickers", [])})
        
        response = thesis_chain.invoke({
            "query": query,
            "financial_data": financial_data_json
        })
        
        return response.content
    except Exception as e:
        logger.error(f"Error generating thesis: {str(e)}\n{traceback.format_exc()}")
        return f"""
        # Investment Thesis Generation Error
        
        We encountered an error while generating your investment thesis. This could be due to:
        
        - API rate limits or connectivity issues
        - Complex or ambiguous query
        - Limited financial data available
        
        Please try again with a more specific query or check your API key configuration.
        
        Error details: {str(e)}
        """

# Configure page with sidebar expanded
st.set_page_config(
    page_title="InvestSageAI - Investment Thesis Generator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    h1, h2, h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stButton button {
        width: 100%;
    }
    .metrics-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        height: 100%;
    }
    .sentiment-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        height: 100%;
    }
    .portfolio-grade {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .grade-A {background-color: #d4edda; color: #155724;}
    .grade-B {background-color: #d1ecf1; color: #0c5460;}
    .grade-C {background-color: #fff3cd; color: #856404;}
    .grade-D {background-color: #f8d7da; color: #721c24;}
    .grade-F {background-color: #f8d7da; color: #721c24;}
    
    [data-testid="stSidebar"] {
        min-width: 350px;
        max-width: 450px;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    .positive-sentiment {
        color: #28a745;
        font-weight: bold;
    }
    .negative-sentiment {
        color: #dc3545;
        font-weight: bold;
    }
    .neutral-sentiment {
        color: #fd7e14;
        font-weight: bold;
    }
    .topic-tag {
        display: inline-block;
        background-color: #e9ecef;
        padding: 3px 8px;
        border-radius: 12px;
                margin-right: 5px;
        font-size: 0.8rem;
    }
    .error-message {
        color: #dc3545;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .success-message {
        color: #155724;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .ticker-analysis-section {
        border: 1px solid #e9ecef;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Update load_portfolio to use the direct CSV approach
def load_portfolio():
    try:
        file_path = PORTFOLIO_FILE
        if os.path.exists(file_path):
            # Check file size to ensure it's not empty
            if os.path.getsize(file_path) > 0:
                # Try reading with pandas
                try:
                    portfolio_df = pd.read_csv(file_path)
                    if len(portfolio_df) > 0:
                        print(f"Successfully loaded portfolio with {len(portfolio_df)} rows")
                        return portfolio_df
                except Exception as e:
                    print(f"Error reading with pandas: {e}")
                    
                # Fallback: try reading with csv module
                try:
                    rows = []
                    with open(file_path, 'r', newline='') as f:
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

# Function to safely save portfolio to CSV
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

# Direct add to portfolio function that works
def direct_add_to_portfolio(ticker, purchase_price, shares):
    try:
        file_path = "portfolio_direct.csv"
        
        # Create row data
        row_data = {
            'Ticker': ticker,
            'Name': ticker,  # Simplified for direct test
            'Current Price': purchase_price,
            'Purchase Price': purchase_price,
            'Shares': shares,
            'Total Value': shares * purchase_price,
            'Profit/Loss': 0,  # No profit/loss initially
            'Profit/Loss (%)': 0,
            'YoY Growth (%)': 0,
            'P/E Ratio': 0,
            'Dividend Yield (%)': 0,
            'Beta': 1,
            'Sector': "Test",
            'Added Date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Check if file exists
        file_exists = os.path.exists(file_path)
        
        # Open the file in append mode if it exists, or write mode if it doesn't
        mode = 'a' if file_exists else 'w'
        with open(file_path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=DEFAULT_PORTFOLIO_COLUMNS)
            
            # Write headers if this is a new file
            if not file_exists:
                writer.writeheader()
            
            # Write the data row
            writer.writerow(row_data)
        
        # Return success and message
        return True, f"Added {ticker} to {file_path}"
    except Exception as e:
        return False, f"Error: {str(e)}"

# Modified add_to_portfolio function with improved debugging
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
        
        # TRY DIRECT APPROACH - Use the working direct file approach
        try:
            # First attempt with main portfolio file
            file_exists = os.path.exists(file_path)
            
            # Add some additional debugging
            print(f"File exists: {file_exists}")
            print(f"Trying to write to: {file_path} in mode {'a' if file_exists else 'w'}")
            
            # Use the csv DictWriter approach that works in the direct method
            mode = 'a' if file_exists else 'w'
            with open(file_path, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=DEFAULT_PORTFOLIO_COLUMNS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_data)
                
            print(f"Successfully wrote to {file_path}")
            
            # Now read back the file to confirm it was written
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                    print(f"File contents length: {len(content)} bytes")
                    
                # Try to read with pandas
                updated_portfolio = pd.read_csv(file_path)
                print(f"Portfolio now has {len(updated_portfolio)} rows")
                
                # Update session state
                st.session_state.portfolio = updated_portfolio
                
                return True
            
        except Exception as e:
            print(f"Error with main portfolio file: {e}")
            traceback.print_exc()
            
            # Fall back to the direct approach that works
            fallback_path = "portfolio_direct.csv"
            print(f"Trying fallback path: {fallback_path}")
            
            try:
                fallback_exists = os.path.exists(fallback_path)
                mode = 'a' if fallback_exists else 'w'
                with open(fallback_path, mode, newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=DEFAULT_PORTFOLIO_COLUMNS)
                    if not fallback_exists:
                        writer.writeheader()
                    writer.writerow(row_data)
                
                # Update the main portfolio file with the fallback contents
                fallback_df = pd.read_csv(fallback_path)
                st.session_state.portfolio = fallback_df
                fallback_df.to_csv(file_path, index=False)
                
                print(f"Used fallback approach, portfolio now has {len(fallback_df)} rows")
                return True
            except Exception as fallback_error:
                print(f"Even fallback approach failed: {fallback_error}")
                return False
            
    except Exception as e:
        print(f"Error in add_to_portfolio: {e}")
        traceback.print_exc()
        return False

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
        total_value = portfolio_df['Total Value'].sum()
        
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

# Function to get cryptocurrency data
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
            yoy_growth = ((current_price - year_ago_price) / year_ago_price) * 100
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
            "YoY Growth": yoy_growth,
            "Beta": info.get('beta', 1.5) if info.get('beta') else 1.5  # Default higher beta for crypto
        }
        
        return {
            "financials": financials,
            "historical_data": historical_data
        }
    except Exception as e:
        logger.error(f"Error fetching crypto data for {ticker}: {str(e)}\n{traceback.format_exc()}")
        return None
        
# Function to get stock data
def get_stock_data(ticker):
    # Check if ticker might be a cryptocurrency
    is_crypto = ticker.upper() in CRYPTO_TICKERS
    
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
            yoy_growth = ((current_price - year_ago_price) / year_ago_price) * 100
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
            "YoY Growth": yoy_growth,
            "Beta": info.get('beta', 1)
        }
        
        return {
            "financials": financials,
            "historical_data": historical_data
        }
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}\n{traceback.format_exc()}")
        # Try as crypto as fallback
        try:
            return get_crypto_data(ticker)
        except:
            return None

# Function to get news sentiment
def get_news_sentiment(ticker):
    try:
        # Simulated news sentiment analysis (fallback if API fails)
        sentiment_scores = np.random.normal(0, 0.3, 1)[0]  # Random score between -1 and 1
        sentiment_score = min(max(sentiment_scores, -1), 1)  # Clamp between -1 and 1
        
        # Generate sentiment summary based on score
        if sentiment_score > 0.3:
            sentiment_summary = "Positive news sentiment with strong market confidence"
            sentiment_label = "Positive"
        elif sentiment_score > 0:
            sentiment_summary = "Slightly positive news sentiment with cautious optimism"
            sentiment_label = "Slightly Positive"
        elif sentiment_score > -0.3:
            sentiment_summary = "Neutral to slightly negative news sentiment"
            sentiment_label = "Neutral"
        else:
            sentiment_summary = "Negative news sentiment with market concerns"
            sentiment_label = "Negative"
        
        # Generate random topics based on ticker
        topics = np.random.choice(
            ["Earnings", "Growth", "Innovation", "Market Share", "Regulation", 
             "Competition", "Leadership", "Products", "Services", "Strategy"],
            size=3, replace=False
        )
        
        # Try to get real sentiment if API key is available
        if os.getenv('ALPHA_VANTAGE_API_KEY'):
            try:
                news_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={os.getenv('ALPHA_VANTAGE_API_KEY')}"
                response = requests.get(news_url, timeout=5)  # Add timeout
                news_data = response.json()
                
                if "feed" in news_data and len(news_data["feed"]) > 0:
                    # Process news items
                    news_items = news_data["feed"][:10]  # Limit to 10 most recent
                    
                    # Calculate average sentiment
                    real_sentiment_scores = []
                    for item in news_items:
                        if "ticker_sentiment" in item:
                            for ticker_sentiment in item["ticker_sentiment"]:
                                if ticker_sentiment["ticker"] == ticker:
                                    real_sentiment_scores.append(float(ticker_sentiment["ticker_sentiment_score"]))
                    
                    if real_sentiment_scores:
                        # Calculate average sentiment
                        avg_sentiment = sum(real_sentiment_scores) / len(real_sentiment_scores)
                        
                        # Determine sentiment label
                        if avg_sentiment > 0.25:
                            sentiment_label = "Very Positive"
                        elif avg_sentiment > 0.1:
                            sentiment_label = "Positive"
                        elif avg_sentiment > -0.1:
                            sentiment_label = "Neutral"
                        elif avg_sentiment > -0.25:
                            sentiment_label = "Negative"
                        else:
                            sentiment_label = "Very Negative"
                        
                        sentiment_score = avg_sentiment
            except Exception as e:
                logger.warning(f"Error getting news sentiment from API: {str(e)}")
                # If API call fails, use the simulated data
                pass
        
        return {
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "sentiment_summary": sentiment_summary,
            "key_topics": topics.tolist()
        }
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        # Fallback sentiment
        return {
            "sentiment_score": 0,
            "sentiment_label": "Neutral",
            "sentiment_summary": "Unable to analyze sentiment due to an error.",
            "key_topics": ["Market", "Trading", "Stocks"]
        }

# Function to extract tickers from query
def extract_tickers(query):
    if 'llm' not in st.session_state or not st.session_state.llm:
        # Fallback if LLM is not available
        # Simple regex to find potential ticker symbols
        import re
        potential_tickers = re.findall(r'\b[A-Z]{1,5}\b', query.upper())
        return [t for t in potential_tickers if t not in ['A', 'I', 'AM', 'BE', 'DO', 'FOR', 'IN', 'IS', 'IT', 'OR', 'THE', 'TO']]
    
    try:
        # Use LLM to extract potential tickers
        extract_prompt = ChatPromptTemplate.from_template(
            """Extract all stock and cryptocurrency ticker symbols from the following query:
            
            {query}
            
            Return only the ticker symbols as a comma-separated list. If no specific tickers are mentioned but companies or cryptocurrencies are referenced, provide their ticker symbols.
            
            For cryptocurrencies, use the standard ticker symbol (like BTC for Bitcoin, ETH for Ethereum).
            """
        )
        
        extract_chain = extract_prompt | st.session_state.llm
        response = extract_chain.invoke({"query": query})
        
        # Parse the response
        tickers = [ticker.strip().upper() for ticker in response.content.split(',')]
        
        # Filter out non-tickers (simple validation)
        valid_tickers = []
        for ticker in tickers:
            # Remove any non-alphanumeric characters
            ticker = ''.join(c for c in ticker if c.isalnum())
            
            # Basic validation (tickers are typically 1-5 characters)
            if 1 <= len(ticker) <= 5:
                valid_tickers.append(ticker)
        
        return valid_tickers
    except Exception as e:
        logger.error(f"Error extracting tickers: {str(e)}")
        # Fallback to simple extraction
        import re
        potential_tickers = re.findall(r'\b[A-Z]{1,5}\b', query.upper())
        return [t for t in potential_tickers if t not in ['A', 'I', 'AM', 'BE', 'DO', 'FOR', 'IN', 'IS', 'IT', 'OR', 'THE', 'TO']]

# Function to generate investment thesis
def generate_thesis(query, financial_data):
    if 'llm' not in st.session_state or not st.session_state.llm:
        return "Unable to generate thesis: LLM not available. Please check your OpenAI API key."
    
    try:
        # Create a summarized version of the financial data
        summarized_data = {}
        
        # Process tickers data
        if "tickers" in financial_data:
            summarized_data["tickers"] = financial_data["tickers"]
        
        # For each ticker, include only the most important financial metrics
        for ticker in financial_data.get("data", {}):
            if ticker not in summarized_data:
                summarized_data[ticker] = {}
            
            # Include basic financials
            if "financials" in financial_data["data"][ticker]:
                financials = financial_data["data"][ticker]["financials"]
                summarized_data[ticker]["financials"] = {
                    "Name": financials.get("Name", ""),
                    "Sector": financials.get("Sector", ""),
                    "Industry": financials.get("Industry", ""),
                    "Current Price": financials.get("Current Price", 0),
                    "Market Cap": financials.get("Market Cap", 0),
                    "P/E Ratio": financials.get("P/E Ratio", 0),
                    "EPS": financials.get("EPS", 0),
                    "Dividend Yield": financials.get("Dividend Yield", 0),
                    "52 Week High": financials.get("52 Week High", 0),
                    "52 Week Low": financials.get("52 Week Low", 0),
                    "YoY Growth": financials.get("YoY Growth", 0),
                    "Beta": financials.get("Beta", 0)
                }
            
            # Include sentiment summary if available
            if "sentiment" in financial_data["data"][ticker]:
                sentiment = financial_data["data"][ticker]["sentiment"]
                summarized_data[ticker]["sentiment"] = {
                    "sentiment_score": sentiment.get("sentiment_score", 0),
                    "sentiment_summary": sentiment.get("sentiment_summary", "")
                }
        
        thesis_prompt = ChatPromptTemplate.from_template(
            """You are a professional investment analyst. Generate a comprehensive investment thesis based on the following query and financial data.
            
            Query: {query}
            
            Financial Data: {financial_data}
            
            Your thesis should include:
            1. Executive Summary
            2. Company Overview
            3. Industry Analysis
            4. Financial Analysis
            5. Valuation
            6. Risk Factors
            7. Investment Recommendation (Buy/Hold/Sell)
            
            Format your response in Markdown with appropriate headings, bullet points, and emphasis.
            """
        )
        
        thesis_chain = thesis_prompt | st.session_state.llm
        
        # Use the custom encoder with summarized data
        try:
            financial_data_json = json.dumps(summarized_data, cls=CustomJSONEncoder, indent=2)
        except Exception as e:
            logger.error(f"Error serializing financial data: {str(e)}")
            # Provide a simplified version as fallback
            financial_data_json = json.dumps({"error": "Data too complex to serialize", "tickers": financial_data.get("tickers", [])})
        
        response = thesis_chain.invoke({
            "query": query,
            "financial_data": financial_data_json
        })
        
        return response.content
    except Exception as e:
        logger.error(f"Error generating thesis: {str(e)}\n{traceback.format_exc()}")
        return f"""
        # Investment Thesis Generation Error
        
        We encountered an error while generating your investment thesis. This could be due to:
        
        - API rate limits or connectivity issues
        - Complex or ambiguous query
        - Limited financial data available
        
        Please try again with a more specific query or check your API key configuration.
        
        Error details: {str(e)}
        """


# Initialize session state after all functions are defined
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_portfolio()
    print(f"Loaded portfolio with {len(st.session_state.portfolio)} rows")

if 'ticker_data' not in st.session_state:
    st.session_state.ticker_data = {}

if 'add_success' not in st.session_state:
    st.session_state.add_success = None

if 'remove_success' not in st.session_state:
    st.session_state.remove_success = None

# Initialize OpenAI
try:
    llm = ChatOpenAI(temperature=0.7)
    st.session_state.llm = llm
except Exception as e:
    logger.error(f"Error initializing OpenAI: {str(e)}")
    st.session_state.llm = None
    
# Sidebar
with st.sidebar:
    st.title("InvestSageAI")
    
    # Portfolio display
    st.subheader("Your Portfolio")
    
    portfolio_df = st.session_state.portfolio
    
    if not portfolio_df.empty:
                # Grade the portfolio
        grade_data = grade_portfolio(portfolio_df)
        grade = grade_data.get("grade", "C")
        
        # Display the grade
        st.markdown(f"""
        <div class="portfolio-grade grade-{grade}">
            {grade}
        </div>
        <p>{grade_data.get("rationale", "")}</p>
        """, unsafe_allow_html=True)
        
        # Display strengths and weaknesses
        if grade_data.get("strengths"):
            st.markdown("**Strengths:**")
            for strength in grade_data.get("strengths", []):
                st.markdown(f"- {strength}")
        
        if grade_data.get("weaknesses"):
            st.markdown("**Weaknesses:**")
            for weakness in grade_data.get("weaknesses", []):
                st.markdown(f"- {weakness}")
        
        if grade_data.get("recommendations"):
            st.markdown("**Recommendations:**")
            for rec in grade_data.get("recommendations", []):
                st.markdown(f"- {rec}")
        
        # Display portfolio table
        st.dataframe(
            portfolio_df[['Ticker', 'Name', 'Current Price', 'YoY Growth (%)', 'Sector']], 
            hide_index=True,
            use_container_width=True
        )
        
        # Remove asset from portfolio
        with st.expander("Remove Asset"):
            remove_ticker = st.selectbox(
                "Select asset to remove",
                options=portfolio_df['Ticker'].tolist(),
                key="remove_ticker"
            )
            
            if st.button("Remove from Portfolio", key="remove_button"):
                if remove_from_portfolio(remove_ticker):
                    st.success(f"Removed {remove_ticker} from your portfolio!")
                    st.rerun()
    else:
        st.info("Your portfolio is empty. Add assets by analyzing them first.")
        
        # Debug section
    with st.expander("Portfolio Debug"):
        if os.path.exists(PORTFOLIO_FILE):
            st.write(f"Portfolio file exists at: {os.path.abspath(PORTFOLIO_FILE)}")
            file_size = os.path.getsize(PORTFOLIO_FILE)
            st.write(f"File size: {file_size} bytes")
            
            # Try to read the file directly
            try:
                direct_df = pd.read_csv(PORTFOLIO_FILE)
                st.write(f"Direct read found {len(direct_df)} rows")
                st.dataframe(direct_df)
            except Exception as e:
                st.error(f"Error reading file directly: {e}")
        else:
            st.write(f"Portfolio file does not exist at: {os.path.abspath(PORTFOLIO_FILE)}")
            
        st.write("Session state portfolio:")
        st.write(f"Rows: {len(st.session_state.portfolio)}")
        st.dataframe(st.session_state.portfolio)
        
        # Add this at the bottom of your sidebar section
with st.sidebar:
    st.markdown("---")
    st.subheader("Direct Portfolio Test")
    
    # Simple inputs
    test_ticker = st.text_input("Test Ticker", value="AAPL")
    test_price = st.number_input("Test Price", value=212.33, step=0.01)
    test_shares = st.number_input("Test Shares", value=1.0, step=0.1)
    
    # Test button
    if st.button("Add Directly to CSV"):
        success, message = direct_add_to_portfolio(test_ticker, test_price, test_shares)
        if success:
            st.success(f"✅ Direct add successful! Check {message}")
            # Force refresh page
            st.rerun()
        else:
            st.error(f"❌ Direct add failed: {message}")
    
    # Show content of the direct CSV
    if os.path.exists("portfolio_direct.csv"):
        st.markdown("### Direct Portfolio File")
        try:
            df = pd.read_csv("portfolio_direct.csv")
            st.write(f"Found {len(df)} rows")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error reading direct portfolio: {e}")
    else:
        st.write("Direct portfolio file not found")
        
# Function for direct portfolio addition (simplified approach that's known to work)
def direct_add_to_portfolio(ticker, price, shares):
    try:
        # Get portfolio file path with absolute path for certainty
        file_path = os.path.abspath(PORTFOLIO_FILE)
        print(f"ABSOLUTE PATH CHECK: Attempting to add {ticker} to portfolio at: {file_path}")
        
        # Check directory existence and permissions
        dir_path = os.path.dirname(file_path)
        print(f"Directory path: {dir_path}")
        print(f"Directory exists: {os.path.exists(dir_path)}")
        print(f"Directory writable: {os.access(dir_path, os.W_OK)}")
        
        # Add values
        current_price = price
        purchase_price = price
        total_value = shares * current_price
        profit_loss = 0  # New position has no P/L
        profit_loss_pct = 0
        
        # Create row data
        row_data = {
            'Ticker': ticker,
            'Name': ticker,  # Simplified for direct test
            'Current Price': current_price,
            'Purchase Price': purchase_price,
            'Shares': shares,
            'Total Value': total_value,
            'Profit/Loss': profit_loss,
            'Profit/Loss (%)': profit_loss_pct,
            'YoY Growth (%)': 0,  # Simplified
            'P/E Ratio': 0,  # Simplified
            'Dividend Yield (%)': 0,  # Simplified
            'Beta': 1,  # Simplified
            'Sector': "Test",  # Simplified
            'Added Date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Check if file exists
        file_exists = os.path.exists(file_path)
        
        # Open the file in append mode if it exists, or write mode if it doesn't
        mode = 'a' if file_exists else 'w'
        with open(file_path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=DEFAULT_PORTFOLIO_COLUMNS)
            
            # Write headers if this is a new file
            if not file_exists:
                writer.writeheader()
            
            # Write the data row
            writer.writerow(row_data)
        
        # Check if it worked
        if os.path.exists(file_path):
            return True, file_path
        else:
            return False, "File not created"
            
    except Exception as e:
        return False, str(e)

# Modify the add_to_portfolio function to have more detailed debugging
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
        
        # TRY DIRECT APPROACH - Use the working direct file as a fallback
        try:
            # First attempt with main portfolio file
            file_exists = os.path.exists(file_path)
            
            # Add some additional debugging
            print(f"File exists: {file_exists}, Path: {file_path}")
            print(f"Trying to write to: {file_path} in mode {'a' if file_exists else 'w'}")
            
            # Use the csv DictWriter approach that works in the direct method
            mode = 'a' if file_exists else 'w'
            with open(file_path, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=DEFAULT_PORTFOLIO_COLUMNS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_data)
                
            print(f"Successfully wrote to {file_path}")
            
            # Now read back the file to confirm it was written
            if os.path.exists(file_path):
                print(f"File exists after write: {os.path.getsize(file_path)} bytes")
                updated_portfolio = pd.read_csv(file_path)
                print(f"Portfolio now has {len(updated_portfolio)} rows")
                
                # Update session state
                st.session_state.portfolio = updated_portfolio
                
                return True
            
        except Exception as e:
            print(f"Error with main portfolio file: {e}")
            traceback.print_exc()
            
            # Fall back to the direct approach that works
            fallback_path = "portfolio_direct.csv"
            print(f"Trying fallback path: {fallback_path}")
            
            try:
                fallback_exists = os.path.exists(fallback_path)
                mode = 'a' if fallback_exists else 'w'
                with open(fallback_path, mode, newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=DEFAULT_PORTFOLIO_COLUMNS)
                    if not fallback_exists:
                        writer.writeheader()
                    writer.writerow(row_data)
                
                # Update the main portfolio file with the fallback contents
                if os.path.exists(fallback_path):
                    fallback_df = pd.read_csv(fallback_path)
                    st.session_state.portfolio = fallback_df
                    fallback_df.to_csv(file_path, index=False)
                    
                    print(f"Used fallback approach, portfolio now has {len(fallback_df)} rows")
                    return True
                else:
                    return False
            except Exception as fallback_error:
                print(f"Even fallback approach failed: {fallback_error}")
                return False
            
    except Exception as e:
        print(f"Error in add_to_portfolio: {e}")
        traceback.print_exc()
        return False

# Main content
st.title("Investment Thesis Generator")

# Create tabs
tab1, tab2 = st.tabs(["Analyze Assets", "Portfolio Analysis"])

# Check for successful portfolio addition
if 'add_success' in st.session_state and st.session_state.add_success:
    ticker = st.session_state.add_success.get("ticker")
    st.success(f"Added {ticker} to your portfolio!")
    # Clear the success message so it doesn't show again on next refresh
    st.session_state.add_success = None

# Tab 1: Analyze Assets
with tab1:
    st.subheader("Analyze Stocks & Cryptocurrencies")
    
    # Query input
    query = st.text_area(
        "Enter your investment query or ticker symbols",
        placeholder="Example: 'Should I invest in AAPL, MSFT, and GOOGL?' or 'Analyze tech stocks with strong growth potential'",
        height=100
    )
    
    analyze_button = st.button("Analyze", key="analyze_button", use_container_width=True)
    
    if query and analyze_button:
        # Extract tickers from query
        tickers = extract_tickers(query)
        
        if not tickers:
            st.warning("No valid ticker symbols found in your query. Please specify some stocks or cryptocurrencies.")
        else:
            st.info(f"Analyzing the following tickers: {', '.join(tickers)}")
            
            # Create progress bar
            progress_bar = st.progress(0)
            
            # Collect financial data for each ticker
            financial_data = {
                "tickers": tickers,
                "data": {}
            }
            
            for i, ticker in enumerate(tickers):
                try:
                    # Get stock data
                    stock_data = get_stock_data(ticker)
                    
                    if stock_data:
                        # Get news sentiment
                        sentiment_data = get_news_sentiment(ticker)
                        
                        # Store data
                        financial_data["data"][ticker] = {
                            "financials": stock_data.get("financials", {}),
                            "historical_data": stock_data.get("historical_data", pd.DataFrame()),
                            "sentiment": sentiment_data
                        }
                        
                        # Store in session state for later use
                        if 'ticker_data' not in st.session_state:
                            st.session_state.ticker_data = {}
                        st.session_state.ticker_data[ticker] = stock_data
                    else:
                        st.warning(f"Could not retrieve data for {ticker}. Skipping.")
                except Exception as e:
                    logger.error(f"Error analyzing {ticker}: {str(e)}\n{traceback.format_exc()}")
                    st.error(f"Error analyzing {ticker}: {str(e)}")
                
                # Update progress
                progress_bar.progress((i + 1) / len(tickers))
            
            # Display individual ticker analysis - WITHOUT EXPANDERS
            for ticker in tickers:
                if ticker in financial_data["data"]:
                    # Use a container with custom styling instead of an expander
                    st.markdown(f"<div class='ticker-analysis-section'><h3>Analysis: {ticker}</h3>", unsafe_allow_html=True)
                    
                    ticker_data = financial_data["data"][ticker]
                    financials = ticker_data.get("financials", {})
                    sentiment = ticker_data.get("sentiment", {})
                    
                    # Create columns for metrics and sentiment
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.markdown(f"### {financials.get('Name', ticker)}")
                        st.markdown(f"**Sector:** {financials.get('Sector', 'Unknown')}")
                        st.markdown(f"**Industry:** {financials.get('Industry', 'Unknown')}")
                        
                        # Financial metrics in cards
                        metrics_cols = st.columns(3)
                        
                        with metrics_cols[0]:
                            st.markdown("""
                            <div class="metrics-card">
                                <div class="metric-label">Current Price</div>
                                <div class="metric-value">${:,.2f}</div>
                            </div>
                            """.format(financials.get("Current Price", 0)), unsafe_allow_html=True)
                        
                        with metrics_cols[1]:
                            st.markdown("""
                            <div class="metrics-card">
                                <div class="metric-label">Market Cap</div>
                                <div class="metric-value">${:,.2f}B</div>
                            </div>
                            """.format(financials.get("Market Cap", 0) / 1e9), unsafe_allow_html=True)
                        
                        with metrics_cols[2]:
                            growth = financials.get("YoY Growth", 0)
                            growth_color = "#28a745" if growth > 0 else "#dc3545"
                            st.markdown("""
                            <div class="metrics-card">
                                <div class="metric-label">YoY Growth</div>
                                <div class="metric-value" style="color: {};">{:,.2f}%</div>
                            </div>
                            """.format(growth_color, growth), unsafe_allow_html=True)
                        
                        # Second row of metrics
                        metrics_cols2 = st.columns(3)
                        
                        with metrics_cols2[0]:
                            pe_ratio = financials.get("P/E Ratio", None)
                            pe_display = "{:,.2f}".format(pe_ratio) if pe_ratio is not None else "N/A"
                            st.markdown("""
                            <div class="metrics-card">
                                <div class="metric-label">P/E Ratio</div>
                                <div class="metric-value">{}</div>
                            </div>
                            """.format(pe_display), unsafe_allow_html=True)
                        
                        with metrics_cols2[1]:
                            dividend = financials.get("Dividend Yield", 0)
                            st.markdown("""
                            <div class="metrics-card">
                                <div class="metric-label">Dividend Yield</div>
                                <div class="metric-value">{:,.2f}%</div>
                            </div>
                            """.format(dividend), unsafe_allow_html=True)
                        
                        with metrics_cols2[2]:
                            beta = financials.get("Beta", 1)
                            beta_color = "#dc3545" if beta > 1.2 else "#28a745" if beta < 0.8 else "#fd7e14"
                            st.markdown("""
                            <div class="metrics-card">
                                <div class="metric-label">Beta</div>
                                <div class="metric-value" style="color: {};">{:,.2f}</div>
                            </div>
                            """.format(beta_color, beta), unsafe_allow_html=True)
                        
                        # Display sentiment analysis
                        sentiment_score = sentiment.get("sentiment_score", 0)
                        sentiment_label = sentiment.get("sentiment_label", "Neutral")
                        sentiment_class = "positive-sentiment" if sentiment_score > 0.1 else "negative-sentiment" if sentiment_score < -0.1 else "neutral-sentiment"
                        
                        st.markdown("<h4>News Sentiment</h4>", unsafe_allow_html=True)
                        st.markdown(f"<span class='{sentiment_class}'>{sentiment_label}</span> ({sentiment_score:.2f})", unsafe_allow_html=True)
                        st.markdown(f"{sentiment.get('sentiment_summary', 'No sentiment data available.')}")
                        
                        # Display key topics
                        if sentiment.get("key_topics"):
                            st.markdown("<div style='margin-top: 10px;'>", unsafe_allow_html=True)
                            for topic in sentiment.get("key_topics", []):
                                st.markdown(f"<span class='topic-tag'>{topic}</span>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                    with col2:
                        # Add to portfolio section
                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.markdown("### Add to Portfolio")

                        # Simple number inputs with unique keys
                        shares = st.number_input(
                            "Number of Shares", 
                            min_value=0.01, 
                            value=1.0,
                            step=0.1,
                            key=f"shares_{ticker}"
                        )

                        purchase_price = st.number_input(
                            "Purchase Price ($)",
                            min_value=0.01,
                            value=financials.get("Current Price", 0),
                            step=0.01,
                            key=f"price_{ticker}"
                        )

                        # Calculate and display position preview
                        position_value = shares * purchase_price
                        st.markdown(f"**Position Value:** ${position_value:.2f}")

                        # Add to Portfolio button with direct feedback
                        if st.button(f"Add {ticker} to Portfolio"):
                            with st.spinner(f"Adding {ticker} to portfolio..."):
                                # Get values directly from the number inputs
                                shares_key = f"shares_{ticker}"
                                price_key = f"price_{ticker}"
                                shares = st.session_state.get(shares_key, 1.0)
                                purchase_price = st.session_state.get(price_key, financials.get("Current Price", 0))
                                
                                # Call the function with explicit parameters
                                success = add_to_portfolio(
                                    st.session_state.ticker_data[ticker],
                                    ticker,
                                    shares,
                                    purchase_price
                                )
                                
                                if success:
                                    st.success(f"✅ Added {ticker} to portfolio!")
                                    time.sleep(0.5)
                                    st.experimental_rerun()
                                else:
                                    st.error(f"❌ Failed to add {ticker} to portfolio")
                    
                    # Close the div container
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Add some spacing between ticker sections
                    st.markdown("<br>", unsafe_allow_html=True)
            
            # Generate thesis
            if financial_data["data"]:
                with st.spinner("Generating investment thesis..."):
                    thesis = generate_thesis(query, financial_data)
                    st.markdown("## Investment Thesis")
                    st.markdown(thesis)
                                    
# Tab 2: Portfolio Analysis
with tab2:
    st.subheader("Portfolio Analysis")
    
    portfolio_df = st.session_state.portfolio
    
    if portfolio_df.empty:
        st.info("Your portfolio is empty. Add assets in the 'Analyze Assets' tab.")
    else:
        # Portfolio analysis code as before
        # Display portfolio summary
        st.markdown("### Portfolio Summary")
        
        # Calculate portfolio metrics
        total_value = portfolio_df['Total Value'].sum()
        total_cost = (portfolio_df['Purchase Price'] * portfolio_df['Shares']).sum()
        total_profit_loss = portfolio_df['Profit/Loss'].sum()
        profit_loss_pct = (total_profit_loss / total_cost * 100) if total_cost > 0 else 0
        
        weighted_growth = ((portfolio_df['YoY Growth (%)'] * portfolio_df['Total Value']).sum() / 
                           total_value) if total_value > 0 else 0
        weighted_dividend = ((portfolio_df['Dividend Yield (%)'] * portfolio_df['Total Value']).sum() / 
                             total_value) if total_value > 0 else 0
        weighted_beta = ((portfolio_df['Beta'] * portfolio_df['Total Value']).sum() / 
                         total_value) if total_value > 0 else 0
        
        # Display metrics in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Portfolio Value", f"${total_value:,.2f}")
            st.metric("Total Profit/Loss", f"${total_profit_loss:,.2f}", 
                     delta=f"{profit_loss_pct:.2f}%")
        
        with col2:
            st.metric("Total Assets", f"{len(portfolio_df)}")
            st.metric("Weighted Risk (Beta)", f"{weighted_beta:.2f}")
            
        # Display key metrics
        metrics_cols = st.columns(3)
        with metrics_cols[0]:
            st.metric("Weighted Growth", f"{weighted_growth:.2f}%")
        with metrics_cols[1]:
            st.metric("Weighted Dividend", f"{weighted_dividend:.2f}%")
        with metrics_cols[2]:
            st.metric("Total Cost Basis", f"${total_cost:,.2f}")
        
        # Add visualization - Sector allocation pie chart
        st.markdown("### Sector Allocation")
        
        # Calculate sector weights based on value
        sector_values = portfolio_df.groupby('Sector')['Total Value'].sum()
        sector_weights = (sector_values / sector_values.sum() * 100).reset_index()
        sector_weights.columns = ['Sector', 'Allocation (%)']
        
        # Show both chart and table
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create pie chart using plotly
            fig = px.pie(sector_weights, values='Allocation (%)', names='Sector',
                         title='Portfolio Allocation by Sector')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(sector_weights.sort_values('Allocation (%)', ascending=False),
                        hide_index=True)
            
        # Add performance visualization
        st.markdown("### Position Performance")
        
        # Calculate performance metrics for visualization
        performance_df = portfolio_df[['Ticker', 'Name', 'Shares', 'Purchase Price', 
                                      'Current Price', 'Total Value', 'Profit/Loss', 
                                      'Profit/Loss (%)']].copy()
        
        # Sort by absolute profit/loss for better visualization
        performance_df = performance_df.sort_values('Profit/Loss', ascending=False)
        
        # Create bar chart of profit/loss by position
        fig2 = px.bar(performance_df, x='Ticker', y='Profit/Loss',
                     color='Profit/Loss', color_continuous_scale=['red', 'green'],
                     title='Profit/Loss by Position ($)',
                     hover_data=['Name', 'Shares', 'Purchase Price', 'Current Price', 
                                'Profit/Loss (%)'])
        st.plotly_chart(fig2, use_container_width=True)
        
        # Display full portfolio table
        st.markdown("### Complete Portfolio")
        
        # Format the dataframe for display
        display_df = portfolio_df.copy()
        
        # Only attempt to format columns that exist
        if 'Current Price' in display_df.columns:
            display_df['Current Price'] = display_df['Current Price'].astype(float).map('${:.2f}'.format)
        if 'Purchase Price' in display_df.columns:
            display_df['Purchase Price'] = display_df['Purchase Price'].astype(float).map('${:.2f}'.format)
        if 'Shares' in display_df.columns:
            display_df['Shares'] = display_df['Shares'].astype(float).map('{:.2f}'.format)
        if 'Total Value' in display_df.columns:
            display_df['Total Value'] = display_df['Total Value'].astype(float).map('${:.2f}'.format)
        if 'Profit/Loss' in display_df.columns:
            display_df['Profit/Loss'] = display_df['Profit/Loss'].astype(float).map('${:.2f}'.format)
        if 'Profit/Loss (%)' in display_df.columns:
            display_df['Profit/Loss (%)'] = display_df['Profit/Loss (%)'].astype(float).map('{:.2f}%'.format)
        if 'YoY Growth (%)' in display_df.columns:
            display_df['YoY Growth (%)'] = display_df['YoY Growth (%)'].astype(float).map('{:.2f}%'.format)
        if 'Dividend Yield (%)' in display_df.columns:
            display_df['Dividend Yield (%)'] = display_df['Dividend Yield (%)'].astype(float).map('{:.2f}%'.format)
        
        st.dataframe(
            display_df.sort_values('Total Value', ascending=False), 
            hide_index=True,
            use_container_width=True
        )

        # Add trade recommendations
        st.markdown("### Trade Recommendations")
        
        # Basic recommendations based on portfolio analysis
        try:
            high_risk_positions = portfolio_df[portfolio_df['Beta'] > 1.5]
            negative_growth_positions = portfolio_df[portfolio_df['YoY Growth (%)'] < 0]
            high_pe_positions = portfolio_df[portfolio_df['P/E Ratio'] > 30]
            
            # Generate recommendations
            recommendations = []
            
            # Overweight sectors
            sector_threshold = 30  # Consider rebalancing if a sector is over 30%
            for _, row in sector_weights.iterrows():
                if row['Allocation (%)'] > sector_threshold:
                    recommendations.append(f"⚠️ Consider reducing exposure to {row['Sector']} sector ({row['Allocation (%)']:.1f}%)")
            
            # High risk positions
            for _, row in high_risk_positions.iterrows():
                recommendations.append(f"🔍 Monitor high-beta position in {row['Name']} (β={row['Beta']:.2f})")
                
            # Negative growth positions
            for _, row in negative_growth_positions.iterrows():
                recommendations.append(f"📉 Review {row['Name']} due to negative YoY growth ({row['YoY Growth (%)']:.1f}%)")
            
            # High P/E positions
            for _, row in high_pe_positions.iterrows():
                recommendations.append(f"💰 Evaluate if {row['Name']} is overvalued (P/E={row['P/E Ratio']:.1f})")
                
            # Display recommendations
            if recommendations:
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            else:
                st.markdown("- No specific trading recommendations at this time.")
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            st.markdown("- Unable to generate recommendations due to an error.")

        # Add timeline-based projections
        st.markdown("### Investment Timeline Projections")
        
        try:
            # Create projection periods
            timeline_col1, timeline_col2, timeline_col3 = st.columns(3)
            
            # Simple projections based on current growth rates
            with timeline_col1:
                st.markdown("#### 1-Year Projection")
                projected_value = total_value * (1 + weighted_growth/100)
                projected_gain = projected_value - total_value
                st.metric("Projected Value", f"${projected_value:,.2f}", 
                         delta=f"${projected_gain:,.2f}")
                         
            with timeline_col2:
                st.markdown("#### 5-Year Projection")
                projected_value_5yr = total_value * (1 + weighted_growth/100)**5
                projected_gain_5yr = projected_value_5yr - total_value
                st.metric("Projected Value", f"${projected_value_5yr:,.2f}", 
                         delta=f"${projected_gain_5yr:,.2f}")
                         
            with timeline_col3:
                st.markdown("#### 10-Year Projection")
                projected_value_10yr = total_value * (1 + weighted_growth/100)**10
                projected_gain_10yr = projected_value_10yr - total_value
                st.metric("Projected Value", f"${projected_value_10yr:,.2f}", 
                         delta=f"${projected_gain_10yr:,.2f}")
        except Exception as e:
            st.error(f"Error generating projections: {e}")
            st.markdown("Unable to generate projections due to an error in calculations.")

# Main execution block
if __name__ == "__main__":
    try:
        # The app is already running through the Streamlit framework
        pass
    except Exception as e:
        logger.critical(f"Critical application error: {str(e)}\n{traceback.format_exc()}")
        st.error("An unexpected error occurred. Please refresh the page or contact support.")