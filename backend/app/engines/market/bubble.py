
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def get_market_risk_data():
    """
    Fetches and calculates market risk indicators.
    Returns a dictionary with metrics and a composite bubble score.
    """
    risk_data = {
        "timestamp": datetime.now().isoformat(),
        "indicators": {},
        "score": 0,
        "rating": "Neutral"
    }
    
    scores = []
    weights = []

    # --- 1. QQQ Deviation Index (Tech Bubble Proxy) ---
    try:
        qqq = yf.Ticker("QQQ")
        # Get 10 years of history to establish a baseline
        hist = qqq.history(period="10y")
        
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            # Calculate 200-day Moving Average
            hist['MA200'] = hist['Close'].rolling(window=200).mean()
            # Calculate Deviation: (Price - MA200) / MA200
            hist['Deviation'] = (hist['Close'] - hist['MA200']) / hist['MA200']
            
            current_deviation = hist['Deviation'].iloc[-1]
            
            # Calculate Percentile Rank of current deviation vs last 10 years
            # This was missing!
            percentile = (hist['Deviation'] < current_deviation).mean() * 100
            
            # Get last 30 days of deviation for chart
            history_data = hist['Deviation'].tail(30).tolist() if not hist.empty else []
            history_data = [x * 100 for x in history_data] # Convert to percentage

            signal = "High Risk" if percentile >= 76 else "Neutral" if percentile >= 45 else "Low Risk"

            risk_data["indicators"]["qqqDeviation"] = {
                "name": "QQQ Deviation Index",
                "value": current_deviation * 100, # Percentage
                "score": percentile, # 0-100
                "description": "Deviation from 200-day Moving Average (10y percentile)",
                "methodology": "Calculates the % difference between QQQ price and its 200-day MA. The score is the percentile rank of this deviation over the last 10 years.",
                "signal": signal,
                "history": history_data
            }
            scores.append(percentile)
            weights.append(1.5) # High weight for tech bubble check
    except Exception as e:
        logger.error(f"Error calculating QQQ Deviation: {e}")

    # --- 2. VIX Index (Fear Gauge) ---
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="6mo") # Fetch more to get 30 days safely
        if not hist.empty:
            current_vix = hist['Close'].iloc[-1]
            history_data = hist['Close'].tail(30).tolist()
            
            # VIX Scoring: 
            # VIX < 12: Complacency (High Bubble Risk) -> Score 80-100
            # VIX 12-18: Normal (Neutral Risk) -> Score 40-80
            # VIX > 18: Fear (Low Bubble Risk) -> Score < 40
            
            # Linear mapping:
            # 10 -> 100
            # 15 -> 60
            # 20 -> 20
            # 30 -> 0
            
            if current_vix <= 12:
                vix_score = 80 + (12 - current_vix) * 10 # 10->100, 12->80
            elif current_vix <= 18:
                vix_score = 40 + (18 - current_vix) * (40/6) # 12->80, 18->40
            else:
                vix_score = max(0, 40 - (current_vix - 18) * 2) # 18->40, 38->0
            
            vix_score = min(100, vix_score)
            signal = "High Risk" if vix_score >= 76 else "Neutral" if vix_score >= 45 else "Low Risk"
            
            risk_data["indicators"]["vix"] = {
                "name": "VIX",
                "value": current_vix,
                "score": vix_score,
                "description": "Market Volatility Index (Inverted for Bubble Risk)",
                "methodology": "Inverted VIX score. Low VIX (<12) implies complacency (High Risk), while High VIX (>18) implies fear (Low Risk).",
                "signal": signal,
                "history": history_data
            }
            scores.append(vix_score)
            weights.append(1.0)
    except Exception as e:
        logger.error(f"Error fetching VIX: {e}")

    # --- 3. Yield Curve (Liquidity Proxy) ---
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="6mo")
        if not hist.empty:
            yield_10y = hist['Close'].iloc[-1] # e.g. 4.25
            history_data = hist['Close'].tail(30).tolist()
            
            # Scoring: Yield < 2% => High Bubble Risk (Score 100). Yield > 5% => Low Risk (Score 0).
            yield_score = max(0, min(100, (5 - yield_10y) / 3 * 100))
            signal = "High Risk" if yield_score >= 76 else "Neutral" if yield_score >= 45 else "Low Risk"
            
            risk_data["indicators"]["yield10y"] = {
                "name": "Liquidity Index (Yield Proxy)",
                "value": yield_10y,
                "score": yield_score,
                "description": "10-Year Treasury Yield (Low rates = High Liquidity)",
                "methodology": "Proxied by 10-Year Treasury Yield. Lower yields (<2%) indicate cheap money/high liquidity (High Risk), while higher yields (>5%) indicate tight liquidity (Low Risk).",
                "signal": signal,
                "history": history_data
            }
            scores.append(yield_score)
            weights.append(0.8)
    except Exception as e:
        logger.error(f"Error fetching Yields: {e}")

    # --- 4. Smart Money Proxy (HYG vs TLT) ---
    try:
        hyg = yf.Ticker("HYG") # High Yield Corp Bond
        tlt = yf.Ticker("TLT") # 20+ Year Treasury
        
        hyg_hist = hyg.history(period="1y")
        tlt_hist = tlt.history(period="1y")
        
        if not hyg_hist.empty and not tlt_hist.empty:
            # Align dates
            df = pd.DataFrame({'HYG': hyg_hist['Close'], 'TLT': tlt_hist['Close']}).dropna()
            df['Ratio'] = df['HYG'] / df['TLT']
            
            history_data = df['Ratio'].tail(30).tolist()

            # Compare current ratio to 200-day MA of ratio
            df['MA200'] = df['Ratio'].rolling(window=200).mean()
            
            if len(df) > 200:
                current_ratio = df['Ratio'].iloc[-1]
                ma200_ratio = df['MA200'].iloc[-1]
                
                # If Ratio > MA200, Risk On (Smart Money buying junk vs safety).
                # Higher Ratio = Higher Bubble Risk.
                deviation = (current_ratio - ma200_ratio) / ma200_ratio
                
                # Score: Deviation > 10% => 100. Deviation < -10% => 0.
                smart_score = max(0, min(100, 50 + deviation * 500))
                signal = "High Risk" if smart_score >= 76 else "Neutral" if smart_score >= 45 else "Low Risk"
                
                risk_data["indicators"]["smartMoney"] = {
                    "name": "Smart Money vs Dumb Money",
                    "value": current_ratio,
                    "score": smart_score,
                    "description": "HYG/TLT Ratio vs 200d MA (Risk-On vs Risk-Off)",
                    "methodology": "Compares High Yield Bonds (HYG) to Treasuries (TLT). A ratio significantly above its 200-day MA indicates 'Risk On' behavior (High Risk).",
                    "signal": signal,
                    "history": history_data
                }
                scores.append(smart_score)
                weights.append(1.2)
    except Exception as e:
        logger.error(f"Error fetching Smart Money: {e}")

    # --- 5. Put/Call Ratio (Sentiment) ---
    try:
        # Try CBOE Total Put/Call Ratio (^CPC) first
        pcr_ticker = yf.Ticker("^CPC")
        hist = pcr_ticker.history(period="6mo")
        
        if not hist.empty:
            current_pcr = hist['Close'].iloc[-1]
            history_data = hist['Close'].tail(30).tolist()
            
            # Scoring: 
            # High P/C (> 1.0) = Fear (Low Bubble Risk) -> Score < 40
            # Low P/C (< 0.7) = Greed (High Bubble Risk) -> Score > 70
            
            # Mapping: 0.6 -> 100 (Extreme Greed), 1.2 -> 0 (Extreme Fear)
            pc_score = max(0, min(100, 100 - (current_pcr - 0.6) * (100/0.6)))
            
            signal = "High Risk" if pc_score >= 70 else "Neutral" if pc_score >= 40 else "Low Risk"
            
            risk_data["indicators"]["putCall"] = {
                "name": "Put/Call Ratio",
                "value": current_pcr,
                "score": pc_score,
                "description": "CBOE Total Put/Call Ratio (Low = Greed)",
                "methodology": "Ratio of Put volume to Call volume. Low ratio (<0.7) indicates extreme greed (High Risk), while high ratio (>1.0) indicates fear (Low Risk).",
                "signal": signal,
                "history": history_data
            }
            scores.append(pc_score)
            weights.append(1.0)
            
        else:
            # Fallback to SPY Options
            spy = yf.Ticker("SPY")
            opts = spy.options
            if opts:
                # Get nearest expiration
                chain = spy.option_chain(opts[0])
                calls = chain.calls
                puts = chain.puts
                
                total_call_vol = calls['volume'].sum()
                total_put_vol = puts['volume'].sum()
                
                if total_call_vol > 0:
                    pc_ratio = total_put_vol / total_call_vol
                    
                    # Mapping: 0.5 -> 100 (Extreme Greed), 1.5 -> 0 (Extreme Fear)
                    pc_score = max(0, min(100, 100 - (pc_ratio - 0.5) * 100))
                    
                    signal = "High Risk" if pc_score >= 76 else "Neutral" if pc_score >= 45 else "Low Risk"
                    
                    # History is hard for options snapshot, use a flat line or empty
                    history_data = [pc_ratio] * 30 
                    
                    risk_data["indicators"]["putCall"] = {
                        "name": "Put/Call Ratio",
                        "value": pc_ratio,
                        "score": pc_score,
                        "description": "SPY Options Volume Ratio (Proxy)",
                        "methodology": "Ratio of Put volume to Call volume on SPY options. Low ratio indicates greed.",
                        "signal": signal,
                        "history": history_data
                    }
                    scores.append(pc_score)
                    weights.append(1.0)
    except Exception as e:
        logger.error(f"Error fetching Put/Call Ratio: {e}")

    # --- 6. Fed Rate Expectations (3-Month Yield Proxy) ---
    try:
        irx = yf.Ticker("^IRX") # 13 Week Treasury Bill
        hist = irx.history(period="6mo")
        if not hist.empty:
            current_rate = hist['Close'].iloc[-1]
            history_data = hist['Close'].tail(30).tolist()
            
            # Scoring: 
            # Low Rates (< 2%) = Stimulative (High Bubble Risk) -> Score > 80
            # High Rates (> 5%) = Restrictive (Low Bubble Risk) -> Score < 20
            
            rate_score = max(0, min(100, (5.5 - current_rate) / 3.5 * 100))
            signal = "High Risk" if rate_score >= 76 else "Neutral" if rate_score >= 45 else "Low Risk"
            
            risk_data["indicators"]["fedRate"] = {
                "name": "Fed Rate Expectation (Proxy)",
                "value": current_rate,
                "score": rate_score,
                "description": "3-Month Treasury Yield (Proxy for Near-Term Rates)",
                "methodology": "Proxied by 3-Month Treasury Yield. Low rates (<2%) are stimulative (High Risk), while high rates (>5%) are restrictive (Low Risk).",
                "signal": signal,
                "history": history_data
            }
            scores.append(rate_score)
            weights.append(0.8)
    except Exception as e:
        logger.error(f"Error fetching Fed Rate Proxy: {e}")

    # --- 7. CNN Fear & Greed Index ---
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            fg_data = data.get('fear_and_greed', {})
            score = fg_data.get('score', 50)
            rating = fg_data.get('rating', 'neutral')
            
            # History
            # The API returns 'fear_and_greed_historical' usually
            # But for now let's just use the current score. 
            # If we want history, we might need to parse the graph data if available.
            # The 'fear_and_greed_historical' key contains 'data' list of objects {x, y, rating}
            
            history_data = []
            hist_data = data.get('fear_and_greed_historical', {}).get('data', [])
            if hist_data:
                # Get last 30 points
                history_data = [item['y'] for item in hist_data[-30:]]
            
            # Ranges:
            # 0-24: Extreme Fear (Low Risk)
            # 25-44: Fear (Low Risk)
            # 45-55: Neutral (Neutral Risk)
            # 56-75: Greed (Neutral/Elevated Risk)
            # 76-100: Extreme Greed (High Risk)
            
            category = ""
            if score <= 24: category = "Extreme Fear"
            elif score <= 44: category = "Fear"
            elif score <= 55: category = "Neutral"
            elif score <= 75: category = "Greed"
            else: category = "Extreme Greed"
            
            signal = "High Risk" if score >= 76 else "Neutral" if score >= 45 else "Low Risk"
            
            risk_data["indicators"]["fearGreed"] = {
                "name": "Fear & Greed Index",
                "value": score,
                "score": score,
                "description": f"CNN Sentiment: {category} (Low = Fear/Safe)",
                "methodology": "Directly fetched from CNN. 0-25: Extreme Fear (Safe), 75-100: Extreme Greed (Risky).",
                "signal": signal,
                "history": history_data
            }
            scores.append(score)
            weights.append(2.0) # High weight as it's a composite itself
    except Exception as e:
        logger.error(f"Error fetching Fear & Greed Index: {e}")


    # --- Calculate Composite Score ---
    if scores:
        total_score = np.average(scores, weights=weights)
        risk_data["score"] = round(total_score, 1)
        
        if total_score > 75:
            risk_data["rating"] = "Extreme Greed / Bubble Risk"
        elif total_score > 55:
            risk_data["rating"] = "Greed / Elevated Risk"
        elif total_score > 45:
            risk_data["rating"] = "Neutral"
        elif total_score > 25:
            risk_data["rating"] = "Fear / Undervalued"
        else:
            risk_data["rating"] = "Extreme Fear / Opportunity"
            
    return risk_data
