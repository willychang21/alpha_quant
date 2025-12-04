import yfinance as yf
import pandas as pd
import numpy as np
from app.domain import schemas
import logging

logger = logging.getLogger(__name__)

def get_backtest_data(allocation: dict, initial_amount: float, monthly_amount: float):
    tickers = list(allocation.keys())
    if not tickers:
        return {"lumpSum": [], "dca": [], "monthlyReturns": [], "metrics": {}}

    # Fetch historical data
    data = yf.download(tickers, period="max", interval="1mo", progress=False)['Close']
    
    # If single ticker, data is Series, convert to DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
        
    # Drop NaN
    data = data.dropna()
    
    if data.empty:
        return {"lumpSum": [], "dca": [], "monthlyReturns": [], "metrics": {}}

    # Calculate returns
    returns = data.pct_change().dropna()
    
    # Normalize weights
    total_alloc = sum(allocation.values())
    weights = np.array([allocation[t] / total_alloc for t in data.columns])
    
    # Portfolio Returns
    portfolio_returns = returns.dot(weights)
    
    # Lump Sum Simulation
    lump_sum_values = [initial_amount]
    current_lump = initial_amount
    
    # DCA Simulation
    dca_values = [monthly_amount] # Start with first contribution
    current_dca = monthly_amount
    total_invested = monthly_amount
    dca_invested = [total_invested]
    
    dates = portfolio_returns.index
    
    lump_sum_points = [{"date": dates[0].strftime('%Y-%m-%d'), "value": initial_amount}]
    dca_points = [{"date": dates[0].strftime('%Y-%m-%d'), "value": monthly_amount, "invested": monthly_amount}]
    
    monthly_returns_map = {}
    
    for date, ret in portfolio_returns.items():
        date_str = date.strftime('%Y-%m-%d')
        
        # Lump Sum
        current_lump = current_lump * (1 + ret)
        lump_sum_points.append({"date": date_str, "value": current_lump})
        
        # DCA
        current_dca = current_dca * (1 + ret) + monthly_amount
        total_invested += monthly_amount
        dca_points.append({"date": date_str, "value": current_dca, "invested": total_invested})
        
        # Monthly Return for Heatmap
        key = f"{date.year}-{date.month - 1}" # JS months are 0-indexed
        monthly_returns_map[key] = ret * 100

    # Format Monthly Returns
    monthly_returns_list = []
    years = sorted(list(set([d.year for d in dates])), reverse=True)
    
    for year in years:
        row = {"year": year, "returns": [None]*12, "total": 0}
        year_rets = []
        for m in range(12):
            key = f"{year}-{m}"
            if key in monthly_returns_map:
                val = monthly_returns_map[key]
                row["returns"][m] = val
                year_rets.append(val/100)
        
        if year_rets:
            total_ret = np.prod([1 + r for r in year_rets]) - 1
            row["total"] = total_ret * 100
        else:
            row["total"] = None
            
        monthly_returns_list.append(row)

    return {
        "lumpSum": lump_sum_points,
        "dca": dca_points,
        "monthlyReturns": monthly_returns_list,
        "metrics": {}
    }

def analyze_portfolio(request: schemas.AnalysisRequest):
    monthly_amount = request.monthlyAmount
    selected_assets = request.selectedAssets
    dynamic_compositions = request.dynamicCompositions or {}
    stock_names = request.stockNames or {}
    
    stock_investments = {} # ticker -> { amount, direct_amount, etfs: {etf_id: amount}, name }
    
    def ensure_stock_entry(ticker, name=None):
        if ticker not in stock_investments:
            # Try to resolve name from stock_names if not provided
            resolved_name = name or stock_names.get(ticker) or ticker
            stock_investments[ticker] = {
                "amount": 0.0,
                "direct": 0.0,
                "etfs": {},
                "name": resolved_name
            }
            
    for asset in selected_assets:
        asset_id = asset['id']
        asset_type = asset.get('type', 'Asset')
        allocation = float(asset['allocation']) if asset['allocation'] else 0.0
        label = asset.get('label', asset_id)
        
        if allocation <= 0:
            continue
            
        # Check if it's an ETF with known composition
        if asset_type == 'ETF' and asset_id in dynamic_compositions:
            composition = dynamic_compositions[asset_id]
            
            total_weight = sum(composition.values())
            scale = 1.0
            if total_weight > 1.1: # If sum is > 1.1, assume it's 0-100
                scale = 0.01
                
            for ticker, weight in composition.items():
                ensure_stock_entry(ticker)
                
                amount_for_stock = allocation * (weight * scale)
                stock_investments[ticker]["amount"] += amount_for_stock
                stock_investments[ticker]["etfs"][asset_id] = stock_investments[ticker]["etfs"].get(asset_id, 0.0) + amount_for_stock
                
        else:
            # Direct or Unknown ETF
            ensure_stock_entry(asset_id, label)
            stock_investments[asset_id]["amount"] += allocation
            stock_investments[asset_id]["direct"] += allocation
            
    # 2. Build Matrix Holdings
    matrix_holdings = []
    
    for ticker, data in stock_investments.items():
        total_amount = data["amount"]
        if total_amount <= 0:
            continue
            
        percent_of_portfolio = (total_amount / monthly_amount * 100) if monthly_amount > 0 else 0
        
        etfs_percent = {}
        for etf_id, amount in data["etfs"].items():
            etfs_percent[etf_id] = (amount / monthly_amount * 100) if monthly_amount > 0 else 0
            
        direct_percent = (data["direct"] / monthly_amount * 100) if monthly_amount > 0 else 0
        
        matrix_holdings.append(schemas.MatrixHolding(
            ticker=ticker,
            name=data["name"],
            percentOfPortfolio=percent_of_portfolio,
            fundsHolding=len(data["etfs"]) + (1 if data["direct"] > 0 else 0),
            priceChangePercent1D=0, # Not calculating for projection
            pl1D=0,
            flag="US",
            account="Projected",
            purchaseDate="-",
            quantity=0,
            averageCost=0,
            lastPrice=0,
            marketValue=total_amount,
            plExclFX="-",
            plFromFX="-",
            pl="-",
            plPercent="-",
            totalReturnPercent="-",
            etfs=etfs_percent,
            direct=direct_percent,
            price=0,
            changePercent=0
        ))
        
    matrix_holdings.sort(key=lambda x: x.percentOfPortfolio, reverse=True)
    
    # 3. Build ETF Headers
    etf_headers = []
    
    active_etfs = set()
    for asset in selected_assets:
        if asset.get('type') == 'ETF':
             active_etfs.add(asset['id'])
             
    for etf_id in active_etfs:
        asset = next((a for a in selected_assets if a['id'] == etf_id), None)
        weight = 0
        if asset:
            alloc = float(asset['allocation']) if asset['allocation'] else 0
            weight = (alloc / monthly_amount * 100) if monthly_amount > 0 else 0
            
        etf_headers.append(schemas.ETFHeader(
            id=etf_id,
            label=asset.get('label', etf_id) if asset else etf_id,
            weight=weight
        ))
        
    return schemas.AnalysisResponse(
        matrixHoldings=matrix_holdings,
        etfHeaders=etf_headers
    )
