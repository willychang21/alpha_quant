from sqlalchemy.orm import Session
from app.core import database
from app.domain import models, schemas
from app.data.providers import yfinance_provider
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def get_portfolios(db: Session):
    return db.query(models.Portfolio).all()

def create_portfolio(db: Session, name: str):
    db_portfolio = models.Portfolio(name=name)
    db.add(db_portfolio)
    db.commit()
    db.refresh(db_portfolio)
    return db_portfolio

def rename_portfolio(db: Session, portfolio_id: int, name: str):
    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if portfolio:
        portfolio.name = name
        db.commit()
        return True
    return False

def delete_portfolio(db: Session, portfolio_id: int):
    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if portfolio:
        db.delete(portfolio)
        db.commit()
        return True
    return False

def get_portfolio_config(db: Session, portfolio_id: int) -> schemas.PortfolioConfig:
    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not portfolio:
        return schemas.PortfolioConfig(etfs=[], direct=[], cash=0, currency="USD")
    
    etfs = []
    direct = []
    
    for holding in portfolio.holdings:
        if holding.type == 'ETF':
            etfs.append(schemas.ETFConfig(
                ticker=holding.ticker,
                quantity=holding.quantity,
                averageCost=holding.average_cost,
                purchaseDate=holding.purchase_date,
                account=holding.account
            ))
        else:
            direct.append(schemas.DirectConfig(
                ticker=holding.ticker,
                alias=holding.alias,
                quantity=holding.quantity,
                averageCost=holding.average_cost,
                purchaseDate=holding.purchase_date,
                account=holding.account
            ))
            
    return schemas.PortfolioConfig(
        etfs=etfs,
        direct=direct,
        cash=portfolio.cash,
        currency=portfolio.currency
    )

def save_portfolio_config(db: Session, portfolio_id: int, config: schemas.PortfolioConfig):
    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not portfolio:
        return False
        
    # Update portfolio cash/currency
    portfolio.cash = config.cash
    portfolio.currency = config.currency
    
    # Clear existing holdings
    db.query(models.Holding).filter(models.Holding.portfolio_id == portfolio_id).delete()
    
    # Add new holdings
    for etf in config.etfs:
        if etf.ticker:
            db.add(models.Holding(
                ticker=etf.ticker,
                quantity=etf.quantity,
                type='ETF',
                portfolio_id=portfolio_id,
                average_cost=etf.averageCost,
                purchase_date=etf.purchaseDate,
                account=etf.account
            ))
            
    for d in config.direct:
        if d.ticker:
            db.add(models.Holding(
                ticker=d.ticker,
                alias=d.alias,
                quantity=d.quantity,
                type='DIRECT',
                portfolio_id=portfolio_id,
                average_cost=d.averageCost,
                purchase_date=d.purchaseDate,
                account=d.account
            ))
            
    db.commit()
    return True

def get_portfolio_data(db: Session, portfolio_id: int):
    config = get_portfolio_config(db, portfolio_id)
    
    holdings_map = {}
    etf_data = {}
    etf_headers = []
    
    total_portfolio_value = 0
    total_day_change = 0
    asset_values = {}
    
    # 1. Calculate Values
    # ETFs
    for etf in config.etfs:
        data = yfinance_provider.get_price_data(etf.ticker)
        if data:
            value = data['price'] * etf.quantity
            day_change_value = data['change'] * etf.quantity
            asset_values[etf.ticker] = {
                **data,
                "value": value,
                "quantity": etf.quantity,
                "day_change_value": day_change_value
            }
            total_portfolio_value += value
            total_day_change += day_change_value
            
    # Direct
    for d in config.direct:
        alias = d.alias or d.ticker
        data = yfinance_provider.get_price_data(d.ticker)
        if data:
            value = data['price'] * d.quantity
            day_change_value = data['change'] * d.quantity
            asset_values[alias] = {
                **data,
                "value": value,
                "quantity": d.quantity,
                "day_change_value": day_change_value,
                "ticker": d.ticker
            }
            total_portfolio_value += value
            total_day_change += day_change_value

    total_day_change_percent = (total_day_change / (total_portfolio_value - total_day_change) * 100) if (total_portfolio_value - total_day_change) != 0 else 0

    # 2. Process ETFs (Holdings Matrix)
    for etf in config.etfs:
        ticker = etf.ticker
        asset_data = asset_values.get(ticker)
        if not asset_data: continue
        
        weight = (asset_data['value'] / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
        etf_headers.append({"id": ticker, "label": ticker, "weight": weight})
        
        try:
            t = asset_data['obj']
            info = t.info
            
            holdings_list = yfinance_provider.fetch_holdings(ticker)
            
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            country = info.get('country', 'Unknown')
            name = info.get('shortName', ticker)
            
            etf_data[ticker] = {
                "price": asset_data['price'],
                "changePercent": asset_data['change_percent'],
                "name": name,
                "holdings": holdings_list,
                "weight": weight,
                "quantity": etf.quantity,
                "marketValue": asset_data['value'],
                "sector": sector,
                "industry": industry,
                "country": country,
                "type": info.get('quoteType', 'ETF')
            }
            
            # If no holdings, add itself to holdingsMap
            if not holdings_list:
                if ticker not in holdings_map:
                    holdings_map[ticker] = {
                        "ticker": ticker,
                        "name": name,
                        "etfs": {e.ticker: 0 for e in config.etfs},
                        "direct": 0,
                        "price": asset_data['price'],
                        "changePercent": asset_data['change_percent'],
                        "sector": sector,
                        "industry": industry,
                        "country": country,
                        "type": "ETF"
                    }
                holdings_map[ticker]["etfs"][ticker] = weight
            else:
                # Distribute weights
                total_holdings_percent = sum(h['percent'] for h in holdings_list)
                scale = 1.0
                if total_holdings_percent > 1.1: # If > 1.1, assume 0-100 scale
                    scale = 0.01
                    
                for h in holdings_list:
                    h_ticker = h['ticker']
                    h_percent = h['percent'] * scale
                    h_weight = weight * h_percent
                    
                    if h_ticker not in holdings_map:
                        holdings_map[h_ticker] = {
                            "ticker": h_ticker,
                            "name": h['name'],
                            "etfs": {e.ticker: 0 for e in config.etfs},
                            "direct": 0,
                            "price": 0, # We don't have price for constituents unless we fetch
                            "changePercent": 0,
                            "sector": "Unknown",
                            "industry": "Unknown",
                            "country": "Unknown",
                            "type": "EQUITY"
                        }
                    holdings_map[h_ticker]["etfs"][ticker] = h_weight

        except Exception as e:
            logger.error(f"Error processing ETF {ticker}: {e}")

    # 3. Process Direct
    for d in config.direct:
        alias = d.alias or d.ticker
        asset_data = asset_values.get(alias)
        if not asset_data: continue
        
        weight = (asset_data['value'] / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
        
        t = asset_data['obj']
        
        # Check if Direct holding is actually a Fund (like VIIIX)
        is_fund = False
        try:
            info = t.info
            quote_type = info.get('quoteType', 'EQUITY')
            if quote_type in ['ETF', 'MUTUALFUND']:
                holdings_list = yfinance_provider.fetch_holdings(d.ticker)
                if holdings_list:
                    is_fund = True
                    # Treat as ETF for matrix purposes
                    etf_headers.append({"id": alias, "label": alias, "weight": weight})
                    
                    total_holdings_percent = sum(h['percent'] for h in holdings_list)
                    scale = 1.0
                    if total_holdings_percent > 1.1:
                        scale = 0.01
                        
                    for h in holdings_list:
                        h_ticker = h['ticker']
                        h_percent = h['percent'] * scale
                        h_weight = weight * h_percent
                        
                        if h_ticker not in holdings_map:
                            holdings_map[h_ticker] = {
                                "ticker": h_ticker,
                                "name": h['name'],
                                "etfs": {e.ticker: 0 for e in config.etfs}, # This might miss the new 'direct-as-etf' key
                                "direct": 0,
                                "price": 0,
                                "changePercent": 0,
                                "sector": "Unknown",
                                "industry": "Unknown",
                                "country": "Unknown",
                                "type": "EQUITY"
                            }
                            # We need to initialize the new key in all existing maps if we add a header dynamically
                            # But simpler: just add it to 'etfs' map of this holding. 
                            # Wait, 'etfs' keys are initialized from config.etfs. 
                            # We need to add this alias to the keys.
                        
                        # Ensure key exists
                        if alias not in holdings_map[h_ticker]["etfs"]:
                             holdings_map[h_ticker]["etfs"][alias] = 0
                             
                        holdings_map[h_ticker]["etfs"][alias] = h_weight
        except:
            pass

        if not is_fund:
            if alias not in holdings_map:
                info = t.info
                holdings_map[alias] = {
                    "ticker": alias,
                    "name": info.get('shortName', alias),
                    "etfs": {e.ticker: 0 for e in config.etfs},
                    "direct": weight,
                    "price": asset_data['price'],
                    "changePercent": asset_data['change_percent'],
                    "sector": info.get('sector', 'Unknown'),
                    "industry": info.get('industry', 'Unknown'),
                    "country": info.get('country', 'Unknown'),
                    "type": info.get('quoteType', 'EQUITY')
                }
            else:
                holdings_map[alias]["direct"] = weight

    # 4. Transform to Matrix
    matrix_holdings = []
    for ticker, h in holdings_map.items():
        total_weight = h['direct'] + sum(h['etfs'].values())
        funds_holding_count = sum(1 for v in h['etfs'].values() if v > 0)
        
        matrix_holdings.append({
            **h,
            "percentOfPortfolio": total_weight,
            "fundsHolding": funds_holding_count,
            "priceChangePercent1D": h['changePercent'],
            "pl1D": 0,
            "flag": "US", # Simplification
            "account": "Account 2",
            "purchaseDate": "-",
            "quantity": 0,
            "averageCost": 0,
            "lastPrice": h['price'],
            "marketValue": 0,
            "plExclFX": "-",
            "plFromFX": "-",
            "pl": "-",
            "plPercent": "-",
            "totalReturnPercent": "-"
        })
        
    matrix_holdings.sort(key=lambda x: x['percentOfPortfolio'], reverse=True)
    
    # 5. Profit Loss Holdings
    profit_loss_holdings = []
    # Add ETFs
    for etf in config.etfs:
        data = etf_data.get(etf.ticker)
        if data:
            profit_loss_holdings.append({
                "ticker": etf.ticker,
                "name": data['name'],
                "priceChangePercent1D": data['changePercent'],
                "pl1D": 0,
                "percentOfPortfolio": data['weight'],
                "flag": "US",
                "account": etf.account,
                "purchaseDate": "-",
                "quantity": data['quantity'],
                "averageCost": 0,
                "lastPrice": data['price'],
                "marketValue": data['marketValue'],
                "plExclFX": "-",
                "plFromFX": "-",
                "pl": "-",
                "plPercent": "-",
                "totalReturnPercent": "-",
                "type": data['type'],
                "sector": data['sector'],
                "industry": data['industry'],
                "country": data['country']
            })
            
    # Add Direct
    for d in config.direct:
        alias = d.alias or d.ticker
        asset_data = asset_values.get(alias)
        if asset_data:
            t = asset_data['obj']
            info = t.info
            profit_loss_holdings.append({
                "ticker": alias,
                "name": info.get('shortName', alias),
                "priceChangePercent1D": asset_data['change_percent'],
                "pl1D": 0,
                "percentOfPortfolio": (asset_data['value'] / total_portfolio_value * 100) if total_portfolio_value > 0 else 0,
                "flag": "US",
                "account": d.account,
                "purchaseDate": "-",
                "quantity": d.quantity,
                "averageCost": 0,
                "lastPrice": asset_data['price'],
                "marketValue": asset_data['value'],
                "plExclFX": "-",
                "plFromFX": "-",
                "pl": "-",
                "plPercent": "-",
                "totalReturnPercent": "-",
                "type": info.get('quoteType', 'EQUITY'),
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown'),
                "country": info.get('country', 'Unknown')
            })

    return {
        "lastUpdated": datetime.now().isoformat(),
        "summary": {
            "totalValue": total_portfolio_value,
            "dayChange": total_day_change,
            "dayChangePercent": total_day_change_percent,
            "holdingsCount": len(config.etfs) + len(config.direct)
        },
        "matrixHoldings": matrix_holdings,
        "profitLossHoldings": profit_loss_holdings,
        "etfHeaders": etf_headers
    }
