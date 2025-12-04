from sqlalchemy.orm import Session
from datetime import date
import pandas as pd
import numpy as np
from quant.backtest.engine import BacktestEngine
import logging

logger = logging.getLogger(__name__)

class WalkForwardBacktester:
    def __init__(self, db: Session):
        self.db = db
        
    def run(self, start_date: date, end_date: date):
        """
        Run the strategy over the specified period using the Event-Driven Engine.
        Returns equity curve and performance metrics.
        """
        engine = BacktestEngine(self.db)
        history_df = engine.run_backtest(start_date, end_date)
        
        if history_df.empty:
            return {
                "equity_curve": [],
                "metrics": {}
            }
            
        # Calculate Metrics
        metrics = self._calculate_metrics(history_df)
        
        return {
            "equity_curve": history_df.to_dict(orient='records'),
            "metrics": metrics
        }

    def _calculate_metrics(self, df: pd.DataFrame):
        if df.empty or 'equity' not in df.columns:
            return {}
            
        # Daily Returns
        df['returns'] = df['equity'].pct_change().fillna(0)
        
        total_return = (df['equity'].iloc[-1] / df['equity'].iloc[0]) - 1
        
        # Annualized Return (CAGR)
        days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        cagr = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # Volatility (Annualized)
        volatility = df['returns'].std() * np.sqrt(252)
        
        # Sharpe Ratio (Rf=4%)
        rf = 0.04
        sharpe = (cagr - rf) / volatility if volatility > 0 else 0
        
        # Max Drawdown
        df['cummax'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['cummax']) / df['cummax']
        max_drawdown = df['drawdown'].min()
        
        return {
            "total_return": total_return,
            "cagr": cagr,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown
        }
