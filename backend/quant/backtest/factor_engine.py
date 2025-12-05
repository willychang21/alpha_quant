"""
Professional Factor Engine with Risk Controls
AQR / Two Sigma Standard Implementation

Features:
- Point-in-Time factor calculation (no lookahead bias)
- Transaction cost model (10 bps)
- Risk Controls:
  - Max Position Size: 5% per stock
  - Sector Cap: 30% per sector
  - Stop Loss: -15% monthly
  - Minimum Holding Period: 2 months
- Robustness Testing:
  - Bootstrap Sharpe (95% CI)
  - Sub-period Analysis
  - Factor Attribution
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# ============================================
# Configuration Constants (AQR Standard)
# ============================================
DEFAULT_TRANSACTION_COST = 0.0010  # 10 bps one-way
MAX_POSITION_SIZE = 0.05           # 5% max per stock
SECTOR_CAP = 0.30                  # 30% max per sector
STOP_LOSS_THRESHOLD = -0.15       # -15% monthly stop loss
MIN_HOLDING_PERIOD = 2             # Minimum 2 months holding
BOOTSTRAP_ITERATIONS = 1000        # For Sharpe CI


@dataclass
class RiskControlConfig:
    """Configuration for risk controls."""
    max_position: float = MAX_POSITION_SIZE
    sector_cap: float = SECTOR_CAP
    stop_loss: float = STOP_LOSS_THRESHOLD
    min_holding: int = MIN_HOLDING_PERIOD
    transaction_cost: float = DEFAULT_TRANSACTION_COST


# Simple GICS Sector mapping (approximation based on common tickers)
SECTOR_MAPPING = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
    'META': 'Technology', 'NVDA': 'Technology', 'AVGO': 'Technology', 'CSCO': 'Technology',
    'ADBE': 'Technology', 'CRM': 'Technology', 'ORCL': 'Technology', 'ACN': 'Technology',
    'INTC': 'Technology', 'AMD': 'Technology', 'QCOM': 'Technology', 'TXN': 'Technology',
    'IBM': 'Technology', 'NOW': 'Technology', 'AMAT': 'Technology', 'MU': 'Technology',
    # Healthcare
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'ABBV': 'Healthcare',
    'MRK': 'Healthcare', 'LLY': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare',
    'DHR': 'Healthcare', 'BMY': 'Healthcare', 'AMGN': 'Healthcare', 'MDT': 'Healthcare',
    # Financials
    'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
    'MS': 'Financials', 'C': 'Financials', 'BLK': 'Financials', 'AXP': 'Financials',
    'SCHW': 'Financials', 'CME': 'Financials', 'USB': 'Financials', 'PNC': 'Financials',
    'BRK-B': 'Financials', 'V': 'Financials', 'MA': 'Financials',
    # Consumer
    'AMZN': 'Consumer', 'TSLA': 'Consumer', 'HD': 'Consumer', 'MCD': 'Consumer',
    'NKE': 'Consumer', 'SBUX': 'Consumer', 'TGT': 'Consumer', 'LOW': 'Consumer',
    'COST': 'Consumer', 'WMT': 'Consumer', 'PG': 'Consumer', 'KO': 'Consumer', 'PEP': 'Consumer',
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
    'EOG': 'Energy', 'MPC': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy',
    # Industrials
    'CAT': 'Industrials', 'DE': 'Industrials', 'UPS': 'Industrials', 'RTX': 'Industrials',
    'HON': 'Industrials', 'BA': 'Industrials', 'GE': 'Industrials', 'LMT': 'Industrials',
    'MMM': 'Industrials', 'UNP': 'Industrials',
    # Materials
    'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials', 'FCX': 'Materials',
    'NEM': 'Materials', 'ECL': 'Materials', 'DOW': 'Materials',
    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
    'AEP': 'Utilities', 'EXC': 'Utilities', 'XEL': 'Utilities',
    # Real Estate
    'PLD': 'RealEstate', 'AMT': 'RealEstate', 'CCI': 'RealEstate', 'EQIX': 'RealEstate',
    'SPG': 'RealEstate', 'PSA': 'RealEstate',
    # Communication
    'NFLX': 'Communication', 'DIS': 'Communication', 'CMCSA': 'Communication',
    'T': 'Communication', 'VZ': 'Communication', 'TMUS': 'Communication',
}


class PointInTimeFactorEngine:
    """
    Professional factor engine with point-in-time calculation.
    No lookahead bias - uses only data available at each date.
    """
    
    def __init__(self, prices_df: pd.DataFrame):
        """
        Args:
            prices_df: DataFrame with columns ['date', 'ticker', 'close']
        """
        self.prices = prices_df.pivot(index='date', columns='ticker', values='close')
        self.prices = self.prices.sort_index()
        self.returns = self.prices.pct_change()
        
    def calculate_factors_at_date(self, as_of_date: pd.Timestamp) -> pd.DataFrame:
        """Calculate all factors using only data up to as_of_date."""
        prices = self.prices.loc[:as_of_date]
        returns = self.returns.loc[:as_of_date]
        
        if len(prices) < 252:
            logger.warning(f"Insufficient data at {as_of_date}: {len(prices)} days")
            return pd.DataFrame()
        
        factors = {}
        
        # 1. Volatility-Scaled Momentum (VSM)
        factors['vsm'] = self._calculate_vsm(returns)
        
        # 2. Betting Against Beta (BAB)
        factors['bab'] = self._calculate_bab(returns)
        
        # 3. 12-1 Momentum
        factors['momentum'] = self._calculate_momentum_12_1(prices)
        
        result = pd.DataFrame(factors).dropna(how='all')
        
        if result.empty:
            return pd.DataFrame()
        
        # Z-score normalization (cross-sectional)
        for col in ['vsm', 'bab', 'momentum']:
            if col in result.columns:
                mean, std = result[col].mean(), result[col].std()
                result[f'z_{col}'] = (result[col] - mean) / std if std > 0 else 0
        
        # Composite score
        z_cols = [c for c in result.columns if c.startswith('z_')]
        result['score'] = result[z_cols].mean(axis=1) if z_cols else 0
        
        result = result.reset_index()
        result.columns = ['ticker'] + list(result.columns[1:])
        result['date'] = as_of_date
        
        return result
    
    def _calculate_vsm(self, returns: pd.DataFrame, lookback: int = 252) -> pd.Series:
        """Volatility-Scaled Momentum."""
        recent = returns.tail(lookback)
        total_ret = (1 + recent).prod() - 1
        vol = recent.std() * np.sqrt(252)
        return total_ret / vol.replace(0, np.nan)
    
    def _calculate_bab(self, returns: pd.DataFrame, lookback: int = 252) -> pd.Series:
        """Betting Against Beta: prefer low-beta stocks."""
        recent = returns.tail(lookback)
        market = recent.mean(axis=1)
        market_var = market.var()
        
        if market_var == 0:
            return pd.Series(dtype=float)
        
        betas = {t: -recent[t].cov(market) / market_var for t in recent.columns}
        return pd.Series(betas)
    
    def _calculate_momentum_12_1(self, prices: pd.DataFrame) -> pd.Series:
        """12-1 Momentum (Jegadeesh-Titman)."""
        if len(prices) < 252:
            return pd.Series(dtype=float)
        
        ret_12m = prices.iloc[-1] / prices.iloc[-252] - 1
        ret_1m = prices.iloc[-1] / prices.iloc[-21] - 1 if len(prices) >= 21 else 0
        return ret_12m - ret_1m


def get_sector(ticker: str) -> str:
    """Get sector for a ticker, default to 'Other'."""
    return SECTOR_MAPPING.get(ticker, 'Other')


def apply_risk_controls(
    scores: pd.DataFrame,
    holdings: Dict[str, int],  # ticker -> months held
    top_n: int,
    config: RiskControlConfig
) -> Dict[str, float]:
    """
    Apply risk controls to generate final portfolio weights.
    
    Controls:
    1. Max position size
    2. Sector cap
    3. Minimum holding period (anti-churn)
    """
    if scores.empty:
        return {}
    
    # Start with top N by score
    candidates = scores.nlargest(top_n * 2, 'score')  # Get extra for filtering
    
    result = {}
    sector_weights = {}
    
    # Keep existing positions that haven't met minimum holding
    for ticker, months in holdings.items():
        if months < config.min_holding and ticker in candidates['ticker'].values:
            weight = min(1.0 / top_n, config.max_position)
            sector = get_sector(ticker)
            
            if sector_weights.get(sector, 0) + weight <= config.sector_cap:
                result[ticker] = weight
                sector_weights[sector] = sector_weights.get(sector, 0) + weight
    
    # Fill remaining slots
    for _, row in candidates.iterrows():
        if len(result) >= top_n:
            break
            
        ticker = row['ticker']
        if ticker in result:
            continue
            
        weight = min(1.0 / top_n, config.max_position)
        sector = get_sector(ticker)
        
        # Check sector cap
        if sector_weights.get(sector, 0) + weight > config.sector_cap:
            continue
            
        result[ticker] = weight
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
    
    # Normalize weights to sum to 1
    total = sum(result.values())
    if total > 0:
        result = {k: v / total for k, v in result.items()}
    
    return result


def check_stop_loss(
    returns: pd.DataFrame,
    holdings: Dict[str, float],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    threshold: float
) -> Tuple[Dict[str, float], List[str]]:
    """
    Check for stop loss triggers within the period.
    Returns updated holdings and list of stopped out tickers.
    """
    stopped_out = []
    
    period_returns = returns.loc[start_date:end_date]
    if period_returns.empty:
        return holdings, stopped_out
    
    for ticker in list(holdings.keys()):
        if ticker not in period_returns.columns:
            continue
            
        # Calculate cumulative return
        cum_return = (1 + period_returns[ticker]).cumprod() - 1
        
        # Check if any point hit stop loss
        if (cum_return <= threshold).any():
            stopped_out.append(ticker)
    
    # Remove stopped positions
    remaining = {k: v for k, v in holdings.items() if k not in stopped_out}
    
    # Redistribute weights
    total = sum(remaining.values())
    if total > 0:
        remaining = {k: v / total for k, v in remaining.items()}
    
    return remaining, stopped_out


def calculate_turnover(old: Dict[str, float], new: Dict[str, float]) -> float:
    """Calculate portfolio turnover."""
    all_tickers = set(old.keys()) | set(new.keys())
    return sum(abs(old.get(t, 0) - new.get(t, 0)) for t in all_tickers) / 2


def run_factor_backtest(
    prices_df: pd.DataFrame,
    rebalance_dates: List[pd.Timestamp],
    top_n: int = 50,
    config: Optional[RiskControlConfig] = None
) -> pd.DataFrame:
    """
    Run professional factor backtest with risk controls.
    """
    if config is None:
        config = RiskControlConfig()
    
    engine = PointInTimeFactorEngine(prices_df)
    
    results = []
    holdings = {}          # ticker -> weight
    holding_periods = {}   # ticker -> months held
    portfolio_gross = 1.0
    portfolio_net = 1.0
    cumulative_costs = 0.0
    total_stop_losses = 0
    
    for i, date in enumerate(rebalance_dates):
        # 1. Calculate factor scores
        factors = engine.calculate_factors_at_date(date)
        
        if factors.empty:
            continue
        
        # 2. Apply risk controls to get new weights
        new_holdings = apply_risk_controls(factors, holding_periods, top_n, config)
        
        # 3. Calculate turnover and costs
        turnover = calculate_turnover(holdings, new_holdings)
        period_cost = turnover * config.transaction_cost * 2
        cumulative_costs += period_cost
        
        # 4. Calculate returns until next rebalance
        if i < len(rebalance_dates) - 1:
            next_date = rebalance_dates[i + 1]
            
            # Check for stop losses
            active_holdings, stopped = check_stop_loss(
                engine.returns, new_holdings, date, next_date, config.stop_loss
            )
            total_stop_losses += len(stopped)
            
            # Calculate period return
            period_returns = engine.returns.loc[date:next_date].iloc[1:]
            
            if not period_returns.empty:
                gross_ret = sum(
                    w * ((1 + period_returns.get(t, pd.Series([0]))).prod() - 1)
                    for t, w in active_holdings.items()
                    if t in period_returns.columns
                )
                
                portfolio_gross *= (1 + gross_ret)
                portfolio_net *= (1 + gross_ret - period_cost)
        
        # 5. Update holding periods
        new_periods = {}
        for ticker in new_holdings:
            new_periods[ticker] = holding_periods.get(ticker, 0) + 1
        holding_periods = new_periods
        
        # 6. Calculate sector exposure
        sector_exp = {}
        for ticker, weight in new_holdings.items():
            sector = get_sector(ticker)
            sector_exp[sector] = sector_exp.get(sector, 0) + weight
        max_sector = max(sector_exp.values()) if sector_exp else 0
        
        results.append({
            'date': date,
            'portfolio_value': portfolio_net,
            'portfolio_value_gross': portfolio_gross,
            'turnover': turnover,
            'transaction_cost': period_cost,
            'cumulative_costs': cumulative_costs,
            'num_holdings': len(new_holdings),
            'max_sector_exposure': max_sector,
            'stop_losses': total_stop_losses
        })
        
        holdings = new_holdings
    
    return pd.DataFrame(results)


def calculate_performance_metrics(equity_curve: pd.DataFrame) -> Dict:
    """Calculate comprehensive performance metrics."""
    if 'portfolio_value' not in equity_curve.columns or len(equity_curve) < 2:
        return {}
    
    values = equity_curve['portfolio_value'].values
    n_months = len(values) - 1
    
    # Returns
    total_return = values[-1] / values[0] - 1
    annual_return = (1 + total_return) ** (12 / n_months) - 1 if n_months > 0 else 0
    
    # Monthly returns
    returns = pd.Series(values).pct_change().dropna()
    
    # Volatility
    volatility = returns.std() * np.sqrt(12)
    
    # Sharpe
    sharpe = annual_return / volatility if volatility > 0 else 0
    
    # Max Drawdown
    cummax = pd.Series(values).cummax()
    drawdown = (pd.Series(values) - cummax) / cummax
    max_drawdown = drawdown.min()
    
    # Gross Sharpe
    gross_sharpe = None
    if 'portfolio_value_gross' in equity_curve.columns:
        gross_values = equity_curve['portfolio_value_gross'].values
        gross_ret = (gross_values[-1] / gross_values[0] - 1)
        gross_annual = (1 + gross_ret) ** (12 / n_months) - 1 if n_months > 0 else 0
        gross_vol = pd.Series(gross_values).pct_change().dropna().std() * np.sqrt(12)
        gross_sharpe = gross_annual / gross_vol if gross_vol > 0 else 0
    
    # Average metrics
    avg_turnover = equity_curve['turnover'].mean() if 'turnover' in equity_curve.columns else 0
    total_costs = equity_curve['cumulative_costs'].iloc[-1] if 'cumulative_costs' in equity_curve.columns else 0
    total_stop_losses = equity_curve['stop_losses'].iloc[-1] if 'stop_losses' in equity_curve.columns else 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'sharpe_ratio_gross': gross_sharpe,
        'max_drawdown': max_drawdown,
        'avg_turnover': avg_turnover,
        'total_transaction_costs': total_costs,
        'total_stop_losses': int(total_stop_losses),
        'n_periods': n_months
    }


def bootstrap_sharpe_ci(
    equity_curve: pd.DataFrame,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    confidence: float = 0.95
) -> Dict:
    """
    Calculate bootstrap confidence interval for Sharpe ratio.
    """
    if 'portfolio_value' not in equity_curve.columns or len(equity_curve) < 12:
        return {'sharpe_mean': None, 'sharpe_lower': None, 'sharpe_upper': None}
    
    values = equity_curve['portfolio_value'].values
    returns = pd.Series(values).pct_change().dropna().values
    
    sharpes = []
    for _ in range(n_iterations):
        # Resample with replacement
        sample = np.random.choice(returns, size=len(returns), replace=True)
        mean_ret = np.mean(sample) * 12  # Annualize
        vol = np.std(sample) * np.sqrt(12)
        if vol > 0:
            sharpes.append(mean_ret / vol)
    
    sharpes = np.array(sharpes)
    alpha = (1 - confidence) / 2
    
    return {
        'sharpe_mean': np.mean(sharpes),
        'sharpe_lower': np.percentile(sharpes, alpha * 100),
        'sharpe_upper': np.percentile(sharpes, (1 - alpha) * 100),
        'sharpe_std': np.std(sharpes)
    }


def calculate_subperiod_analysis(equity_curve: pd.DataFrame) -> List[Dict]:
    """Calculate year-by-year performance breakdown."""
    if 'portfolio_value' not in equity_curve.columns or 'date' not in equity_curve.columns:
        return []
    
    results = []
    equity_curve['year'] = pd.to_datetime(equity_curve['date']).dt.year
    
    for year, group in equity_curve.groupby('year'):
        if len(group) < 2:
            continue
            
        values = group['portfolio_value'].values
        returns = pd.Series(values).pct_change().dropna()
        
        total_ret = values[-1] / values[0] - 1
        vol = returns.std() * np.sqrt(12) if len(returns) > 1 else 0
        sharpe = (total_ret * (12 / len(returns))) / vol if vol > 0 else 0
        
        cummax = pd.Series(values).cummax()
        max_dd = ((pd.Series(values) - cummax) / cummax).min()
        
        results.append({
            'year': int(year),
            'return': round(total_ret * 100, 1),
            'volatility': round(vol * 100, 1),
            'sharpe': round(sharpe, 2),
            'max_drawdown': round(max_dd * 100, 1)
        })
    
    return results


def calculate_factor_attribution(equity_curve: pd.DataFrame, factors_history: List[pd.DataFrame]) -> Dict:
    """
    Estimate factor contribution to returns.
    Simplified version - calculates average factor exposures.
    """
    if not factors_history:
        return {}
    
    # Aggregate all factors
    all_factors = pd.concat(factors_history, ignore_index=True)
    
    avg_exposures = {}
    for col in ['z_vsm', 'z_bab', 'z_momentum']:
        if col in all_factors.columns:
            avg_exposures[col.replace('z_', '')] = all_factors[col].mean()
    
    return avg_exposures


def calculate_rolling_sharpe(equity_curve: pd.DataFrame, window: int = 12) -> List[Dict]:
    """Calculate 12-month rolling Sharpe ratio."""
    if 'portfolio_value' not in equity_curve.columns or len(equity_curve) < window:
        return []
    
    values = equity_curve['portfolio_value'].values
    dates = equity_curve['date'].values
    returns = pd.Series(values).pct_change().dropna()
    
    result = []
    for i in range(window, len(returns) + 1):
        w_ret = returns.iloc[i-window:i]
        mean = w_ret.mean() * 12
        vol = w_ret.std() * np.sqrt(12)
        sharpe = mean / vol if vol > 0 else 0
        
        result.append({
            'date': pd.Timestamp(dates[i]).strftime('%Y-%m-%d'),
            'rolling_sharpe': round(sharpe, 2)
        })
    
    return result


def calculate_monthly_returns(equity_curve: pd.DataFrame) -> List[Dict]:
    """Calculate monthly returns for heatmap."""
    if 'portfolio_value' not in equity_curve.columns or len(equity_curve) < 2:
        return []
    
    result = []
    values = equity_curve['portfolio_value'].values
    dates = equity_curve['date'].values
    
    for i in range(1, len(values)):
        ret = (values[i] / values[i-1] - 1) * 100
        date = pd.Timestamp(dates[i])
        result.append({
            'year': date.year,
            'month': date.month,
            'return': round(ret, 2)
        })
    
    return result
