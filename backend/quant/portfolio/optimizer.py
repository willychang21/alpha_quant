from sqlalchemy.orm import Session
from typing import Dict, List, Any, Literal
from quant.data.models import Security, MarketDataDaily, ModelSignals, PortfolioTargets
from datetime import date, timedelta
import pandas as pd
import numpy as np
import cvxpy as cp
from sklearn.covariance import LedoitWolf
# Tier-1: Advanced Optimizers
from quant.portfolio.advanced_optimizers import HRPOptimizer, BlackLittermanModel
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(self, db: Session):
        self.db = db

    def run_optimization(
        self, 
        optimization_date: date, 
        top_n: int = 50,  # Optimize top 50
        max_weight: float = 0.10,  # Max 10% per stock
        risk_aversion: float = 1.0,
        optimizer: Literal['hrp', 'bl', 'mvo'] = 'hrp'  # Tier-1: Choose optimizer
    ):
        """
        Run portfolio optimization on top ranked stocks.
        
        Args:
            optimizer: 'hrp' (Hierarchical Risk Parity), 'bl' (Black-Litterman), 'mvo' (Mean-Variance)
        """
        # 1. Get Top Ranked Stocks (ranking_v3 with Tier-1 upgrades)
        top_picks = self.db.query(ModelSignals)\
            .filter(ModelSignals.date == optimization_date, ModelSignals.model_name == 'ranking_v3')\
            .order_by(ModelSignals.rank.asc())\
            .limit(top_n)\
            .all()
        
        # Fallback to v2 if v3 not available
        if not top_picks:
            logger.info("No ranking_v3 signals found, trying ranking_v2...")
            top_picks = self.db.query(ModelSignals)\
                .filter(ModelSignals.date == optimization_date, ModelSignals.model_name == 'ranking_v2')\
                .order_by(ModelSignals.rank.asc())\
                .limit(top_n)\
                .all()
            
        if not top_picks:
            logger.warning("No ranked stocks found for optimization (ranking_v2).")
            return
            
        sids = [p.sid for p in top_picks]
        tickers = [p.security.ticker for p in top_picks]
        
        # Map SID to Alpha Score
        # We use the raw score (which is a composite Z-score) as a proxy for expected return
        # A score of +3.0 implies high expected return.
        alpha_scores = np.array([p.score for p in top_picks])
        
        # 2. Get Historical Prices (1 Year) for Covariance
        # Try fetching from DB first, but if empty, use YFinanceProvider
        from core.adapters.yfinance_provider import YFinanceProvider
        yf_provider = YFinanceProvider()
        
        # We need 1 year of history for covariance
        # We can fetch this in bulk for the top_n tickers
        
        try:
            # Use yf.download for efficiency
            import yfinance as yf
            logger.info(f"Fetching 1y history for {len(tickers)} tickers for optimization...")
            bulk_history = yf.download(tickers, period="1y", group_by='ticker', threads=True, progress=False)
            
            # Process into pivot format (Date x Ticker)
            # bulk_history columns are (Price, Ticker) or just Price if 1 ticker
            
            pivot_df = pd.DataFrame()
            
            if len(tickers) == 1:
                # Single ticker case
                ticker = tickers[0]
                if not bulk_history.empty:
                    close = bulk_history['Close'] if 'Close' in bulk_history.columns else bulk_history
                    pivot_df[sids[0]] = close
            else:
                # Multi ticker case
                # Extract 'Close' for each ticker
                for sid, ticker in zip(sids, tickers):
                    try:
                        if isinstance(bulk_history.columns, pd.MultiIndex):
                            # Try ('Close', 'TICKER')
                            if ticker in bulk_history.columns.get_level_values(1):
                                series = bulk_history.xs(ticker, level=1, axis=1)['Close']
                            elif ticker in bulk_history.columns.get_level_values(0):
                                series = bulk_history[ticker]['Close']
                            else:
                                continue
                        else:
                            # Maybe flat columns if flattened? Unlikely with group_by='ticker'
                            continue
                            
                        pivot_df[sid] = series
                    except KeyError:
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to fetch history for optimization: {e}")
            return

        # Handle missing data (ffill)
        pivot_df = pivot_df.ffill().dropna()
        
        if pivot_df.empty:
            logger.warning("Insufficient price data after cleaning.")
            return
            
        # Ensure alignment between alpha_scores and covariance matrix
        # pivot_df columns are SIDs. We need to match the order of 'sids' list.
        valid_sids = [sid for sid in sids if sid in pivot_df.columns]
        
        if len(valid_sids) < len(sids):
            logger.warning(f"Dropped {len(sids) - len(valid_sids)} assets due to missing price history.")
            
        # Filter and reorder alpha scores
        sid_to_score = {p.sid: p.score for p in top_picks}
        aligned_alphas = np.array([sid_to_score[sid] for sid in valid_sids])
        
        # Calculate Returns
        returns_df = pivot_df[valid_sids].pct_change().dropna()
        
        # Map SID to ticker for later use
        sid_to_ticker = {p.sid: p.security.ticker for p in top_picks}
        ticker_to_sid = {v: k for k, v in sid_to_ticker.items()}
        returns_df_named = returns_df.rename(columns=sid_to_ticker)
        
        # --- Tier-1: Choose Optimizer ---
        
        if optimizer == 'hrp':
            # Hierarchical Risk Parity (more robust, no matrix inversion)
            logger.info("🔷 Using HRP Optimizer (Tier-1)")
            
            hrp = HRPOptimizer()
            hrp_weights = hrp.optimize(returns_df_named)
            
            if not hrp_weights:
                logger.warning("HRP optimization failed, falling back to MVO.")
                optimizer = 'mvo'
            else:
                # Map ticker weights back to SID weights
                optimal_weights = {ticker_to_sid.get(t, t): w for t, w in hrp_weights.items()}
                
                # Store
                final_sids = [sid for sid in valid_sids if sid in optimal_weights]
                final_weights = [optimal_weights[sid] for sid in final_sids]
                
                self._store_targets(final_sids, final_weights, optimization_date, optimizer='hrp')
                
                logger.info(f"✅ HRP Optimization complete. Top 5 weights:")
                for sid, w in sorted(zip(final_sids, final_weights), key=lambda x: x[1], reverse=True)[:5]:
                    ticker = sid_to_ticker.get(sid, str(sid))
                    logger.info(f"   {ticker}: {w:.2%}")
                
                return {'sids': final_sids, 'weights': final_weights, 'optimizer': 'hrp'}
        
        if optimizer == 'bl':
            # Black-Litterman Model (combines market equilibrium with alpha views)
            logger.info("🔶 Using Black-Litterman Optimizer (Tier-1)")
            
            try:
                # Fetch market caps for equilibrium returns
                aligned_tickers = [sid_to_ticker[sid] for sid in valid_sids]
                logger.info(f"Fetching market caps for {len(aligned_tickers)} tickers...")
                
                market_caps = {}
                volatilities = {}
                
                for ticker in aligned_tickers:
                    try:
                        info = yf.Ticker(ticker).info
                        mkt_cap = info.get('marketCap', 0)
                        if mkt_cap and mkt_cap > 0:
                            market_caps[ticker] = mkt_cap
                        # Annualized volatility from returns
                        if ticker in returns_df_named.columns:
                            vol = returns_df_named[ticker].std() * np.sqrt(252)
                            volatilities[ticker] = vol if vol > 0 else 0.25
                    except:
                        continue
                
                if len(market_caps) < 5:
                    logger.warning("Insufficient market cap data for Black-Litterman, falling back to HRP.")
                    optimizer = 'hrp'
                else:
                    # Prepare inputs for Black-Litterman
                    valid_bl_tickers = [t for t in aligned_tickers if t in market_caps and t in volatilities]
                    
                    market_caps_series = pd.Series({t: market_caps[t] for t in valid_bl_tickers})
                    cov_matrix = returns_df_named[valid_bl_tickers].cov() * 252  # Annualized
                    
                    # Z-scores from ranking signals
                    z_scores = {sid_to_ticker[p.sid]: p.score for p in top_picks if sid_to_ticker.get(p.sid) in valid_bl_tickers}
                    vol_dict = {t: volatilities[t] for t in valid_bl_tickers}
                    
                    # Run Black-Litterman
                    bl = BlackLittermanModel(tau=0.05, risk_aversion=2.5)
                    bl_weights = bl.optimize(cov_matrix, market_caps_series, z_scores, vol_dict, ic=0.05)
                    
                    if not bl_weights:
                        logger.warning("Black-Litterman failed, falling back to HRP.")
                        optimizer = 'hrp'
                    else:
                        # Map ticker weights back to SID weights
                        optimal_weights = {ticker_to_sid.get(t, t): w for t, w in bl_weights.items()}
                        
                        final_sids = [sid for sid in valid_sids if sid in optimal_weights]
                        final_weights = [optimal_weights[sid] for sid in final_sids]
                        
                        self._store_targets(final_sids, final_weights, optimization_date, optimizer='bl')
                        
                        logger.info(f"✅ Black-Litterman Optimization complete. Top 5 weights:")
                        for sid, w in sorted(zip(final_sids, final_weights), key=lambda x: x[1], reverse=True)[:5]:
                            ticker = sid_to_ticker.get(sid, str(sid))
                            logger.info(f"   {ticker}: {w:.2%}")
                        
                        return {'sids': final_sids, 'weights': final_weights, 'optimizer': 'bl'}
                        
            except Exception as e:
                logger.error(f"Black-Litterman optimization error: {e}")
                optimizer = 'mvo'
        
        # --- Original MVO Path ---
        if optimizer == 'mvo':
            logger.info("🔸 Using MVO Optimizer (fallback)")
            
            # 3. Estimate Covariance (Ledoit-Wolf)
            lw = LedoitWolf()
            covariance_matrix = lw.fit(returns_df).covariance_
            
            # 4. Optimize (CVXPY)
            n_assets = len(valid_sids)
            w = cp.Variable(n_assets)
            
            # Scale alpha to be comparable to variance
            scaled_alpha = aligned_alphas * 0.0005  # 1 Z-score = 5bps daily alpha
            
            risk = cp.quad_form(w, covariance_matrix)
            ret = scaled_alpha @ w
            
            objective = cp.Maximize(ret - risk_aversion * risk)
            
            constraints = [
                cp.sum(w) == 1,  # Fully invested
                w >= 0,  # Long only
                w <= max_weight  # Diversification constraint
            ]
            
            prob = cp.Problem(objective, constraints)
            
            try:
                prob.solve()
            except Exception as e:
                logger.error(f"Solver failed: {e}")
                return
                
            if w.value is None:
                logger.error("Optimization failed (unbounded or infeasible).")
            return
            
        optimal_weights = w.value
        
        # Clean weights
        optimal_weights[optimal_weights < 0.001] = 0
        optimal_weights /= np.sum(optimal_weights)
        
        # 5. Store Results
        self._store_targets(valid_sids, optimal_weights, optimization_date)
        
        # Log results
        logger.info("Optimization successful. Top Allocations:")
        
        # Map back to tickers
        sid_to_ticker = {p.sid: p.security.ticker for p in top_picks}
        aligned_tickers = [sid_to_ticker[sid] for sid in valid_sids]
        
        allocations = sorted(zip(aligned_tickers, optimal_weights), key=lambda x: x[1], reverse=True)
        for ticker, weight in allocations[:10]:
            logger.info(f"{ticker}: {weight:.1%}")
            
        return allocations

    def _store_targets(self, sids, weights, date, optimizer: str = 'mvo'):
        model_name = f'{optimizer}_v1'  # mvo_v1 or hrp_v1
        
        self.db.query(PortfolioTargets).filter(
            PortfolioTargets.date == date,
            PortfolioTargets.model_name == model_name
        ).delete()
        
        targets = []
        for sid, weight in zip(sids, weights):
            if weight > 0.001:  # Filter tiny weights
                targets.append(PortfolioTargets(
                    sid=sid,
                    date=date,
                    model_name=model_name,
                    weight=float(weight)
                ))
                
        self.db.bulk_save_objects(targets)
        self.db.commit()
        logger.info(f"Stored {len(targets)} portfolio targets with {model_name}.")

