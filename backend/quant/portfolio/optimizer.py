"""Portfolio Optimizer Module.

Implements multiple optimization strategies via Registry Pattern:
- HRP (Hierarchical Risk Parity)
- Black-Litterman
- MVO (Mean-Variance Optimization)
- Kelly Criterion

All optimizers are loaded dynamically from quant.plugins.optimizers.
New optimizers can be added as plugins without modifying this file.
"""

from sqlalchemy.orm import Session
from typing import Dict, List, Any, Literal, Optional, Tuple, Union
from quant.data.models import Security, MarketDataDaily, ModelSignals, PortfolioTargets
from datetime import date, timedelta
import pandas as pd
import numpy as np
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from quant.portfolio.risk_control import apply_vol_targeting
import yfinance as yf

# Import infrastructure
from core.structured_logger import get_structured_logger
from core.rate_limiter import get_yfinance_rate_limiter
from config.quant_config import get_optimization_config, OptimizationConfig

# Registry Pattern - all optimizers and risk models loaded from here
from quant.core.registry import registry

logger = get_structured_logger("PortfolioOptimizer")


class PortfolioOptimizer:
    """Portfolio optimizer with configurable constraints.
    
    Uses OptimizationConfig for default parameters. All defaults can be
    overridden via environment variables or constructor arguments.
    """
    
    def __init__(self, db: Session, config: Optional[OptimizationConfig] = None):
        """Initialize PortfolioOptimizer.
        
        Args:
            db: SQLAlchemy database session
            config: Optional OptimizationConfig, uses singleton if not provided
        """
        self.db = db
        self.config = config or get_optimization_config()
        self._rate_limiter = get_yfinance_rate_limiter()
        
        # Initialize Registry with plugins
        registry.discover_plugins("quant.plugins")
        logger.info(
            f"Registry initialized: {len(registry.list_optimizers())} optimizers, "
            f"{len(registry.list_risk_models())} risk models"
        )

    def get_optimizer(self, name: str, params: Optional[Dict] = None):
        """Get an optimizer instance from the Registry.
        
        Args:
            name: Registered optimizer name ('HRP', 'MVO', 'BlackLitterman', 'Kelly')
            params: Optional parameters for the optimizer
            
        Returns:
            Instantiated optimizer plugin.
        """
        optimizer_cls = registry.get_optimizer(name)
        return optimizer_cls(params=params or {})

    def check_risk_constraints(
        self,
        weights: pd.Series,
        sectors: Optional[Dict[str, str]] = None,
        betas: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, bool, Optional[str]]]:
        """Check portfolio weights against all registered risk constraints.
        
        Args:
            weights: Portfolio weights indexed by ticker.
            sectors: Optional ticker -> sector mapping.
            betas: Optional ticker -> beta mapping.
            
        Returns:
            List of (constraint_name, is_valid, error_message) tuples.
        """
        results = []
        
        for name in registry.list_risk_models():
            try:
                risk_cls = registry.get_risk_model(name)
                risk_model = risk_cls(params={})
                is_valid, msg = risk_model.check_constraints(
                    weights, sectors=sectors, betas=betas
                )
                results.append((name, is_valid, msg))
                if not is_valid:
                    logger.warning(f"Risk constraint '{name}' violated: {msg}")
            except Exception as e:
                logger.warning(f"Risk model '{name}' check failed: {e}")
                results.append((name, True, f"Check skipped: {e}"))
                
        return results

    def run_optimization(
        self, 
        optimization_date: date, 
        top_n: Optional[int] = None,
        max_weight: Optional[float] = None,
        risk_aversion: Optional[float] = None,
        optimizer: Literal['hrp', 'bl', 'mvo', 'kelly'] = 'hrp',
        target_vol: Optional[float] = None,
        sector_constraints: bool = False,
        beta_constraints: bool = False
    ) -> Optional[Union[Dict[str, Any], List[Tuple[str, float]]]]:
        """Run portfolio optimization on top ranked stocks.
        
        Args:
            optimization_date: Date for optimization
            top_n: Number of top stocks to optimize (default from config)
            max_weight: Maximum weight per stock (default from config)
            risk_aversion: Risk aversion parameter (default from config)
            optimizer: Optimization method ('hrp', 'bl', 'mvo', 'kelly')
            target_vol: Target volatility for vol targeting (optional)
            sector_constraints: Enable sector weight constraints
            beta_constraints: Enable beta constraints
            
        Returns:
            Optimization results or None if insufficient data
        """
        # Use config defaults if not specified
        top_n = top_n if top_n is not None else self.config.top_n
        max_weight = max_weight if max_weight is not None else self.config.max_weight
        risk_aversion = risk_aversion if risk_aversion is not None else self.config.risk_aversion
        target_vol = target_vol if target_vol is not None else self.config.target_vol
        
        # 1. Get Top Ranked Stocks from Parquet SignalStore
        from quant.data.signal_store import get_signal_store
        import json
        
        store = get_signal_store()
        
        # Try ranking_v3 first
        signals_df = store.get_signals(
            model_name='ranking_v3',
            start_date=optimization_date,
            end_date=optimization_date,
            limit=top_n
        )
        
        # Fallback to v2 if v3 not available
        if signals_df.empty:
            logger.info("No ranking_v3 signals found, trying ranking_v2...")
            signals_df = store.get_signals(
                model_name='ranking_v2',
                start_date=optimization_date,
                end_date=optimization_date,
                limit=top_n
            )
        
        # Fallback to latest available signals if none for exact date
        if signals_df.empty:
            logger.info("No signals for exact date, trying latest available...")
            signals_df = store.get_latest_signals(model_name='ranking_v3', limit=top_n)
            
        if signals_df.empty:
            signals_df = store.get_latest_signals(model_name='ranking_v2', limit=top_n)
            
        if signals_df.empty:
            logger.warning("No ranked stocks found for optimization.")
            return
        
        # Get tickers and scores from Parquet data
        tickers = signals_df['ticker'].tolist()
        alpha_scores = signals_df['score'].values
        
        # Create a mapping for metadata (for sector/beta constraints)
        ticker_metadata = {}
        for _, row in signals_df.iterrows():
            meta = {}
            if 'metadata_json' in row and row['metadata_json']:
                try:
                    meta = json.loads(row['metadata_json']) if isinstance(row['metadata_json'], str) else row['metadata_json']
                except:
                    pass
            ticker_metadata[row['ticker']] = {
                'score': row['score'],
                'rank': row['rank'],
                'metadata': meta
            }
        
        logger.info(f"Loaded {len(tickers)} signals from Parquet for optimization")
        
        # 2. Get Historical Prices (1 Year) for Covariance from Parquet or yfinance
        from quant.data.parquet_io import ParquetReader, get_data_lake_path
        from datetime import timedelta
        
        # Try Parquet first
        reader = ParquetReader(str(get_data_lake_path()))
        start_date = optimization_date - timedelta(days=365)
        
        prices_df = reader.read_prices(
            tickers=tickers,
            start_date=start_date,
            end_date=optimization_date,
            columns=['date', 'ticker', 'close']
        )
        
        if not prices_df.empty:
            logger.info(f"Loaded {len(prices_df)} price records from Parquet")
            # Remove duplicates before pivot (keep last value for each date/ticker)
            prices_df = prices_df.drop_duplicates(subset=['date', 'ticker'], keep='last')
            # Pivot to Date x Ticker format
            pivot_df = prices_df.pivot(index='date', columns='ticker', values='close')
        else:
            # Fallback to yfinance
            logger.info("No Parquet data, fetching from yfinance...")
            import yfinance as yf
            bulk_history = yf.download(tickers, period="1y", group_by='ticker', threads=True, progress=False)
            
            pivot_df = pd.DataFrame()
            
            if len(tickers) == 1:
                ticker = tickers[0]
                if not bulk_history.empty:
                    close = bulk_history['Close'] if 'Close' in bulk_history.columns else bulk_history
                    pivot_df[ticker] = close
            else:
                for ticker in tickers:
                    try:
                        if isinstance(bulk_history.columns, pd.MultiIndex):
                            if ticker in bulk_history.columns.get_level_values(1):
                                series = bulk_history.xs(ticker, level=1, axis=1)['Close']
                            elif ticker in bulk_history.columns.get_level_values(0):
                                series = bulk_history[ticker]['Close']
                            else:
                                continue
                        else:
                            continue
                        pivot_df[ticker] = series
                    except KeyError:
                        continue

        # Handle missing data (ffill)
        pivot_df = pivot_df.ffill().dropna()
        
        if pivot_df.empty:
            logger.warning("Insufficient price data after cleaning.")
            return
            
        # Ensure alignment between alpha_scores and covariance matrix
        # pivot_df columns are tickers
        valid_tickers = [t for t in tickers if t in pivot_df.columns]
        
        if len(valid_tickers) < len(tickers):
            logger.warning(f"Dropped {len(tickers) - len(valid_tickers)} assets due to missing price history.")
            
        # Filter and reorder alpha scores
        ticker_to_score = {t: ticker_metadata[t]['score'] for t in valid_tickers}
        aligned_alphas = np.array([ticker_to_score[t] for t in valid_tickers])
        
        # Calculate Returns
        returns_df = pivot_df[valid_tickers].pct_change().dropna()
        returns_df_named = returns_df  # Already named by ticker
        
        # --- Prepare Constraints (Phase 8) ---
        sector_mapper = None
        sector_limits = None
        beta_vector = None
        target_beta = None
        
        if sector_constraints or beta_constraints:
            if sector_constraints:
                logger.info("Preparing Sector Constraints...")
                sectors = []
                for ticker in valid_tickers:
                    meta = ticker_metadata.get(ticker, {}).get('metadata', {})
                    sectors.append(meta.get('sector', 'Unknown'))
                
                unique_sectors = sorted(list(set(sectors)))
                n_sectors = len(unique_sectors)
                n_assets = len(valid_tickers)
                
                sector_mapper = np.zeros((n_sectors, n_assets))
                for i, sec_name in enumerate(sectors):
                    row_idx = unique_sectors.index(sec_name)
                    sector_mapper[row_idx, i] = 1.0
                    
                # Limit max 30% per sector
                sector_limits = np.full(n_sectors, 0.30)
                logger.info(f"Sector Constraints: {n_sectors} sectors found.")
                
            if beta_constraints:
                logger.info("Preparing Beta Constraints...")
                betas = []
                for ticker in valid_tickers:
                    meta = ticker_metadata.get(ticker, {}).get('metadata', {})
                    betas.append(meta.get('beta', 1.0))
                
                beta_vector = np.array(betas)
                target_beta = 1.0
                logger.info(f"Beta Constraints: Target Beta = {target_beta}")

        # --- Run Optimizer from Registry ---
        cov_matrix = returns_df_named.cov() * 252  # Annualized
        
        if optimizer == 'hrp':
            logger.info("🔷 Using HRP Optimizer (Registry Plugin)")
            
            hrp_plugin = self.get_optimizer('HRP', {'linkage_method': 'ward'})
            weights_series = hrp_plugin.optimize(returns_df_named, cov_matrix)
            
            if weights_series.empty or weights_series.sum() == 0:
                logger.warning("HRP optimization failed, falling back to MVO.")
                optimizer = 'mvo'
            else:
                final_weights = weights_series.to_dict()
                final_tickers = [t for t in valid_tickers if t in final_weights]
                final_weight_list = [final_weights[t] for t in final_tickers]
                
                self._store_targets(final_tickers, final_weight_list, optimization_date, optimizer='hrp')
                
                logger.info(f"✅ HRP Optimization complete. Top 5 weights:")
                for ticker, w in sorted(zip(final_tickers, final_weight_list), key=lambda x: x[1], reverse=True)[:5]:
                    logger.info(f"   {ticker}: {w:.2%}")
                
                return {'tickers': final_tickers, 'weights': final_weight_list, 'optimizer': 'hrp'}
        
        if optimizer == 'bl':
            logger.info("🔶 Using Black-Litterman Optimizer (Registry Plugin)")
            
            try:
                # Fetch market caps for equilibrium returns
                logger.info(f"Fetching market caps for {len(valid_tickers)} tickers...")
                
                market_caps = {}
                volatilities = {}
                
                for ticker in valid_tickers:
                    try:
                        info = yf.Ticker(ticker).info
                        mkt_cap = info.get('marketCap', 0)
                        if mkt_cap and mkt_cap > 0:
                            market_caps[ticker] = mkt_cap
                        if ticker in returns_df_named.columns:
                            vol = returns_df_named[ticker].std() * np.sqrt(252)
                            volatilities[ticker] = vol if vol > 0 else 0.25
                    except:
                        continue
                
                if len(market_caps) < 5:
                    logger.warning("Insufficient market cap data for Black-Litterman, falling back to HRP.")
                    optimizer = 'hrp'
                else:
                    valid_bl_tickers = [t for t in valid_tickers if t in market_caps and t in volatilities]
                    bl_cov = returns_df_named[valid_bl_tickers].cov() * 252
                    z_scores = {t: ticker_metadata[t]['score'] for t in valid_bl_tickers}
                    
                    bl_plugin = self.get_optimizer('BlackLitterman', {
                        'tau': 0.05,
                        'risk_aversion': 2.5,
                        'ic': 0.05
                    })
                    
                    weights_series = bl_plugin.optimize(
                        returns_df_named[valid_bl_tickers],
                        bl_cov,
                        market_caps=market_caps,
                        z_scores=z_scores,
                        volatilities=volatilities
                    )
                    
                    if weights_series.empty:
                        logger.warning("Black-Litterman failed, falling back to HRP.")
                        optimizer = 'hrp'
                    else:
                        final_weights = weights_series.to_dict()
                        final_tickers = [t for t in valid_tickers if t in final_weights]
                        final_weight_list = [final_weights[t] for t in final_tickers]
                        
                        self._store_targets(final_tickers, final_weight_list, optimization_date, optimizer='bl')
                        
                        logger.info(f"✅ Black-Litterman Optimization complete. Top 5 weights:")
                        for ticker, w in sorted(zip(final_tickers, final_weight_list), key=lambda x: x[1], reverse=True)[:5]:
                            logger.info(f"   {ticker}: {w:.2%}")
                        
                        return {'tickers': final_tickers, 'weights': final_weight_list, 'optimizer': 'bl'}
                        
            except Exception as e:
                logger.error(f"Black-Litterman optimization error: {e}")
                optimizer = 'mvo'
        
        if optimizer == 'kelly':
            logger.info("🚀 Using Kelly Optimizer (Registry Plugin)")
            
            try:
                kelly_plugin = self.get_optimizer('Kelly', {
                    'fractional_kelly': 0.5,
                    'max_leverage': 1.0
                })
                
                expected_returns = pd.Series(aligned_alphas * 0.05, index=valid_tickers)
                
                weights_series = kelly_plugin.optimize(expected_returns, cov_matrix)
                optimal_weights = weights_series.values
                
            except Exception as e:
                logger.error(f"Kelly optimization error: {e}")
                optimizer = 'mvo'
                
        if optimizer == 'mvo':
            logger.info("🔸 Using MVO Optimizer (Registry Plugin)")
            
            mvo_plugin = self.get_optimizer('MVO', {
                'risk_aversion': risk_aversion,
                'max_weight': max_weight,
                'long_only': True
            })
            
            expected_returns = pd.Series(aligned_alphas * 0.0005, index=valid_tickers)
            weights_series = mvo_plugin.optimize(expected_returns, cov_matrix)
            optimal_weights = weights_series.values
        
        # Clean weights
        optimal_weights[optimal_weights < 0.001] = 0
        optimal_weights /= np.sum(optimal_weights)
        
        # --- Tier-2: Volatility Targeting ---
        if target_vol:
            # Phase 9: System Confidence Integration
            confidence_multiplier = self._get_system_confidence()
            adjusted_target_vol = target_vol * confidence_multiplier
            
            logger.info(f"🎯 Applying Volatility Targeting (Base: {target_vol:.1%}, Adj: {adjusted_target_vol:.1%})")
            
            # Convert weights to Series for function
            weights_series = pd.Series(optimal_weights, index=valid_tickers)
            
            # We need prices with tickers as columns
            prices_df_for_vol = pivot_df[valid_tickers]
            
            scaled_weights_series = apply_vol_targeting(
                weights_series, 
                prices_df_for_vol, 
                target_vol=adjusted_target_vol
            )
            
            # Update optimal_weights
            optimal_weights = np.array([scaled_weights_series.get(t, 0.0) for t in valid_tickers])
            
            logger.info(f"   Leverage after Vol Targeting: {np.sum(optimal_weights):.2f}x")

        # 5. Store Results
        self._store_targets(valid_tickers, optimal_weights, optimization_date, optimizer=optimizer)
        
        # Log results
        logger.info("Optimization successful. Top Allocations:")
        
        allocations = sorted(zip(valid_tickers, optimal_weights), key=lambda x: x[1], reverse=True)
        for ticker, weight in allocations[:10]:
            logger.info(f"{ticker}: {weight:.1%}")
            
        return allocations

    def _get_system_confidence(self) -> float:
        """
        Fetches the latest Meta-Labeling confidence score (Sharpe Ratio) from ModelRegistry.
        Returns a multiplier between 0.5 and 1.2.
        """
        try:
            from quant.mlops.registry import ModelRegistry
            registry = ModelRegistry(experiment_name="test_genetic_algo")
            best_run = registry.get_best_run(metric_name="best_fitness", mode="max")
            
            if best_run is None:
                return 1.0
                
            # Get Sharpe Ratio (best_fitness)
            # Note: keys might have 'metrics.' prefix or not depending on how it's returned
            metrics = best_run.filter(regex="metrics.").to_dict()
            sharpe = metrics.get('metrics.best_fitness', 0.0)
            
            logger.info(f"🧠 System Confidence (Sharpe): {sharpe:.2f}")
            
            # Logic: 
            # Sharpe > 1.0 -> High Confidence (1.2x)
            # Sharpe < 0.0 -> Low Confidence (0.5x)
            # Linear interpolation in between
            
            # Base multiplier = 0.5 + 0.5 * Sharpe
            multiplier = 0.5 + 0.5 * sharpe
            
            # Clip between 0.5 and 1.2
            multiplier = max(0.5, min(multiplier, 1.2))
            
            logger.info(f"⚖️  Confidence Multiplier: {multiplier:.2f}x")
            return multiplier
            
        except Exception as e:
            logger.warning(f"Failed to fetch system confidence: {e}")
            return 1.0

    def _store_targets(self, tickers, weights, target_date, optimizer: str = 'mvo'):
        """
        Store portfolio targets to Parquet via SignalStore.
        
        Args:
            tickers: List of ticker symbols
            weights: List of portfolio weights
            target_date: Date for the targets
            optimizer: Optimizer name (hrp, bl, mvo, kelly)
        """
        from quant.data.signal_store import get_signal_store
        
        model_name = f'{optimizer}_v1'  # mvo_v1 or hrp_v1
        
        # Prepare targets DataFrame
        targets_data = []
        for ticker, weight in zip(tickers, weights):
            if weight > 0.001:  # Filter tiny weights
                targets_data.append({
                    'ticker': ticker,
                    'weight': float(weight)
                })
        
        if not targets_data:
            logger.warning("No significant weights to store")
            return
        
        # Write to Parquet
        store = get_signal_store()
        targets_df = pd.DataFrame(targets_data)
        result = store.write_targets(target_date, model_name, targets_df)
        
        logger.info(f"Stored {result['rows_written']} portfolio targets ({model_name}) to Parquet.")

