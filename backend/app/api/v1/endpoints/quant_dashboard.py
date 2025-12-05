from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime

from quant.mlops.registry import ModelRegistry
from quant.risk.var import calculate_component_var
from quant.risk.hedging import calculate_tail_hedge_cost
from quant.execution.algo import VWAPExecution

router = APIRouter()

@router.get("/ml/signals", response_model=Dict[str, Any])
def get_ml_signals():
    """
    Returns the latest ML signal metrics from MLflow.
    """
    try:
        registry = ModelRegistry(experiment_name="test_genetic_algo") # Using the one we created
        best_run = registry.get_best_run(metric_name="best_fitness", mode="max")
        
        if best_run is None:
            return {
                "status": "no_data",
                "metrics": {},
                "params": {}
            }
            
        # Strip prefixes
        metrics = {k.replace("metrics.", ""): v for k, v in best_run.filter(regex="metrics.").to_dict().items()}
        params = {k.replace("params.", ""): v for k, v in best_run.filter(regex="params.").to_dict().items()}

        return {
            "status": "success",
            "run_id": best_run.run_id,
            "metrics": metrics,
            "params": params
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk/metrics", response_model=Dict[str, Any])
def get_risk_metrics(
    portfolio_value: float = 1_000_000,
    volatility: float = 0.15
):
    """
    Returns current risk metrics (VaR, Hedge Cost) based on the latest Optimized Portfolio.
    """
    try:
        from app.core.database import SessionLocal
        from quant.data.models import PortfolioTargets, Security
        
        db = SessionLocal()
        
        # 1. Fetch Latest Portfolio Targets
        latest_date = db.query(PortfolioTargets.date).order_by(PortfolioTargets.date.desc()).first()
        
        if not latest_date:
            # Fallback to dummy if no portfolio exists
            weights = np.array([0.6, 0.4]) 
            tickers = ["DUMMY_A", "DUMMY_B"]
            cov_matrix = np.array([[0.04, 0.01], [0.01, 0.01]])
        else:
            targets = db.query(PortfolioTargets).filter(PortfolioTargets.date == latest_date[0]).all()
            if not targets:
                 weights = np.array([0.6, 0.4])
                 tickers = ["DUMMY_A", "DUMMY_B"]
                 cov_matrix = np.array([[0.04, 0.01], [0.01, 0.01]])
            else:
                # Construct weights array
                weights = np.array([t.weight for t in targets])
                tickers = [t.security.ticker for t in targets]
                
                # Fetch Real Covariance (Simplified: Fetch 1y history for these tickers)
                # For speed, we might just use a proxy or cached cov. 
                # Here we will simulate a cov matrix based on the number of assets to keep it fast for the dashboard
                # In production, this should be cached.
                n = len(weights)
                # Create a synthetic correlation matrix (0.5 correlation)
                corr = np.full((n, n), 0.5)
                np.fill_diagonal(corr, 1.0)
                # Assume 20% vol for all
                vols = np.full(n, 0.20)
                cov_matrix = np.outer(vols, vols) * corr

        # Normalize weights just in case
        if weights.sum() > 0:
            weights = weights / weights.sum()

        # 2. Component VaR
        p_var, m_var, c_var = calculate_component_var(
            weights, cov_matrix, portfolio_value=portfolio_value
        )
        
        # 3. Tail Hedge Cost
        hedge_cost = calculate_tail_hedge_cost(
            portfolio_value=portfolio_value,
            spot_price=400, # Dummy SPY
            volatility=volatility
        )
        
        db.close()
        
        return {
            "var": {
                "portfolio_var": p_var,
                "component_var": c_var.tolist(),
                "weights": weights.tolist(),
                "tickers": tickers # Add tickers for UI
            },
            "hedge": hedge_cost
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/execution/vwap", response_model=Dict[str, Any])
def get_vwap_schedule(
    shares: int = 10000,
    volume: int = 1_000_000,
    volatility: float = 0.02
):
    """
    Returns a VWAP execution schedule and impact cost.
    """
    try:
        algo = VWAPExecution()
        schedule = algo.generate_schedule(total_shares=shares)
        
        impact_bps = algo.estimate_impact_cost(shares, volume, volatility)
        
        return {
            "schedule": schedule.to_dict(orient="records"),
            "impact_cost_bps": impact_bps,
            "total_shares": shares
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/weekly/trades", response_model=List[Dict[str, Any]])
def get_weekly_trades():
    """
    Returns the latest weekly trade list.
    """
    try:
        from app.core.database import SessionLocal
        from quant.reporting.trade_list import TradeListGenerator
        from quant.data.models import PortfolioTargets
        
        db = SessionLocal()
        
        # Find latest date with targets (Try Kelly first, then MVO)
        latest_rec = db.query(PortfolioTargets.date, PortfolioTargets.model_name)\
            .filter(PortfolioTargets.model_name.in_(['kelly_v1', 'mvo_v1']))\
            .order_by(PortfolioTargets.date.desc())\
            .first()
            
        if not latest_rec:
            db.close()
            return []
            
        target_date = latest_rec.date
        model_name = latest_rec.model_name
            
        generator = TradeListGenerator(db)
        df = generator.generate_trade_list(target_date, model_name=model_name)
        db.close()
        
        if df.empty:
            return []
            
        # Convert to dict
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/summary", response_model=Dict[str, Any])
def get_dashboard_summary():
    """
    Aggregated endpoint for Quant Command Center 2.0.
    Returns: Regime, Attribution, and Trades in a single payload.
    """
    try:
        from app.core.database import SessionLocal
        from quant.data.models import PortfolioTargets, ModelSignals
        from quant.reporting.trade_list import TradeListGenerator
        import json
        
        db = SessionLocal()
        
        # ========== 1. REGIME ==========
        registry = ModelRegistry(experiment_name="test_genetic_algo")
        best_run = registry.get_best_run(metric_name="best_fitness", mode="max")
        
        confidence_score = 0.0
        if best_run is not None:
            metrics = best_run.filter(regex="metrics.").to_dict()
            confidence_score = metrics.get('metrics.best_fitness', 0.0)
        
        # Calculate multiplier (same logic as optimizer)
        confidence_multiplier = max(0.5, min(0.5 + 0.5 * confidence_score, 1.2))
        base_target_vol = 0.15
        adjusted_target_vol = base_target_vol * confidence_multiplier
        
        # Determine regime state
        if confidence_multiplier >= 1.0:
            regime_state = "AGGRESSIVE"
        elif confidence_multiplier >= 0.7:
            regime_state = "NEUTRAL"
        else:
            regime_state = "DEFENSIVE"
        
        regime = {
            "state": regime_state,
            "confidence_score": round(confidence_score, 4),
            "confidence_multiplier": round(confidence_multiplier, 2),
            "base_target_vol": base_target_vol,
            "adjusted_target_vol": round(adjusted_target_vol, 4)
        }
        
        # ========== 2. ATTRIBUTION (Factor Exposures) ==========
        # Fetch latest ModelSignals to get factor scores for the portfolio
        latest_target = db.query(PortfolioTargets)\
            .filter(PortfolioTargets.model_name.in_(['kelly_v1', 'mvo_v1']))\
            .order_by(PortfolioTargets.date.desc())\
            .first()
        
        attribution = {
            "value": 0.0,
            "momentum": 0.0,
            "quality": 0.0,
            "low_risk": 0.0,
            "sentiment": 0.0
        }
        
        if latest_target:
            target_date = latest_target.date
            targets = db.query(PortfolioTargets)\
                .filter(PortfolioTargets.date == target_date)\
                .filter(PortfolioTargets.model_name.in_(['kelly_v1', 'mvo_v1']))\
                .all()
            
            # Get corresponding signals
            sids = [t.sid for t in targets]
            weights = {t.sid: t.weight for t in targets}
            
            signals = db.query(ModelSignals)\
                .filter(ModelSignals.date == target_date)\
                .filter(ModelSignals.model_name == 'ranking_v3')\
                .filter(ModelSignals.sid.in_(sids))\
                .all()
            
            # Fallback to v2
            if not signals:
                signals = db.query(ModelSignals)\
                    .filter(ModelSignals.date == target_date)\
                    .filter(ModelSignals.model_name == 'ranking_v2')\
                    .filter(ModelSignals.sid.in_(sids))\
                    .all()
            
            # Weight-average the factor scores
            total_weight = sum(weights.values())
            if signals and total_weight > 0:
                value_sum = momentum_sum = quality_sum = low_risk_sum = sentiment_sum = 0.0
                
                for sig in signals:
                    w = weights.get(sig.sid, 0.0) / total_weight
                    try:
                        meta = json.loads(sig.metadata_json) if sig.metadata_json else {}
                        # Use actual keys from metadata
                        value_sum += w * meta.get('upside', meta.get('valuation_composite', 0.0))
                        momentum_sum += w * meta.get('vsm', 0.0)
                        quality_sum += w * meta.get('qmj', meta.get('roe', 0.0))
                        low_risk_sum += w * meta.get('bab', 0.0)
                        sentiment_sum += w * meta.get('revisions', meta.get('sentiment', 0.0))
                    except:
                        pass
                
                attribution = {
                    "value": round(value_sum, 2),
                    "momentum": round(momentum_sum, 2),
                    "quality": round(quality_sum, 2),
                    "low_risk": round(low_risk_sum, 2),
                    "sentiment": round(sentiment_sum, 2)
                }
        
        # ========== 3. TRADES ==========
        trades = []
        if latest_target:
            target_date = latest_target.date
            generator = TradeListGenerator(db)
            df = generator.generate_trade_list(target_date, model_name='kelly_v1')
            
            if df.empty:
                df = generator.generate_trade_list(target_date, model_name='mvo_v1')
            
            if not df.empty:
                # Build ticker -> signal score map first
                from quant.data.models import Security
                ticker_to_score = {}
                for sig in signals:
                    ticker = sig.security.ticker if sig.security else None
                    if ticker:
                        ticker_to_score[ticker] = sig.score
                
                # Add alpha score and reason
                for _, row in df.iterrows():
                    ticker = row['Ticker']
                    alpha_score = ticker_to_score.get(ticker, 0.0)
                    
                    # Determine conviction level
                    if alpha_score >= 2.0:
                        conviction = "High"
                    elif alpha_score >= 1.0:
                        conviction = "Medium"
                    else:
                        conviction = "Low"
                    
                    # Reason for sizing
                    reason = "Standard Sizing"
                    if confidence_multiplier < 1.0:
                        reason = f"Vol Target Reduced ({int((1-confidence_multiplier)*100)}%)"
                    
                    trades.append({
                        "ticker": row['Ticker'],
                        "name": row.get('Name', row['Ticker']),
                        "sector": row.get('Sector', 'Unknown'),
                        "alpha_score": round(alpha_score, 2),
                        "conviction": conviction,
                        "raw_weight": round(row['Weight'] / confidence_multiplier, 4) if confidence_multiplier > 0 else row['Weight'],
                        "final_weight": round(row['Weight'], 4),
                        "shares": row.get('Shares', 0),
                        "value": round(row.get('Value', 0), 2),
                        "reason": reason
                    })
        
        db.close()
        
        return {
            "regime": regime,
            "attribution": attribution,
            "trades": trades,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest/run", response_model=Dict[str, Any])
def run_backtest(
    start_year: int = Query(2021, description="Start year for backtest"),
    end_year: int = Query(2024, description="End year for backtest"),
    top_n: int = Query(50, description="Number of top stocks to hold"),
    benchmarks: str = Query("SPY,QQQ", description="Comma-separated benchmark tickers")
):
    """
    Run walk-forward backtest and return equity curves with benchmark comparison.
    """
    try:
        from app.core.database import SessionLocal
        from quant.data.models import Security, MarketDataDaily
        from quant.backtest.factor_engine import (
            PointInTimeFactorEngine,
            run_factor_backtest,
            calculate_performance_metrics
        )
        import yfinance as yf
        
        db = SessionLocal()
        
        # 1. Load price data
        data_start = f"{start_year - 1}-01-01"
        data_end = f"{end_year}-12-31"
        
        query = db.query(
            MarketDataDaily.date,
            Security.ticker,
            MarketDataDaily.close
        ).join(Security, Security.sid == MarketDataDaily.sid)\
         .filter(MarketDataDaily.date >= data_start)\
         .filter(MarketDataDaily.date <= data_end)\
         .filter(MarketDataDaily.close.isnot(None))
        
        data = query.all()
        
        if not data:
            db.close()
            return {"error": "No price data available. Run download_history.py first."}
        
        prices_df = pd.DataFrame(data, columns=['date', 'ticker', 'close'])
        prices_df['date'] = pd.to_datetime(prices_df['date'])
        
        # 2. Run backtest
        rebalance_dates = pd.date_range(
            start=f"{start_year}-01-01",
            end=f"{end_year}-12-31",
            freq='ME'
        ).tolist()
        
        equity_curve = run_factor_backtest(prices_df, rebalance_dates, top_n=top_n)
        
        if equity_curve.empty:
            db.close()
            return {"error": "Backtest failed - insufficient data"}
        
        # 3. Get benchmark data
        benchmark_list = [b.strip() for b in benchmarks.split(',')]
        benchmark_curves = {}
        
        for ticker in benchmark_list:
            try:
                bench_df = yf.download(
                    ticker,
                    start=f"{start_year}-01-01",
                    end=f"{end_year}-12-31",
                    progress=False
                )
                if not bench_df.empty:
                    # Normalize to start at 1.0
                    closes = bench_df['Close'].values
                    normalized = closes / closes[0]
                    # Resample to monthly
                    bench_monthly = bench_df['Close'].resample('ME').last()
                    bench_normalized = bench_monthly / bench_monthly.iloc[0]
                    benchmark_curves[ticker] = [
                        {"date": d.strftime('%Y-%m-%d'), "value": float(v)}
                        for d, v in zip(bench_normalized.index, bench_normalized.values)
                    ]
            except Exception as e:
                print(f"Failed to fetch {ticker}: {e}")
        
        # 4. Calculate metrics (now includes gross/net Sharpe, turnover, costs)
        strategy_metrics = calculate_performance_metrics(equity_curve)
        
        # Calculate benchmark metrics
        benchmark_metrics = {}
        for ticker, curve in benchmark_curves.items():
            if len(curve) >= 2:
                values = [p['value'] for p in curve]
                total_return = values[-1] / values[0] - 1
                n_months = len(values) - 1
                annual_return = (1 + total_return) ** (12 / n_months) - 1 if n_months > 0 else 0
                returns = pd.Series(values).pct_change().dropna()
                volatility = returns.std() * np.sqrt(12)
                sharpe = annual_return / volatility if volatility > 0 else 0
                
                # Max drawdown
                cummax = pd.Series(values).cummax()
                drawdown = (pd.Series(values) - cummax) / cummax
                max_dd = drawdown.min()
                
                benchmark_metrics[ticker] = {
                    "total_return": round(total_return * 100, 2),
                    "annual_return": round(annual_return * 100, 2),
                    "volatility": round(volatility * 100, 2),
                    "sharpe_ratio": round(sharpe, 2),
                    "max_drawdown": round(max_dd * 100, 2)
                }
        
        # 5. Format strategy equity curves (net and gross)
        strategy_curve = [
            {"date": row['date'].strftime('%Y-%m-%d'), "value": float(row['portfolio_value'])}
            for _, row in equity_curve.iterrows()
        ]
        
        gross_curve = []
        if 'portfolio_value_gross' in equity_curve.columns:
            gross_curve = [
                {"date": row['date'].strftime('%Y-%m-%d'), "value": float(row['portfolio_value_gross'])}
                for _, row in equity_curve.iterrows()
            ]
        
        # 6. Calculate drawdown series
        values = equity_curve['portfolio_value'].values
        cummax = pd.Series(values).cummax()
        drawdowns = ((pd.Series(values) - cummax) / cummax * 100).tolist()
        drawdown_curve = [
            {"date": equity_curve.iloc[i]['date'].strftime('%Y-%m-%d'), "drawdown": drawdowns[i]}
            for i in range(len(drawdowns))
        ]
        
        # 7. Turnover series
        turnover_curve = []
        if 'turnover' in equity_curve.columns:
            turnover_curve = [
                {"date": row['date'].strftime('%Y-%m-%d'), "turnover": round(row['turnover'] * 100, 1)}
                for _, row in equity_curve.iterrows()
            ]
        
        # 8. Rolling Sharpe (12-month)
        from quant.backtest.factor_engine import (
            calculate_rolling_sharpe, calculate_monthly_returns,
            bootstrap_sharpe_ci, calculate_subperiod_analysis
        )
        rolling_sharpe = calculate_rolling_sharpe(equity_curve, window=12)
        
        # 9. Monthly returns for heatmap
        monthly_returns = calculate_monthly_returns(equity_curve)
        
        # 10. Bootstrap Sharpe CI (95%)
        bootstrap_result = bootstrap_sharpe_ci(equity_curve, n_iterations=500)
        
        # 11. Subperiod Analysis (yearly breakdown)
        subperiod = calculate_subperiod_analysis(equity_curve)
        
        db.close()
        
        return {
            "strategy": {
                "name": f"Factor Model (Top {top_n})",
                "curve": strategy_curve,
                "curve_gross": gross_curve,
                "metrics": {
                    "total_return": round(strategy_metrics.get('total_return', 0) * 100, 2),
                    "annual_return": round(strategy_metrics.get('annual_return', 0) * 100, 2),
                    "volatility": round(strategy_metrics.get('volatility', 0) * 100, 2),
                    "sharpe_ratio": round(strategy_metrics.get('sharpe_ratio', 0), 2),
                    "sharpe_ratio_gross": round(strategy_metrics.get('sharpe_ratio_gross', 0) or 0, 2),
                    "max_drawdown": round(strategy_metrics.get('max_drawdown', 0) * 100, 2),
                    "avg_turnover": round(strategy_metrics.get('avg_turnover', 0) * 100, 1),
                    "total_transaction_costs": round(strategy_metrics.get('total_transaction_costs', 0) * 100, 2),
                    "total_stop_losses": strategy_metrics.get('total_stop_losses', 0)
                }
            },
            "benchmarks": {
                ticker: {
                    "curve": curve,
                    "metrics": benchmark_metrics.get(ticker, {})
                }
                for ticker, curve in benchmark_curves.items()
            },
            "drawdown": drawdown_curve,
            "turnover": turnover_curve,
            "rolling_sharpe": rolling_sharpe,
            "monthly_returns": monthly_returns,
            "robustness": {
                "bootstrap_sharpe": {
                    "mean": round(bootstrap_result.get('sharpe_mean', 0) or 0, 2),
                    "lower_95": round(bootstrap_result.get('sharpe_lower', 0) or 0, 2),
                    "upper_95": round(bootstrap_result.get('sharpe_upper', 0) or 0, 2),
                    "std": round(bootstrap_result.get('sharpe_std', 0) or 0, 2)
                },
                "subperiod": subperiod
            },
            "risk_controls": {
                "max_position": "5%",
                "sector_cap": "30%",
                "stop_loss": "-15%",
                "min_holding": "2 months"
            },
            "period": f"{start_year}-{end_year}",
            "transaction_cost_bps": 10,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



