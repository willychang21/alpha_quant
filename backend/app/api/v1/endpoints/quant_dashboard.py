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
    Now uses Parquet SignalStore for portfolio targets.
    """
    try:
        from quant.data.signal_store import get_signal_store
        
        store = get_signal_store()
        
        # 1. Fetch Latest Portfolio Targets from Parquet
        # Try different optimizer models
        targets_df = None
        for model in ['kelly_v1', 'mvo_v1', 'hrp_v1']:
            targets_df = store.get_latest_targets(model_name=model)
            if not targets_df.empty:
                break
        
        if targets_df is None or targets_df.empty:
            # Fallback to dummy if no portfolio exists
            weights = np.array([0.6, 0.4]) 
            tickers = ["DUMMY_A", "DUMMY_B"]
            cov_matrix = np.array([[0.04, 0.01], [0.01, 0.01]])
        else:
            # Construct weights array from Parquet data
            weights = targets_df['weight'].values
            tickers = targets_df['ticker'].tolist()
            
            # Simulate a cov matrix based on the number of assets
            n = len(weights)
            corr = np.full((n, n), 0.5)
            np.fill_diagonal(corr, 1.0)
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
    Returns the latest weekly trade list from Parquet SignalStore.
    """
    try:
        from quant.data.signal_store import get_signal_store
        
        store = get_signal_store()
        
        # Find latest targets from Parquet
        targets_df = None
        model_name = None
        for model in ['kelly_v1', 'mvo_v1', 'hrp_v1']:
            targets_df = store.get_latest_targets(model_name=model)
            if not targets_df.empty:
                model_name = model
                break
        
        if targets_df is None or targets_df.empty:
            return []
        
        # Convert to trade list format
        trades = []
        for _, row in targets_df.iterrows():
            trades.append({
                'Ticker': row['ticker'],
                'Weight': float(row['weight']),
                'Action': 'BUY' if row['weight'] > 0 else 'HOLD',
                'Model': model_name
            })
        
        return trades
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/summary", response_model=Dict[str, Any])
def get_dashboard_summary():
    """
    Aggregated endpoint for Quant Command Center 2.0.
    Returns: Regime, Attribution, and Trades in a single payload.
    Now uses Parquet SignalStore.
    """
    try:
        from quant.data.signal_store import get_signal_store
        import json
        
        store = get_signal_store()
        
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
        
        # ========== 2. ATTRIBUTION (Factor Exposures) from Parquet ==========
        attribution = {
            "value": 0.0,
            "momentum": 0.0,
            "quality": 0.0,
            "low_risk": 0.0,
            "sentiment": 0.0
        }
        
        # Get latest targets from Parquet
        targets_df = None
        for model in ['kelly_v1', 'mvo_v1']:
            targets_df = store.get_latest_targets(model_name=model)
            if not targets_df.empty:
                break
        
        # Get latest signals from Parquet
        signals_df = store.get_latest_signals(model_name='ranking_v3', limit=500)
        if signals_df.empty:
            signals_df = store.get_latest_signals(model_name='ranking_v2', limit=500)
        
        if targets_df is not None and not targets_df.empty and not signals_df.empty:
            # Build ticker -> weight mapping
            weights = dict(zip(targets_df['ticker'], targets_df['weight']))
            total_weight = sum(weights.values())
            
            # Calculate weighted attribution
            value_sum = momentum_sum = quality_sum = low_risk_sum = sentiment_sum = 0.0
            
            for _, sig in signals_df.iterrows():
                ticker = sig['ticker']
                if ticker not in weights:
                    continue
                    
                w = weights[ticker] / total_weight if total_weight > 0 else 0
                
                try:
                    meta = {}
                    if 'metadata_json' in sig and sig['metadata_json']:
                        meta = json.loads(sig['metadata_json']) if isinstance(sig['metadata_json'], str) else sig['metadata_json']
                    
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
        
        # ========== 3. TRADES from Parquet ==========
        trades = []
        if targets_df is not None and not targets_df.empty:
            # Build ticker -> signal score map
            ticker_to_score = {}
            if not signals_df.empty:
                ticker_to_score = dict(zip(signals_df['ticker'], signals_df['score']))
            
            # Build trades from targets
            for _, row in targets_df.iterrows():
                ticker = row['ticker']
                weight = row['weight']
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
                    "ticker": ticker,
                    "name": ticker,
                    "sector": "Unknown",
                    "alpha_score": round(alpha_score, 2),
                    "conviction": conviction,
                    "raw_weight": round(weight / confidence_multiplier, 4) if confidence_multiplier > 0 else weight,
                    "final_weight": round(weight, 4),
                    "shares": 0,
                    "value": 0,
                    "reason": reason
                })
        
        return {
            "regime": regime,
            "attribution": attribution,
            "trades": trades,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sector-rotation", response_model=Dict[str, Any])
def get_sector_rotation():
    """
    Returns sector rotation analysis using RRG (Relative Rotation Graph) methodology.
    Shows which sectors are receiving capital inflows (Improving) vs outflows (Weakening).
    """
    try:
        from quant.features.capital_flow.sector_rotation import SectorRotationAnalyzer
        
        analyzer = SectorRotationAnalyzer()
        results = analyzer.analyze_all_sectors(period='6mo')
        
        if not results:
            return {
                "status": "no_data",
                "sectors": [],
                "quadrant_summary": {}
            }
        
        # Format results for frontend
        sectors = []
        quadrant_counts = {'Leading': 0, 'Improving': 0, 'Weakening': 0, 'Lagging': 0}
        
        for symbol, result in results.items():
            sectors.append({
                'symbol': result.symbol,
                'name': result.sector_name,
                'rs_ratio': round(result.rs_ratio, 2),
                'rs_momentum': round(result.rs_momentum, 2),
                'quadrant': result.quadrant,
                'previous_quadrant': result.previous_quadrant,
                'transition_signal': result.transition_signal,
                'score': analyzer.get_quadrant_score(result.quadrant)
            })
            quadrant_counts[result.quadrant] = quadrant_counts.get(result.quadrant, 0) + 1
        
        # Sort by RS Momentum (strongest momentum first)
        sectors.sort(key=lambda x: x['rs_momentum'], reverse=True)
        
        # Identify capital flow direction
        improving_sectors = [s for s in sectors if s['quadrant'] == 'Improving']
        leading_sectors = [s for s in sectors if s['quadrant'] == 'Leading']
        weakening_sectors = [s for s in sectors if s['quadrant'] == 'Weakening']
        lagging_sectors = [s for s in sectors if s['quadrant'] == 'Lagging']
        
        return {
            "status": "success",
            "sectors": sectors,
            "quadrant_summary": quadrant_counts,
            "capital_inflow": [s['name'] for s in improving_sectors + leading_sectors],
            "capital_outflow": [s['name'] for s in weakening_sectors + lagging_sectors],
            "hot_sectors": [s['name'] for s in improving_sectors],  # Early signal
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sector-stocks/{sector_name}", response_model=Dict[str, Any])
def get_sector_stocks(sector_name: str):
    """
    Returns stocks within a specific sector with their individual money flow scores.
    Calculates MFI/OBV on-the-fly if not present in stored metadata.
    """
    try:
        from quant.data.signal_store import get_signal_store
        from quant.features.capital_flow.money_flow import MoneyFlowCalculator
        import json
        import yfinance as yf
        
        store = get_signal_store()
        mfc = MoneyFlowCalculator(mfi_period=14)
        
        # Get latest signals
        signals_df = store.get_latest_signals(model_name='ranking_v3', limit=500)
        if signals_df.empty:
            signals_df = store.get_latest_signals(model_name='ranking_v2', limit=500)
        
        if signals_df.empty:
            return {
                "status": "no_data",
                "sector": sector_name,
                "stocks": []
            }
        
        # Filter by sector and collect tickers that need MFI/OBV calculation
        sector_stocks = []
        tickers_need_calc = []
        
        for _, row in signals_df.iterrows():
            try:
                meta = {}
                if 'metadata_json' in row and row['metadata_json']:
                    meta = json.loads(row['metadata_json']) if isinstance(row['metadata_json'], str) else row['metadata_json']
                
                stock_sector = meta.get('sector', 'Unknown')
                
                # Match sector name (case-insensitive, partial match)
                if sector_name.lower() in stock_sector.lower() or stock_sector.lower() in sector_name.lower():
                    ticker = row['ticker']
                    mfi = meta.get('mfi')
                    obv_zscore = meta.get('obv_zscore')
                    money_flow = meta.get('money_flow')
                    
                    # Check if we need to calculate on-the-fly
                    needs_calc = (mfi is None or mfi == 50) and (obv_zscore is None or obv_zscore == 0)
                    
                    sector_stocks.append({
                        'ticker': ticker,
                        'score': row['score'],
                        'rank': row['rank'],
                        'meta': meta,
                        'needs_calc': needs_calc
                    })
                    
                    if needs_calc:
                        tickers_need_calc.append(ticker)
            except Exception:
                continue
        
        # Bulk fetch price data for tickers that need calculation (max 20 to avoid timeout)
        mfi_obv_cache = {}
        if tickers_need_calc:
            calc_tickers = tickers_need_calc[:20]  # Limit to avoid timeout
            try:
                bulk_data = yf.download(calc_tickers, period='3mo', group_by='ticker', progress=False, threads=True)
                
                for ticker in calc_tickers:
                    try:
                        if len(calc_tickers) > 1:
                            if ticker in bulk_data.columns.get_level_values(0):
                                hist = bulk_data[ticker].dropna()
                            else:
                                continue
                        else:
                            hist = bulk_data.dropna()
                        
                        if hist.empty or len(hist) < 20:
                            continue
                        
                        # Handle MultiIndex columns from yfinance
                        if isinstance(hist.columns, pd.MultiIndex):
                            hist.columns = hist.columns.get_level_values(0)
                        
                        # Calculate MFI
                        mfi_series = mfc.calculate_mfi(
                            hist['High'], hist['Low'], hist['Close'], hist['Volume']
                        )
                        mfi_val = float(mfi_series.iloc[-1]) if not mfi_series.empty else 50.0
                        
                        # Calculate OBV Z-score
                        obv_series = mfc.calculate_obv(hist['Close'], hist['Volume'])
                        obv_norm = mfc.normalize_obv(obv_series)
                        obv_z = float(obv_norm.iloc[-1]) if not obv_norm.empty else 0.0
                        
                        # Calculate money flow score from MFI and OBV
                        # MFI < 30 = oversold (bullish), > 70 = overbought (bearish)
                        # OBV Z > 0 = accumulation, < 0 = distribution
                        mfi_signal = (50 - mfi_val) / 50  # -1 to 1, positive when oversold
                        money_flow_calc = (mfi_signal * 0.4) + (obv_z * 0.6)
                        
                        mfi_obv_cache[ticker] = {
                            'mfi': mfi_val,
                            'obv_zscore': obv_z,
                            'money_flow': money_flow_calc
                        }
                    except Exception:
                        continue
            except Exception:
                pass
        
        # Build final stock list
        stocks = []
        for item in sector_stocks:
            ticker = item['ticker']
            meta = item['meta']
            
            # Use cached calculation if available, otherwise use stored values
            if ticker in mfi_obv_cache:
                mfi = mfi_obv_cache[ticker]['mfi']
                obv_zscore = mfi_obv_cache[ticker]['obv_zscore']
                money_flow = mfi_obv_cache[ticker]['money_flow']
            else:
                mfi = meta.get('mfi', 50) or 50
                obv_zscore = meta.get('obv_zscore', 0) or 0
                money_flow = meta.get('money_flow', 0) or 0
            
            capital_flow = meta.get('capital_flow', 0) or 0
            sector_flow = meta.get('sector_flow', 0) or 0
            
            # Determine flow signal
            if money_flow > 0.5:
                flow_signal = 'Strong Inflow'
            elif money_flow > 0:
                flow_signal = 'Inflow'
            elif money_flow < -0.5:
                flow_signal = 'Strong Outflow'
            elif money_flow < 0:
                flow_signal = 'Outflow'
            else:
                flow_signal = 'Neutral'
            
            stocks.append({
                'ticker': ticker,
                'score': round(item['score'], 2),
                'rank': int(item['rank']),
                'capital_flow': round(capital_flow, 2),
                'money_flow': round(money_flow, 2),
                'sector_flow': round(sector_flow, 2),
                'mfi': round(mfi, 1),
                'obv_zscore': round(obv_zscore, 2),
                'flow_signal': flow_signal,
                'vsm': round(meta.get('vsm', 0) or 0, 2),
                'sentiment': round(meta.get('sentiment', 0) or 0, 2)
            })
        
        # Sort by money_flow (strongest inflow first)
        stocks.sort(key=lambda x: x['money_flow'], reverse=True)
        
        # Summary stats
        avg_money_flow = sum(s['money_flow'] for s in stocks) / len(stocks) if stocks else 0
        inflow_count = len([s for s in stocks if s['money_flow'] > 0])
        outflow_count = len([s for s in stocks if s['money_flow'] < 0])
        
        return {
            "status": "success",
            "sector": sector_name,
            "stock_count": len(stocks),
            "avg_money_flow": round(avg_money_flow, 2),
            "inflow_count": inflow_count,
            "outflow_count": outflow_count,
            "stocks": stocks,
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
    Now uses Parquet data lake for price data.
    """
    try:
        from quant.data.parquet_io import ParquetReader, get_data_lake_path
        from quant.backtest.factor_engine import (
            PointInTimeFactorEngine,
            run_factor_backtest,
            calculate_performance_metrics
        )
        import yfinance as yf
        from datetime import date
        
        # 1. Load price data from Parquet data lake
        reader = ParquetReader(str(get_data_lake_path()))
        
        start_date = date(start_year - 1, 1, 1)
        end_date = date(end_year, 12, 31)
        
        prices_df = reader.read_prices(
            start_date=start_date,
            end_date=end_date,
            columns=['date', 'ticker', 'close']
        )
        
        if prices_df.empty:
            return {"error": "No price data available in data lake. Run startup_catch_up.py first."}
        
        # Convert date column to datetime for backtest engine
        prices_df['date'] = pd.to_datetime(prices_df['date'])
        
        # 2. Run backtest
        rebalance_dates = pd.date_range(
            start=f"{start_year}-01-01",
            end=f"{end_year}-12-31",
            freq='ME'
        ).tolist()
        
        equity_curve, trade_logs = run_factor_backtest(prices_df, rebalance_dates, top_n=top_n)
        
        if equity_curve.empty:
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
            "trades": trade_logs,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



