from sqlalchemy.orm import Session
from quant.data.models import Security, ModelSignals
from quant.features.fundamental import ValueFactor, QualityFactor, GrowthFactor
from quant.features.technical import MomentumFactor
from quant.features.volatility import VolatilityScaledMomentum
from quant.features.beta import BettingAgainstBeta
from quant.features.quality import QualityMinusJunk
from quant.features.pipeline import FactorPipeline
from quant.valuation.orchestrator import ValuationOrchestrator
from core.adapters.yfinance_provider import YFinanceProvider
# Tier-1 Upgrades
from quant.features.accruals import AccrualsAnomaly
from quant.features.ivol import IdiosyncraticVolatility, AmihudIlliquidity
from quant.features.pead import PostEarningsAnnouncementDrift, EarningsMomentum
from quant.features.sentiment import NewsSentimentFactor
from quant.regime.hmm import RegimeDetector, DynamicFactorWeights
from datetime import date
import pandas as pd
import numpy as np
import json
import logging
import asyncio

logger = logging.getLogger(__name__)

class RankingEngine:
    def __init__(self, db: Session):
        self.db = db
        self.market_data = YFinanceProvider()
        self.valuation_orchestrator = ValuationOrchestrator()
        
        # Initialize Feature Generators
        self.value_gen = ValueFactor()
        # self.quality_gen = QualityFactor() # Replaced by QMJ
        self.growth_gen = GrowthFactor()
        self.momentum_gen = MomentumFactor()
        
        # New Factors
        self.vsm_gen = VolatilityScaledMomentum()
        self.bab_gen = BettingAgainstBeta()
        self.qmj_gen = QualityMinusJunk()
        
        # Tier-1 Alpha Factors
        self.accruals_gen = AccrualsAnomaly()
        self.ivol_gen = IdiosyncraticVolatility(lookback=21)
        self.illiq_gen = AmihudIlliquidity(lookback=21)
        
        # P3: PEAD (Post-Earnings Announcement Drift)
        self.pead_gen = PostEarningsAnnouncementDrift(lookback_days=60, decay_halflife=30)
        self.earnings_mom_gen = EarningsMomentum()
        
        # P4: NLP Sentiment
        self.sentiment_gen = NewsSentimentFactor(lookback_days=7, max_articles=5)
        
        # Regime Detection
        self.regime_detector = RegimeDetector(n_states=2, lookback=252)
        self.current_regime = "Unknown"

    async def run_ranking(self, ranking_date: date):
        """
        Run the ranking process for the given date.
        """
        securities = self.db.query(Security).filter(Security.active == True).all()
        
        data = []
        
        # 1. Collect Raw Data
        import yfinance as yf
        
        # Fetch SPY Benchmark Data first
        logger.info("Fetching benchmark (SPY) data...")
        try:
            spy_history = yf.download("SPY", period="2y", progress=False)
            logger.info(f"SPY Raw Shape: {spy_history.shape}")
            logger.info(f"SPY Raw Columns: {spy_history.columns}")
            
            # Extract Close Series
            if isinstance(spy_history.columns, pd.MultiIndex):
                # Structure is usually (Price, Ticker) e.g. ('Close', 'SPY')
                try:
                    spy_history = spy_history['Close']['SPY']
                except KeyError:
                    # Try alternative structure or just take first column
                    logger.warning("Could not find ['Close']['SPY'], taking iloc[:, 0]")
                    spy_history = spy_history.iloc[:, 0]
            elif 'Close' in spy_history.columns:
                spy_history = spy_history['Close']
            
            # Ensure it's a Series
            if isinstance(spy_history, pd.DataFrame):
                spy_history = spy_history.squeeze()
            
            logger.info(f"SPY Extracted Shape: {spy_history.shape}")
            
            if spy_history.empty:
                logger.warning("SPY history is empty.")
            else:
                # --- Tier-1: HMM Regime Detection ---
                spy_returns = spy_history.pct_change().dropna()
                if len(spy_returns) > 100:
                    self.regime_detector.fit(spy_returns)
                    self.current_regime, regime_probs = self.regime_detector.predict_regime(spy_returns.iloc[-30:])
                    logger.info(f"🎯 Current Market Regime: {self.current_regime}")
                    logger.info(f"   Regime Probabilities: {regime_probs}")
                    
        except Exception as e:
            logger.error(f"Failed to fetch SPY data: {e}")
            spy_history = pd.Series()

        all_tickers = [sec.ticker for sec in securities]
        logger.info(f"Processing {len(all_tickers)} tickers in chunks...")
        
        chunk_size = 50
        for i in range(0, len(securities), chunk_size):
            chunk_securities = securities[i:i + chunk_size]
            chunk_tickers = [s.ticker for s in chunk_securities]
            logger.info(f"Processing chunk {i//chunk_size + 1}: {len(chunk_tickers)} tickers")
            
            # Bulk download for chunk
            try:
                bulk_history = yf.download(chunk_tickers, period="2y", group_by='ticker', threads=True)
            except Exception as e:
                logger.error(f"Bulk download failed for chunk {i}: {e}")
                bulk_history = pd.DataFrame()

            semaphore = asyncio.Semaphore(1) # Reduce concurrency to avoid rate limits

            async def process_ticker(sec):
                async with semaphore:
                    await asyncio.sleep(1.0) # Add delay
                    try:
                        # Extract History
                        history = pd.DataFrame()
                        if not bulk_history.empty:
                            if len(chunk_tickers) > 1:
                                try:
                                    if sec.ticker in bulk_history.columns.get_level_values(0):
                                         history = bulk_history[sec.ticker].copy()
                                except KeyError:
                                    pass
                            else:
                                history = bulk_history.copy()
                        
                        # Fallback: Individual fetch if missing
                        if history.empty:
                            try:
                                history = await self.market_data.get_history(sec.ticker)
                            except:
                                pass
                                
                        if history.empty:
                            return None
                            
                        # Fetch Info (Fundamentals) - Optional
                        ticker_data = {}
                        try:
                            ticker_data = await self.market_data.get_ticker_data(sec.ticker)
                        except:
                            pass
                        
                        # --- Compute Factors ---
                        
                        # 1. Volatility Scaled Momentum
                        vsm_score = self.vsm_gen.compute(history)
                        if isinstance(vsm_score, pd.Series): vsm_score = float(vsm_score.iloc[-1]) if not vsm_score.empty else 0.0
                        
                        # 2. Betting Against Beta
                        bab_score = self.bab_gen.compute(history, benchmark_history=spy_history)
                        if isinstance(bab_score, pd.Series): bab_score = float(bab_score.iloc[-1]) if not bab_score.empty else 0.0
                        
                        # 3. Quality Minus Junk (Composite)
                        # QMJ returns a Series of dicts or just a Series with one dict?
                        # My implementation returns pd.Series([{'roe': ...}])
                        qmj_raw = self.qmj_gen.compute(history, ticker_data.get('info', {}))
                        qmj_components = {}
                        if not qmj_raw.empty and isinstance(qmj_raw.iloc[-1], dict):
                            qmj_components = qmj_raw.iloc[-1]

                        # 4. Valuation Upside
                        upside = 0.0
                        try:
                            # Map YFinance keys to ValuationOrchestrator keys
                            valuation_data = {
                                'info': ticker_data.get('info', {}),
                                'income_stmt': ticker_data.get('income'),
                                'balance_sheet': ticker_data.get('balance'),
                                'cashflow': ticker_data.get('cashflow')
                            }
                            
                            # get_valuation is synchronous
                            val_res = self.valuation_orchestrator.get_valuation(sec.ticker, valuation_data)
                            if val_res and val_res.upside:
                                upside = val_res.upside
                                # Trap for suspicious value
                                if abs(upside - 10.741022879973459) < 1e-6:
                                    logger.critical(f"SUSPICIOUS UPSIDE DETECTED for {sec.ticker}: {upside}")
                                    logger.critical(f"Price: {ticker_data.get('info', {}).get('currentPrice')}")
                                    logger.critical(f"Fair Value: {val_res.fair_value}")
                                    logger.critical(f"Model: {val_res.model_name}")
                        except Exception as e:
                            logger.warning(f"Valuation failed for {sec.ticker}: {e}")
                        
                        # Legacy Factors (for reference or fallback)
                        mom_score = self.momentum_gen.compute(history)
                        if isinstance(mom_score, pd.Series): mom_score = float(mom_score.iloc[-1]) if not mom_score.empty else 0.0
                        
                        # 5. PEAD (Post-Earnings Announcement Drift) - P3
                        pead_score = 0.0
                        try:
                            pead_raw = self.pead_gen.compute(history, ticker_data.get('info', {}), ticker=sec.ticker)
                            if isinstance(pead_raw, pd.Series) and not pead_raw.empty:
                                pead_score = float(pead_raw.iloc[-1])
                        except Exception as e:
                            logger.debug(f"PEAD calculation failed for {sec.ticker}: {e}")
                        
                        # 6. News Sentiment (P4)
                        sentiment_score = 0.0
                        try:
                            sentiment_raw = self.sentiment_gen.compute(history, ticker_data.get('info', {}), ticker=sec.ticker)
                            if isinstance(sentiment_raw, pd.Series) and not sentiment_raw.empty:
                                sentiment_score = float(sentiment_raw.iloc[-1])
                        except Exception as e:
                            logger.debug(f"Sentiment calculation failed for {sec.ticker}: {e}")
                        
                        # Construct Result Dict
                        result = {
                            'sid': sec.sid,
                            'ticker': sec.ticker,
                            'sector': ticker_data.get('info', {}).get('sector', 'Unknown'),
                            'momentum': mom_score,  # Legacy
                            'volatility_scaled_momentum': vsm_score,
                            'betting_against_beta': bab_score,
                            'upside': upside,
                            'pead': pead_score,  # P3: PEAD
                            'sentiment': sentiment_score,  # P4: NLP Sentiment
                            # QMJ Components
                            'roe': qmj_components.get('roe', 0.0),
                            'gross_margin': qmj_components.get('gross_margin', 0.0),
                            'debt_to_equity': qmj_components.get('debt_to_equity', 0.0)
                        }
                        
                        return result
                    except Exception as e:
                        logger.error(f"Error processing {sec.ticker}: {e}")
                        return None

            tasks = [process_ticker(sec) for sec in chunk_securities]
            chunk_results = await asyncio.gather(*tasks)
            
            # Add valid results to main list
            for r in chunk_results:
                if r is not None:
                    data.append(r)
            
            # Small delay between chunks
            await asyncio.sleep(1)
                
        if not data:
            logger.warning("No sufficient data for ranking.")
            return
            
        df = pd.DataFrame(data)
        
        # Fill NaNs
        df.fillna(0.0, inplace=True)
        
        # --- Factor Pipeline ---
        # Winsorize -> Z-Score -> Neutralize -> Composite
        df = FactorPipeline.process_factors(df, sector_col='sector')
        
        # --- Tier-1: Dynamic Factor Weighting based on Regime ---
        # Get regime-adaptive weights
        weights = DynamicFactorWeights.get_weights(self.current_regime)
        logger.info(f"📊 Using {self.current_regime} regime weights: {weights}")
        
        # Note: process_factors creates 'z_{factor}' and 'z_{factor}_neutral'
        # We use neutralized scores for best results
        
        df['score'] = (
            weights['vsm'] * df.get('z_volatility_scaled_momentum_neutral', 0) +
            weights['bab'] * df.get('z_betting_against_beta_neutral', 0) +
            weights['qmj'] * df.get('quality', 0) +  # Quality is already a composite Z-score from pipeline
            weights['upside'] * df.get('z_upside_neutral', 0) +
            weights.get('pead', 0) * df.get('z_pead_neutral', df.get('pead', 0)) +  # P3: PEAD
            weights.get('sentiment', 0) * df.get('z_sentiment_neutral', df.get('sentiment', 0))  # P4: Sentiment
        ).fillna(0.0)
        
        # Rank
        df['rank'] = df['score'].rank(ascending=False, method='first')
        df = df.sort_values('rank')
        
        # Store
        self._store_signals(df, ranking_date)
        
        return df.head(20)

    def _store_signals(self, df: pd.DataFrame, ranking_date: date):
        # Clear existing for this date/model
        self.db.query(ModelSignals).filter(
            ModelSignals.date == ranking_date,
            ModelSignals.model_name == 'ranking_v3'  # Version 3 with Tier-1 upgrades
        ).delete()
        
        signals = []
        for _, row in df.iterrows():
            meta = {
                'vsm': row.get('volatility_scaled_momentum', 0),
                'bab': row.get('betting_against_beta', 0),
                'qmj': row.get('quality', 0),
                'upside': row.get('upside', 0),
                'pead': row.get('pead', 0),  # P3: PEAD
                'sentiment': row.get('sentiment', 0),  # P4: Sentiment
                'z_vsm': row.get('z_volatility_scaled_momentum_neutral', 0),
                'z_bab': row.get('z_betting_against_beta_neutral', 0),
                'z_upside': row.get('z_upside_neutral', 0),
                'z_pead': row.get('z_pead_neutral', row.get('pead', 0)),  # P3: PEAD z-score
                'z_sentiment': row.get('z_sentiment_neutral', row.get('sentiment', 0)),  # P4: Sentiment z-score
                # Tier-1: Regime info
                'regime': self.current_regime,
                'weights_used': DynamicFactorWeights.get_weights(self.current_regime)
            }
            
            signals.append(ModelSignals(
                sid=int(row['sid']),
                date=ranking_date,
                model_name='ranking_v3',  # Version 3 with Tier-1 upgrades
                score=row['score'],
                rank=int(row['rank']),
                metadata_json=json.dumps(meta)
            ))
            
        self.db.bulk_save_objects(signals)
        self.db.commit()
        logger.info(f"Stored {len(signals)} ranking_v3 signals (Regime: {self.current_regime}).")

