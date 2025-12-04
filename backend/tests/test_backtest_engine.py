import pytest
from unittest.mock import MagicMock, patch
from datetime import date, timedelta
import pandas as pd
from quant.backtest.engine import BacktestEngine
from quant.backtest.execution.slippage import FixedSlippage

@pytest.fixture
def mock_db():
    return MagicMock()

@pytest.fixture
def engine(mock_db):
    # Mock RankingEngine and PortfolioOptimizer
    with patch('quant.backtest.engine.RankingEngine') as MockRanking, \
         patch('quant.backtest.engine.PortfolioOptimizer') as MockOptimizer:
        
        eng = BacktestEngine(mock_db, initial_capital=100000.0)
        eng.ranking_engine = MockRanking.return_value
        eng.optimizer = MockOptimizer.return_value
        yield eng

def test_initialization(engine):
    assert engine.cash == 100000.0
    assert engine.holdings == {}
    assert len(engine.history) == 0

def test_execution_buy(engine):
    # Mock price
    engine._get_price = MagicMock(return_value=100.0)
    
    # Execute Buy 10 shares of AAPL
    # Price = 100. Slippage (10bps) = 0.1. Exec Price = 100.1
    # Cost = 10 * 100.1 = 1001.0
    engine._execute_order("AAPL", 10, date.today())
    
    assert engine.holdings["AAPL"] == 10
    assert engine.cash < 100000.0
    # Check rough cost
    expected_cost = 10 * 100 * (1 + 0.001) # 10bps spread
    assert engine.cash == pytest.approx(100000.0 - expected_cost, rel=1e-3)

def test_execution_sell(engine):
    engine._get_price = MagicMock(return_value=100.0)
    engine.cash = 0
    engine.holdings = {"AAPL": 10}
    
    # Execute Sell 10 shares
    # Price = 100. Slippage = 0.1. Exec Price = 99.9
    # Proceeds = 10 * 99.9 = 999.0
    engine._execute_order("AAPL", -10, date.today())
    
    assert "AAPL" not in engine.holdings
    assert engine.cash > 0
    expected_proceeds = 10 * 100 * (1 - 0.001)
    assert engine.cash == pytest.approx(expected_proceeds, rel=1e-3)

def test_rebalance(engine):
    # Mock Ranking and Optimizer
    engine.ranking_engine.run_ranking = MagicMock()
    # Use 0.45 to leave room for slippage/costs
    engine.optimizer.run_optimization = MagicMock(return_value=[("AAPL", 0.45), ("GOOG", 0.45)])
    
    # Mock Prices
    engine._get_price = MagicMock(side_effect=lambda ticker, d: 100.0 if ticker == "AAPL" else 200.0)
    
    # Run Rebalance
    engine._rebalance(date.today())
    
    # Total Equity = 100k. Target AAPL = 45k, GOOG = 45k.
    # AAPL Qty = 450. GOOG Qty = 225.
    
    assert engine.holdings["AAPL"] == pytest.approx(450, rel=0.1) 
    assert engine.holdings["GOOG"] == pytest.approx(225, rel=0.1)
    assert engine.cash > 0 # Should have ~10k left

def test_run_backtest(engine):
    # Mock Rebalance to do nothing to avoid complex logic in loop
    engine._rebalance = MagicMock()
    engine._log_state = MagicMock()
    
    start = date(2023, 1, 1)
    end = date(2023, 1, 5)
    
    history = engine.run_backtest(start, end)
    
    # Should call rebalance once (on start date)
    engine._rebalance.assert_called()
    # Should log state daily (5 days)
    assert engine._log_state.call_count >= 5
    assert isinstance(history, pd.DataFrame)
