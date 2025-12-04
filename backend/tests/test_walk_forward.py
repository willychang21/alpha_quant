import pytest
from unittest.mock import MagicMock, patch
from datetime import date
import pandas as pd
from quant.backtest.walk_forward import WalkForwardBacktester

@pytest.fixture
def mock_db():
    return MagicMock()

@pytest.fixture
def wf_tester(mock_db):
    return WalkForwardBacktester(mock_db)

def test_run_walk_forward(wf_tester):
    # Mock BacktestEngine
    with patch('quant.backtest.walk_forward.BacktestEngine') as MockEngine:
        engine_instance = MockEngine.return_value
        
        # Mock run_backtest return value
        # Create a dummy history DataFrame
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        equity = [100000 * (1.01 ** i) for i in range(10)]
        history_df = pd.DataFrame({
            'date': dates,
            'equity': equity,
            'cash': [1000] * 10,
            'holdings_count': [5] * 10
        })
        engine_instance.run_backtest.return_value = history_df
        
        start = date(2023, 1, 1)
        end = date(2023, 1, 10)
        
        result = wf_tester.run(start, end)
        
        # Verify Engine was called
        engine_instance.run_backtest.assert_called_with(start, end)
        
        # Verify Metrics
        assert 'metrics' in result
        metrics = result['metrics']
        assert metrics['total_return'] > 0
        assert metrics['sharpe_ratio'] > 0
        assert metrics['max_drawdown'] == 0 # Monotonic increase
        
        # Verify Equity Curve
        assert 'equity_curve' in result
        assert len(result['equity_curve']) == 10
