import sys
import os
import pandas as pd
import numpy as np
import unittest

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from quant.backtest.simulator import PortfolioSimulator

class TestBacktestDeterminism(unittest.TestCase):
    def setUp(self):
        # Create deterministic mock data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        np.random.seed(42) # Set seed for data generation
        
        self.history = pd.DataFrame({
            'AAPL': np.random.normal(1.001, 0.02, 100).cumprod() * 100,
            'GOOGL': np.random.normal(1.001, 0.02, 100).cumprod() * 100
        }, index=dates)
        
        self.allocation = {'AAPL': 0.6, 'GOOGL': 0.4}
        self.monthly_amount = 1000.0
        
    def test_simulation_determinism(self):
        """
        Ensure that running the simulation twice with identical inputs produces identical outputs.
        """
        # Run 1
        result1 = PortfolioSimulator.simulate_dca(
            self.history, self.allocation, self.monthly_amount
        )
        
        # Run 2
        result2 = PortfolioSimulator.simulate_dca(
            self.history, self.allocation, self.monthly_amount
        )
        
        # Check Metrics
        self.assertEqual(
            result1['metrics']['final_value_dca'], 
            result2['metrics']['final_value_dca']
        )
        self.assertEqual(
            result1['metrics']['total_return_dca'], 
            result2['metrics']['total_return_dca']
        )
        
        # Check Curve Data (Sample)
        self.assertEqual(len(result1['dca']), len(result2['dca']))
        self.assertEqual(result1['dca'][-1]['value'], result2['dca'][-1]['value'])
        
    def test_cost_sensitivity_determinism(self):
        """
        Ensure cost sensitivity analysis is deterministic.
        """
        df1 = PortfolioSimulator.run_cost_sensitivity_analysis(
            self.history, self.allocation, self.monthly_amount
        )
        
        df2 = PortfolioSimulator.run_cost_sensitivity_analysis(
            self.history, self.allocation, self.monthly_amount
        )
        
        pd.testing.assert_frame_equal(df1, df2)
        
    def test_cost_impact(self):
        """
        Verify that higher costs reduce final value.
        """
        res_0bps = PortfolioSimulator.simulate_dca(
            self.history, self.allocation, self.monthly_amount, transaction_cost_rate=0.0
        )
        
        res_10bps = PortfolioSimulator.simulate_dca(
            self.history, self.allocation, self.monthly_amount, transaction_cost_rate=0.001
        )
        
        val_0 = res_0bps['metrics']['final_value_dca']
        val_10 = res_10bps['metrics']['final_value_dca']
        
        self.assertGreater(val_0, val_10)

if __name__ == '__main__':
    unittest.main()
