import numpy as np
import pandas as pd
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class VWAPExecution:
    def __init__(self, volume_profile: np.ndarray = None):
        """
        VWAP Execution Algorithm.
        
        Args:
            volume_profile (np.ndarray): Historical volume profile (percentage of daily volume per bin).
                                         If None, assumes "U-shaped" profile.
        """
        if volume_profile is None:
            # Default U-shaped profile (Morning high, Mid-day low, Close high)
            # 13 bins (30-min intervals for 6.5 hour trading day)
            self.volume_profile = np.array([
                0.15, 0.10, 0.08, 0.06, 0.05, 0.05, 0.05, # Morning to Lunch
                0.05, 0.06, 0.07, 0.08, 0.09, 0.11        # Afternoon to Close
            ])
            # Normalize
            self.volume_profile /= self.volume_profile.sum()
        else:
            self.volume_profile = volume_profile / volume_profile.sum()
            
    def generate_schedule(self, total_shares: int, start_time: str = "09:30", end_time: str = "16:00") -> pd.DataFrame:
        """
        Generates an execution schedule based on VWAP profile.
        
        Args:
            total_shares (int): Total shares to execute.
            start_time (str): Start time (HH:MM).
            end_time (str): End time (HH:MM).
            
        Returns:
            pd.DataFrame: Schedule with columns ['Time', 'Shares', 'Pct'].
        """
        n_bins = len(self.volume_profile)
        
        # Generate timestamps
        # Assuming 30 min bins for simplicity matching default profile
        times = pd.date_range(start=f"2023-01-01 {start_time}", periods=n_bins, freq="30min").time
        
        shares_per_bin = np.round(total_shares * self.volume_profile).astype(int)
        
        # Adjust rounding error
        diff = total_shares - shares_per_bin.sum()
        if diff != 0:
            # Add/subtract from last bin
            shares_per_bin[-1] += diff
            
        schedule = pd.DataFrame({
            'Time': times,
            'Shares': shares_per_bin,
            'Pct': self.volume_profile
        })
        
        return schedule
        
    def estimate_impact_cost(self, total_shares: int, daily_volume: int, volatility: float) -> float:
        """
        Estimates market impact cost using Square Root Law.
        Cost (bps) = Y * sigma * sqrt(Q / V)
        Y is a constant (typically 0.5 to 1.0).
        
        Args:
            total_shares (int): Order size (Q).
            daily_volume (int): Average Daily Volume (V).
            volatility (float): Daily volatility (sigma).
            
        Returns:
            float: Estimated impact cost in basis points.
        """
        if daily_volume == 0:
            return 0.0
            
        participation_rate = total_shares / daily_volume
        
        # Square Root Formula: Cost = 1.0 * sigma * sqrt(participation)
        # Result is in decimal (e.g. 0.001 = 10 bps)
        impact = 1.0 * volatility * np.sqrt(participation_rate)
        
        return impact * 10000 # Convert to bps
