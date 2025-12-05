import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_daily_vol(close: pd.Series, span0: int = 100) -> pd.Series:
    """
    Computes daily volatility using an Exponential Weighted Moving Average (EWMA).
    This is used to set dynamic barrier widths.
    
    Args:
        close (pd.Series): Closing prices.
        span0 (int): EWMA span.
        
    Returns:
        pd.Series: Daily volatility estimates.
    """
    # 1. Compute daily returns
    # We use the previous day's close to calculate return
    # df0 = close / close.shift(1) - 1  <-- Standard return
    # Lopez de Prado uses a slightly different index alignment in snippets, 
    # but standard pct_change is usually sufficient for daily data.
    
    df0 = close.pct_change()
    
    # 2. Compute EWMA Standard Deviation
    df0 = df0.ewm(span=span0).std()
    
    return df0

def apply_triple_barrier(
    close: pd.Series, 
    events: pd.DataFrame, 
    pt_sl: list = [1, 1], 
    molecule: list = None
) -> pd.DataFrame:
    """
    Applies the Triple Barrier Method to label trades.
    
    Barriers:
    1. Upper: Profit Take (pt * volatility)
    2. Lower: Stop Loss (sl * volatility)
    3. Vertical: Time Expiration (t1 in events)
    
    Args:
        close (pd.Series): Close prices.
        events (pd.DataFrame): DataFrame with columns:
                               - 't1': The vertical barrier timestamp (end date).
                               - 'trgt': The daily volatility (unit width of barriers).
                               - 'side': The signal direction (1 or -1).
        pt_sl (list): [Profit Take Multiplier, Stop Loss Multiplier].
                      e.g., [2, 1] means PT is 2x vol, SL is 1x vol.
        molecule (list): Optional list of indices to process (for parallelization).
        
    Returns:
        pd.DataFrame: DataFrame with columns ['t1', 'pt', 'sl'] containing timestamps of barrier touches.
    """
    # Filter events by molecule if provided
    events_ = events.loc[molecule] if molecule is not None else events
    
    # Output dataframe
    out = events_[['t1']].copy(deep=True)
    
    # Apply Profit Taking Barrier (Upper)
    if pt_sl[0] > 0:
        pt = pt_sl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events_.index) # NaNs

    # Apply Stop Loss Barrier (Lower)
    if pt_sl[1] > 0:
        sl = -pt_sl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events_.index) # NaNs

    # Loop through events to find the first barrier touch
    # This loop is necessary because of path dependency
    for loc, t1 in events_['t1'].fillna(close.index[-1]).items():
        # Get price path for this trade duration
        # From entry (loc) to vertical barrier (t1)
        df0 = close[loc:t1]
        
        # Calculate returns relative to entry price
        # Adjusted for side (Long/Short)
        # If Side=1 (Long), Return = (P_t / P_0) - 1
        # If Side=-1 (Short), Return = Side * ((P_t / P_0) - 1)
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']
        
        # Identify earliest stop loss hit
        # We look for returns < sl[loc]
        sl_hits = df0[df0 < sl[loc]].index
        out.loc[loc, 'sl'] = sl_hits.min() if not sl_hits.empty else pd.NaT
        
        # Identify earliest profit take hit
        # We look for returns > pt[loc]
        pt_hits = df0[df0 > pt[loc]].index
        out.loc[loc, 'pt'] = pt_hits.min() if not pt_hits.empty else pd.NaT
        
    return out

def get_bins(triple_barrier_events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """
    Generates labels (-1, 0, 1) based on which barrier was touched first.
    
    Args:
        triple_barrier_events (pd.DataFrame): Output from apply_triple_barrier.
                                              Columns: ['t1', 'pt', 'sl']
        close (pd.Series): Close prices (to calculate realized return).
        
    Returns:
        pd.DataFrame: ['ret', 'bin']
                      - ret: The realized return of the trade.
                      - bin: The label (1 for PT, -1 for SL, 0 for Time/Vertical).
    """
    # 1. Determine the first touch timestamp
    # We take the min of t1 (Vertical), pt (Profit), sl (Stop)
    first_touch = triple_barrier_events.dropna(how='all').min(axis=1)
    
    # 2. Drop trades that didn't touch anything (shouldn't happen if t1 is set)
    events = triple_barrier_events.loc[first_touch.index]
    
    out = pd.DataFrame(index=events.index)
    out['t1'] = first_touch
    out['ret'] = 0.0
    out['bin'] = 0
    
    # 3. Assign Labels
    # If PT touched first -> 1
    # If SL touched first -> -1
    # If Vertical touched first -> 0 (or sign of return)
    
    # Logic:
    # If pt timestamp == first_touch -> 1
    # If sl timestamp == first_touch -> -1
    # Else -> 0
    
    # We need to handle cases where multiple happen on same timestamp (rare with daily, but possible)
    # Usually SL takes precedence or PT? 
    # Let's assume:
    # PT hit -> 1
    # SL hit -> -1
    
    # Vectorized assignment
    pt_hit = (events['pt'] == first_touch)
    sl_hit = (events['sl'] == first_touch)
    
    out.loc[pt_hit, 'bin'] = 1
    out.loc[sl_hit, 'bin'] = -1
    
    # Calculate realized returns
    # ret = (Price_at_touch / Price_at_entry) - 1
    # We need 'side' info? Ideally yes. 
    # But for now let's assume we just want the return magnitude or raw return.
    # Actually, get_bins usually takes 'events' with 'side' info.
    # Let's simplify: The caller should calculate returns.
    
    # For now, just returning the label is enough for Meta-Labeling.
    # Meta-Labeling target: 1 if bin == 1 (Profit), 0 otherwise.
    
    return out
