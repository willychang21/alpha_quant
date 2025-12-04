from quant.backtest.execution.slippage import FixedSlippage, VolumeShareSlippage
from quant.backtest.execution.fill_model import LiquidityConstrainedFill

def test_execution_models():
    print("Testing Execution Models...")
    
    # 1. Fixed Slippage
    price = 100.0
    qty = 10
    model = FixedSlippage(spread_bps=10) # 0.1% spread -> 0.1 cost
    exec_price = model.calculate_price(price, qty)
    print(f"Fixed Slippage (Buy): {price} -> {exec_price}")
    assert exec_price > price
    
    # 2. Volume Share Slippage
    vol_model = VolumeShareSlippage(price_impact_coeff=0.1)
    # High participation (50% of volume) -> High impact
    exec_price_vol = vol_model.calculate_price(price, 500, volume=1000, volatility=0.02)
    print(f"Volume Impact (Buy): {price} -> {exec_price_vol}")
    assert exec_price_vol > price
    
    # 3. Fill Model
    fill_model = LiquidityConstrainedFill(max_participation=0.1)
    desired = 1000
    volume = 5000 # Max fill = 500
    filled = fill_model.get_fill_quantity(desired, volume)
    print(f"Fill Model: Desired {desired} -> Filled {filled}")
    assert filled == 500
    
    print("Test Complete")

if __name__ == "__main__":
    test_execution_models()
