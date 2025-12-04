import sys
import os
import logging

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from quant.model_registry.registry import ModelRegistry
from quant.model_registry.schema import ModelMetadata, ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def register_initial_models():
    registry = ModelRegistry()
    
    # 1. DCF Model
    dcf_meta = ModelMetadata(
        model_id="val_dcf_growth_v1",
        version="1.0.0",
        type="valuation",
        author="quant_dev",
        description="Discounted Cash Flow model with sector-specific growth assumptions and Gordon Growth terminal value.",
        status="prod",
        config=ModelConfig(
            parameters={
                "projection_years": "dynamic (5 or 10)",
                "terminal_method": "gordon_growth",
                "wacc_source": "capm_bottom_up"
            },
            inputs=["revenue", "ebitda", "fcf", "net_debt", "shares"],
            outputs=["fair_value", "upside", "wacc", "growth_rate"]
        ),
        tags=["fundamental", "intrinsic_value", "cash_flow"]
    )
    registry.register(dcf_meta)
    
    # 2. DDM Model
    ddm_meta = ModelMetadata(
        model_id="val_ddm_gordon_v1",
        version="1.0.0",
        type="valuation",
        author="quant_dev",
        description="Dividend Discount Model for Financial Services. Falls back to Excess Return Model if no dividends.",
        status="prod",
        config=ModelConfig(
            parameters={
                "method": "gordon_growth",
                "fallback": "excess_return"
            },
            inputs=["dividend_rate", "roe", "payout_ratio", "book_value"],
            outputs=["fair_value", "dividend_yield", "cost_of_equity"]
        ),
        tags=["financials", "dividends", "banks"]
    )
    registry.register(ddm_meta)
    
    # 3. REIT Model
    reit_meta = ModelMetadata(
        model_id="val_reit_ffo_v1",
        version="1.0.0",
        type="valuation",
        author="quant_dev",
        description="REIT Valuation based on Price/FFO multiples, adjusted for sub-sector premiums.",
        status="prod",
        config=ModelConfig(
            parameters={
                "multiple_source": "sector_premium_map"
            },
            inputs=["net_income", "depreciation", "shares"],
            outputs=["fair_value", "ffo", "ffo_per_share"]
        ),
        tags=["real_estate", "reit", "ffo"]
    )
    registry.register(reit_meta)
    
    print("Successfully registered initial models.")

if __name__ == "__main__":
    register_initial_models()
