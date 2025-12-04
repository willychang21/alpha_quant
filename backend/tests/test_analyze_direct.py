import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import schemas, service

def test_analyze():
    request = schemas.AnalysisRequest(
        monthlyAmount=1000,
        selectedAssets=[
            {"id": "AAPL", "type": "EQUITY", "allocation": 500, "label": "Apple"},
            {"id": "SMH", "type": "ETF", "allocation": 500, "label": "Semiconductor ETF"}
        ],
        dynamicCompositions={
            "SMH": {"NVDA": 0.2, "TSM": 0.1} # Mock composition
        }
    )
    
    response = service.analyze_portfolio(request)
    print("Matrix Holdings:")
    for h in response.matrixHoldings:
        print(f"{h.ticker}: {h.percentOfPortfolio}% (Funds: {h.fundsHolding}, Direct: {h.direct}%)")
        
    print("\nETF Headers:")
    for h in response.etfHeaders:
        print(f"{h.id}: {h.weight}%")

if __name__ == "__main__":
    test_analyze()
