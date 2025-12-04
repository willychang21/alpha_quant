from sqlalchemy import func
from app.core.database import SessionLocal
from quant.data.models import ModelSignals

def check_signals():
    print("\nLatest Signals Metadata Sample (ranking_v2):")
    session = SessionLocal() 
    latest_date = session.query(func.max(ModelSignals.date)).scalar()
    signals = session.query(ModelSignals).filter(
        ModelSignals.date == latest_date,
        ModelSignals.model_name == 'ranking_v2'
    ).limit(5).all()
    
    for s in signals:
        print(f"Ticker: {s.security.ticker}, Score: {s.score}, Rank: {s.rank}")
        print(f"Metadata: {s.metadata_json}")
        
    session.close()

if __name__ == "__main__":
    check_signals()
