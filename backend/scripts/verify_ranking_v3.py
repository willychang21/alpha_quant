"""
Quick test to generate RankingEngine v3 signals and verify API.
"""
import asyncio
import sys
from datetime import date

# Add backend to path
sys.path.insert(0, '/Users/zhangjunjie/Documents/DCA/backend')

from app.core.database import SessionLocal
from quant.selection.ranking import RankingEngine
from quant.data.models import ModelSignals, Security
from sqlalchemy import func
import json

async def run_ranking_test():
    """Run ranking on a few test tickers to verify v3 works."""
    db = SessionLocal()
    
    try:
        print("=" * 60)
        print("RANKING ENGINE V3 VERIFICATION")
        print("=" * 60)
        
        # 1. Run ranking engine
        print("\n1. Running RankingEngine v3...")
        engine = RankingEngine(db)
        
        # Run for today
        today = date.today()
        results = await engine.run_ranking(today)
        
        if results is not None and not results.empty:
            print(f"\n✅ Ranking complete! Generated {len(results)} signals")
            print("\nTop 10 Picks:")
            print(results[['ticker', 'score', 'rank', 'pead', 'sentiment']].head(10).to_string())
        else:
            print("❌ No results generated")
            return False
        
        # 2. Verify signals stored with v3 model name
        print("\n2. Verifying ModelSignals in database...")
        v3_signals = db.query(ModelSignals).filter(
            ModelSignals.date == today,
            ModelSignals.model_name == 'ranking_v3'
        ).limit(5).all()
        
        if v3_signals:
            print(f"✅ Found {len(v3_signals)} ranking_v3 signals in DB")
            for s in v3_signals[:3]:
                meta = json.loads(s.metadata_json) if s.metadata_json else {}
                print(f"\n  Ticker: {s.security.ticker}")
                print(f"  Score: {s.score:.4f}")
                print(f"  Rank: {s.rank}")
                print(f"  Regime: {meta.get('regime', 'N/A')}")
                print(f"  PEAD: {meta.get('pead', 'N/A')}")
                print(f"  Sentiment: {meta.get('sentiment', 'N/A')}")
        else:
            print("⚠️ No ranking_v3 signals found in DB")
        
        # 3. Check latest signals date
        latest_date = db.query(func.max(ModelSignals.date)).scalar()
        print(f"\n3. Latest signal date in DB: {latest_date}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()


if __name__ == "__main__":
    success = asyncio.run(run_ranking_test())
    
    print("\n" + "=" * 60)
    if success:
        print("✅ RANKING ENGINE V3 VERIFICATION PASSED!")
    else:
        print("❌ RANKING ENGINE V3 VERIFICATION FAILED!")
    print("=" * 60)
