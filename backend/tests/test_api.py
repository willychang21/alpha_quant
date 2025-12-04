import requests
import time

BASE_URL = "http://localhost:8000/api/v1"

def test_api():
    print("Testing API...")
    
    # Wait for server to start
    for i in range(10):
        try:
            requests.get(f"{BASE_URL}/portfolios")
            break
        except:
            time.sleep(1)
            
    # 1. Get Portfolios
    try:
        r = requests.get(f"{BASE_URL}/portfolios")
        assert r.status_code == 200
        portfolios = r.json()
        print(f"Portfolios: {len(portfolios)}")
        
        if not portfolios:
            print("No portfolios found, creating one...")
            r = requests.post(f"{BASE_URL}/portfolios", params={"name": "Test Portfolio"})
            assert r.status_code == 200
            portfolios = [r.json()]
            
        pid = portfolios[0]['id']
        print(f"Using Portfolio ID: {pid}")
        
        # 2. Get Portfolio Data
        r = requests.get(f"{BASE_URL}/portfolio/{pid}")
        assert r.status_code == 200
        data = r.json()
        print("Portfolio Data fetched successfully")
        print(f"Total Value: {data['summary']['totalValue']}")
        
        # 3. Search
        r = requests.get(f"{BASE_URL}/search?q=AAPL")
        assert r.status_code == 200
        results = r.json()
        print(f"Search Results: {len(results)}")
        
        # 4. Backtest
        payload = {
            "allocation": {"AAPL": 50, "GOOG": 50},
            "initialAmount": 10000,
            "monthlyAmount": 100
        }
        r = requests.post(f"{BASE_URL}/backtest", json=payload)
        assert r.status_code == 200
        backtest = r.json()
        print(f"Backtest Lump Sum Points: {len(backtest['lumpSum'])}")
        
        print("ALL TESTS PASSED")
        
    except Exception as e:
        print(f"TEST FAILED: {e}")

if __name__ == "__main__":
    test_api()
