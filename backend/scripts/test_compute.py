from compute.job_runner import submit_job
from compute.tasks import TaskType
import time

def test_submission():
    print("Submitting Valuation Job...")
    job_id_1 = submit_job(TaskType.VALUATION, {"ticker": "AAPL", "model": "dcf_v1"})
    print(f"Submitted Job ID: {job_id_1}")

    print("Submitting Backtest Job...")
    job_id_2 = submit_job(TaskType.BACKTEST, {"strategy": "momentum", "start_date": "2023-01-01"})
    print(f"Submitted Job ID: {job_id_2}")

if __name__ == "__main__":
    test_submission()
