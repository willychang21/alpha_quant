import time
import logging
import signal
import sys
from .broker import get_broker
from .tasks import JobPayload
from .handlers import execute_task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QuantWorker")

class Worker:
    def __init__(self, queues=['quant_tasks_default']):
        self.broker = get_broker()
        self.queues = queues
        self.running = True
        
        # Handle signals
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def shutdown(self, signum, frame):
        logger.info("Received shutdown signal. Stopping worker...")
        self.running = False

    def run(self):
        logger.info(f"Worker started. Listening on queues: {self.queues}")
        
        while self.running:
            try:
                # Round-robin poll queues
                for q in self.queues:
                    job_data = self.broker.dequeue_job(q, timeout=1)
                    
                    if job_data:
                        job = JobPayload(**job_data)
                        logger.info(f"Processing job {job.job_id} ({job.task_type})")
                        
                        start_time = time.time()
                        result = execute_task(job)
                        duration = time.time() - start_time
                        
                        logger.info(f"Job {job.job_id} finished in {duration:.2f}s. Result: {result}")
                        # TODO: Store result in Redis/DB
                        
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(1)
                
        logger.info("Worker stopped.")

if __name__ == "__main__":
    worker = Worker()
    worker.run()
