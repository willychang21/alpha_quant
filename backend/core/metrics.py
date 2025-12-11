"""Metrics Collector for Job Execution.

JSONL-based metrics persistence for job performance tracking.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class JobMetric:
    """Metric record for a completed job."""
    job_name: str
    status: str  # 'success' or 'failure'
    duration_seconds: float
    timestamp: datetime


class MetricsCollector:
    """JSONL-based metrics persistence.
    
    Appends job metrics to a JSONL file for later analysis.
    """
    
    def __init__(self, metrics_file: str = "logs/metrics.jsonl"):
        """Initialize metrics collector.
        
        Args:
            metrics_file: Path to JSONL file for metrics storage
        """
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    def record(self, metric: JobMetric) -> None:
        """Append metric to JSONL file.
        
        Args:
            metric: JobMetric instance to record
        """
        try:
            data = asdict(metric)
            data['timestamp'] = metric.timestamp.isoformat()
            
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(data) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
    
    def record_job(
        self, 
        job_name: str, 
        status: str, 
        duration_seconds: float
    ) -> None:
        """Convenience method to record a job metric.
        
        Args:
            job_name: Name/type of the job
            status: 'success' or 'failure'
            duration_seconds: How long the job took
        """
        metric = JobMetric(
            job_name=job_name,
            status=status,
            duration_seconds=duration_seconds,
            timestamp=datetime.now()
        )
        self.record(metric)
    
    def read_metrics(self) -> list:
        """Read all metrics from file.
        
        Returns:
            List of metric dictionaries
        """
        if not self.metrics_file.exists():
            return []
        
        metrics = []
        with open(self.metrics_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        metrics.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return metrics


# Singleton instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector(metrics_file: str = "logs/metrics.jsonl") -> MetricsCollector:
    """Get or create MetricsCollector singleton."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(metrics_file=metrics_file)
    return _metrics_collector
