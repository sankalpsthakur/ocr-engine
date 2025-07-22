"""Monitoring utilities for OCR Engine"""

import time
import psutil
import torch
from typing import Dict, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import threading
from dataclasses import dataclass, field

from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    endpoint: str
    start_time: float
    end_time: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    
    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None
    
    @property
    def success(self) -> bool:
        return self.status_code is not None and 200 <= self.status_code < 300


class MetricsCollector:
    """Collects and aggregates application metrics"""
    
    def __init__(self, window_minutes: int = 5):
        self.window_minutes = window_minutes
        self.metrics: Dict[str, list] = defaultdict(list)
        self.lock = threading.Lock()
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background thread to clean old metrics"""
        def cleanup():
            while True:
                time.sleep(60)  # Check every minute
                self._cleanup_old_metrics()
        
        thread = threading.Thread(target=cleanup, daemon=True)
        thread.start()
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than window"""
        cutoff_time = time.time() - (self.window_minutes * 60)
        
        with self.lock:
            for endpoint in list(self.metrics.keys()):
                self.metrics[endpoint] = [
                    m for m in self.metrics[endpoint]
                    if m.start_time > cutoff_time
                ]
                if not self.metrics[endpoint]:
                    del self.metrics[endpoint]
    
    def record_request(self, metric: RequestMetrics):
        """Record a completed request"""
        with self.lock:
            self.metrics[metric.endpoint].append(metric)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics"""
        stats = {}
        
        with self.lock:
            for endpoint, metrics in self.metrics.items():
                successful = [m for m in metrics if m.success]
                failed = [m for m in metrics if not m.success]
                durations = [m.duration_ms for m in successful if m.duration_ms]
                
                endpoint_stats = {
                    'total_requests': len(metrics),
                    'successful_requests': len(successful),
                    'failed_requests': len(failed),
                    'success_rate': len(successful) / len(metrics) if metrics else 0,
                    'avg_duration_ms': sum(durations) / len(durations) if durations else 0,
                    'min_duration_ms': min(durations) if durations else 0,
                    'max_duration_ms': max(durations) if durations else 0,
                }
                
                # Calculate percentiles
                if durations:
                    sorted_durations = sorted(durations)
                    endpoint_stats['p50_duration_ms'] = sorted_durations[len(sorted_durations) // 2]
                    endpoint_stats['p95_duration_ms'] = sorted_durations[int(len(sorted_durations) * 0.95)]
                    endpoint_stats['p99_duration_ms'] = sorted_durations[int(len(sorted_durations) * 0.99)]
                
                stats[endpoint] = endpoint_stats
        
        return stats


class ResourceMonitor:
    """Monitor system resources"""
    
    @staticmethod
    def get_current_resources() -> Dict[str, Any]:
        """Get current resource usage"""
        process = psutil.Process()
        
        resources = {
            'timestamp': datetime.utcnow().isoformat(),
            'cpu': {
                'percent': process.cpu_percent(interval=0.1),
                'count': psutil.cpu_count(),
            },
            'memory': {
                'rss_mb': process.memory_info().rss / 1024 / 1024,
                'vms_mb': process.memory_info().vms / 1024 / 1024,
                'percent': process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / 1024 / 1024,
                'total_mb': psutil.virtual_memory().total / 1024 / 1024,
            },
            'disk': {
                'usage_percent': psutil.disk_usage('/').percent,
                'free_gb': psutil.disk_usage('/').free / 1024 / 1024 / 1024,
            }
        }
        
        # Add GPU info if available
        if torch.cuda.is_available():
            resources['gpu'] = {
                'name': torch.cuda.get_device_name(0),
                'memory_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'memory_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'memory_total_mb': torch.cuda.get_device_properties(0).total_memory / 1024 / 1024,
                'utilization_percent': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else None,
            }
        
        return resources
    
    @staticmethod
    def check_resource_health() -> Dict[str, Any]:
        """Check if resources are healthy"""
        resources = ResourceMonitor.get_current_resources()
        
        issues = []
        
        # Check memory
        if resources['memory']['percent'] > 90:
            issues.append(f"High memory usage: {resources['memory']['percent']:.1f}%")
        
        # Check disk
        if resources['disk']['usage_percent'] > 90:
            issues.append(f"High disk usage: {resources['disk']['usage_percent']:.1f}%")
        
        # Check GPU if available
        if 'gpu' in resources:
            gpu_percent = (resources['gpu']['memory_allocated_mb'] / 
                         resources['gpu']['memory_total_mb'] * 100)
            if gpu_percent > 90:
                issues.append(f"High GPU memory usage: {gpu_percent:.1f}%")
        
        return {
            'healthy': len(issues) == 0,
            'issues': issues,
            'resources': resources
        }


# Global metrics collector
metrics_collector = MetricsCollector()


def record_request_start(endpoint: str) -> RequestMetrics:
    """Record the start of a request"""
    return RequestMetrics(endpoint=endpoint, start_time=time.time())


def record_request_end(metric: RequestMetrics, status_code: int, error: Optional[str] = None):
    """Record the end of a request"""
    metric.end_time = time.time()
    metric.status_code = status_code
    metric.error = error
    metrics_collector.record_request(metric)
    
    # Log if slow request
    if metric.duration_ms and metric.duration_ms > 5000:
        logger.warning(f"Slow request detected", extra={
            'endpoint': metric.endpoint,
            'duration_ms': metric.duration_ms,
            'status_code': status_code
        })


def get_application_metrics() -> Dict[str, Any]:
    """Get comprehensive application metrics"""
    return {
        'request_stats': metrics_collector.get_stats(),
        'resource_health': ResourceMonitor.check_resource_health(),
        'window_minutes': metrics_collector.window_minutes
    }