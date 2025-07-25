"""
Enhanced Performance Monitoring Utilities
========================================

Provides real-time performance monitoring, profiling, and optimization tracking
for the HunyuanVideo 3D application.
"""

import time
import psutil
import torch
import sqlite3
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque, defaultdict
from contextlib import contextmanager
from functools import wraps
import traceback
import gc
import json

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Complete system performance snapshot"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    gpu_memory_mb: Optional[float]
    gpu_utilization: Optional[float]
    active_threads: int
    open_files: int
    metrics: Dict[str, PerformanceMetric] = field(default_factory=dict)


class PerformanceMonitor:
    """
    Advanced performance monitoring system with optimization tracking.
    
    Features:
    - Real-time performance metrics
    - Automatic bottleneck detection
    - Before/after optimization comparison
    - Memory leak detection
    - Database query profiling
    - GPU utilization tracking
    """
    
    def __init__(self, 
                 history_size: int = 1000,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        self.history_size = history_size
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self.snapshots = deque(maxlen=history_size)
        self.alert_thresholds = alert_thresholds or {
            "cpu_percent": 90.0,
            "memory_percent": 85.0,
            "gpu_memory_percent": 90.0,
            "response_time_ms": 1000.0
        }
        self.alerts: List[Dict[str, Any]] = []
        self.optimization_benchmarks: Dict[str, Dict[str, Any]] = {}
        self._monitoring = False
        self._monitor_thread = None
        self.process = psutil.Process()
        
    @contextmanager
    def profile_operation(self, operation_name: str, metadata: Optional[Dict] = None):
        """
        Context manager for profiling specific operations.
        
        Usage:
            with monitor.profile_operation("database_query", {"query": "SELECT ..."}):
                result = db.execute(query)
        """
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            start_gpu_memory = 0
        
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            end_memory = self.process.memory_info().rss / 1024 / 1024
            memory_delta = end_memory - start_memory
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_memory_delta = end_gpu_memory - start_gpu_memory
            else:
                gpu_memory_delta = 0
            
            # Record metrics
            metric = PerformanceMetric(
                name=operation_name,
                value=duration * 1000,  # Convert to milliseconds
                unit="ms",
                metadata={
                    "memory_delta_mb": memory_delta,
                    "gpu_memory_delta_mb": gpu_memory_delta,
                    **(metadata or {})
                }
            )
            
            self.record_metric(metric)
            
            # Check for alerts
            if duration * 1000 > self.alert_thresholds.get("response_time_ms", float('inf')):
                self.add_alert(
                    "slow_operation",
                    f"Operation '{operation_name}' took {duration*1000:.1f}ms",
                    metric
                )
    
    def profile_function(self, func: Callable) -> Callable:
        """
        Decorator for automatic function profiling.
        
        Usage:
            @monitor.profile_function
            def expensive_operation():
                # ... code ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.profile_operation(func.__name__):
                return func(*args, **kwargs)
        return wrapper
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        self.metrics_history[metric.name].append(metric)
        
        # Log if it's notably slow
        if metric.unit == "ms" and metric.value > 100:
            logger.info(f"Performance: {metric.name} took {metric.value:.1f}ms")
    
    def take_snapshot(self) -> PerformanceSnapshot:
        """Take a complete system performance snapshot"""
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = self.process.memory_percent()
        
        try:
            open_files = len(self.process.open_files())
        except:
            open_files = 0
        
        try:
            threads = self.process.num_threads()
        except:
            threads = threading.active_count()
        
        # GPU metrics
        gpu_memory_mb = None
        gpu_utilization = None
        if torch.cuda.is_available():
            try:
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                # Note: gpu_utilization requires nvidia-ml-py
                # gpu_utilization = ...
            except:
                pass
        
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            gpu_memory_mb=gpu_memory_mb,
            gpu_utilization=gpu_utilization,
            active_threads=threads,
            open_files=open_files
        )
        
        self.snapshots.append(snapshot)
        self.check_alerts(snapshot)
        
        return snapshot
    
    def check_alerts(self, snapshot: PerformanceSnapshot):
        """Check if any thresholds are exceeded"""
        if snapshot.cpu_percent > self.alert_thresholds.get("cpu_percent", 100):
            self.add_alert("high_cpu", f"CPU usage: {snapshot.cpu_percent:.1f}%", snapshot)
        
        if snapshot.memory_percent > self.alert_thresholds.get("memory_percent", 100):
            self.add_alert("high_memory", f"Memory usage: {snapshot.memory_percent:.1f}%", snapshot)
        
        if snapshot.gpu_memory_mb and torch.cuda.is_available():
            total_gpu = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            gpu_percent = (snapshot.gpu_memory_mb / total_gpu) * 100
            if gpu_percent > self.alert_thresholds.get("gpu_memory_percent", 100):
                self.add_alert("high_gpu_memory", f"GPU memory: {gpu_percent:.1f}%", snapshot)
    
    def add_alert(self, alert_type: str, message: str, data: Any):
        """Add a performance alert"""
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": time.time(),
            "data": data
        }
        self.alerts.append(alert)
        logger.warning(f"Performance Alert: {message}")
    
    def start_monitoring(self, interval: float = 1.0):
        """Start background monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        
        def monitor_loop():
            while self._monitoring:
                try:
                    self.take_snapshot()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")
    
    def get_metrics_summary(self, metric_name: str, last_n: int = 100) -> Dict[str, float]:
        """Get summary statistics for a metric"""
        metrics = list(self.metrics_history[metric_name])[-last_n:]
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "latest": values[-1] if values else 0,
            "p50": self._percentile(values, 50),
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99)
        }
    
    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile"""
        if not values:
            return 0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (p / 100)
        f = int(k)
        c = k - f
        if f + 1 < len(sorted_values):
            return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
        return sorted_values[f]
    
    def detect_memory_leak(self, threshold_mb: float = 100, duration_minutes: float = 5) -> bool:
        """Detect potential memory leaks"""
        if len(self.snapshots) < 2:
            return False
        
        duration_seconds = duration_minutes * 60
        cutoff_time = time.time() - duration_seconds
        
        recent_snapshots = [s for s in self.snapshots if s.timestamp > cutoff_time]
        
        if len(recent_snapshots) < 10:
            return False
        
        # Check if memory is consistently increasing
        memory_values = [s.memory_mb for s in recent_snapshots]
        memory_increase = memory_values[-1] - memory_values[0]
        
        if memory_increase > threshold_mb:
            # Check if it's a consistent increase (not just spikes)
            increasing_count = sum(
                1 for i in range(1, len(memory_values))
                if memory_values[i] > memory_values[i-1]
            )
            
            if increasing_count > len(memory_values) * 0.7:  # 70% increasing
                self.add_alert(
                    "memory_leak",
                    f"Potential memory leak: {memory_increase:.1f}MB increase over {duration_minutes} minutes",
                    {"start_mb": memory_values[0], "end_mb": memory_values[-1]}
                )
                return True
        
        return False
    
    def benchmark_optimization(self, 
                             name: str,
                             original_func: Callable,
                             optimized_func: Callable,
                             test_args: tuple = (),
                             test_kwargs: dict = None,
                             iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark original vs optimized implementation.
        
        Returns detailed comparison metrics.
        """
        test_kwargs = test_kwargs or {}
        
        # Warmup
        for _ in range(min(10, iterations // 10)):
            original_func(*test_args, **test_kwargs)
            optimized_func(*test_args, **test_kwargs)
        
        # Benchmark original
        original_times = []
        original_memory = []
        
        gc.collect()
        for _ in range(iterations):
            start_mem = self.process.memory_info().rss
            start_time = time.perf_counter()
            
            original_func(*test_args, **test_kwargs)
            
            original_times.append(time.perf_counter() - start_time)
            original_memory.append(self.process.memory_info().rss - start_mem)
        
        # Benchmark optimized
        optimized_times = []
        optimized_memory = []
        
        gc.collect()
        for _ in range(iterations):
            start_mem = self.process.memory_info().rss
            start_time = time.perf_counter()
            
            optimized_func(*test_args, **test_kwargs)
            
            optimized_times.append(time.perf_counter() - start_time)
            optimized_memory.append(self.process.memory_info().rss - start_mem)
        
        # Calculate statistics
        original_stats = self.get_metrics_summary("_temp_", 0)
        original_stats.update({
            "mean": sum(original_times) / len(original_times),
            "min": min(original_times),
            "max": max(original_times),
            "p50": self._percentile(original_times, 50),
            "p95": self._percentile(original_times, 95),
            "p99": self._percentile(original_times, 99),
            "mean_memory": sum(original_memory) / len(original_memory)
        })
        
        optimized_stats = {
            "mean": sum(optimized_times) / len(optimized_times),
            "min": min(optimized_times),
            "max": max(optimized_times),
            "p50": self._percentile(optimized_times, 50),
            "p95": self._percentile(optimized_times, 95),
            "p99": self._percentile(optimized_times, 99),
            "mean_memory": sum(optimized_memory) / len(optimized_memory)
        }
        
        # Calculate improvements
        time_improvement = original_stats["mean"] / optimized_stats["mean"]
        memory_improvement = (
            original_stats["mean_memory"] / optimized_stats["mean_memory"]
            if optimized_stats["mean_memory"] > 0 else 1.0
        )
        
        benchmark_result = {
            "name": name,
            "iterations": iterations,
            "original": original_stats,
            "optimized": optimized_stats,
            "improvement": {
                "time_factor": time_improvement,
                "time_percent": (time_improvement - 1) * 100,
                "memory_factor": memory_improvement,
                "memory_percent": (memory_improvement - 1) * 100
            },
            "timestamp": time.time()
        }
        
        self.optimization_benchmarks[name] = benchmark_result
        
        # Log results
        logger.info(f"Optimization '{name}' benchmark results:")
        logger.info(f"  Time improvement: {time_improvement:.2f}x ({benchmark_result['improvement']['time_percent']:.1f}% faster)")
        logger.info(f"  Memory improvement: {memory_improvement:.2f}x")
        logger.info(f"  Original: {original_stats['mean']*1000:.2f}ms avg")
        logger.info(f"  Optimized: {optimized_stats['mean']*1000:.2f}ms avg")
        
        return benchmark_result
    
    def export_report(self, output_path: Path):
        """Export performance report to file"""
        report = {
            "timestamp": time.time(),
            "duration_seconds": (
                self.snapshots[-1].timestamp - self.snapshots[0].timestamp
                if len(self.snapshots) > 1 else 0
            ),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024**3,
                "gpu_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            },
            "metrics_summary": {
                name: self.get_metrics_summary(name)
                for name in self.metrics_history
            },
            "optimization_benchmarks": self.optimization_benchmarks,
            "alerts": self.alerts[-100:],  # Last 100 alerts
            "memory_leak_detected": self.detect_memory_leak()
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report exported to {output_path}")


class DatabaseProfiler:
    """
    Specialized profiler for database operations.
    
    Tracks query performance and identifies slow queries.
    """
    
    def __init__(self, db_path: Path, slow_query_threshold_ms: float = 100):
        self.db_path = db_path
        self.slow_query_threshold_ms = slow_query_threshold_ms
        self.query_stats: Dict[str, List[float]] = defaultdict(list)
        self.slow_queries: List[Dict[str, Any]] = []
        
    @contextmanager
    def profile_query(self, query: str, params: Optional[tuple] = None):
        """Profile a database query"""
        start_time = time.perf_counter()
        
        # Normalize query for grouping
        normalized_query = self._normalize_query(query)
        
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.query_stats[normalized_query].append(duration_ms)
            
            if duration_ms > self.slow_query_threshold_ms:
                self.slow_queries.append({
                    "query": query,
                    "params": params,
                    "duration_ms": duration_ms,
                    "timestamp": time.time()
                })
                logger.warning(f"Slow query ({duration_ms:.1f}ms): {query[:100]}...")
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for statistics grouping"""
        # Remove whitespace and lowercase
        normalized = " ".join(query.split()).lower()
        
        # Replace values with placeholders
        import re
        # Replace quoted strings
        normalized = re.sub(r"'[^']*'", "'?'", normalized)
        # Replace numbers
        normalized = re.sub(r'\b\d+\b', '?', normalized)
        
        return normalized
    
    def get_query_statistics(self) -> List[Dict[str, Any]]:
        """Get statistics for all tracked queries"""
        stats = []
        
        for query, times in self.query_stats.items():
            if times:
                stats.append({
                    "query": query,
                    "count": len(times),
                    "total_ms": sum(times),
                    "mean_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "p95_ms": self._percentile(times, 95)
                })
        
        # Sort by total time descending
        stats.sort(key=lambda x: x["total_ms"], reverse=True)
        
        return stats
    
    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile"""
        if not values:
            return 0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (p / 100)
        f = int(k)
        c = k - f
        if f + 1 < len(sorted_values):
            return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
        return sorted_values[f]
    
    def analyze_index_usage(self) -> Dict[str, Any]:
        """Analyze database index usage"""
        with sqlite3.connect(self.db_path) as conn:
            # Get index list
            cursor = conn.execute("""
                SELECT name, tbl_name 
                FROM sqlite_master 
                WHERE type = 'index' AND name NOT LIKE 'sqlite_%'
            """)
            indexes = cursor.fetchall()
            
            # Get table stats
            table_stats = {}
            for index_name, table_name in indexes:
                if table_name not in table_stats:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                    table_stats[table_name] = cursor.fetchone()[0]
            
            return {
                "indexes": [{"name": idx[0], "table": idx[1]} for idx in indexes],
                "table_row_counts": table_stats,
                "recommendations": self._get_index_recommendations()
            }
    
    def _get_index_recommendations(self) -> List[str]:
        """Get index recommendations based on slow queries"""
        recommendations = []
        
        # Analyze slow queries for missing indexes
        where_columns = defaultdict(int)
        order_columns = defaultdict(int)
        
        for slow_query in self.slow_queries:
            query = slow_query["query"].lower()
            
            # Extract WHERE clause columns
            import re
            where_matches = re.findall(r'where\s+(\w+)\s*=', query)
            for col in where_matches:
                where_columns[col] += 1
            
            # Extract ORDER BY columns
            order_matches = re.findall(r'order\s+by\s+(\w+)', query)
            for col in order_matches:
                order_columns[col] += 1
        
        # Recommend indexes for frequently filtered columns
        for col, count in where_columns.items():
            if count > 10:
                recommendations.append(
                    f"Consider adding index on '{col}' (used in {count} slow queries)"
                )
        
        return recommendations


# Global monitor instance
_global_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


# Convenience decorators
def profile_operation(name: str):
    """Decorator for profiling operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            with monitor.profile_operation(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def monitor_memory_leak(threshold_mb: float = 100):
    """Decorator to monitor functions for memory leaks"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            start_memory = monitor.process.memory_info().rss / 1024 / 1024
            
            result = func(*args, **kwargs)
            
            end_memory = monitor.process.memory_info().rss / 1024 / 1024
            memory_increase = end_memory - start_memory
            
            if memory_increase > threshold_mb:
                logger.warning(
                    f"Potential memory leak in {func.__name__}: "
                    f"{memory_increase:.1f}MB increase"
                )
            
            return result
        return wrapper
    return decorator