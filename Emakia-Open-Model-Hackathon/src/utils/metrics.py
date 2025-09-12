"""
Metrics utilities for the Emakia Validator Agent.

This module provides functionality for collecting and monitoring application metrics.
"""

import time
import threading
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
from loguru import logger


class MetricsCollector:
    """
    Collects and manages application metrics.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize the metrics collector.
        
        Args:
            max_history: Maximum number of historical records to keep
        """
        self.max_history = max_history
        self.lock = threading.Lock()
        
        # Metrics storage
        self.validation_metrics = defaultdict(list)
        self.classification_metrics = defaultdict(list)
        self.error_metrics = defaultdict(list)
        self.performance_metrics = defaultdict(list)
        
        # Counters
        self.counters = defaultdict(int)
        
        # Timers
        self.timers = {}
        
        logger.info("Metrics collector initialized")
    
    def record_validation(self, content_type: str, result: Dict[str, Any]) -> None:
        """
        Record validation metrics.
        
        Args:
            content_type: Type of content validated
            result: Validation result
        """
        with self.lock:
            timestamp = datetime.now()
            metric = {
                'timestamp': timestamp,
                'content_type': content_type,
                'is_valid': result.get('validation', {}).get('is_valid', False),
                'confidence': result.get('validation', {}).get('confidence', 0.0),
                'model_provider': result.get('validation', {}).get('model_provider', 'unknown'),
                'violations_count': len(result.get('validation', {}).get('violations', [])),
                'processing_time': result.get('metadata', {}).get('processing_time', 0.0)
            }
            
            self.validation_metrics[content_type].append(metric)
            self._trim_history(self.validation_metrics[content_type])
            
            # Update counters
            self.counters[f'validations_{content_type}'] += 1
            if metric['is_valid']:
                self.counters[f'valid_validations_{content_type}'] += 1
            else:
                self.counters[f'invalid_validations_{content_type}'] += 1
    
    def record_classification(self, content_type: str, result: Dict[str, Any]) -> None:
        """
        Record classification metrics.
        
        Args:
            content_type: Type of content classified
            result: Classification result
        """
        with self.lock:
            timestamp = datetime.now()
            metric = {
                'timestamp': timestamp,
                'content_type': content_type,
                'category': result.get('classification', {}).get('category', 'unknown'),
                'confidence': result.get('classification', {}).get('confidence', 0.0),
                'model_provider': result.get('classification', {}).get('model_provider', 'unknown'),
                'threshold_met': result.get('classification', {}).get('threshold_met', False),
                'processing_time': result.get('metadata', {}).get('processing_time', 0.0)
            }
            
            self.classification_metrics[content_type].append(metric)
            self._trim_history(self.classification_metrics[content_type])
            
            # Update counters
            self.counters[f'classifications_{content_type}'] += 1
            self.counters[f'classifications_{metric["category"]}'] += 1
    
    def record_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None) -> None:
        """
        Record error metrics.
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Additional context
        """
        with self.lock:
            timestamp = datetime.now()
            metric = {
                'timestamp': timestamp,
                'error_type': error_type,
                'error_message': error_message,
                'context': context or {}
            }
            
            self.error_metrics[error_type].append(metric)
            self._trim_history(self.error_metrics[error_type])
            
            # Update counters
            self.counters[f'errors_{error_type}'] += 1
            self.counters['total_errors'] += 1
    
    def record_performance(self, operation: str, duration: float, metadata: Dict[str, Any] = None) -> None:
        """
        Record performance metrics.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            metadata: Additional metadata
        """
        with self.lock:
            timestamp = datetime.now()
            metric = {
                'timestamp': timestamp,
                'operation': operation,
                'duration': duration,
                'metadata': metadata or {}
            }
            
            self.performance_metrics[operation].append(metric)
            self._trim_history(self.performance_metrics[operation])
            
            # Update counters
            self.counters[f'operations_{operation}'] += 1
    
    def start_timer(self, operation: str) -> str:
        """
        Start a timer for an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Timer ID
        """
        timer_id = f"{operation}_{int(time.time() * 1000)}"
        self.timers[timer_id] = time.time()
        return timer_id
    
    def stop_timer(self, timer_id: str) -> Optional[float]:
        """
        Stop a timer and return duration.
        
        Args:
            timer_id: Timer ID
            
        Returns:
            Duration in seconds, or None if timer not found
        """
        if timer_id in self.timers:
            start_time = self.timers.pop(timer_id)
            duration = time.time() - start_time
            self.record_performance(timer_id.split('_')[0], duration)
            return duration
        return None
    
    def get_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get metrics summary for the specified time period.
        
        Args:
            hours: Number of hours to include in summary
            
        Returns:
            Metrics summary
        """
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            summary = {
                'period_hours': hours,
                'timestamp': datetime.now().isoformat(),
                'counters': dict(self.counters),
                'validation_summary': self._get_validation_summary(cutoff_time),
                'classification_summary': self._get_classification_summary(cutoff_time),
                'error_summary': self._get_error_summary(cutoff_time),
                'performance_summary': self._get_performance_summary(cutoff_time)
            }
            
            return summary
    
    def _get_validation_summary(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Get validation metrics summary."""
        summary = {}
        
        for content_type, metrics in self.validation_metrics.items():
            recent_metrics = [m for m in metrics if m['timestamp'] >= cutoff_time]
            
            if recent_metrics:
                total = len(recent_metrics)
                valid_count = sum(1 for m in recent_metrics if m['is_valid'])
                avg_confidence = sum(m['confidence'] for m in recent_metrics) / total
                avg_processing_time = sum(m['processing_time'] for m in recent_metrics) / total
                
                summary[content_type] = {
                    'total_validations': total,
                    'valid_count': valid_count,
                    'invalid_count': total - valid_count,
                    'success_rate': valid_count / total if total > 0 else 0.0,
                    'avg_confidence': avg_confidence,
                    'avg_processing_time': avg_processing_time
                }
        
        return summary
    
    def _get_classification_summary(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Get classification metrics summary."""
        summary = {}
        
        for content_type, metrics in self.classification_metrics.items():
            recent_metrics = [m for m in metrics if m['timestamp'] >= cutoff_time]
            
            if recent_metrics:
                total = len(recent_metrics)
                category_counts = defaultdict(int)
                avg_confidence = sum(m['confidence'] for m in recent_metrics) / total
                avg_processing_time = sum(m['processing_time'] for m in recent_metrics) / total
                
                for metric in recent_metrics:
                    category_counts[metric['category']] += 1
                
                summary[content_type] = {
                    'total_classifications': total,
                    'category_distribution': dict(category_counts),
                    'avg_confidence': avg_confidence,
                    'avg_processing_time': avg_processing_time
                }
        
        return summary
    
    def _get_error_summary(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Get error metrics summary."""
        summary = {}
        
        for error_type, metrics in self.error_metrics.items():
            recent_metrics = [m for m in metrics if m['timestamp'] >= cutoff_time]
            
            if recent_metrics:
                summary[error_type] = {
                    'total_errors': len(recent_metrics),
                    'recent_errors': recent_metrics[-5:]  # Last 5 errors
                }
        
        return summary
    
    def _get_performance_summary(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Get performance metrics summary."""
        summary = {}
        
        for operation, metrics in self.performance_metrics.items():
            recent_metrics = [m for m in metrics if m['timestamp'] >= cutoff_time]
            
            if recent_metrics:
                total = len(recent_metrics)
                avg_duration = sum(m['duration'] for m in recent_metrics) / total
                min_duration = min(m['duration'] for m in recent_metrics)
                max_duration = max(m['duration'] for m in recent_metrics)
                
                summary[operation] = {
                    'total_operations': total,
                    'avg_duration': avg_duration,
                    'min_duration': min_duration,
                    'max_duration': max_duration
                }
        
        return summary
    
    def _trim_history(self, metrics_list: List[Dict[str, Any]]) -> None:
        """Trim metrics history to maximum size."""
        if len(metrics_list) > self.max_history:
            metrics_list[:] = metrics_list[-self.max_history:]
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self.lock:
            self.validation_metrics.clear()
            self.classification_metrics.clear()
            self.error_metrics.clear()
            self.performance_metrics.clear()
            self.counters.clear()
            self.timers.clear()
            
        logger.info("All metrics reset")
    
    def export_metrics(self, format: str = 'json') -> str:
        """
        Export metrics in specified format.
        
        Args:
            format: Export format ('json' or 'csv')
            
        Returns:
            Exported metrics string
        """
        summary = self.get_summary()
        
        if format.lower() == 'json':
            import json
            return json.dumps(summary, indent=2, default=str)
        elif format.lower() == 'csv':
            # Simple CSV export for key metrics
            lines = ['metric,value']
            for key, value in summary['counters'].items():
                lines.append(f'{key},{value}')
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on metrics collector.
        
        Returns:
            Health status information
        """
        with self.lock:
            return {
                'status': 'healthy',
                'component': 'metrics_collector',
                'total_metrics': sum(len(metrics) for metrics in self.validation_metrics.values()) +
                               sum(len(metrics) for metrics in self.classification_metrics.values()) +
                               sum(len(metrics) for metrics in self.error_metrics.values()) +
                               sum(len(metrics) for metrics in self.performance_metrics.values()),
                'active_timers': len(self.timers),
                'counters_total': sum(self.counters.values())
            }
