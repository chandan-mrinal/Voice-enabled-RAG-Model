"""
High-Precision Sub-Millisecond Latency Tracker & Analytics Accumulator
Calculates exact P50, P70, and P100 latency distributions.
"""
import time
import numpy as np
from typing import Dict, Any, List
from .models import LatencyBreakdown

class LatencyTracker:
    def __init__(self):
        self.history: List[Dict[str, float]] = []

    def record(self, breakdown: LatencyBreakdown):
        self.history.append(breakdown.model_dump())

    def compute_percentiles(self, metric_name: str = 'total_ms') -> Dict[str, float]:
        if not self.history:
            return {'p50': 0.0, 'p70': 0.0, 'p100': 0.0}
        
        values = [h.get(metric_name, 0.0) for h in self.history]
        return {
            'p50': float(np.percentile(values, 50)),
            'p70': float(np.percentile(values, 70)),
            'p100': float(np.percentile(values, 100))
        }

    def get_full_analytics(self) -> Dict[str, Any]:
        return {
            'total_queries': len(self.history),
            'total_latency': self.compute_percentiles('total_ms'),
            'retrieval_latency': self.compute_percentiles('chunking_and_retrieval_ms'),
            'stt_latency': self.compute_percentiles('stt_ms'),
            'llm_latency': self.compute_percentiles('harness_and_llm_ms'),
            'under_200ms_percentage': (
                sum(1 for h in self.history if h.get('total_ms', 0) <= 200.0) / max(1, len(self.history))
            ) * 100.0
        }
