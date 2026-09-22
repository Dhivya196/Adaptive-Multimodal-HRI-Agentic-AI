"""
Metrics computation utilities for the HRI evaluation framework.
"""

import numpy as np
from typing import List, Dict, Any, Tuple

def calculate_accuracy(expected: List[Any], actual: List[Any]) -> float:
    if not expected:
        return 0.0
    correct = sum(1 for e, a in zip(expected, actual) if e == a)
    return (correct / len(expected)) * 100.0

def calculate_precision_recall_f1(expected: List[str], actual: List[str], target_class: str) -> Tuple[float, float, float]:
    tp = sum(1 for e, a in zip(expected, actual) if e == target_class and a == target_class)
    fp = sum(1 for e, a in zip(expected, actual) if e != target_class and a == target_class)
    fn = sum(1 for e, a in zip(expected, actual) if e == target_class and a != target_class)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision * 100.0, recall * 100.0, f1 * 100.0

def calculate_latency_stats(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    
    return {
        "mean": np.mean(latencies),
        "median": np.median(latencies),
        "min": np.min(latencies),
        "max": np.max(latencies),
        "std": np.std(latencies)
    }

def print_table(title: str, headers: List[str], rows: List[List[Any]]):
    print("\n" + "-" * 80)
    print(title.center(80))
    print("-" * 80)
    
    col_widths = [max(len(str(item)) for item in col) for col in zip(*([headers] + rows))]
    
    header_str = " | ".join(f"{str(h):<{w}}" for h, w in zip(headers, col_widths))
    print(f"| {header_str} |")
    print("-" * 80)
    
    for row in rows:
        row_str = " | ".join(f"{str(item):<{w}}" for item, w in zip(row, col_widths))
        print(f"| {row_str} |")
    print("-" * 80)

