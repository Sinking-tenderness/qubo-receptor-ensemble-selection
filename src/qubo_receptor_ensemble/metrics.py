"""Statistical helpers.

Consolidated from ``scripts/experimental/unidock/audit_unidock_gpu_equivalence.py``
and ``scripts/run_stage63_cross_target_rank_pair_failure_diagnosis.py``;
behavior is identical to the originals.
"""

from __future__ import annotations

import math
import statistics


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation inputs differ or are too short")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return 1.0 if left == right else 0.0
    return numerator / (left_scale * right_scale)
