import time

import numpy as np
from typing import List, Optional
from enum import Enum


class Status(str, Enum):
    DEFECT = "defect"
    GOOD = "good"


class JudgmentMethod(str, Enum):
    VOTING = "voting"
    AVERAGE = "average"
    ANY = "any"
    AT_LEAST_TWO = "at_least_two"


def _any(labels):
    return np.any(labels)


def _voting(labels):
    true_count = np.sum(labels)
    false_count = len(labels) - true_count
    return true_count >= false_count


def _average(scores, threshold):
    if len(scores) == 0:
        return Status.DEFECT  # Default status when no scores are available
    average_score = np.mean(scores)
    return average_score >= threshold


def _at_least_two(labels):
    true_count = np.sum(labels)
    if len(labels) >= 2:
        return true_count >= 2
    return True


class PredictionResult:
    def __init__(
            self,
            labels: Optional[List[bool]] = None,
            scores: Optional[List[float]] = None,
            threshold: Optional[float] = None,
            method: JudgmentMethod = JudgmentMethod.VOTING,
            timestamp: float = None,
    ):

        self.labels = np.array(labels, dtype=bool) if labels is not None else np.array([], dtype=bool)
        self.scores = np.array(scores) if scores is not None else np.array([])
        self.threshold = threshold
        self.method = method
        self.timestamp = timestamp

        if self.threshold is None and self.method == JudgmentMethod.AVERAGE:
            raise ValueError("Threshold must be defined for AVERAGE method")

        self._calculate_final_status_func = {
            JudgmentMethod.ANY: _any,
            JudgmentMethod.VOTING: _voting,
            JudgmentMethod.AVERAGE: lambda scores_: _average(scores_, self.threshold),
            JudgmentMethod.AT_LEAST_TWO: _at_least_two
        }[self.method]

        self.final_labels_status = self.calculate_final_status()
        self.to_push_at = time.perf_counter()

    def calculate_final_status(self) -> Status:
        if self.method == JudgmentMethod.AVERAGE:
            return self._calculate_final_status_func(self.scores)
        return self._calculate_final_status_func(self.labels)

    def add_prediction(self, label: bool, score: float, timestamp: float):
        if self.method == JudgmentMethod.AVERAGE:
            label = score >= self.threshold
        self.labels = np.append(self.labels, label)
        self.scores = np.append(self.scores, score)
        self.final_labels_status = self.calculate_final_status()
        self.timestamp = timestamp

    def get_final_labels(self) -> List[bool]:
        if self.threshold is None:
            return self.labels.tolist()
        else:
            return (self.scores >= self.threshold).tolist()

    def set_push_at(self, timestamp: float):
        self.to_push_at = timestamp

    def __repr__(self):
        return (f"PredictionResult(labels={self.labels.tolist()}, scores={self.scores.tolist()}, "
                f"threshold={self.threshold}, method={self.method}, final_labels_status={self.final_labels_status})")
