r"""
Standalone AUC Unit Tests for Deepfake Detection Pipeline
==========================================================
Tests the Wilcoxon-Mann-Whitney AUC implementation in evaluate_dataset.py.
Run: .\venv\Scripts\python.exe -m pytest tests/test_auc_unit.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from evaluate_dataset import calculate_classification_metrics


def compute_auc_only(y_true, y_scores):
    """Helper: run through calculate_classification_metrics to extract AUC."""
    y_pred = [1 if s >= 0.50 else 0 for s in y_scores]
    m = calculate_classification_metrics(y_true, y_pred, y_scores)
    return m['roc_auc']


class TestAUCImplementation:
    """Verify AUC uses correct convention: y_true=1=Fake, score=fake_score."""

    def test_perfect_classifier_auc_is_1(self):
        """Perfect classifier: fake videos always score higher than real."""
        y_true = [0, 0, 1, 1]      # 0=Real, 1=Fake
        y_scores = [0.2, 0.3, 0.7, 0.8]  # fake_probability (higher = more fake)
        print(f"\n  labels:  {y_true}")
        print(f"  scores:  {y_scores}")
        auc = compute_auc_only(y_true, y_scores)
        print(f"  AUC:     {auc}")
        assert abs(auc - 1.0) < 0.01, f"Expected AUC=1.0, got {auc}"

    def test_inverted_classifier_auc_is_0(self):
        """Inverted classifier: fake videos always score lower than real."""
        y_true = [0, 0, 1, 1]
        y_scores = [0.7, 0.8, 0.2, 0.3]
        print(f"\n  labels:  {y_true}")
        print(f"  scores:  {y_scores}")
        auc = compute_auc_only(y_true, y_scores)
        print(f"  AUC:     {auc}")
        assert abs(auc - 0.0) < 0.01, f"Expected AUC=0.0, got {auc}"

    def test_random_classifier_auc_is_0_5(self):
        """Random classifier: mixed ranking produces AUC~0.5."""
        y_true = [0, 1, 0, 1]
        y_scores = [0.4, 0.4, 0.6, 0.6]
        print(f"\n  labels:  {y_true}")
        print(f"  scores:  {y_scores}")
        auc = compute_auc_only(y_true, y_scores)
        print(f"  AUC:     {auc}")
        assert abs(auc - 0.5) < 0.01, f"Expected AUC=0.5 for tied scores, got {auc}"

    def test_single_pair_correct(self):
        """Single real + single fake, correct ordering."""
        y_true = [0, 1]
        y_scores = [0.4, 0.6]
        print(f"\n  labels:  {y_true}")
        print(f"  scores:  {y_scores}")
        auc = compute_auc_only(y_true, y_scores)
        print(f"  AUC:     {auc}")
        assert abs(auc - 1.0) < 0.01, f"Expected AUC=1.0, got {auc}"

    def test_single_pair_inverted(self):
        """Single real + single fake, inverted ordering."""
        y_true = [0, 1]
        y_scores = [0.6, 0.4]
        print(f"\n  labels:  {y_true}")
        print(f"  scores:  {y_scores}")
        auc = compute_auc_only(y_true, y_scores)
        print(f"  AUC:     {auc}")
        assert abs(auc - 0.0) < 0.01, f"Expected AUC=0.0, got {auc}"

    def test_exp_c_scores_produce_auc_1(self):
        """Verify Exp C (QMC-FD only) actual scores reproduce AUC=1.0."""
        # Actual scores from evaluation: real_02, real_03, fake_02, fake_03
        y_true = [0, 0, 1, 1]
        y_scores = [0.3353, 0.3736, 0.3991, 0.4347]
        print(f"\n  Exp C actual data:")
        print(f"  labels:  {y_true}   (0=real, 1=fake)")
        print(f"  scores:  {y_scores}  (QMC-FD fake scores)")
        auc = compute_auc_only(y_true, y_scores)
        print(f"  AUC:     {auc}")
        assert abs(auc - 1.0) < 0.01, f"Exp C AUC should be 1.0 (correct ranking), got {auc}"

    def test_exp_d_scores_produce_auc_0(self):
        """Verify Exp D (Full System) actual scores reproduce AUC=0.0 (inverted ranking)."""
        # Actual scores from evaluation: real_02, real_03, fake_02, fake_03
        y_true = [0, 0, 1, 1]
        y_scores = [0.4487, 0.6318, 0.3275, 0.3753]
        print(f"\n  Exp D actual data:")
        print(f"  labels:  {y_true}   (0=real, 1=fake)")
        print(f"  scores:  {y_scores}  (Full system fake scores)")
        auc = compute_auc_only(y_true, y_scores)
        print(f"  AUC:     {auc}")
        assert abs(auc - 0.0) < 0.01, (
            f"Exp D AUC should be 0.0 (fakes rank lower than reals due to ViT domain gap), got {auc}"
        )

    def test_auc_uses_scores_not_hard_labels(self):
        """AUC must use continuous scores, not hard predicted labels."""
        # Scenario: hard labels would give 50% correct,
        # but continuous scores rank perfectly
        y_true = [0, 0, 1, 1]
        y_scores = [0.3, 0.35, 0.65, 0.7]   # Correct ranking
        y_pred = [0, 1, 0, 1]                 # 50% accuracy (mixed)
        m = calculate_classification_metrics(y_true, y_pred, y_scores)
        print(f"\n  AUC from continuous scores: {m['roc_auc']}")
        print(f"  Accuracy from hard labels:  {m['accuracy']}")
        assert abs(m['roc_auc'] - 1.0) < 0.01, "AUC should use scores, not hard labels"

    def test_auc_not_using_real_probability(self):
        """AUC must use fake_probability, NOT real_probability (which would invert it)."""
        y_true = [0, 0, 1, 1]
        fake_scores = [0.2, 0.3, 0.7, 0.8]
        real_scores = [1.0 - s for s in fake_scores]  # [0.8, 0.7, 0.3, 0.2]
        
        auc_with_fake = compute_auc_only(y_true, fake_scores)
        auc_with_real = compute_auc_only(y_true, real_scores)
        print(f"\n  AUC with fake_scores (correct): {auc_with_fake}")
        print(f"  AUC with real_scores (wrong):   {auc_with_real}")
        
        assert abs(auc_with_fake - 1.0) < 0.01, "AUC with fake_scores should be 1.0"
        assert abs(auc_with_real - 0.0) < 0.01, "AUC with real_scores should be 0.0 (inverted)"
