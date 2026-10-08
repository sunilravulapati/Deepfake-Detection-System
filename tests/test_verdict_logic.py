"""
Unit Tests for verdict_logic.py
================================
Tests that interpretation bands work correctly and that probability is
never conflated with accuracy in any direction.

Run: .\\venv\\Scripts\\python.exe -m pytest tests/test_verdict_logic.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from verdict_logic import interpret_verdict, VERDICT_BANDS


class TestVerdictInterpretation:
    """Verify that P(fake) maps correctly to interpretation bands."""

    def test_high_fake_prob_is_highly_likely_manipulated(self):
        verd = interpret_verdict(p_fake=0.85, p_real=0.15)
        assert verd['verdict_tier'] == 'warning'
        assert 'Highly Likely' in verd['verdict_label']
        assert verd['manipulation_pct'] == 85
        assert verd['real_pct'] == 15

    def test_medium_high_fake_prob_is_likely_manipulated(self):
        verd = interpret_verdict(p_fake=0.72, p_real=0.28)
        assert verd['verdict_tier'] == 'danger'
        assert 'Likely Manipulated' in verd['verdict_label']
        assert verd['manipulation_pct'] == 72

    def test_boundary_fake_prob_is_inconclusive(self):
        verd = interpret_verdict(p_fake=0.50, p_real=0.50)
        assert verd['verdict_tier'] == 'uncertain'
        assert verd['is_inconclusive'] is True

    def test_low_fake_prob_is_likely_authentic(self):
        verd = interpret_verdict(p_fake=0.18, p_real=0.82)
        assert verd['verdict_tier'] == 'safe'
        assert 'Likely Authentic' in verd['verdict_label']
        assert verd['manipulation_pct'] == 18
        assert not verd['is_inconclusive']

    def test_zero_fake_prob(self):
        verd = interpret_verdict(p_fake=0.0, p_real=1.0)
        assert verd['verdict_tier'] == 'safe'
        assert verd['manipulation_pct'] == 0

    def test_full_fake_prob(self):
        verd = interpret_verdict(p_fake=1.0, p_real=0.0)
        assert verd['verdict_tier'] == 'warning'
        assert verd['manipulation_pct'] == 100

    def test_manipulation_pct_is_not_accuracy(self):
        """The manipulation percentage must NOT be labeled as accuracy."""
        verd = interpret_verdict(p_fake=0.82, p_real=0.18)
        # The dict must NOT have an 'accuracy' key
        assert 'accuracy' not in verd
        # The manipulation_pct must correspond to p_fake, not 1-p_fake
        assert verd['manipulation_pct'] == 82
        # Must not have a field called "confidence" that could be mistaken for system accuracy
        assert 'confidence' not in verd

    def test_48_percent_fake_is_NOT_labeled_as_48_percent_real(self):
        """
        Critical: P(fake)=0.48 must NOT display '48% Real'.
        It should show 'Manipulation probability: 48%' as inconclusive.
        """
        verd = interpret_verdict(p_fake=0.48, p_real=0.52)
        # Must be labeled as inconclusive
        assert verd['is_inconclusive'] is True
        # manipulation_pct must be 48 (the fake score), NOT 52
        assert verd['manipulation_pct'] == 48
        # real_pct must be 52
        assert verd['real_pct'] == 52
        # verdict must NOT contain 'authentic' since it's inconclusive
        assert 'Authentic' not in verd['verdict_label']

    def test_boundary_at_40_percent(self):
        """P(fake)=0.40 is exactly the inconclusive lower boundary."""
        verd = interpret_verdict(p_fake=0.40, p_real=0.60)
        assert verd['is_inconclusive'] is True
        assert verd['verdict_tier'] == 'uncertain'

    def test_boundary_at_39_percent(self):
        """P(fake)=0.39 is just below the boundary → Likely Authentic."""
        verd = interpret_verdict(p_fake=0.39, p_real=0.61)
        assert not verd['is_inconclusive']
        assert verd['verdict_tier'] == 'safe'

    def test_boundary_at_60_percent(self):
        """P(fake)=0.60 transitions from uncertain to danger."""
        verd = interpret_verdict(p_fake=0.60, p_real=0.40)
        assert verd['verdict_tier'] == 'danger'
        assert not verd['is_inconclusive']

    def test_boundary_at_80_percent(self):
        """P(fake)=0.80 transitions to highest tier."""
        verd = interpret_verdict(p_fake=0.80, p_real=0.20)
        assert verd['verdict_tier'] == 'warning'

    def test_clamping_above_1(self):
        verd = interpret_verdict(p_fake=1.5, p_real=-0.5)
        assert verd['manipulation_pct'] == 100

    def test_clamping_below_0(self):
        verd = interpret_verdict(p_fake=-0.1, p_real=1.1)
        assert verd['manipulation_pct'] == 0

    def test_result_dict_has_required_keys(self):
        """Ensure the dict always returns all required keys."""
        verd = interpret_verdict(p_fake=0.55, p_real=0.45)
        required = [
            'verdict_label', 'verdict_tier', 'manipulation_pct',
            'real_pct', 'interpretation', 'is_inconclusive',
            'p_fake', 'p_real'
        ]
        for key in required:
            assert key in verd, f"Missing key: {key}"
